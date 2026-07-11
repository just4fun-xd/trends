"""Sizer (realized/garch), ensemble and portfolio-sweep tests.

Invariants checked (synthetic, no network):
  - make_sizer: registry, bounds [0, max_leverage], NaN warm-up
    holds the applied value;
  - mr_ensemble: position in [0, 1], index matches bars;
  - trend_mr_combo: position in [0, 1], weight validation;
  - vol_sweep_basket: portfolio columns present, the portfolio DD is
    no worse (not deeper) than the worst-case per-instrument
    (diversification cannot worsen equal-weight DD vs the worst one);
  - the sweep turnover is drift-aware (matches the engine formula).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.bars import Bars
from core.engine import drift_turnover
from core.sizing import make_sizer
from diagnostics.vol_sweep import vol_sweep_basket, vol_sweep_single
from strategies.ensemble import mr_ensemble, trend_mr_combo


def _make_bars(n: int = 900, seed: int = 7, drift: float = 0.0002,
               symbol: str = "SYN") -> Bars:
    """Synthetic instrument with clustered vol (for GARCH)."""
    rng = np.random.default_rng(seed)
    # Simple vol clustering: two-regime sigma.
    regime = (np.sin(np.arange(n) / 60.0) > 0).astype(float)
    sigma = 0.008 + 0.012 * regime
    rets = drift + sigma * rng.standard_normal(n)
    close = pd.Series(
        100.0 * np.cumprod(1.0 + rets),
        index=pd.date_range("2021-01-04", periods=n, freq="B"),
    )
    high = close * 1.005
    low = close * 0.995
    return Bars(open=close.shift(1).fillna(close.iloc[0]), high=high,
                low=low, close=close, bars_per_year=252.0, symbol=symbol)


def test_make_sizer_registry_and_bounds():
    bars = _make_bars()
    for name in ("realized", "garch"):
        sizer = make_sizer(name, target_vol=0.20, max_leverage=2.0)
        mult = sizer(bars)
        assert mult.index.equals(bars.index)
        assert (mult >= 0).all()
        assert (mult <= 2.0 + 1e-12).all()
        # After warm-up the sizer must be active.
        assert float(mult.iloc[-100:].mean()) > 0
    with pytest.raises(KeyError):
        make_sizer("hmm")


def test_mr_ensemble_bounds_and_index():
    bars = _make_bars(seed=11)
    pos = mr_ensemble(bars)
    assert pos.index.equals(bars.index)
    assert (pos >= -1e-12).all()
    assert (pos <= 1.0 + 1e-12).all()
    # Granularity: an ensemble of 4 binary legs gives multiples of 0.25.
    vals = np.unique(np.round(pos.values, 6))
    assert set(vals).issubset({0.0, 0.25, 0.5, 0.75, 1.0})


def test_trend_mr_combo_bounds_and_weight_validation():
    bars = _make_bars(seed=13)
    pos = trend_mr_combo(bars, w_trend=0.5)
    assert pos.index.equals(bars.index)
    assert (pos >= -1e-12).all()
    assert (pos <= 1.0 + 1e-12).all()
    with pytest.raises(ValueError):
        trend_mr_combo(bars, w_trend=1.5)


def test_vol_sweep_portfolio_columns_and_dd_dominance():
    baskets = {f"S{i}": _make_bars(seed=100 + i) for i in range(4)}

    def sig(bars):
        # A simple always-long signal: isolates the sizer effect.
        return pd.Series(1.0, index=bars.index)

    df = vol_sweep_basket(baskets, sig, target_vols=(0.15, 0.30),
                          cost=0.0002)
    for col in ("port_return", "port_dd", "port_sharpe",
                "port_passes_dd", "worst_dd"):
        assert col in df.columns
    # Diversification: equal-weight portfolio DD no deeper than worst-case.
    assert (df["port_dd"] >= df["worst_dd"] - 1e-12).all()


def test_sweep_turnover_matches_engine_formula():
    bars = _make_bars(seed=21)

    def sig(bars_):
        return pd.Series(1.0, index=bars_.index)

    df, _ = vol_sweep_single(bars, sig, target_vols=(0.20,),
                             cost=0.0002)
    # Rebuild turnover by hand with the same engine formula.
    sizer = make_sizer("realized", target_vol=0.20, max_leverage=6.0)
    pos = sig(bars) * sizer(bars)
    prev = pos.shift(1).fillna(0.0)
    manual = float(drift_turnover(prev, bars.returns()).sum())
    years = len(bars) / bars.bars_per_year
    assert df.loc[0.20, "turnover_ann"] == pytest.approx(
        manual / years, rel=1e-9)


def test_garch_sizer_reacts_to_vol_regimes():
    """The GARCH sizer must give a smaller size in a high-vol regime
    than in a low-vol one (otherwise it is not a sizer)."""
    bars = _make_bars(seed=33, n=1200)
    mult = make_sizer("garch", target_vol=0.20)(bars)
    rv = bars.returns().rolling(30).std()
    hi = mult[rv > rv.quantile(0.8)].mean()
    lo = mult[rv < rv.quantile(0.2)].mean()
    assert hi < lo


def test_trend_ensemble_bounds_and_registry():
    from runners.run_basket import STRATEGIES
    from strategies.ensemble import trend_ensemble
    bars = _make_bars(seed=51, n=900, drift=0.0008)
    pos = trend_ensemble(bars)
    assert pos.index.equals(bars.index)
    assert (pos >= -1e-9).all() and (pos <= 1.0 + 1e-9).all()
    # On a drift the ensemble should be long a noticeable share of time.
    assert pos.iloc[300:].mean() > 0.3
    assert "trend_ens" in STRATEGIES


def test_portfolio_vol_target_scales_and_no_lookahead():
    from core.sizing import portfolio_vol_target
    rng = np.random.default_rng(9)
    # Quiet portfolio series ~3% annual vol (like a combo after parity).
    rets = pd.Series(
        0.0002 + 0.002 * rng.standard_normal(1500),
        index=pd.date_range("2019-01-02", periods=1500, freq="B"),
    )
    scaled, lev = portfolio_vol_target(rets, target_vol=0.10,
                                       max_leverage=4.0)
    raw_vol = rets.std() * np.sqrt(252)
    new_vol = scaled.iloc[100:].std() * np.sqrt(252)
    assert new_vol > raw_vol * 2          # vol actually raised
    assert abs(new_vol - 0.10) < 0.03     # and close to target
    assert (lev <= 4.0 + 1e-9).all()      # leverage cap
    # No look-ahead: leverage at bar t does not depend on rets[t]. The
    # spike is absolute (0.05 >> sigma) so the window vol surely jumps
    # and leverage drops off the cap on the NEXT bar.
    rets2 = rets.copy()
    rets2.iloc[500] = 0.05
    _, lev2 = portfolio_vol_target(rets2, target_vol=0.10)
    assert lev2.iloc[500] == lev.iloc[500]
    assert lev2.iloc[501] != lev.iloc[501]


def test_funding_rate_reduces_leveraged_return():
    """Funding cost lowers the leveraged portfolio return but does not
    affect the leverage weights (same vol-targeting implementation)."""
    from core.sizing import portfolio_vol_target
    rng = np.random.default_rng(2)
    n = 1000
    rets = pd.Series(
        0.0004 + 0.008 * rng.standard_normal(n),
        index=pd.date_range("2019-01-02", periods=n, freq="B"),
    )
    free, lev_free = portfolio_vol_target(
        rets, target_vol=0.25, max_leverage=6.0, funding_rate=0.0)
    paid, lev_paid = portfolio_vol_target(
        rets, target_vol=0.25, max_leverage=6.0, funding_rate=0.06)
    # Leverage (position weights) does not change with funding_rate.
    pd.testing.assert_series_equal(lev_free, lev_paid)
    # Return after funding is strictly lower (when leverage is active).
    active = lev_free > 1.0 + 1e-9
    assert (paid[active] < free[active]).mean() > 0.95


def test_funding_only_charges_borrowed_portion():
    """With leverage<=1 (no borrowing) funding_rate must subtract
    nothing — we pay only for (leverage-1), not the whole size."""
    from core.sizing import portfolio_vol_target
    rng = np.random.default_rng(6)
    n = 500
    # Very high vol -> target_vol is easily reached with leverage<1.
    rets = pd.Series(
        0.0 + 0.05 * rng.standard_normal(n),
        index=pd.date_range("2019-01-02", periods=n, freq="B"),
    )
    free, lev = portfolio_vol_target(
        rets, target_vol=0.05, max_leverage=6.0, funding_rate=0.0)
    paid, lev2 = portfolio_vol_target(
        rets, target_vol=0.05, max_leverage=6.0, funding_rate=0.06)
    no_borrow = lev <= 1.0 + 1e-9
    # Where leverage <=1 (no borrowing) funding must not change return.
    if no_borrow.any():
        assert np.allclose(
            free[no_borrow].values, paid[no_borrow].values, atol=1e-12)


def test_breakeven_funding_rate_bounds_and_monotonicity():
    """breakeven returns the rate separating profit/loss: just below it
    net return > 0, just above < 0."""
    from core.sizing import breakeven_funding_rate, portfolio_vol_target
    rng = np.random.default_rng(41)
    n = 1200
    rets = pd.Series(
        0.0003 + 0.006 * rng.standard_normal(n),
        index=pd.date_range("2019-01-02", periods=n, freq="B"),
    )
    be = breakeven_funding_rate(rets, target_vol=0.25, max_leverage=4.0)
    assert 0.0 < be < 0.50

    def net(fr):
        scaled, _ = portfolio_vol_target(
            rets, target_vol=0.25, max_leverage=4.0, funding_rate=fr)
        return float((1 + scaled).cumprod().iloc[-1] - 1.0)

    assert net(max(be - 0.02, 0.0)) > 0
    assert net(min(be + 0.02, 0.49)) < 0


def test_breakeven_zero_when_already_unprofitable():
    from core.sizing import breakeven_funding_rate
    n = 500
    # A purely losing series with no funding -> breakeven = 0.
    rets = pd.Series(
        -0.001, index=pd.date_range("2019-01-02", periods=n, freq="B"))
    be = breakeven_funding_rate(rets, target_vol=0.20, max_leverage=4.0)
    assert be == 0.0


def test_breakeven_returns_hi_when_always_profitable():
    from core.sizing import breakeven_funding_rate
    rng = np.random.default_rng(50)
    n = 1200
    # Very high Sharpe -> profitable even at funding=hi.
    rets = pd.Series(
        0.003 + 0.003 * rng.standard_normal(n),
        index=pd.date_range("2019-01-02", periods=n, freq="B"),
    )
    be = breakeven_funding_rate(
        rets, target_vol=0.25, max_leverage=4.0, hi=0.10)
    assert be == 0.10


def test_member_contribution_loo_and_corr():
    """LOO diagnostic: ballast (a zero member) is detected, a harmful
    member (anti-signal) gives a negative delta, the correlation matrix
    contains an ENSEMBLE column."""
    from diagnostics.member_contribution import member_contribution

    # Strong drift + fixed seeds: always-long catches the whole drift
    # (deterministically helps), always-short goes against it
    # (deterministically hurts). Momentum members are unusable here:
    # iid synthetic, momentum has no edge, its contribution is a coin flip.
    baskets = {f"S{i}": _make_bars(seed=200 + i, drift=0.0015)
               for i in range(3)}

    def good(bars):
        return pd.Series(1.0, index=bars.index)

    def zero(bars):
        return pd.Series(0.0, index=bars.index)

    def bad(bars):
        return pd.Series(-1.0, index=bars.index)

    members = {"good": good, "zero": zero, "bad": bad}
    res = member_contribution(baskets, members, sizer=None)

    assert set(res["solo"].index) == set(members)
    assert set(res["loo"].index) == set(members)
    assert "ENSEMBLE" in res["corr"].columns
    # Zero member: solo Sharpe exactly 0.
    assert res["solo"].loc["zero", "sharpe"] == 0.0
    # An anti-signal on a drift must hurt the ensemble: better without it.
    assert res["loo"].loc["bad", "delta"] < 0
    # A good member must help: worse without it.
    assert res["loo"].loc["good", "delta"] > 0


def test_mr_kelt_confirm_registered_and_bounded():
    from runners.run_basket import STRATEGIES
    from strategies.ensemble import mr_keltner_confirm
    bars = _make_bars(seed=71, n=800)
    pos = mr_keltner_confirm(bars)
    assert pos.index.equals(bars.index)
    assert (pos >= -1e-9).all() and (pos <= 1.0 + 1e-9).all()
    # A pair of two binary members -> values are multiples of 0.5.
    vals = set(np.unique(np.round(pos.values, 6)))
    assert vals.issubset({0.0, 0.5, 1.0})
    assert "mr_kelt_confirm" in STRATEGIES
