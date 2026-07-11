"""Regime-layer and cross-section tests — updated by the 2026-07 audit.

New regressions:
  - Router RISK PARITY: under AlwaysRange the OU leg is scaled by the
    shared vol_target, not firing raw +-1.0 (there was a x3-4 risk
    asymmetry).
  - z-score NaN robustness: a flat price segment does not break
    positions or create phantom exits (std->NaN => hold state).
  - Cross-section rebalance sparsity: weights change ~once a month,
    not every bar (top-20% boundary jitter now costs).

Kept: sanity router(AlwaysTrend)==champion, OU math, honest
HMM/Markowitz stubs.

Run: python -m tests.test_regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine, vol_target_size
from core.engine_portfolio import run_portfolio
from regime.detector import (
    AlwaysTrendDetector,
    HMMDetector,
    Regime,
    RegimeDetector,
    VolatilityRegimeDetector,
)
from regime.router import regime_router
from strategies import cross_sectional as xs
from strategies.donchian import donchian_est_macd_4step_take
from strategies.ou import ou_fit, ou_zscore


class AlwaysRangeDetector(RegimeDetector):
    """Test detector: always RANGE — isolates the router OU leg."""

    def detect(self, bars: Bars) -> pd.DataFrame:
        """P(RANGE)=1 over the whole period."""
        df = pd.DataFrame(0.0, index=bars.index,
                          columns=[r.value for r in Regime])
        df[Regime.RANGE.value] = 1.0
        return df


def _ou_series(n=500, theta=0.1, mu=100, sigma=1.5, seed=0) -> Bars:
    """Synthetic OU series — truly mean-reverting."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + rng.normal(0, sigma)
    close = pd.Series(x, index=idx)
    hl = sigma
    high = close + hl * rng.uniform(0.3, 1.0, n)
    low = close - hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="OU")


def _trend_bars(n=400, seed=1) -> Bars:
    """Trending series for the router sanity check."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps = rng.normal(0.0015, 0.008, n)
    close = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
    hl = close * 0.006
    high = close + hl * rng.uniform(0.3, 1.0, n)
    low = close - hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="TREND")


def _flat_gap_bars(n1=150, nflat=60, n2=150, seed=4) -> Bars:
    """OU series with a FLAT segment in the middle (std=0 -> z=NaN).

    2026-07 audit regression: a NaN in z used to drop the position to 0
    for a bar (phantom exit/entry with double cost), and the proposed
    fix std.replace(0, 1e-9) would give |z|~1e8 and a false signal.
    """
    rng = np.random.default_rng(seed)
    n = n1 + nflat + n2
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = np.empty(n)
    x[0] = 100.0
    for i in range(1, n1):
        x[i] = x[i - 1] + 0.12 * (100 - x[i - 1]) + rng.normal(0, 1.2)
    x[n1:n1 + nflat] = x[n1 - 1]
    for i in range(n1 + nflat, n):
        x[i] = x[i - 1] + 0.12 * (100 - x[i - 1]) + rng.normal(0, 1.2)
    close = pd.Series(x, index=idx)
    high = close + 0.5
    low = close - 0.5
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="FLATGAP")


def test_router_degenerates_to_champion() -> None:
    """Sanity: router under AlwaysTrend == champion (layout identity).

    Relies on champion == raw * vol_size — if the risk-parity refactor
    broke the identity, this test fails first.
    """
    bars = _trend_bars()
    routed = regime_router(bars, AlwaysTrendDetector())
    direct = donchian_est_macd_4step_take(bars)
    diff = (routed - direct).abs().max()
    assert diff < 1e-9, f"Router diverged from champion: {diff:.2e}"
    print(f"  [ok] Router(AlwaysTrend) == champion: diff {diff:.2e}")


def test_router_risk_parity_in_range() -> None:
    """CRITICAL (2026-07 audit): the OU leg is scaled, not raw +-1.

    Under AlwaysRange the router must emit ou_raw * vol_size: positions
    are leverage-capped and NOT equal to a bare unit of capital.
    """
    bars = _ou_series()
    routed = regime_router(bars, AlwaysRangeDetector())
    raw = ou_zscore(bars)
    expected = (raw * vol_target_size(bars, 0.15)).fillna(0.0)
    diff = (routed - expected).abs().max()
    assert diff < 1e-9, f"OU leg scaled wrong: {diff:.2e}"

    active = routed[raw != 0]
    assert (active.abs() <= 2.0 + 1e-9).all(), "Leverage 2.0 breached"
    med = active.abs().median()
    assert abs(med - 1.0) > 0.05, (
        f"Positions ~1.0 (median {med:.2f}) — vol targeting not applied"
    )
    print(f"  [ok] Risk parity: OU leg = raw*vol (median |pos| "
          f"{med:.2f}, would be 1.00 without scaling), leverage <= 2.0")


def test_ou_nan_flat_segment() -> None:
    """Regression: a flat segment yields no NaN and no phantom trades."""
    bars = _flat_gap_bars()
    pos = ou_zscore(bars)
    assert not pos.isna().any(), "NaN leaked into positions"
    # Inside a fully flat window (std->NaN) the state is held.
    inner = pos.iloc[150 + 25:150 + 60]
    assert inner.nunique() == 1, (
        f"Phantom trades on the flat segment: {inner.nunique()} "
        f"unique states"
    )
    res = run_engine(bars, pos)
    assert np.isfinite(res.equity).all(), "NaN poisoned the equity curve"
    print(f"  [ok] Flat segment: no NaN, state held "
          f"({inner.iloc[0]:+.0f} all 35 bars), equity finite")


def test_router_position_buffer() -> None:
    """Regression (Gemini audit): position hysteresis damps jitter.

    Under a jittery detector (probs jump every bar) position_buffer
    should reduce the number of final-position changes. With buffer=0
    behaviour is unchanged (identity with champion under AlwaysTrend).
    """
    bars = _trend_bars()
    # buffer=0 does not change the router==champion identity.
    routed0 = regime_router(bars, AlwaysTrendDetector(),
                            position_buffer=0.0)
    direct = donchian_est_macd_4step_take(bars)
    assert (routed0 - direct).abs().max() < 1e-9, (
        "buffer=0 broke the router==champion identity"
    )

    # Jittery detector: probs noise around 0.5 every bar.
    class JitterDetector(RegimeDetector):
        def detect(self, b: Bars) -> pd.DataFrame:
            rng = np.random.default_rng(7)
            p = 0.5 + rng.normal(0, 0.05, len(b))
            p = np.clip(p, 0, 1)
            df = pd.DataFrame(0.0, index=b.index,
                              columns=[r.value for r in Regime])
            df[Regime.TREND.value] = p
            df[Regime.RANGE.value] = 1 - p
            return df

    det = JitterDetector()
    raw = regime_router(bars, det, position_buffer=0.0)
    buf = regime_router(bars, det, position_buffer=0.05)
    ch_raw = (raw.diff().abs() > 1e-9).sum()
    ch_buf = (buf.diff().abs() > 1e-9).sum()
    assert ch_buf < ch_raw, (
        f"Buffer does not damp jitter: {ch_buf} vs {ch_raw}"
    )
    print(f"  [ok] position_buffer: position changes {ch_raw} -> {ch_buf} "
          f"under a jittery detector; buffer=0 keeps the identity")


def test_ou_adf_rejects_random_walk() -> None:
    """Regression (Gemini audit): the ADF filter rejects a random walk.

    Dickey-Fuller bias: OLS underestimates b in a finite sample, so on
    a pure random walk (true b=0) the sign of theta almost always gives
    a false mean-reversion. The ADF filter must reject these. We check
    on 50 random walks that the false-positive rate is near the nominal
    alpha (not 90%+ as it would be without the filter).
    """
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        print("  [skip] statsmodels not installed — ADF filter skipped")
        return
    false_pos = 0
    for s in range(50):
        rng = np.random.default_rng(s + 500)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
        if ou_fit(rw)["well_defined"]:
            false_pos += 1
    rate = false_pos / 50
    assert rate < 0.20, (
        f"ADF lets through {rate:.0%} of random walks — filter broken"
    )
    # And a real OU still passes.
    rng = np.random.default_rng(0)
    x = np.zeros(500)
    x[0] = 100
    for i in range(1, 500):
        x[i] = x[i - 1] + 0.1 * (100 - x[i - 1]) + rng.normal(0, 1.5)
    fit = ou_fit(pd.Series(x))
    assert fit["well_defined"], "ADF falsely rejected a real OU"
    print(f"  [ok] ADF filter: random walk falsely passes {rate:.0%} "
          f"(was ~94%), real OU passes (p={fit['adf_pvalue']:.3f})")


def test_ou_math_valid() -> None:
    """ou_fit recovers half-life on a synthetic OU series."""
    bars = _ou_series(theta=0.1)
    fit = ou_fit(bars.close)
    assert fit["well_defined"], "OU fit did not converge on a clear OU series"
    assert 3 < fit["half_life"] < 15, f"half_life={fit['half_life']:.1f}"
    print(f"  [ok] ou_fit: theta={fit['theta']:.3f}, "
          f"half_life={fit['half_life']:.1f}d (math valid)")


def test_ou_profits_in_range() -> None:
    """OU z-score is profitable on a mean-reverting series (native env)."""
    bars = _ou_series(theta=0.12)
    res = run_engine(bars, ou_zscore(bars), cost=0.0)
    assert res.total_return > 0, (
        f"OU lost on a mean-reverting series: {res.total_return:.1%}"
    )
    print(f"  [ok] OU on a range: ret={res.total_return:+.1%} "
          f"(native env — RANGE)")


def test_hmm_stub_raises() -> None:
    """HMMDetector honestly raises NotImplementedError (not faked)."""
    bars = _trend_bars()
    try:
        HMMDetector().detect(bars)
        raise AssertionError("HMM stub should have raised")
    except NotImplementedError as e:
        assert "track 2.2" in str(e)
    print("  [ok] HMMDetector — honest stub (NotImplementedError)")


def test_markowitz_stub_raises() -> None:
    """markowitz_momentum honestly raises NotImplementedError."""
    prices = pd.DataFrame(np.random.rand(100, 5).cumsum(axis=0) + 100)
    try:
        xs.markowitz_momentum(prices)
        raise AssertionError("Markowitz stub should have raised")
    except NotImplementedError as e:
        assert "track 1.2" in str(e)
    print("  [ok] markowitz_momentum — honest stub")


def test_vol_regime_detector() -> None:
    """VolatilityRegimeDetector gives a valid regime distribution."""
    bars = _trend_bars()
    probs = VolatilityRegimeDetector().detect(bars)
    row_sums = probs.sum(axis=1)
    valid = row_sums[row_sums > 0]
    assert np.allclose(valid, 1.0, atol=1e-6), "Probabilities do not sum to 1"
    print(f"  [ok] VolRegimeDetector: probabilities sum to 1, "
          f"columns {list(probs.columns)}")


def test_dual_momentum_monthly_rebalance() -> None:
    """Dual momentum: gross~2 (MN) and weights change ~once a month.

    Audit regression: daily re-ranking jittered the top-20% boundary and
    became costly with drift costs; now rebalance_every=21.
    """
    rng = np.random.default_rng(3)
    n, m = 300, 10
    idx = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (n, m)), axis=0)),
        index=idx, columns=[f"A{i}" for i in range(m)],
    )
    w = xs.dual_momentum(prices, market_neutral=True, abs_filter_sma=None)
    change_days = int((w.diff().abs().sum(axis=1) > 1e-12).sum())
    assert change_days < 30, (
        f"Weights change {change_days} of {n} days — not a monthly rebalance"
    )
    res = run_portfolio(prices, w, cost=0.0)
    active = res.gross.iloc[200:][res.gross.iloc[200:] > 0]
    assert len(active) > 50, "Weights did not form"
    print(f"  [ok] Dual momentum MN: gross median {active.median():.2f}, "
          f"weight changes {change_days} (monthly rebalance), "
          f"total {res.total_return:+.1%}")


if __name__ == "__main__":
    print("Regime-layer and cross-section tests (2026-07 audit):")
    test_router_degenerates_to_champion()
    test_router_risk_parity_in_range()
    test_ou_nan_flat_segment()
    test_ou_adf_rejects_random_walk()
    test_router_position_buffer()
    test_ou_math_valid()
    test_ou_profits_in_range()
    test_hmm_stub_raises()
    test_markowitz_stub_raises()
    test_vol_regime_detector()
    test_dual_momentum_monthly_rebalance()
    print("All regime-layer tests passed.")
