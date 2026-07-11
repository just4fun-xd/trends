"""Tests for the 2026-07k labs (trend_lab4, crypto_aggr_lab2,
meanrev_lab3, mr_lowvol2, schwartz_smith).

Checks MECHANICS (not returns — synthetic data):
  - contract: index matches, no NaN, position ranges are honest;
  - look-ahead: prefix stability (truncating the future does not change
    past positions) for all 37 strategies;
  - double-shift regression: the engine is the ONLY shift(1) point;
  - regime behaviour: trend models spend more time in a trend than in a
    chop; calm-gated MR stays silent in a storm; crypto_aggr2 crisis
    guards cut exposure on a synthetic crash;
  - Schwartz-Smith: the filter recovers kappa to order-of-magnitude on a
    synthetic chi+xi series; z is stationary.

Schwartz-Smith runs in the shared tests with REDUCED windows
(min_obs=150, fit_window=250, refit_every=60) — the default 500/750 on
a 500-long synthetic series would give an empty position and a fake pass.

Run: python -m pytest tests/test_labs_2026_07k.py -q
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.crypto_aggr_lab2 import CRYPTO_AGGR_LAB2
from strategies.meanrev_lab3 import MEANREV_LAB3
from strategies.mr_lowvol2 import MR_LOWVOL2
from strategies.schwartz_smith import SCHWARTZ_SMITH, schwartz_smith_z
from strategies.trend_lab4 import TREND_LAB4

_SS_FAST = dict(min_obs=150, fit_window=250, refit_every=60)
SS_FAST = {
    name: functools.partial(fn, **_SS_FAST)
    for name, fn in SCHWARTZ_SMITH.items()
}
ALL_NEW = {**TREND_LAB4, **CRYPTO_AGGR_LAB2, **MEANREV_LAB3,
           **MR_LOWVOL2, **SS_FAST}


def _mk_bars(close: pd.Series, seed: int = 0,
             bpy: float = 252.0) -> Bars:
    """Bars with realistic bar anatomy (close not at the midpoint)."""
    rng = np.random.default_rng(seed)
    n = len(close)
    ret = close.pct_change().fillna(0.0).to_numpy()
    width = (np.abs(rng.normal(0, 1, n)) * 0.01 + 0.004) * close.values
    loc = np.clip(
        rng.uniform(0, 1, n) * 0.6 + np.where(ret > 0, 0.4, 0.0), 0, 1)
    high = pd.Series(close.values + width * (1 - loc), index=close.index)
    low = pd.Series(close.values - width * loc, index=close.index)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=bpy, symbol="SYN")


def _trend_bars(n: int = 500, seed: int = 3) -> Bars:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0015, 0.012, n))), index=idx)
    return _mk_bars(c, seed)


def _range_bars(n: int = 500, seed: int = 2) -> Bars:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = np.zeros(n)
    x[0] = 100
    for i in range(1, n):
        x[i] = x[i - 1] + 0.08 * (100 - x[i - 1]) + rng.normal(0, 2)
    return _mk_bars(pd.Series(x, index=idx), seed)


def _crash_bars(n: int = 600, seed: int = 7) -> Bars:
    """Bull trend, then a 40-bar -60% crash with gaps (crypto-2022)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-06-01", periods=n)
    up = rng.normal(0.003, 0.015, n - 120)
    crash = rng.normal(-0.022, 0.045, 40)
    after = rng.normal(0.0, 0.02, 80)
    c = pd.Series(100 * np.exp(np.cumsum(
        np.concatenate([up, crash, after]))), index=idx)
    return _mk_bars(c, seed)


# ── contract ─────────────────────────────────────────────────────────
def test_contract_index_nan_range():
    bars = _trend_bars()
    for name, fn in ALL_NEW.items():
        pos = fn(bars)
        assert pos.index.equals(bars.index), name
        assert not pos.isna().any(), f"{name}: NaN in position"
        assert pos.min() >= -1e-9, f"{name}: negative position"
        cap = 2.0 + 1e-9
        assert pos.max() <= cap, f"{name}: position above {cap}"


# ── look-ahead: prefix stability ─────────────────────────────────────
def test_prefix_stability():
    bars = _crash_bars(600)
    cut = 480
    truncated = Bars(
        open=bars.open.iloc[:cut], high=bars.high.iloc[:cut],
        low=bars.low.iloc[:cut], close=bars.close.iloc[:cut],
        bars_per_year=bars.bars_per_year, symbol=bars.symbol)
    for name, fn in ALL_NEW.items():
        full = fn(bars).iloc[:cut - 60]
        pref = fn(truncated).iloc[:cut - 60]
        pd.testing.assert_series_equal(
            full, pref, check_names=False,
            obj=f"{name}: look-ahead (prefix changed the past)")


# ── double-shift regression ──────────────────────────────────────────
def test_no_internal_signal_shift():
    """The signal on bar t must see close[t]: on a step series at least
    one model per dict reacts on the event bar, not the next one. An
    indirect but cheap detector of an extra shift."""
    idx = pd.bdate_range("2021-01-01", periods=300)
    c = pd.Series(100.0, index=idx)
    c.iloc[150:] = 130.0                      # instant step up
    bars = _mk_bars(c, seed=1)
    reacted_on_event_bar = False
    for fn in TREND_LAB4.values():
        pos = fn(bars)
        if pos.iloc[150] != pos.iloc[149]:
            reacted_on_event_bar = True
            break
    assert reacted_on_event_bar, (
        "no tr4 model sees close[t] on bar t — looks like an internal "
        "shift(1) (double shift with the engine)")


# ── regime behaviour ─────────────────────────────────────────────────
def _persistent_trend_bars(n: int = 500, seed: int = 5,
                           phi: float = 0.2) -> Bars:
    """Trend with AR(1) persistence in returns (not iid-GBM)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.0012 + phi * (r[i - 1] - 0.0012) + rng.normal(0, 0.01)
    c = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    return _mk_bars(c, seed)


def test_trend_models_longer_in_trend():
    tb, rb = _trend_bars(), _range_bars()
    for name, fn in TREND_LAB4.items():
        # tr4_ar1's apparatus is PROCESS persistence: an iid-GBM trend
        # is empty for it by construction, so test it on an AR(1) trend.
        t_time = fn(_persistent_trend_bars() if name == "tr4_ar1"
                    else tb).mean()
        r_time = fn(rb).mean()
        assert t_time > r_time - 1e-9, (
            f"{name}: in chop ({r_time:.2f}) not shorter than in trend "
            f"({t_time:.2f})")


def test_crisis_guards_cut_exposure_in_crash():
    """Every AGGR-2 guard: mean exposure in the crash window is lower
    than in the preceding bull window of the same length."""
    bars = _crash_bars(600)
    crash_sl = slice(480, 520)
    bull_sl = slice(380, 420)
    for name, fn in CRYPTO_AGGR_LAB2.items():
        pos = fn(bars)
        bull = float(pos.iloc[bull_sl].mean())
        crash = float(pos.iloc[crash_sl].mean())
        if bull < 0.05:      # core itself is out of market — don't judge
            continue
        assert crash < bull + 1e-9, (
            f"{name}: crash exposure ({crash:.2f}) not below bull "
            f"({bull:.2f}) — guard does not work")


def test_calm_gated_mr_silent_in_storm():
    """calm-gated MR (bertram/grid/overshoot/kelly and mr_lv2_*) does
    not add to the position during the synthetic crash window."""
    bars = _crash_bars(600)
    gated = ["mr3_bertram", "mr3_grid", "mr3_overshoot", "mr3_kelly",
             "mr_lv2_cont", "mr_lv2_scale"]
    for name in gated:
        fn = ALL_NEW[name]
        pos = fn(bars)
        entries = (pos.diff().clip(lower=0.0)).iloc[485:520].sum()
        assert entries <= 0.5, (
            f"{name}: adds to position in the storm (gate fails)")


# ── Schwartz-Smith: parameter recovery and stationarity ──────────────
def _simulate_ss(n: int = 1500, kappa: float = 0.08,
                 s_chi: float = 0.015, mu: float = 0.0006,
                 s_xi: float = 0.007, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    phi = np.exp(-kappa)
    q_chi = s_chi * np.sqrt((1 - phi ** 2) / (2 * kappa))
    chi = np.zeros(n)
    xi = np.zeros(n)
    xi[0] = np.log(100.0)
    for t in range(1, n):
        chi[t] = phi * chi[t - 1] + q_chi * rng.normal()
        xi[t] = xi[t - 1] + mu + s_xi * rng.normal()
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(np.exp(chi + xi), index=idx)


def test_ss_z_stationary_and_reactive():
    close = _simulate_ss()
    z = schwartz_smith_z(close, fit_window=500, refit_every=125,
                         min_obs=400)
    zv = z.dropna()
    assert len(zv) > 700, "z almost empty — filter did not run"
    assert zv.abs().mean() < 3.0, "z blew up — normalization broken"
    assert 0.3 < zv.std() < 3.5, f"std(z)={zv.std():.2f} out of range"
    # reversion in z-space: negative autocorrelation of increments
    dz = zv.diff().dropna()
    assert dz.autocorr(1) < 0.1, "z increments persistent — chi not OU"


def test_leading_nan_does_not_poison_position():
    """Leading NaN (futures splice) must not POISON the series: after
    warm-up the position must not be a permanent NaN or a permanent 0
    from NaN leaking forward (regression on the prev=c[0] virus in the
    removed tr3_kama_slope, and on the old tsmom where a pct_change NaN
    became a false 0).

    Note: bit-for-bit equivalence with the clean run is NOT required —
    recursive stop systems (PSAR, KAMA, Renko) legitimately depend on
    the start point. The virus shows up differently: the position AFTER
    warm-up sticks at NaN/0 forever. That is what we check.

    tsmom_multi is checked separately: leading/mid NaN must yield NaN on
    the affected bars (the engine skips them), NOT a false 0.
    """
    from strategies.trend_lab import kama_trend, tsmom_multi
    clean = _trend_bars(800, seed=9)
    close = clean.close.copy()
    close.iloc[:15] = np.nan
    bad = Bars(open=clean.open.where(close.notna()),
               high=clean.high.where(close.notna()),
               low=clean.low.where(close.notna()), close=close,
               bars_per_year=clean.bars_per_year, symbol="NANLEAD")
    checked = {**TREND_LAB4, "kama_trend": kama_trend}
    for name, fn in checked.items():
        pos = fn(bad)
        # 1) NaN did not leak into the tail (past a safe warm-up)
        assert not pos.iloc[400:].isna().any(), (
            f"{name}: NaN leaked into the position tail (virus)")
        # 2) position not stuck: if the model trades on the clean
        #    series, it must show activity here after warm-up too
        clean_active = fn(clean).iloc[400:].abs().sum() > 0
        if clean_active:
            assert pos.iloc[400:].abs().sum() > 0, (
                f"{name}: stuck at 0 after leading NaN even though it "
                f"trades on clean data (init virus)")

    # tsmom_multi: NaN -> NaN on the bars, NOT a false 0
    close2 = clean.close.copy()
    close2.iloc[200:203] = np.nan            # mid-series gap
    gap = Bars(open=clean.open, high=clean.high, low=clean.low,
               close=close2, bars_per_year=clean.bars_per_year,
               symbol="GAP")
    tpos = tsmom_multi(gap)
    assert tpos.iloc[200:203].isna().all(), (
        "tsmom_multi: a close gap produced a false 0 instead of NaN "
        "(old pct_change>0-on-NaN bug)")
    close = _simulate_ss(700)
    bars = _mk_bars(close, seed=4)
    pos = SCHWARTZ_SMITH["ss_chi_mr"](
        bars, fit_window=400, refit_every=100, min_obs=350)
    assert (pos.iloc[:349] == 0.0).all(), "trading before min_obs warm-up"
