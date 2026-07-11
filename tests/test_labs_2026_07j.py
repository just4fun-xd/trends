"""Tests for the 2026-07j labs (meanrev_lab2, trend_lab3, crypto_aggr).

Checks MECHANICS (not returns — synthetic data):
  - contract: index matches, no NaN, position ranges are honest;
  - look-ahead: prefix stability (truncating the future does not change
    past positions) for ALL 40 strategies;
  - double-shift regression: the engine is the only shift(1) point, the
    labs do not shift the signal themselves (2026-07j fix);
  - regime behaviour: trend models spend more time in a trend than in a
    chop; gated MR stays silent in a trend.

Run: python -m pytest tests/test_labs_2026_07j.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.crypto_aggr_lab import CRYPTO_AGGR_LAB
from strategies.meanrev_lab2 import MEANREV_LAB2
from strategies.trend_lab2 import _shift01
from strategies.trend_lab3 import TREND_LAB3

ALL_NEW = {**MEANREV_LAB2, **TREND_LAB3, **CRYPTO_AGGR_LAB}


def _mk_bars(close: pd.Series, seed: int = 0) -> Bars:
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
                close=close, bars_per_year=252.0, symbol="SYN")


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


def test_contract_all_new() -> None:
    """Index, NaN, position ranges for all 40 new strategies."""
    bars = _trend_bars()
    for name, fn in ALL_NEW.items():
        pos = fn(bars)
        assert pos.index.equals(bars.index), name
        assert not pos.isna().any(), f"{name}: NaN in position"
        assert pos.max() <= 2.5 + 1e-9, f"{name}: position > 2.5"
        assert pos.min() >= -1.0 - 1e-9, f"{name}: position < -1"
        if name not in ("ca_squeeze_pop", "ca_burst", "ca_short_break"):
            assert pos.min() >= -1e-9, f"{name}: long strategy shorts"


def test_no_lookahead_prefix_stability() -> None:
    """Truncating future bars does not change past positions.

    Standard look-ahead detector: compute the position on the full
    series and on a prefix; if the strategy is honest the positions on
    the common segment match (a small tail is allowed only for pivot
    models, where confirming a swing after the fact is part of the
    mechanism — but even there the past is not rewritten beyond the
    pivot window).
    """
    bars = _trend_bars(400)
    cut = 320
    bars_cut = Bars(
        open=bars.open.iloc[:cut], high=bars.high.iloc[:cut],
        low=bars.low.iloc[:cut], close=bars.close.iloc[:cut],
        bars_per_year=252.0, symbol="SYN")
    for name, fn in ALL_NEW.items():
        full = fn(bars).iloc[:cut]
        pref = fn(bars_cut)
        pd.testing.assert_series_equal(
            full, pref, check_names=False,
            obj=f"{name}: look-ahead (prefix changed)")


def test_double_shift_removed() -> None:
    """Regression for the 2026-07j fix: lab helpers do NOT shift.

    The engine run_engine is the only shift(1) point. Before the fix
    trend_lab2/_impulse_lab/kalman/carver_mr/monday_range/ou_trend
    shifted the signal themselves -> a t+2 vs t+1 lag against donchian
    in every bootstrap comparison (a handicap for the champion).
    """
    idx = pd.bdate_range("2020-01-01", periods=5)
    sig = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0], index=idx)
    pd.testing.assert_series_equal(_shift01(sig), sig)


def test_trend_models_prefer_trend() -> None:
    """Every trend model is in the market longer in a trend than chop."""
    bt, br = _trend_bars(), _range_bars()
    for name, fn in TREND_LAB3.items():
        if name == "tr3_vr_trend":
            continue  # on iid synthetic VR=1: gate closed everywhere — ok
        in_tr = float((fn(bt) > 0).mean())
        in_rg = float((fn(br) > 0).mean())
        assert in_tr > in_rg, (
            f"{name}: in trend {in_tr:.2f} <= in chop {in_rg:.2f}")


def test_gated_mr_silent_in_trend() -> None:
    """Regime-gated MR stays almost silent in a steady trend."""
    bt = _trend_bars()
    for name in ("mr2_entropy", "mr2_vr", "mr2_percb_bw"):
        frac = float((MEANREV_LAB2[name](bt) > 0).mean())
        assert frac < 0.15, f"{name}: {frac:.2f} in market on a trend"


def test_pyramids_capped_and_exit() -> None:
    """Pyramids are capped and actually exit the position."""
    bt = _trend_bars(700)
    for name in ("ca_turbo_don", "ca_pyramid_max"):
        pos = CRYPTO_AGGR_LAB[name](bt)
        assert pos.max() <= 2.5 + 1e-9, name
        assert (pos == 0).any(), f"{name}: never exits"


def test_short_side_only_short() -> None:
    """ca_short_break never opens longs."""
    pos = CRYPTO_AGGR_LAB["ca_short_break"](_range_bars())
    assert pos.max() <= 1e-9
    assert pos.min() >= -1.0 - 1e-9
