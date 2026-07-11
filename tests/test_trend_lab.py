"""Tests for the trend lab and the vol-percentile gate.

Invariants:
  - every TREND_LAB model: Bars -> position contract, index matches,
    range [0, 1] (long-only);
  - tsmom catches a synthetic trend (long position on drift);
  - chandelier: peak-trailing closes the position on a deep pullback;
  - vol_percentile_gate: 1 in a calm regime, 0 on a vol explosion of a
    "beyond the historical distribution" scale (covid test);
  - the gate does not disable the strategy during warm-up (=1 until the
    window is filled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.overlays import vol_percentile_gate, with_vol_gate
from strategies.trend_lab import TREND_LAB, tsmom


def _bars_from_close(close: pd.Series, symbol: str = "SYN") -> Bars:
    return Bars(open=close.shift(1).fillna(close.iloc[0]),
                high=close * 1.005, low=close * 0.995, close=close,
                bars_per_year=252.0, symbol=symbol)


def _trend_bars(n: int = 900, seed: int = 5) -> Bars:
    rng = np.random.default_rng(seed)
    rets = 0.0008 + 0.01 * rng.standard_normal(n)
    close = pd.Series(
        100.0 * np.cumprod(1.0 + rets),
        index=pd.date_range("2021-01-04", periods=n, freq="B"),
    )
    return _bars_from_close(close)


def test_trend_lab_contract_and_bounds():
    bars = _trend_bars()
    for name, fn in TREND_LAB.items():
        pos = fn(bars)
        assert pos.index.equals(bars.index), name
        # NaN on warm-up/gaps is ALLOWED (the engine shifts and
        # fillna(0) skips such bars — honester than a false 0; tsmom
        # fix 10.07.26). Bounds are checked on valid values.
        valid = pos.dropna()
        assert (valid >= -1e-9).all(), name
        assert (valid <= 1.0 + 1e-9).all(), name
        # but the tail (after warm-up) must not be all NaN
        assert pos.iloc[300:].notna().any(), f"{name}: tail all NaN"


def test_tsmom_long_on_drift():
    bars = _trend_bars(seed=9)
    pos = tsmom(bars)
    # On a steady positive drift TSMOM is almost always long after the
    # 252-bar warm-up.
    assert pos.iloc[300:].mean() > 0.8


def test_chandelier_exits_on_crash():
    n = 600
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    # Drift 0.8%/bar > synthetic high wick (+0.5%), otherwise close
    # would never break yesterday's rolling-max of highs.
    up = 100.0 * (1.008 ** np.arange(400))
    crash = up[-1] * (0.97 ** np.arange(1, 201))
    close = pd.Series(np.concatenate([up, crash]), index=idx)
    bars = _bars_from_close(close)
    pos = TREND_LAB["chandelier"](bars)
    assert pos.iloc[350] == 1.0          # in trend — in position
    assert pos.iloc[-50:].sum() == 0.0   # after crash — exited


def test_vol_gate_blocks_covid_style_explosion():
    n = 900
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    rng = np.random.default_rng(3)
    rets = 0.0003 + 0.008 * rng.standard_normal(n)
    # "Covid": 40 bars with vol many times the whole history.
    rets[700:740] = 0.10 * rng.standard_normal(40)
    close = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)
    bars = _bars_from_close(close)
    gate = vol_percentile_gate(bars)
    assert set(np.unique(gate.values)).issubset({0.0, 1.0})
    # Before the explosion (after warm-up) — open, during it — closed.
    assert gate.iloc[600:695].mean() > 0.9
    assert gate.iloc[715:745].mean() < 0.2


def test_with_vol_gate_wraps_contract():
    bars = _trend_bars(seed=17)

    def always_long(b):
        return pd.Series(1.0, index=b.index)

    gated = with_vol_gate(always_long)
    pos = gated(bars)
    assert pos.index.equals(bars.index)
    assert set(np.unique(pos.values)).issubset({0.0, 1.0})


def test_gate_warmup_is_open():
    bars = _trend_bars(seed=21, n=300)  # shorter than rank_window
    gate = vol_percentile_gate(bars, rank_window=500)
    # min_periods=250 not reached at the start — gate is open.
    assert gate.iloc[:100].min() == 1.0


def test_kama_not_poisoned_by_warmup_nan():
    """Regression: a NaN in sc at the warm-up boundary used to poison
    the KAMA recursion forever (position a permanent 0). After the fix,
    on a drifting series KAMA should spend a noticeable share of time
    long."""
    bars = _trend_bars(seed=0, n=500)
    pos = TREND_LAB["kama"](bars)
    assert pos.iloc[50:].mean() > 0.2
