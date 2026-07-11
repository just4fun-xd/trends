"""Tests for the mean-reversion lab (10 bb_rsi variants).

Checks the MECHANICS of each variant (not returns — synthetic data):
  - time_stop actually caps the position duration;
  - atr_stop cuts a tail the base bb_rsi does not cut;
  - scaled/ladder give positions in [0, 1] with the right granularity;
  - short — mirror control (positions <= 0);
  - a comparison table of all 10 on a range and a trend.

Run: python -m tests.test_meanrev_lab
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine
from strategies.bollinger import bollinger_rsi
from strategies.meanrev_lab import MEANREV_LAB


def _range_bars(n=600, seed=2) -> Bars:
    """Range (OU) — the native environment of mean-reversion.

    theta=0.05 (half-life ~14 bars): the slow Wilder RSI(14) has time to
    reach oversold before reversion. With a fast OU (hl~5) RSI<30 almost
    never happens and half the variants do not trade.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = np.zeros(n)
    x[0] = 100
    for i in range(1, n):
        x[i] = x[i - 1] + 0.05 * (100 - x[i - 1]) + rng.normal(0, 2.2)
    close = pd.Series(x, index=idx)
    hl = 1.2
    high = close + hl * rng.uniform(0.3, 1.0, n)
    low = close - hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="RANGE")


def _crash_bars(n=300, seed=3) -> Bars:
    """Slow sell-off with no bounce — a falling knife (stress for long MR)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps = np.concatenate([
        rng.normal(0.0005, 0.008, 100),        # calm stretch
        rng.normal(-0.004, 0.012, n - 100),     # prolonged sell-off
    ])
    close = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
    sp = close * 0.008
    high = close + sp * rng.uniform(0.3, 1.0, n)
    low = close - sp * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="CRASH")


def test_all_variants_valid_positions() -> None:
    """All 10 give valid positions: length, no NaN, range."""
    bars = _range_bars()
    for name, fn in MEANREV_LAB.items():
        pos = fn(bars)
        assert len(pos) == len(bars), f"{name}: length"
        assert not pos.isna().any(), f"{name}: NaN in positions"
        if name == "mr_short":
            assert (pos <= 0).all(), f"{name}: short control went long"
        else:
            assert (pos >= 0).all(), f"{name}: long went short"
        assert pos.abs().max() <= 1.0 + 1e-9, f"{name}: |pos| > 1"
    print(f"  [ok] all {len(MEANREV_LAB)} variants: positions valid")


def test_time_stop_limits_holding() -> None:
    """mr_time_stop: a continuous position lasts no more than max_hold bars."""
    bars = _crash_bars()   # sell-off: the RSI exit does not fire for long
    pos = MEANREV_LAB["mr_time_stop"](bars)
    # Lengths of continuous blocks of position 1.0.
    runs, cur = [], 0
    for v in pos.values:
        if v > 0:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    max_run = max(runs) if runs else 0
    assert max_run <= 10, f"time-stop did not fire: held {max_run} bars"
    print(f"  [ok] mr_time_stop: max hold {max_run} <= 10 bars "
          f"(an expired hypothesis is closed)")


def test_atr_stop_cuts_tail() -> None:
    """mr_atr_stop loses less than the base bb_rsi in a sell-off (tail cut).

    Regression on the Tesla case (base DD -53%): the RSI exit is not a
    risk stop, the ATR stop is.
    """
    bars = _crash_bars()
    base = run_engine(bars, bollinger_rsi(bars), cost=0.0)
    stopped = run_engine(bars, MEANREV_LAB["mr_atr_stop"](bars), cost=0.0)
    assert stopped.max_drawdown >= base.max_drawdown - 1e-9, (
        f"ATR stop did not improve DD: {stopped.max_drawdown:.1%} vs "
        f"{base.max_drawdown:.1%}"
    )
    print(f"  [ok] mr_atr_stop in sell-off: DD {stopped.max_drawdown:+.1%} "
          f"vs base {base.max_drawdown:+.1%} (tail cut)")


def test_ladder_discrete_levels() -> None:
    """mr_ladder: positions strictly from {0, 0.5, 1.0} (hard cap)."""
    bars = _range_bars()
    pos = MEANREV_LAB["mr_ladder"](bars)
    levels = set(np.round(pos.unique(), 6))
    assert levels <= {0.0, 0.5, 1.0}, f"Extra levels: {levels}"
    print(f"  [ok] mr_ladder: levels {sorted(levels)} (not martingale)")


def test_scaled_continuous() -> None:
    """mr_scaled: continuous sizing in [0, 1], deeper z — larger size."""
    bars = _range_bars()
    pos = MEANREV_LAB["mr_scaled"](bars)
    active = pos[pos > 0]
    assert len(active) > 5, "scaled opened no positions on the range"
    assert active.nunique() > 3, "sizing is not continuous"
    print(f"  [ok] mr_scaled: {active.nunique()} size levels, "
          f"max {active.max():.2f}")


def test_lab_comparison_table() -> None:
    """Comparison table of the 10 variants on a range (native env)."""
    bars = _range_bars()
    print("\n  Lab on a range (synthetic, cost=2bps):")
    print(f"  {'variant':14s} {'return':>8s} {'maxDD':>8s} {'sharpe':>7s}")
    print("  " + "-" * 40)
    for name, fn in MEANREV_LAB.items():
        res = run_engine(bars, fn(bars))
        print(f"  {name:14s} {res.total_return:>+7.1%} "
              f"{res.max_drawdown:>+7.1%} {res.sharpe:>+7.2f}")
    print("  (numbers are synthetic — only for mechanics checking)")


if __name__ == "__main__":
    print("Mean-reversion lab tests:")
    test_all_variants_valid_positions()
    test_time_stop_limits_holding()
    test_atr_stop_cuts_tail()
    test_ladder_discrete_levels()
    test_scaled_continuous()
    test_lab_comparison_table()
    print("\nAll lab tests passed.")
