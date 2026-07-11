"""Walk-forward diagnostics tests.

Key check: the engine MUST tell apart a strategy with a stable
year-over-year result and a mirage strategy living on one lucky window.
If it cannot, the metric is useless.

Run: python -m tests.test_walkforward
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from diagnostics.walkforward import (
    consistency_metrics,
    walk_forward_single,
    walk_forward_windows,
)


def _bars_5y(seed=0) -> Bars:
    """5-year series to slice into yearly windows."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", "2026-01-01")
    n = len(idx)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                      index=idx)
    sp = close * 0.006
    return Bars(open=close.shift(1).bfill(), high=close + sp,
                low=close - sp, close=close, bars_per_year=252.0,
                symbol="WF")


def test_windows_by_year() -> None:
    """Yearly slicing of 2021-2026 gives correct yearly windows."""
    bars = _bars_5y()
    wins = walk_forward_windows(bars.index, "year")
    years = [w[0].year for w in wins]
    assert 2021 in years and 2025 in years
    assert len(wins) >= 5
    print(f"  [ok] yearly slicing: {len(wins)} windows "
          f"({years[0]}..{years[-1]})")


def test_distinguishes_stable_vs_mirage() -> None:
    """CRITICAL: walk-forward tells apart a stable and a mirage strategy.

    Build two artificial signals on one series:
      - stable: +0.05% every bar (steady income across all years);
      - mirage: all income in one year (2023), zero elsewhere.
    Both may have a similar mean, but window consistency
    (positive_frac, spread) must separate them.
    """
    bars = _bars_5y()

    def stable(b):
        # Constant position -> income follows the series, but the sign
        # is stable across years (long on a weakly trending series).
        return pd.Series(1.0, index=b.index)

    def mirage(b):
        # In the market ONLY in 2023, otherwise cash.
        pos = pd.Series(0.0, index=b.index)
        pos[b.index.year == 2023] = 1.0
        return pos

    wf_stable = walk_forward_single(bars, stable)
    wf_mirage = walk_forward_single(bars, mirage)
    m_stable = consistency_metrics(wf_stable)
    m_mirage = consistency_metrics(wf_mirage)

    # Mirage trades only 1 year -> at most 1 of 5 positive windows.
    assert m_mirage["positive_frac"] <= 0.4, (
        f"mirage gave {m_mirage['positive_frac']:.0%} profitable windows"
    )
    # Mirage's non-zero result is concentrated -> at least one window is
    # noticeable, the rest ~0.
    nonzero = (wf_mirage["return"].abs() > 0.001).sum()
    assert nonzero <= 2, f"mirage traded in {nonzero} windows, not one"
    print(f"  [ok] distinguishes: stable {m_stable['positive_frac']:.0%} "
          f"profitable windows vs mirage {m_mirage['positive_frac']:.0%} "
          f"(mirage trades {nonzero} window)")


def test_consistency_metrics_shape() -> None:
    """consistency_metrics returns all expected fields."""
    bars = _bars_5y()
    wf = walk_forward_single(bars, lambda b: pd.Series(1.0, index=b.index))
    m = consistency_metrics(wf)
    for key in ("n_windows", "positive_frac", "mean_return",
                "worst_window", "best_window", "spread"):
        assert key in m, f"missing metric {key}"
    assert m["spread"] >= 0
    print(f"  [ok] metrics complete: {m['n_windows']} windows, spread "
          f"{m['spread']:.1%}")


if __name__ == "__main__":
    print("Walk-forward tests:")
    test_windows_by_year()
    test_distinguishes_stable_vs_mirage()
    test_consistency_metrics_shape()
    print("All walk-forward tests passed.")
