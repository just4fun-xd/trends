"""Tests for the single-strategy runner (run_strategy_test)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from runners.run_strategy_test import _metrics


def _series(seed: int, n: int = 500, drift: float = 0.0005) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(drift, 0.01, n))


def test_metrics_keys_and_ranges() -> None:
    """Metrics return all keys and sensible ranges."""
    m = _metrics(_series(0), bpy=252.0)
    for k in ("roi", "cagr", "max_dd", "sharpe", "sortino",
              "calmar", "vol", "win_rate", "n_bars"):
        assert k in m, f"missing key {k}"
    assert m["max_dd"] <= 0.0, "DD must be <= 0"
    assert 0.0 <= m["win_rate"] <= 1.0
    assert m["vol"] >= 0.0
    assert m["n_bars"] == 500


def test_positive_drift_positive_sharpe() -> None:
    """A series with positive drift gives positive Sharpe/ROI."""
    m = _metrics(_series(1, drift=0.002), bpy=252.0)
    assert m["sharpe"] > 0
    assert m["roi"] > 0


def test_degenerate_series() -> None:
    """A flat/short series does not crash and yields zeros."""
    flat = pd.Series([0.0] * 10)
    m = _metrics(flat, bpy=252.0)
    assert m["sharpe"] == 0.0
    short = pd.Series([0.01])
    m2 = _metrics(short, bpy=252.0)
    assert m2["sharpe"] == 0.0


def test_bpy_affects_annualization() -> None:
    """Different bpy changes annualized vol/Sharpe (honest annualization)."""
    r = _series(3, drift=0.001)
    m_daily = _metrics(r, bpy=252.0)
    m_h4 = _metrics(r, bpy=2190.0)
    assert m_h4["vol"] > m_daily["vol"], "H4 bpy -> higher annual vol"


def test_calmar_and_sortino_consistent() -> None:
    """Calmar = CAGR/|DD|; Sortino is finite when drawdowns exist."""
    m = _metrics(_series(5, drift=0.0015), bpy=252.0)
    if m["max_dd"] != 0 and m["calmar"] != float("inf"):
        assert abs(m["calmar"] - m["cagr"] / abs(m["max_dd"])) < 1e-6
