"""leverage_sweep tests: finds the max return within DD<40%.

Invariants:
  - monotonicity: on a quiet (low-vol) synthetic series a larger
    leverage cap should give more return until DD<40% binds —
    otherwise the sweep does not do what it claims;
  - best_leverage returns a point from the passing subset;
  - lev_hit_cap correctly detects "hit the cap": if target_vol is
    unreachable at the given cap, the fraction of days at the cap
    must be high;
  - with an empty passing set (all points exceed DD) best_leverage is
    None, not an error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diagnostics.port_lev_sweep import best_leverage, leverage_sweep


def _quiet_combo(n: int = 1500, seed: int = 5,
                 daily_vol: float = 0.001) -> pd.Series:
    """Low-vol synthetic — a combo after vol-parity
    (real project combos: annual vol 1.6-2.9%)."""
    rng = np.random.default_rng(seed)
    mu = 1.2 * daily_vol  # Sharpe ~1.2
    rets = mu + daily_vol * rng.standard_normal(n)
    return pd.Series(
        rets, index=pd.date_range("2019-01-02", periods=n, freq="B"))


def test_higher_cap_increases_return_until_dd_breaks():
    combo = _quiet_combo()
    df = leverage_sweep(
        combo, target_vols=(0.30,), max_leverage_grid=(2.0, 8.0),
    )
    low = df.loc[(0.30, 2.0)]
    high = df.loc[(0.30, 8.0)]
    # On a quiet series a larger cap -> more realized leverage ->
    # more return (Sharpe > 0 was built into the series).
    assert high["avg_lev"] >= low["avg_lev"]
    assert high["return"] >= low["return"]


def test_best_leverage_picks_max_return_among_passing():
    combo = _quiet_combo(seed=11)
    df = leverage_sweep(
        combo, target_vols=(0.15, 0.30, 0.60),
        max_leverage_grid=(2.0, 6.0),
    )
    best = best_leverage(df)
    assert best is not None
    passing = df[df["passes_dd"]]
    assert best == passing["return"].idxmax()
    assert df.loc[best, "passes_dd"]


def test_lev_hit_cap_detects_unreachable_target():
    # A very high target_vol with a low cap -> the cap should bind
    # almost always (after the window warms up).
    combo = _quiet_combo(seed=17, daily_vol=0.0008)
    df = leverage_sweep(
        combo, target_vols=(0.90,), max_leverage_grid=(2.0,),
    )
    row = df.loc[(0.90, 2.0)]
    assert row["lev_hit_cap"] > 0.8
    assert row["avg_lev"] == pytest.approx(2.0, abs=0.05)


def test_best_leverage_none_when_all_fail_dd():
    # An extreme cap on a volatile series -> DD almost surely breaks
    # 40%; best_leverage must not crash but return None.
    rng = np.random.default_rng(23)
    n = 800
    rets = pd.Series(
        0.05 * rng.standard_normal(n),
        index=pd.date_range("2019-01-02", periods=n, freq="B"),
    )
    df = leverage_sweep(
        rets, target_vols=(2.0,), max_leverage_grid=(20.0,),
    )
    assert not df["passes_dd"].any()
    assert best_leverage(df) is None


def test_sweep_grid_shape_and_columns():
    combo = _quiet_combo(seed=3, n=600)
    tvs = (0.10, 0.20)
    caps = (2.0, 4.0, 6.0)
    df = leverage_sweep(combo, target_vols=tvs,
                        max_leverage_grid=caps)
    assert len(df) == len(tvs) * len(caps)
    for col in ("return", "max_dd", "sharpe", "avg_lev",
                "lev_hit_cap", "passes_dd"):
        assert col in df.columns


def test_leverage_sweep_funding_rate_lowers_return():
    """funding_rate>0 in the grid should lower return relative to
    funding_rate=0 wherever borrowing is actually used (avg_lev>1),
    and not change avg_lev/lev_hit_cap (funding does not affect weights)."""
    combo = _quiet_combo(seed=29)
    free = leverage_sweep(
        combo, target_vols=(0.30,), max_leverage_grid=(6.0,),
        funding_rate=0.0,
    )
    paid = leverage_sweep(
        combo, target_vols=(0.30,), max_leverage_grid=(6.0,),
        funding_rate=0.08,
    )
    key = (0.30, 6.0)
    assert paid.loc[key, "avg_lev"] == pytest.approx(
        free.loc[key, "avg_lev"])
    assert paid.loc[key, "return"] < free.loc[key, "return"]
