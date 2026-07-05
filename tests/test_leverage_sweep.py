"""Тесты leverage_sweep: находит максимум доходности в рамках DD<40%.

Инварианты:
  - монотонность: на тихом (низковолатильном) синтетическом ряду
    больший кэп плеча должен давать больший доход, пока не упрёмся
    в DD<40% — иначе sweep не делает то, что заявлено;
  - best_leverage возвращает точку из passing-подмножества;
  - lev_hit_cap корректно детектирует «упёрлись в потолок»: если
    target_vol недостижим при данном кэпе, доля дней у потолка
    должна быть высокой;
  - при пустом passing (все точки превышают DD) best_leverage — None,
    а не падает с ошибкой.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diagnostics.port_lev_sweep import best_leverage, leverage_sweep


def _quiet_combo(n: int = 1500, seed: int = 5,
                 daily_vol: float = 0.001) -> pd.Series:
    """Синтетика с низкой волой — имитация комбо после vol-parity
    (реальные комбо в проекте: годовая вола 1.6-2.9%)."""
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
    # На тихом ряду больший кэп -> больше реализованное плечо ->
    # больше доходность (Sharpe > 0 закладывался в конструкции ряда).
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
    # Очень высокий target_vol с низким кэпом -> кэп должен связывать
    # почти всегда (после прогрева окна).
    combo = _quiet_combo(seed=17, daily_vol=0.0008)
    df = leverage_sweep(
        combo, target_vols=(0.90,), max_leverage_grid=(2.0,),
    )
    row = df.loc[(0.90, 2.0)]
    assert row["lev_hit_cap"] > 0.8
    assert row["avg_lev"] == pytest.approx(2.0, abs=0.05)


def test_best_leverage_none_when_all_fail_dd():
    # Экстремальный кэп на волатильном ряду -> DD почти наверняка
    # пробивает 40%; best_leverage не должен падать, а вернуть None.
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
    """funding_rate>0 в сетке должен снижать return относительно
    funding_rate=0 везде, где реально используется заём (avg_lev>1),
    и не менять avg_lev/lev_hit_cap (funding не влияет на веса)."""
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
