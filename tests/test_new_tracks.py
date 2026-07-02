"""Тесты новых треков с ГОДОВОЙ РАЗБИВКОЙ (аудит 2026-07).

Показывает по годам return и DD в процентах для каждого алгоритма и
инструмента — прямой ответ на запрос ревью. Данные синтетические
(Yahoo недоступен в песочнице), проверяется механика + формат вывода;
локально те же прогоны идут на реальных данных.

Запуск: python -m tests.test_new_tracks
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine
from core.engine_portfolio import run_portfolio
from diagnostics.yearly import (
    format_matrix,
    format_yearly_table,
    yearly_breakdown,
    yearly_matrix,
)
from strategies import cross_sectional as xs
from strategies import seasonal
from strategies.pairs import kalman_beta, run_pair_kalman


def _gas_like_bars(seed=0) -> Bars:
    """6-летний ряд с сезонным паттерном (осень-зима сильнее)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", "2026-01-01")
    n = len(idx)
    # Сезонный дрейф: плюс в авг-ноя, минус весной.
    month = idx.month.values
    seasonal_drift = np.where(np.isin(month, [8, 9, 10, 11]), 0.0018,
                              np.where(np.isin(month, [3, 4, 5, 6]),
                                       -0.0010, 0.0002))
    noise = rng.normal(0, 0.018, n)
    close = pd.Series(100 * np.exp(np.cumsum(seasonal_drift + noise)),
                      index=idx)
    sp = close * 0.01
    high = close + sp * rng.uniform(0.3, 1.0, n)
    low = close - sp * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="GAS")


def _equity_panel(m=12, seed=1) -> pd.DataFrame:
    """Панель акций: пара мегакапов + спокойные имена (для DM-трека)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", "2026-01-01")
    n = len(idx)
    cols, data = [], {}
    for i in range(m):
        # Первые два — «прыгучие мегакапы», остальные спокойнее.
        drift = 0.0008 if i < 2 else 0.0004
        vol = 0.030 if i < 2 else 0.014
        px = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        name = f"MEGA{i}" if i < 2 else f"CALM{i}"
        cols.append(name)
        data[name] = px
    return pd.DataFrame(data, index=idx)


def test_seasonal_yearly() -> None:
    """Сезонные стратегии: годовая разбивка return/DD."""
    bars = _gas_like_bars()
    variants = {
        "seasonal_gas": seasonal.seasonal_gas(bars),
        "donch_seasonal": seasonal.donchian_seasonal(bars),
        "donch_seas_VT": seasonal.donchian_seasonal_voltarget(bars),
    }
    print("\n=== СЕЗОННЫЕ: годовая разбивка (синтетика) ===")
    equities = {}
    for name, pos in variants.items():
        res = run_engine(bars, pos)
        equities[name] = res.equity
        yb = yearly_breakdown(res.equity, res.bars_per_year)
        print("\n" + format_yearly_table(yb, f"[{name}] на GAS"))
        assert res.passes_dd(0.60), f"{name} экстремальный DD"
    # Сводная матрица год × вариант.
    print("\n" + format_matrix(
        yearly_matrix(equities, 252.0, "return"),
        "Сезонные — return по годам (год × вариант)"
    ))
    print("  [ok] сезонные отработали, годовая разбивка построена")


def test_dualmom_research_yearly() -> None:
    """Dual momentum research-треки: годовая разбивка портфельно."""
    prices = _equity_panel()
    benchmark = prices.mean(axis=1)  # синтетический «рынок»
    variants = {
        "DM_tilt": xs.dual_momentum_tilt(prices, benchmark),
        "DM_regime": xs.dual_momentum_regime(prices, benchmark),
        "DM_volscaled": xs.dual_momentum_volscaled(prices),
    }
    print("\n=== DUAL MOMENTUM RESEARCH: годовая разбивка ===")
    equities = {}
    for name, w in variants.items():
        res = run_portfolio(prices, w, cost=0.0002)
        equities[name] = res.equity
        yb = yearly_breakdown(res.equity, res.bars_per_year)
        print("\n" + format_yearly_table(yb, f"[{name}] портфель"))
    print("\n" + format_matrix(
        yearly_matrix(equities, 252.0, "return"),
        "DM research — return по годам (год × вариант)"
    ))
    print("\n" + format_matrix(
        yearly_matrix(equities, 252.0, "max_dd"),
        "DM research — MaxDD по годам (год × вариант)"
    ))
    # volscaled должен ограничивать концентрацию мегакапов -> обычно
    # мягче по DD, чем tilt. Не жёсткий ассерт (синтетика), но проверим
    # что все дали валидные кривые.
    for name, eq in equities.items():
        assert np.isfinite(eq).all(), f"{name}: NaN в equity"
    print("  [ok] три DM research-трека отработали")


def test_kalman_beta_valid() -> None:
    """Kalman-бета восстанавливает известное соотношение (математика)."""
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=500)
    b = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)), index=idx)
    true_beta = 1.5
    a = true_beta * b + rng.normal(0, 2, 500)  # A = 1.5*B + шум
    a = pd.Series(a, index=idx)
    beta = kalman_beta(a, b)
    # После прогрева бета должна сойтись к ~1.5.
    converged = beta.iloc[100:].mean()
    assert abs(converged - true_beta) < 0.3, (
        f"Kalman-бета не сошлась: {converged:.2f} vs {true_beta}"
    )
    print(f"\n  [ok] Kalman-бета: сошлась к {converged:.2f} "
          f"(истинная {true_beta}) — математика валидна")


def test_kalman_pair_yearly() -> None:
    """Kalman-пара: годовая разбивка (research, край не подтверждён)."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", "2026-01-01")
    n = len(idx)
    # Коинтегрированная пара: общий фактор + расходящийся спред.
    common = np.cumsum(rng.normal(0, 1, n))
    a = pd.Series(100 + common + rng.normal(0, 3, n), index=idx)
    b = pd.Series(100 + common + rng.normal(0, 3, n), index=idx)
    res = run_pair_kalman(a, b)
    yb = yearly_breakdown(res.equity, res.bars_per_year)
    print("\n=== KALMAN PAIRS (research): годовая разбивка ===")
    print(format_yearly_table(yb, "[kalman_pair] synth-спред"))
    assert np.isfinite(res.equity).all(), "NaN в equity пары"
    print("  [ok] Kalman-пара отработала (research — не боевой трек)")


if __name__ == "__main__":
    print("Тесты новых треков (годовая разбивка, аудит 2026-07):")
    test_seasonal_yearly()
    test_dualmom_research_yearly()
    test_kalman_beta_valid()
    test_kalman_pair_yearly()
    print("\nВсе тесты новых треков пройдены.")
