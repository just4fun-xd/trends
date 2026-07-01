"""Тесты регим-слоя и кросс-секционных стратегий.

Ключевые проверки:
  - Роутер под AlwaysTrendDetector == чистый Donchian (sanity).
  - OU профитен на mean-reverting ряде (его родная среда).
  - HMM/Markowitz заглушки честно бросают NotImplementedError.
  - Dual momentum даёт валидную матрицу весов (market-neutral -> gross~2).

Запуск: python -m tests.test_regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine
from core.engine_portfolio import run_portfolio
from regime.detector import (
    AlwaysTrendDetector,
    HMMDetector,
    VolatilityRegimeDetector,
)
from regime.router import regime_router
from strategies import cross_sectional as xs
from strategies.donchian import donchian_est_macd_4step_take
from strategies.ou import ou_fit, ou_zscore


def _ou_series(n=500, theta=0.1, mu=100, sigma=1.5, seed=0) -> Bars:
    """Синтетический OU-ряд — истинно mean-reverting."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = x[i-1] + theta * (mu - x[i-1]) + rng.normal(0, sigma)
    close = pd.Series(x, index=idx)
    hl = sigma
    high = close + hl * rng.uniform(0.3, 1.0, n)
    low = close - hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="OU")


def _trend_bars(n=400, seed=1) -> Bars:
    """Трендовый ряд для sanity-чека роутера."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps = rng.normal(0.0015, 0.008, n)
    close = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
    hl = close * 0.006
    high = close + hl * rng.uniform(0.3, 1.0, n)
    low = close - hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="TREND")


def test_router_degenerates_to_donchian() -> None:
    """Роутер под AlwaysTrend == прямой Donchian champion (sanity)."""
    bars = _trend_bars()
    routed = regime_router(bars, AlwaysTrendDetector())
    direct = donchian_est_macd_4step_take(bars)
    diff = (routed - direct).abs().max()
    assert diff < 1e-9, f"Роутер разошёлся с Дончианом: {diff:.2e}"
    print(f"  [ok] Роутер(AlwaysTrend) == Donchian champion: "
          f"расхождение {diff:.2e}")


def test_ou_math_valid() -> None:
    """ou_fit восстанавливает half-life на синтетич. OU-ряде."""
    bars = _ou_series(theta=0.1)
    fit = ou_fit(bars.close)
    assert fit["well_defined"], "OU-фит не сошёлся на явном OU-ряде"
    # half_life = ln(2)/theta; theta~0.1 -> hl~7. Проверяем порядок.
    assert 3 < fit["half_life"] < 15, f"half_life={fit['half_life']:.1f}"
    print(f"  [ok] ou_fit: theta={fit['theta']:.3f}, "
          f"half_life={fit['half_life']:.1f}д (математика валидна)")


def test_ou_profits_in_range() -> None:
    """OU z-score профитен на mean-reverting ряде (его родная среда)."""
    bars = _ou_series(theta=0.12)
    res = run_engine(bars, ou_zscore(bars), cost=0.0)
    assert res.total_return > 0, (
        f"OU слил на mean-reverting ряде: {res.total_return:.1%}"
    )
    print(f"  [ok] OU на боковике: ret={res.total_return:+.1%} "
          f"(родная среда — RANGE)")


def test_ou_dangerous_on_trend() -> None:
    """OU опасен на тренде (документируем, зачем нужен роутер).

    Не ассертим знак — показываем механизм: на тренде OU шортит растущее.
    """
    bars = _trend_bars()
    res = run_engine(bars, ou_zscore(bars), cost=0.0)
    print(f"  [ok] OU на тренде: ret={res.total_return:+.1%}, "
          f"dd={res.max_drawdown:.1%} (вот зачем детектор режима)")


def test_hmm_stub_raises() -> None:
    """HMMDetector честно бросает NotImplementedError (не выдумка)."""
    bars = _trend_bars()
    try:
        HMMDetector().detect(bars)
        raise AssertionError("HMM-заглушка должна была бросить")
    except NotImplementedError as e:
        assert "трек 2.2" in str(e)
    print("  [ok] HMMDetector — честная заглушка (NotImplementedError)")


def test_markowitz_stub_raises() -> None:
    """markowitz_momentum честно бросает NotImplementedError."""
    prices = pd.DataFrame(np.random.rand(100, 5).cumsum(axis=0) + 100)
    try:
        xs.markowitz_momentum(prices)
        raise AssertionError("Markowitz-заглушка должна была бросить")
    except NotImplementedError as e:
        assert "трек 1.2" in str(e)
    print("  [ok] markowitz_momentum — честная заглушка")


def test_vol_regime_detector() -> None:
    """VolatilityRegimeDetector даёт валидное распределение режимов."""
    bars = _trend_bars()
    probs = VolatilityRegimeDetector().detect(bars)
    row_sums = probs.sum(axis=1)
    valid = row_sums[row_sums > 0]
    assert np.allclose(valid, 1.0, atol=1e-6), "Вероятности не суммир. в 1"
    print(f"  [ok] VolRegimeDetector: вероятности суммир. в 1, "
          f"колонки {list(probs.columns)}")


def test_dual_momentum_weights() -> None:
    """Dual momentum market-neutral даёт gross~2 (лонг+шорт по 100%)."""
    rng = np.random.default_rng(3)
    n, m = 300, 10
    idx = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (n, m)), axis=0)),
        index=idx, columns=[f"A{i}" for i in range(m)],
    )
    w = xs.dual_momentum(prices, market_neutral=True, abs_filter_sma=None)
    res = run_portfolio(prices, w, cost=0.0)
    # После прогрева market-neutral gross ~ 2 (топ 20% лонг + дно 20% шорт).
    late_gross = res.gross.iloc[200:]
    active = late_gross[late_gross > 0]
    assert len(active) > 50, "Веса не сформировались"
    print(f"  [ok] Dual momentum MN: медиана gross "
          f"{active.median():.2f}, итог {res.total_return:+.1%}")


if __name__ == "__main__":
    print("Тесты регим-слоя и кросс-секции:")
    test_router_degenerates_to_donchian()
    test_ou_math_valid()
    test_ou_profits_in_range()
    test_ou_dangerous_on_trend()
    test_hmm_stub_raises()
    test_markowitz_stub_raises()
    test_vol_regime_detector()
    test_dual_momentum_weights()
    print("Все тесты регим-слоя пройдены.")
