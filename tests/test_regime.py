"""Тесты регим-слоя и кросс-секции — обновлены аудитом 2026-07.

Новые регрессии:
  - РИСК-ПАРИТЕТ роутера: под AlwaysRange OU-нога масштабирована общим
    vol_target, а не лупит сырым ±1.0 (была асимметрия риска x3-4).
  - NaN-устойчивость z-score: плоский сегмент цены не ломает позиции
    и не создаёт фантомных выходов (std->NaN => держим состояние).
  - Разреженность ребаланса кросс-секции: веса меняются ~раз в месяц,
    а не каждый бар (дрожание границы топ-20% теперь платно).

Сохранены: sanity роутер(AlwaysTrend)==champion, математика OU,
честные заглушки HMM/Markowitz.

Запуск: python -m tests.test_regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine, vol_target_size
from core.engine_portfolio import run_portfolio
from regime.detector import (
    AlwaysTrendDetector,
    HMMDetector,
    Regime,
    RegimeDetector,
    VolatilityRegimeDetector,
)
from regime.router import regime_router
from strategies import cross_sectional as xs
from strategies.donchian import donchian_est_macd_4step_take
from strategies.ou import ou_fit, ou_zscore


class AlwaysRangeDetector(RegimeDetector):
    """Тестовый детектор: всегда RANGE — изолирует OU-ногу роутера."""

    def detect(self, bars: Bars) -> pd.DataFrame:
        """P(RANGE)=1 на всём периоде."""
        df = pd.DataFrame(0.0, index=bars.index,
                          columns=[r.value for r in Regime])
        df[Regime.RANGE.value] = 1.0
        return df


def _ou_series(n=500, theta=0.1, mu=100, sigma=1.5, seed=0) -> Bars:
    """Синтетический OU-ряд — истинно mean-reverting."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + rng.normal(0, sigma)
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


def _flat_gap_bars(n1=150, nflat=60, n2=150, seed=4) -> Bars:
    """OU-ряд с ПЛОСКИМ сегментом в середине (std=0 -> z=NaN).

    Регрессия аудита 2026-07: раньше NaN в z ронял позицию в 0 на бар
    (фантомный выход/вход с двойными издержками), а предлагавшийся
    фикс std.replace(0, 1e-9) дал бы |z|~1e8 и ложный сигнал.
    """
    rng = np.random.default_rng(seed)
    n = n1 + nflat + n2
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = np.empty(n)
    x[0] = 100.0
    for i in range(1, n1):
        x[i] = x[i - 1] + 0.12 * (100 - x[i - 1]) + rng.normal(0, 1.2)
    x[n1:n1 + nflat] = x[n1 - 1]
    for i in range(n1 + nflat, n):
        x[i] = x[i - 1] + 0.12 * (100 - x[i - 1]) + rng.normal(0, 1.2)
    close = pd.Series(x, index=idx)
    high = close + 0.5
    low = close - 0.5
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="FLATGAP")


def test_router_degenerates_to_champion() -> None:
    """Sanity: роутер под AlwaysTrend == champion (тождество раскладки).

    Держится на champion == raw * vol_size — если рефакторинг
    риск-паритета сломал тождество, тест падает первым.
    """
    bars = _trend_bars()
    routed = regime_router(bars, AlwaysTrendDetector())
    direct = donchian_est_macd_4step_take(bars)
    diff = (routed - direct).abs().max()
    assert diff < 1e-9, f"Роутер разошёлся с champion: {diff:.2e}"
    print(f"  [ok] Роутер(AlwaysTrend) == champion: расхождение {diff:.2e}")


def test_router_risk_parity_in_range() -> None:
    """КРИТИЧНО (аудит 2026-07): OU-нога масштабирована, не сырая ±1.

    Под AlwaysRange роутер обязан выдавать ou_raw * vol_size: позиции
    ограничены плечом и НЕ равны голой единице капитала.
    """
    bars = _ou_series()
    routed = regime_router(bars, AlwaysRangeDetector())
    raw = ou_zscore(bars)
    expected = (raw * vol_target_size(bars, 0.15)).fillna(0.0)
    diff = (routed - expected).abs().max()
    assert diff < 1e-9, f"OU-нога не так масштабирована: {diff:.2e}"

    active = routed[raw != 0]
    assert (active.abs() <= 2.0 + 1e-9).all(), "Пробито плечо 2.0"
    med = active.abs().median()
    assert abs(med - 1.0) > 0.05, (
        f"Позиции ~1.0 (медиана {med:.2f}) — vol targeting не применён"
    )
    print(f"  [ok] Риск-паритет: OU-нога = raw*vol (медиана |pos| "
          f"{med:.2f}, было бы 1.00 без масштаба), плечо <= 2.0")


def test_ou_nan_flat_segment() -> None:
    """Регрессия: плоский сегмент не даёт NaN и фантомных сделок."""
    bars = _flat_gap_bars()
    pos = ou_zscore(bars)
    assert not pos.isna().any(), "NaN просочился в позиции"
    # Внутри полностью плоского окна (std->NaN) состояние держится.
    inner = pos.iloc[150 + 25:150 + 60]
    assert inner.nunique() == 1, (
        f"Фантомные сделки на плоском сегменте: {inner.nunique()} "
        f"уникальных состояний"
    )
    res = run_engine(bars, pos)
    assert np.isfinite(res.equity).all(), "NaN отравил кривую капитала"
    print(f"  [ok] Плоский сегмент: NaN нет, состояние держится "
          f"({inner.iloc[0]:+.0f} все 35 баров), equity конечна")


def test_ou_math_valid() -> None:
    """ou_fit восстанавливает half-life на синтетич. OU-ряде."""
    bars = _ou_series(theta=0.1)
    fit = ou_fit(bars.close)
    assert fit["well_defined"], "OU-фит не сошёлся на явном OU-ряде"
    assert 3 < fit["half_life"] < 15, f"half_life={fit['half_life']:.1f}"
    print(f"  [ok] ou_fit: theta={fit['theta']:.3f}, "
          f"half_life={fit['half_life']:.1f}д (математика валидна)")


def test_ou_profits_in_range() -> None:
    """OU z-score профитен на mean-reverting ряде (родная среда)."""
    bars = _ou_series(theta=0.12)
    res = run_engine(bars, ou_zscore(bars), cost=0.0)
    assert res.total_return > 0, (
        f"OU слил на mean-reverting ряде: {res.total_return:.1%}"
    )
    print(f"  [ok] OU на боковике: ret={res.total_return:+.1%} "
          f"(родная среда — RANGE)")


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
    assert np.allclose(valid, 1.0, atol=1e-6), "Вероятности не в сумме 1"
    print(f"  [ok] VolRegimeDetector: вероятности суммир. в 1, "
          f"колонки {list(probs.columns)}")


def test_dual_momentum_monthly_rebalance() -> None:
    """Dual momentum: gross~2 (MN) и веса меняются ~раз в месяц.

    Регрессия аудита: ежедневный ре-ранг дрожал границей топ-20% и с
    drift-издержками стал платным; теперь rebalance_every=21.
    """
    rng = np.random.default_rng(3)
    n, m = 300, 10
    idx = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (n, m)), axis=0)),
        index=idx, columns=[f"A{i}" for i in range(m)],
    )
    w = xs.dual_momentum(prices, market_neutral=True, abs_filter_sma=None)
    change_days = int((w.diff().abs().sum(axis=1) > 1e-12).sum())
    assert change_days < 30, (
        f"Веса меняются {change_days} дней из {n} — ребаланс не месячный"
    )
    res = run_portfolio(prices, w, cost=0.0)
    active = res.gross.iloc[200:][res.gross.iloc[200:] > 0]
    assert len(active) > 50, "Веса не сформировались"
    print(f"  [ok] Dual momentum MN: gross медиана {active.median():.2f}, "
          f"смен весов {change_days} (месячный ребаланс), "
          f"итог {res.total_return:+.1%}")


if __name__ == "__main__":
    print("Тесты регим-слоя и кросс-секции (аудит 2026-07):")
    test_router_degenerates_to_champion()
    test_router_risk_parity_in_range()
    test_ou_nan_flat_segment()
    test_ou_math_valid()
    test_ou_profits_in_range()
    test_hmm_stub_raises()
    test_markowitz_stub_raises()
    test_vol_regime_detector()
    test_dual_momentum_monthly_rebalance()
    print("Все тесты регим-слоя пройдены.")
