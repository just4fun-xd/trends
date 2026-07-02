"""Портфельный движок для кросс-секционных стратегий.

Параллельный путь к run_engine. Посерийные стратегии дают позицию по
одному инструменту; кросс-секционные (Dual Momentum, Markowitz) дают
МАТРИЦУ весов — сколько каждого инструмента держать в каждый день.

Архитектурная граница (жёсткая, из уроков проекта): кросс-секционные
стратегии НЕЛЬЗЯ прогонять посерийным циклом — получается мусор.
run_portfolio отдельный. Адаптер positions_to_weights мостит их для
sanity-чека: на одном инструменте оба движка обязаны совпасть до 0.0%.

Единственный shift(1) тот же — веса применяются к завтрашним доходностям.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioResult:
    """Результат прогона кросс-секционной стратегии на корзине.

    Attributes:
        equity: Кривая капитала портфеля.
        total_return: Итоговая доходность.
        max_drawdown: Худшая просадка портфеля (<= 0).
        weights: Применённая матрица весов (после shift).
        gross: Ряд валовой экспозиции (сумма |весов|) по времени.
        bars_per_year: Баров в году для аннуализации Sharpe (честное
            поле вместо magic-атрибута Series — pandas 3.0 CoW терял
            атрибуты при копировании).
    """

    equity: pd.Series
    total_return: float
    max_drawdown: float
    weights: pd.DataFrame
    gross: pd.Series
    bars_per_year: float = 252.0

    @property
    def sharpe(self) -> float:
        """Годовой Sharpe кривой капитала портфеля (безрисковая = 0)."""
        rets = self.equity.pct_change().dropna()
        if rets.std() == 0 or len(rets) < 2:
            return 0.0
        return float(
            rets.mean() / rets.std() * np.sqrt(self.bars_per_year)
        )

    def passes_dd(self, limit: float = 0.40) -> bool:
        """Проходит ли портфель лимит просадки.

        Args:
            limit: Порог DD (0.40 = 40%).

        Returns:
            True если |max_drawdown| <= limit.
        """
        return abs(self.max_drawdown) <= limit


def run_portfolio(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    bars_per_year: float = 252.0,
    cost: float = 0.0002,
    trade_start: str | None = None,
) -> PortfolioResult:
    """Прогоняет матрицу весов через портфельный движок.

    Логика (векторно по всей матрице):
        returns   = prices.pct_change()            # доходность каждого
        pnl_t     = Σ_i weights[i].shift(1) * returns[i]
        drifted_i = w_i(1+r_i) / (1+pnl)           # вес после движения цен
        turnover  = Σ_i |w_i,t − drifted_i,t-1|    # ребаланс против дрейфа
        strat     = pnl - turnover * cost
        equity    = (1 + strat).cumprod()

    МОДЕЛЬ (аудит 2026-07): P&L-формула Σ w·r предполагает ежедневную
    ребалансировку к целевым весам — значит, оборот обязан учитывать
    дрейф. Старый diff() целевых весов давал нулевой оборот при
    постоянном целевом весе, хотя удержание постоянной ДОЛИ NAV требует
    ежедневного ребаланса против движения цен. Нормализация дрейфа
    делит на (1+pnl) ДО издержек — издержки за бар ~bps, погрешность
    нормализации второго порядка, пренебрежимо.

    Args:
        prices: Матрица цен close, индекс — даты, колонки — инструменты.
        weights: Матрица весов от стратегии, тот же shape что prices.
            Веса — доли капитала; сумма может быть !=1 (gross exposure).
            shift делает движок.
        bars_per_year: Для аннуализации (пробрасывается в Sharpe).
        cost: Издержки на единицу оборота.
        trade_start: Дата начала торговли или None.

    Returns:
        PortfolioResult с кривой капитала и метриками.

    Raises:
        ValueError: Если shape весов и цен не совпадают.
    """
    if not weights.index.equals(prices.index):
        raise ValueError("weights и prices: разные индексы")
    if list(weights.columns) != list(prices.columns):
        raise ValueError("weights и prices: разные колонки")

    returns = prices.pct_change()
    prev_w = weights.shift(1).fillna(0.0)

    # P&L портфеля = построчная сумма вклад инструментов.
    pnl = (prev_w * returns).sum(axis=1)

    # Дрейфовавшие веса к концу бара: w(1+r) на выросший NAV (1+pnl).
    denom = (1.0 + pnl).where((1.0 + pnl).abs() > 1e-9)
    drifted = prev_w.mul(1.0 + returns).div(denom, axis=0).shift(1)

    # Оборот = ребаланс от дрейфовавших весов к новым целевым.
    turnover = (prev_w - drifted.fillna(0.0)).abs().sum(axis=1).fillna(0.0)

    strat = pnl - turnover * cost

    if trade_start is not None:
        strat = strat.copy()
        strat[strat.index < pd.Timestamp(trade_start)] = 0.0

    strat = strat.fillna(0.0)
    equity = (1.0 + strat).cumprod()

    drawdown = equity / equity.cummax() - 1.0
    gross = prev_w.abs().sum(axis=1)

    return PortfolioResult(
        equity=equity,
        total_return=float(equity.iloc[-1] - 1.0),
        max_drawdown=float(drawdown.min()),
        weights=prev_w,
        gross=gross,
        bars_per_year=bars_per_year,
    )


def positions_to_weights(
    position: pd.Series, symbol: str
) -> pd.DataFrame:
    """Оборачивает посерийную позицию в одноколоночную матрицу весов.

    Мост между движками для sanity-чека: run_portfolio на результате
    этого адаптера обязан совпасть с run_engine на той же позиции до
    0.0%. Если не совпадает — движки разъехались, ловим сразу.

    Args:
        position: Позиция по одному инструменту от посерийной стратегии.
        symbol: Имя инструмента (станет именем колонки).

    Returns:
        DataFrame с единственной колонкой symbol = position.
    """
    return position.to_frame(name=symbol)


def sanity_check_engines(
    bars, position, cost: float = 0.0002
) -> float:
    """Проверяет совпадение run_engine и run_portfolio на 1 инструменте.

    Прогоняет одну позицию обоими движками и возвращает абсолютную
    разницу итоговых доходностей. Должна быть ~0 (< 1e-9). Используется
    в тестах миграции.

    Args:
        bars: Данные инструмента.
        position: Позиция для проверки.
        cost: Издержки (одинаковые для обоих движков).

    Returns:
        |total_return_engine - total_return_portfolio|.
    """
    from core.engine import run_engine

    r_single = run_engine(bars, position, cost=cost)
    w = positions_to_weights(position, bars.symbol)
    prices = bars.close.to_frame(name=bars.symbol)
    r_port = run_portfolio(
        prices, w, bars_per_year=bars.bars_per_year, cost=cost
    )
    return abs(r_single.total_return - r_port.total_return)
