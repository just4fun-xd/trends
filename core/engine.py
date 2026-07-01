"""Движок бэктеста для посерийных (per-instrument) стратегий.

Считает деньги: position -> equity -> drawdown. Единственное место в
проекте, где живёт shift(1) — защита от look-ahead. Стратегия возвращает
«сырую» позицию (числовой ряд: 1.0 = полный лонг, 0 = кэш, дробное при
vol targeting, отрицательное = шорт). Движок сам сдвигает и считает.
Стратегия НЕ может это обойти — контракт узкий.

Отличие от старого движка: аннуализация волатильности берёт bars_per_year
из Bars, а не хардкодит 252. Это делает H4/интрадей (требование 5)
корректным без спец-веток: на дневных 252, на H4 ~1512, формула одна.

Контракт возврата тот же: (equity, total_return, max_drawdown). Числа из
старых .md должны воспроизводиться бит-в-бит — это тест миграции.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.bars import Bars


@dataclass(frozen=True)
class BacktestResult:
    """Результат прогона одной стратегии на одном инструменте.

    Attributes:
        equity: Кривая капитала (мультипликативная, старт 1.0).
        total_return: Итоговая доходность за период (equity[-1] - 1).
        max_drawdown: Худшая просадка от вершины (<= 0).
        position: Фактически применённая позиция (после shift).
        symbol: Инструмент.
    """

    equity: pd.Series
    total_return: float
    max_drawdown: float
    position: pd.Series
    symbol: str = ""

    @property
    def sharpe(self) -> float:
        """Годовой Sharpe кривой капитала (безрисковая = 0).

        Использует bars_per_year, зашитый при расчёте (см. run_engine,
        где Sharpe кладётся в атрибут кривой). Здесь — из daily-ряда.
        """
        rets = self.equity.pct_change().dropna()
        if rets.std() == 0 or len(rets) < 2:
            return 0.0
        bpy = getattr(self.equity, "_bars_per_year", 252.0)
        return float(rets.mean() / rets.std() * np.sqrt(bpy))

    def passes_dd(self, limit: float = 0.40) -> bool:
        """Проходит ли жёсткий лимит максимальной просадки.

        Args:
            limit: Порог DD (положительное число, 0.40 = 40%).

        Returns:
            True если |max_drawdown| <= limit.
        """
        return abs(self.max_drawdown) <= limit


def run_engine(
    bars: Bars,
    position: pd.Series,
    trade_start: str | None = None,
    cost: float = 0.0002,
) -> BacktestResult:
    """Прогоняет позицию через движок, возвращает метрики.

    Единственная точка shift(1) в проекте. Логика:
        returns  = close.pct_change()
        strat    = returns * position.shift(1)   # торгуем по вчера
        strat   -= turnover * cost               # издержки на смену позиции
        equity   = (1 + strat).cumprod()
        drawdown = equity / equity.cummax() - 1

    Args:
        bars: Данные инструмента (даёт close и bars_per_year).
        position: Сырая позиция от стратегии, тот же индекс что bars.
            Числовой ряд, НЕ bool. shift делает движок.
        trade_start: Дата начала торговли (прогрев индикаторов до неё).
            None = с первого бара.
        cost: Издержки на единицу оборота позиции (0.0002 = 2 bps).

    Returns:
        BacktestResult с кривой капитала и метриками.

    Raises:
        ValueError: Если индекс position не совпадает с bars.
    """
    if not position.index.equals(bars.index):
        raise ValueError(f"position и bars не выровнены ({bars.symbol})")

    returns = bars.returns()
    prev_pos = position.shift(1).fillna(0.0)

    # Оборот = |изменение позиции|; издержки платятся при входе/выходе/ресайзе.
    turnover = prev_pos.diff().abs().fillna(0.0)

    strat = returns * prev_pos - turnover * cost

    if trade_start is not None:
        # До trade_start P&L обнуляется (индикаторы грелись, не торговали).
        strat = strat.copy()
        strat[strat.index < pd.Timestamp(trade_start)] = 0.0

    strat = strat.fillna(0.0)
    equity = (1.0 + strat).cumprod()
    equity._bars_per_year = bars.bars_per_year  # для Sharpe

    drawdown = equity / equity.cummax() - 1.0

    return BacktestResult(
        equity=equity,
        total_return=float(equity.iloc[-1] - 1.0),
        max_drawdown=float(drawdown.min()),
        position=prev_pos,
        symbol=bars.symbol,
    )


def vol_target_size(
    bars: Bars,
    target_vol: float = 0.15,
    lookback: int = 30,
    max_leverage: float = 2.0,
) -> pd.Series:
    """Множитель размера позиции для таргетирования волатильности.

    size = target_vol / realized_vol, с потолком по плечу. realized_vol
    аннуализируется через bars.bars_per_year — корректно на любом
    таймфрейме (252 дневные, ~1512 H4), формула одна.

    Vol targeting масштабирует риск И доходность одним множителем — не
    создаёт прибыль, только меняет масштаб (вывод EMA-трека). Настоящая
    сила — на уровне портфеля (выравнивает вклад инструментов).

    Args:
        bars: Данные инструмента.
        target_vol: Целевая годовая волатильность (0.15 = 15%).
        lookback: Окно оценки реализованной волатильности.
        max_leverage: Потолок множителя (защита от деления на ~0 vol).

    Returns:
        Ряд множителей размера [0, max_leverage].
    """
    daily_vol = bars.returns().rolling(lookback).std()
    annual_vol = daily_vol * np.sqrt(bars.bars_per_year)
    size = (target_vol / annual_vol).clip(upper=max_leverage)
    return size.fillna(0.0)
