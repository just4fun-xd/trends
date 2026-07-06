"""Движок бэктеста для посерийных (per-instrument) стратегий.

Считает деньги: position -> equity -> drawdown. Единственное место в
проекте, где живёт shift(1) — защита от look-ahead. Стратегия возвращает
«сырую» позицию (числовой ряд: 1.0 = полный лонг, 0 = кэш, дробное при
vol targeting, отрицательное = шорт). Движок сам сдвигает и считает.

МОДЕЛЬ ИСПОЛНЕНИЯ (зафиксирована явно, аудит 2026-07):
  - Позиция = доля NAV, ребалансируемая к целевой КАЖДЫЙ бар.
  - Формула P&L (equity *= 1 + pos*r) валидна ТОЛЬКО при ежедневной
    ребалансировке — значит, оборот обязан учитывать дрейф весов.
  - Оборот_t = |target_t − drifted_{t-1}|, где drifted — вчерашний вес
    после движения цены: w(1+r)/(1+w·r). Для w=1.0 дрейф нулевой
    (buy-and-hold бесплатен), для дробных весов ребаланс платный.
    Старая формула diff() занижала издержки при дробных позициях.

Аннуализация волатильности берёт bars_per_year из Bars, а не хардкодит
252 — H4/интрадей корректны без спец-веток.
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
        bars_per_year: Баров в году — для аннуализации Sharpe. Хранится
            честным полем, а не magic-атрибутом Series (pandas 3.0 CoW
            терял атрибут при копировании -> Sharpe тихо считался с 252
            на H4-данных).
        symbol: Инструмент.
    """

    equity: pd.Series
    total_return: float
    max_drawdown: float
    position: pd.Series
    bars_per_year: float = 252.0
    symbol: str = ""
    rf: float = 0.0

    @property
    def sharpe(self) -> float:
        """Годовой excess-Sharpe кривой капитала (вычет rf).

        Соглашение идентично diagnostics.bootstrap._sharpe:
        excess = mean − rf/bars_per_year, затем аннуализация.
        При rf=0 сводится к прежнему поведению.
        """
        rets = self.equity.pct_change().dropna()
        if rets.std() == 0 or len(rets) < 2:
            return 0.0
        excess = rets.mean() - self.rf / self.bars_per_year
        return float(excess / rets.std() * np.sqrt(self.bars_per_year))

    def passes_dd(self, limit: float = 0.40) -> bool:
        """Проходит ли жёсткий лимит максимальной просадки.

        Args:
            limit: Порог DD (положительное число, 0.40 = 40%).

        Returns:
            True если |max_drawdown| <= limit.
        """
        return abs(self.max_drawdown) <= limit


def drift_turnover(prev_pos: pd.Series, returns: pd.Series) -> pd.Series:
    """Оборот с учётом дрейфа веса (общая формула для обоих движков).

    Вчерашний целевой вес w за день дрейфует до w(1+r)/(1+w·r)
    (позиция выросла на r, NAV — на w·r). Сегодняшний оборот — расстояние
    от нового целевого веса до этого дрейфовавшего:

        turnover_t = |prev_pos_t − drifted_{t-1}|

    Свойства: w=1.0 -> drifted=1.0 -> buy-and-hold бесплатен; w=0 ->
    бесплатен; вход 0->w платит |w|; удержание дробного w платит
    ~|w(1-w)r| в день (ребаланс против дрейфа).

    Args:
        prev_pos: Применяемая позиция (уже после shift(1)).
        returns: Побарные доходности инструмента.

    Returns:
        Ряд оборота >= 0.
    """
    denom = 1.0 + prev_pos * returns
    # denom -> 0 означает потерю ~100% NAV за бар — за пределами
    # осмысленного бэктеста; защищаемся от численного взрыва.
    denom = denom.where(denom.abs() > 1e-9)
    drifted = (prev_pos * (1.0 + returns) / denom).shift(1)
    return (prev_pos - drifted.fillna(0.0)).abs().fillna(0.0)


def run_engine(
    bars: Bars,
    position: pd.Series,
    trade_start: str | None = None,
    cost: float = 0.0002,
    rf: float = 0.0,
) -> BacktestResult:
    """Прогоняет позицию через движок, возвращает метрики.

    Единственная точка shift(1) в проекте. Логика:
        returns  = close.pct_change()
        strat    = returns * position.shift(1)     # торгуем по вчера
        strat   -= drift_turnover(...) * cost      # издержки с дрейфом
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

    turnover = drift_turnover(prev_pos, returns)

    strat = returns * prev_pos - turnover * cost

    if trade_start is not None:
        # До trade_start P&L обнуляется (индикаторы грелись, не торговали).
        strat = strat.copy()
        strat[strat.index < pd.Timestamp(trade_start)] = 0.0

    strat = strat.fillna(0.0)
    equity = (1.0 + strat).cumprod()

    drawdown = equity / equity.cummax() - 1.0

    return BacktestResult(
        equity=equity,
        total_return=float(equity.iloc[-1] - 1.0),
        max_drawdown=float(drawdown.min()),
        position=prev_pos,
        bars_per_year=bars.bars_per_year,
        symbol=bars.symbol,
        rf=rf,
    )


def vol_target_size(
    bars: Bars,
    target_vol: float = 0.15,
    lookback: int = 30,
    max_leverage: float = 2.0,
    buffer: float = 0.10,
) -> pd.Series:
    """Множитель размера позиции для таргетирования волатильности.

    size = target_vol / realized_vol, с потолком по плечу. realized_vol
    аннуализируется через bars.bars_per_year — корректно на любом
    таймфрейме (252 дневные, ~1512 H4), формула одна.

    БУФЕР РЕБАЛАНСИРОВКИ (аудит 2026-07): сырой множитель дрожит каждый
    бар вместе с rolling std, и с drift-aware издержками это дрожание
    платное («смерть от тысячи порезов»). Поэтому применённый размер
    обновляется до сырого ТОЛЬКО когда расхождение превышает buffer
    (относительно применённого). buffer=0 возвращает старое непрерывное
    поведение. Вводит состояние -> цикл; look-ahead нет (только прошлое).

    Vol targeting масштабирует риск И доходность одним множителем — не
    создаёт прибыль, только меняет масштаб (вывод EMA-трека).

    Args:
        bars: Данные инструмента.
        target_vol: Целевая годовая волатильность (0.15 = 15%).
        lookback: Окно оценки реализованной волатильности.
        max_leverage: Потолок множителя (защита от деления на ~0 vol).
        buffer: Мёртвая зона ребалансировки (0.10 = не трогаем позицию,
            пока новый размер в пределах ±10% от применённого).

    Returns:
        Ряд множителей размера [0, max_leverage].
    """
    bar_vol = bars.returns().rolling(lookback).std()
    annual_vol = bar_vol * np.sqrt(bars.bars_per_year)
    raw = (target_vol / annual_vol).clip(upper=max_leverage)
    raw_v = raw.to_numpy()

    out = np.zeros(len(raw_v))
    applied = 0.0
    for i in range(len(raw_v)):
        r = raw_v[i]
        if np.isnan(r):
            out[i] = applied  # прогрев: держим текущее (0 в начале)
            continue
        if applied == 0.0:
            applied = r
        elif abs(r - applied) > buffer * abs(applied):
            applied = r
        out[i] = applied
    return pd.Series(out, index=bars.index)
