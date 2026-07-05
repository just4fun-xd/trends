"""Оверлеи поверх сигналов: защита от структурных коллапсов.

vol_percentile_gate — ответ на провал mr_ens в CL апрель-2020 (−64%
realized / −86% GARCH при отрицательной цене WTI). Механизм провала:
long-only mean-reversion не отличает «откат в тренде» (торгуемая
перепроданность) от «структурный коллапс рынка» (падающий нож). ATR-стоп
не спасает — при гэпах через экстремальные уровни ATR теряет смысл.

Диагностический признак коллапса, видимый БЕЗ look-ahead: реализованная
волатильность выходит за пределы собственного исторического
распределения. В апреле-2020 20-дневная вола WTI была выше ЛЮБОГО
значения за предшествующие годы. Гейт: позиция разрешена только пока
текущая вола ниже q-го перцентиля своей trailing-истории.

Это НЕ закрытый HMM-роутер и НЕ закрытый EMA200-фильтр: те угадывали
режим ТРЕНДА (и опаздывали); гейт меряет режим ВОЛАТИЛЬНОСТИ — прямую
причину провала — одним параметром (перцентиль).

Гейт умножается на позицию, т.е. блокирует и новые входы, и принудительно
закрывает удерживаемые позиции при взрыве волы — хвост режется с обеих
сторон. Look-ahead нет: rolling-ранг на баре t использует данные <= t,
торговля идёт через shift(1) движка.
"""

from __future__ import annotations

import pandas as pd

from core.bars import Bars


def vol_percentile_gate(
    bars: Bars,
    vol_lookback: int = 20,
    rank_window: int = 500,
    pctl: float = 0.90,
) -> pd.Series:
    """Бинарный гейт: 1 пока вола ниже перцентиля своей истории.

    Args:
        bars: Данные инструмента.
        vol_lookback: Окно реализованной волы (баров).
        rank_window: Trailing-окно распределения для ранга (~2 года
            дневных). Меньше — быстрее адаптация, но шумнее хвост.
        pctl: Порог перцентиля (0.90 = блок в верхних 10% волы).

    Returns:
        Ряд {0.0, 1.0}; на прогреве (нет ранга) — 1.0, чтобы гейт не
        выключал стратегию на старте истории.
    """
    vol = bars.returns().rolling(vol_lookback).std()
    rank = vol.rolling(rank_window, min_periods=rank_window // 2).rank(
        pct=True
    )
    gate = (rank < pctl).astype(float)
    return gate.where(~rank.isna(), 1.0)


def with_vol_gate(
    strategy_fn,
    vol_lookback: int = 20,
    rank_window: int = 500,
    pctl: float = 0.90,
):
    """Оборачивает стратегию гейтом: position * gate.

    Args:
        strategy_fn: Базовая стратегия (Bars -> position).
        vol_lookback: Окно волы гейта.
        rank_window: Окно распределения.
        pctl: Порог перцентиля.

    Returns:
        Новая функция стратегии с тем же контрактом.
    """

    def gated(bars: Bars) -> pd.Series:
        """Позиция базовой стратегии, обнулённая при взрыве волы."""
        return strategy_fn(bars) * vol_percentile_gate(
            bars, vol_lookback, rank_window, pctl
        )

    return gated
