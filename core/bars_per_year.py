"""Фактический bars_per_year из индекса панели (не константа).

Зачем: на дневках 252 — честная константа. На H4 фьючерсов число
баров в году зависит от сессии инструмента (CME Globex ~23ч, но
воскресное открытие и праздники дают рваньё): CL ~1589, у газа/
металлов/зерна своё. Зашитые 1512 (=6×252) смещают Sharpe у каждого
инструмента по-разному — аннуализация едет на sqrt(факт/1512).

Решение: считать bpy из самого индекса. Медиана по ПОЛНЫМ годам
(первый/последний год обычно частичные — отбрасываем) устойчивее
среднего span-оценки к рваным краям выборки.

Использование в DatabentoSource.load: вместо хардкода
    bars_per_year = 1512.0 if interval == "4h" else 252.0
подставить
    bars_per_year = infer_bars_per_year(index, interval)
"""

from __future__ import annotations

import pandas as pd


def infer_bars_per_year(index: pd.DatetimeIndex, interval: str) -> float:
    """Оценивает bars_per_year из фактического индекса.

    Для дневок возвращает каноническую 252 (константа честна). Для
    интрадея — медиану числа баров по полным календарным годам
    выборки; при нехватке лет откатывается к span-оценке.

    Args:
        index: DatetimeIndex панели (одного инструмента или общий).
        interval: Таймфрейм ('1d' — вернёт 252.0; иначе считает).

    Returns:
        Оценка bars_per_year (float). Дефолт 252.0 на вырожденном
        входе (< 2 баров).
    """
    if interval == "1d":
        return 252.0
    if index is None or len(index) < 2:
        return 252.0

    per_year = index.to_series().groupby(index.year).size()
    if len(per_year) > 2:
        # Отбрасываем частичные крайние годы, берём медиану полных.
        full = per_year.iloc[1:-1]
        if len(full) > 0:
            return float(full.median())
    # Мало лет — span-оценка по всему диапазону.
    span_years = (index[-1] - index[0]).days / 365.25
    if span_years <= 0:
        return 252.0
    return float(len(index) / span_years)
