"""Monday range breakout — недельный пробой опорного диапазона.

Бенчмарк Александра (~100%/год на крипте интрадей, пережил крах BTC).
Спецификация — docs/MONDAY_RANGE.md.

Идея: в начале недели фиксируется коридор [range_low, range_high] по
опорному периоду (понедельник целиком на daily, или первые N баров
недели на интрадей). Дальше неделю торгуем ПРОБОЙ коридора: цена вышла
вверх -> лонг, вниз -> шорт (или 0 для long-only). Это трендовая
ставка на продолжение импульса, не реверсия.

Работает на любом таймфрейме: неделя определяется по календарю
(isocalendar), опорный период — первые `ref_bars` баров недели. На
daily ref_bars=1 (понедельник); на H4 ref_bars=6 (первые сутки);
на H1 ref_bars=24.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars


def monday_range(
    bars: Bars,
    ref_bars: int = 1,
    long_only: bool = False,
    buffer: float = 0.0,
) -> pd.Series:
    """Позиция по пробою недельного опорного диапазона.

    В начале каждой ISO-недели берутся первые `ref_bars` баров как
    опорный период; их max/min образуют коридор. Затем:
      close > range_high*(1+buffer) -> +1 (лонг),
      close < range_low*(1-buffer)  -> -1 (шорт) или 0 (long_only),
      внутри коридора -> держим предыдущую позицию.
    Коридор сбрасывается на границе недели. Сдвиг 1 бар (пробой на t
    торгуется на t+1) исключает look-ahead.

    Args:
        bars: Данные инструмента (любой таймфрейм).
        ref_bars: Сколько первых баров недели формируют коридор
            (daily=1 пн; H4=6 сутки; H1=24).
        long_only: True -> без шорта (крипта-спот).
        buffer: Отступ пробоя в долях (0.001 = 0.1%) — фильтр ложных
            пробоев у самой границы.

    Returns:
        position в {-1, 0, +1}, сдвинута на 1 бар.
    """
    idx = bars.close.index
    # ISO (год, неделя) для нарезки недель по календарю.
    iso = pd.DataFrame({
        "year": idx.isocalendar().year.to_numpy(),
        "week": idx.isocalendar().week.to_numpy(),
    }, index=idx)
    week_id = (iso["year"].astype(int) * 100 + iso["week"].astype(int))

    high = bars.high.to_numpy(dtype=float)
    low = bars.low.to_numpy(dtype=float)
    close = bars.close.to_numpy(dtype=float)
    wid = week_id.to_numpy()

    n = len(close)
    pos = np.zeros(n)
    cur_week = None
    seen = 0          # баров опорного периода в текущей неделе
    r_hi, r_lo = np.nan, np.nan
    state = 0
    for i in range(n):
        if wid[i] != cur_week:
            # Новая неделя: сброс коридора и счётчиков.
            cur_week = wid[i]
            seen = 0
            r_hi, r_lo = -np.inf, np.inf
            state = 0
        if seen < ref_bars:
            # Ещё формируем опорный диапазон — расширяем границы.
            r_hi = max(r_hi, high[i])
            r_lo = min(r_lo, low[i])
            seen += 1
            pos[i] = float(state)
            continue
        # Коридор готов — торгуем пробой.
        up = r_hi * (1.0 + buffer)
        dn = r_lo * (1.0 - buffer)
        if close[i] > up:
            state = 1
        elif close[i] < dn:
            state = 0 if long_only else -1
        pos[i] = float(state)
    # ФИКС 2026-07j: двойной shift удалён (движок сдвигает сам).
    return pd.Series(pos, index=idx).fillna(0.0)


def monday_range_h4(bars: Bars) -> pd.Series:
    """Long-only H4-пресет: опорный период — первые сутки недели."""
    return monday_range(bars, ref_bars=6, long_only=True)


def monday_range_h1(bars: Bars) -> pd.Series:
    """Long-only H1-пресет: опорный период — первые сутки недели."""
    return monday_range(bars, ref_bars=24, long_only=True)


MONDAY_RANGE = {
    "brk_monday_range": monday_range,        # daily long+short
    "brk_monday_h4": monday_range_h4,        # крипта H4 long-only
    "brk_monday_h1": monday_range_h1,        # крипта H1 long-only
}
