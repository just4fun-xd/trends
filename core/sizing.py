"""Реестр сайзеров: единый выбор realized-VT / GARCH-VT для раннеров.

Проблема, которую решает модуль: core/garch.py был готов, но НЕ был
подключён ни к одному раннеру — флаг --vt везде жёстко брал
vol_target_size (rolling realized). Теперь у всех раннеров один
контракт:

    sizer = make_sizer("garch", target_vol=0.25)
    pos = raw_signal * sizer(bars)

Имена реестра:
    realized — rolling std за 30 баров (vol_target_size, статус-кво);
    garch    — GARCH(1,1) one-step-ahead прогноз (garch_vol_target_size).

Оба сайзера делят потолок плеча и буфер ребалансировки (аудит 2026-07),
поэтому A/B «realized vs garch» изолирует ровно одну переменную —
оценку волатильности в знаменателе.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from core.bars import Bars
from core.engine import vol_target_size
from core.garch import garch_vol_target_size

SizerFn = Callable[[Bars], pd.Series]

SIZERS: dict[str, Callable] = {
    "realized": vol_target_size,
    "garch": garch_vol_target_size,
}


def make_sizer(
    name: str = "realized",
    target_vol: float = 0.15,
    max_leverage: float = 2.0,
    buffer: float = 0.10,
) -> SizerFn:
    """Фабрика сайзера по имени реестра.

    Args:
        name: 'realized' или 'garch'.
        target_vol: Целевая годовая волатильность.
        max_leverage: Потолок множителя.
        buffer: Мёртвая зона ребалансировки.

    Returns:
        Функция Bars -> Series множителей позиции.

    Raises:
        KeyError: Неизвестное имя сайзера.
    """
    if name not in SIZERS:
        raise KeyError(
            f"нет сайзера {name!r}; доступны: {sorted(SIZERS)}"
        )
    base = SIZERS[name]

    def sizer(bars: Bars) -> pd.Series:
        """Множитель позиции для данного инструмента."""
        return base(
            bars, target_vol=target_vol,
            max_leverage=max_leverage, buffer=buffer,
        )

    return sizer
