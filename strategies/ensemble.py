"""Ансамбли на уровне СИГНАЛА (позиции), не P&L.

Два ансамбля, два разных механизма:

1. mr_ensemble — усреднение позиций K вариантов mean-reversion лаборатории.
   Механизм: 10 вариантов на одном периоде = multiple testing; выбор
   «лучшего» — подгонка под выборку. Ансамбль убирает сам выбор: если
   у вариантов общий сигнал (реверсия) и разные шумовые компоненты
   (выходы: ATR-стоп / time-stop / Keltner / подтверждение), усреднение
   сохраняет сигнал и режет дисперсию оценки. Позиция становится
   дробной [0..1] — «сколько вариантов сейчас в лонге» — что само по
   себе мягкий сайзинг по согласию (conviction sizing).

2. trend_mr_combo — champion (Donchian, тренд) + mr_ensemble (реверсия)
   на ОДНОМ инструменте, веса на уровне позиции. Механизм тот же, что
   у run_sleeves (противоположные режимы -> низкая корреляция P&L), но
   внутри инструмента: в трендовые годы работает тренд-нога, в боковые
   — MR-нога, кэш обеих ног не пересекается по построению.

Отличие от run_sleeves: там комбинация ДНЕВНЫХ P&L разных корзин,
здесь — позиций на одном инструменте. Сигнальный ансамбль дешевле по
обороту: противоположные изменения ног неттингуются ДО издержек.

Все функции соблюдают посерийный контракт (Bars -> position) и
регистрируются в STRATEGIES раннера. VT-слой (realized или GARCH)
накладывается снаружи флагом --vt/--sizer, внутри ансамбля VT нет —
иначе двойное таргетирование.
"""

from __future__ import annotations

import pandas as pd

from core.bars import Bars
from strategies.donchian import donchian_champion_raw
from strategies.meanrev_lab import (
    mr_atr_stop,
    mr_confirm,
    mr_keltner,
    mr_time_stop,
)

# Дефолтный состав MR-ансамбля: варианты с ОБЩИМ входом-механизмом
# (перепроданность) и РАЗНЫМИ механизмами выхода/фильтра — именно
# такая смесь даёт декорреляцию ошибок, а не K копий одного сигнала.
MR_ENSEMBLE_MEMBERS = (
    mr_atr_stop,    # риск-стоп: обрезает хвост
    mr_time_stop,   # срок годности реверсии
    mr_keltner,     # другая мера растяжения (ATR-полосы)
    mr_confirm,     # вход только по подтверждению разворота
)


def mr_ensemble(bars: Bars, members=MR_ENSEMBLE_MEMBERS) -> pd.Series:
    """Равновзвешенное среднее позиций MR-вариантов.

    Args:
        bars: Данные инструмента.
        members: Кортеж функций-вариантов (Bars -> position 0/1).

    Returns:
        position в [0, 1]: доля вариантов, находящихся в лонге.
    """
    acc = None
    for fn in members:
        p = fn(bars).fillna(0.0)
        acc = p if acc is None else acc + p
    return acc / float(len(members))


def trend_mr_combo(
    bars: Bars, w_trend: float = 0.5,
) -> pd.Series:
    """Комбо тренд + реверсия на одном инструменте (уровень позиции).

    position = w_trend * donchian_champion_raw + (1-w_trend) * mr_ensemble

    Сырые ноги (без VT), масштаб задаётся внешним сайзером. Пирамида
    champion'а даёт ногу в [0..1] ступенями, MR-ансамбль — [0..1]
    дробно; итог в [0..1] при любых весах из [0,1].

    Args:
        bars: Данные инструмента.
        w_trend: Вес тренд-ноги (0.5 = поровну).

    Returns:
        position: Комбинированная позиция [0, 1].
    """
    if not 0.0 <= w_trend <= 1.0:
        raise ValueError("w_trend должен быть в [0, 1]")
    trend = donchian_champion_raw(bars).fillna(0.0)
    mr = mr_ensemble(bars)
    return w_trend * trend + (1.0 - w_trend) * mr


ENSEMBLES = {
    "mr_ens": mr_ensemble,
    "combo_tmr": trend_mr_combo,
}
