"""Regime Router — переключает стратегии под детектором режима (трек 2.3).

Это архитектурный ответ Александру на «hundreds of strategies that switch
for different modes of the market» — реализованный ЧЕСТНО через
математику детектора, а не ручным переключением.

Ключевая идея (из OU_RESULTS.md): Donchian и OU беззащитны против разных
режимов — тренд-стратегии тонут в боковике, OU тонет в тренде. Роутер
даёт каждой её родной режим:
    P(TREND) высокая -> Donchian champion
    P(RANGE) высокая -> OU z-score
    P(CRISIS) высокая -> кэш (позиция 0)

Роутер РАБОТАЕТ уже сейчас с любым RegimeDetector. С HMMDetector он
станет полноценным (когда трек 2.2 пройден); с VolatilityRegimeDetector —
отлаживается архитектурно; с AlwaysTrendDetector — вырождается в чистый
Donchian (проверяемо против прямого прогона).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import vol_target_size
from regime.detector import AlwaysTrendDetector, Regime, RegimeDetector
from strategies.donchian import donchian_champion_raw
from strategies.ou import ou_zscore


def _apply_position_buffer(
    pos: pd.Series, buffer: float
) -> pd.Series:
    """Гистерезис на итоговой позиции: не ребалансируем на микро-сдвиги.

    Обновляем применённую позицию, только когда она отходит от текущей
    больше чем на buffer (абсолютная мёртвая зона). Гасит дрожание,
    которое иначе платно через drift-turnover движок. buffer=0 ->
    поведение не меняется (возвращаем как есть).

    Args:
        pos: Сырая итоговая позиция роутера.
        buffer: Абсолютная мёртвая зона (0.05 = не трогаем, пока
            |new - applied| <= 0.05). 0 -> выключено.

    Returns:
        Позиция после гистерезиса.
    """
    if buffer <= 0:
        return pos
    raw = pos.to_numpy()
    out = np.zeros(len(raw))
    applied = 0.0
    for i in range(len(raw)):
        v = raw[i]
        if np.isnan(v):
            out[i] = applied
            continue
        if abs(v - applied) > buffer:
            applied = v
        out[i] = applied
    return pd.Series(out, index=pos.index)


def regime_router(
    bars: Bars,
    detector: RegimeDetector | None = None,
    target_vol: float = 0.15,
    position_buffer: float = 0.0,
) -> pd.Series:
    """Смешивает СЫРЫЕ стратегии по вероятностям режимов, затем один
    общий vol_target на итог.

    РИСК-ПАРИТЕТ РЕЖИМОВ (аудит 2026-07, критическая правка): раньше
    роутер складывал разнокалиберные ноги — Дончиан приходил уже ужатый
    внутренним vol targeting (~0.2-0.5), а OU лупил сырым ±1.0. В RANGE-
    режиме роутер нёс в разы больше риска, чем в TREND. Теперь обе ноги
    сырые [-1..1], а волатильностный бюджет накладывается ОДИН раз на
    смешанный сигнал:

        mixed = P(TREND)·donchian_raw + P(RANGE)·ou_raw    # CRISIS -> 0
        pos   = mixed · vol_target_size(bars, target_vol)

    Тождество сохранено: champion = raw · vol_size, поэтому под
    AlwaysTrendDetector роутер по-прежнему В ТОЧНОСТИ равен champion
    (проверяется тестом) — при position_buffer=0.

    ГИСТЕРЕЗИС ПОЗИЦИИ (аудит 4, Gemini; задел под HMM): position_buffer
    гасит дрожание ИТОГОВОЙ позиции. Мотив верен: когда probs даёт
    реальная HMM, они меняются каждый бар (0.60->0.63->0.58), и буфер
    внутри vol_target_size это НЕ ловит — дрожат probs, а не размер.
    Небуферизованный mixed микро-ребалансируется, drift-движок берёт
    плату. По умолчанию 0 (поведение не меняется). ВАЖНО: это грубый
    буфер на выходе; более правильный гистерезис режимов — dwell-time
    на самих probs (не переключать режим, пока P устойчиво > порога N
    баров), он не имеет проблемы знака на границе. Пойдёт вместе с
    реализацией HMM (трек 2.2), где и появится источник дрожания.

    Args:
        bars: Данные инструмента.
        detector: Детектор режима. None -> AlwaysTrendDetector (роутер
            вырождается в чистый champion — база для sanity-чека).
        target_vol: Единый волатильностный бюджет итоговой позиции.
        position_buffer: Мёртвая зона ребаланса итоговой позиции (0 ->
            выключено; ~0.05 разумно для дрожащей HMM).

    Returns:
        position: смешанная позиция с единым риск-бюджетом.
    """
    if detector is None:
        detector = AlwaysTrendDetector()

    probs = detector.detect(bars)

    trend_raw = donchian_champion_raw(bars)
    range_raw = ou_zscore(bars)
    # CRISIS -> кэш (нулевая позиция), вклад ноль.

    mixed = (
        probs[Regime.TREND.value] * trend_raw
        + probs[Regime.RANGE.value] * range_raw
    )
    size = vol_target_size(bars, target_vol)
    pos = (mixed * size).fillna(0.0)
    return _apply_position_buffer(pos, position_buffer)
