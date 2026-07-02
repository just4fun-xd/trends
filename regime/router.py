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

import pandas as pd

from core.bars import Bars
from core.engine import vol_target_size
from regime.detector import AlwaysTrendDetector, Regime, RegimeDetector
from strategies.donchian import donchian_champion_raw
from strategies.ou import ou_zscore


def regime_router(
    bars: Bars,
    detector: RegimeDetector | None = None,
    target_vol: float = 0.15,
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
    (проверяется тестом).

    Мягкое смешивание (не жёсткое переключение) — избегает рывков на
    границе режимов.

    Args:
        bars: Данные инструмента.
        detector: Детектор режима. None -> AlwaysTrendDetector (роутер
            вырождается в чистый champion — база для sanity-чека).
        target_vol: Единый волатильностный бюджет итоговой позиции.

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
    return (mixed * size).fillna(0.0)
