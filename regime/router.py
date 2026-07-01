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
from regime.detector import AlwaysTrendDetector, Regime, RegimeDetector
from strategies.donchian import donchian_est_macd_4step_take
from strategies.ou import ou_zscore


def regime_router(
    bars: Bars,
    detector: RegimeDetector | None = None,
    prob_threshold: float = 0.5,
) -> pd.Series:
    """Смешивает стратегии по вероятностям режимов от детектора.

    Позиция = взвешенная сумма стратегий по P(режим):
        pos = P(TREND)·donchian + P(RANGE)·ou + P(CRISIS)·0

    Мягкое смешивание (не жёсткое переключение) — избегает рывков на
    границе режимов. Каждая под-стратегия считается на всём ряду, затем
    взвешивается вероятностью своего режима в каждый момент.

    Args:
        bars: Данные инструмента.
        detector: Детектор режима. None -> AlwaysTrendDetector (роутер
            вырождается в чистый Donchian — база для sanity-чека).
        prob_threshold: Порог отсечки (режимы с P ниже игнорируются;
            сейчас мягкое смешивание, порог зарезервирован под жёсткий
            режим при необходимости).

    Returns:
        position: смешанная позиция по режимам.
    """
    if detector is None:
        detector = AlwaysTrendDetector()

    probs = detector.detect(bars)

    trend_pos = donchian_est_macd_4step_take(bars)
    range_pos = ou_zscore(bars)
    # CRISIS -> кэш (нулевая позиция), вклад ноль.

    mixed = (
        probs[Regime.TREND.value] * trend_pos
        + probs[Regime.RANGE.value] * range_pos
    )
    return mixed.fillna(0.0)
