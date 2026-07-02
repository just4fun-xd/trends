"""EMA trend-following — полный трек из EMA_cheat_sheet.md.

Вывод трека (зафиксирован): EMA-моментум на сырье не имеет устойчивого
края, кроме золота в трендовые годы. Все приёмы улучшения — управление
РИСКОМ, ни один не создаёт прибыль. Единственный проходящий DD<40% —
ema_ensemble_voltarget (max -21%). На АКЦИЯХ работает (структурный
восходящий дрейф). Портировано под контракт Bars: стратегия берёт
bars.close, возвращает position. Движок сам делает shift(1).

Все стратегии возвращают pd.Series position на индексе bars:
1.0 = полный лонг, 0 = кэш, дробное = vol targeting, <0 = шорт.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import vol_target_size

# Лесенка пар ансамбля 1:4 по Карверу (геометрическая).
CARVER_PAIRS = ((5, 20), (10, 40), (20, 80), (40, 160), (64, 256))


def ema_cross(bars: Bars, fast: int = 12, slow: int = 21) -> pd.Series:
    """Базовый EMA-кроссовер: лонг когда fast EMA выше slow EMA.

    Вывод: нестабилен по годам на сырье (Gas -64%). Бенчмарк, не боевая.

    Args:
        bars: Данные инструмента.
        fast: Период быстрой EMA.
        slow: Период медленной EMA.

    Returns:
        position: 1.0 когда fast>slow, иначе 0.
    """
    ef = bars.close.ewm(span=fast, adjust=False).mean()
    es = bars.close.ewm(span=slow, adjust=False).mean()
    return (ef > es).astype(float)


def ema_cross_stop(
    bars: Bars, fast: int = 12, slow: int = 21, stop: float = 0.10
) -> pd.Series:
    """EMA-кроссовер со стоп-лоссом от цены входа.

    Вывод: стоп лечит не ту болезнь. Наша просадка — сумма мелких
    убытков в боковике, стоп ограничивает глубину ОДНОГО. Не помогает.
    Требует последовательной логики (состояние = цена входа) — цикл.

    Args:
        bars: Данные инструмента.
        fast: Быстрая EMA.
        slow: Медленная EMA.
        stop: Доля просадки от входа для выхода (0.10 = 10%).

    Returns:
        position: 1.0 в лонге до срабатывания стопа, иначе 0.
    """
    ef = bars.close.ewm(span=fast, adjust=False).mean()
    es = bars.close.ewm(span=slow, adjust=False).mean()
    bull = (ef > es).values
    low = bars.low.values
    close = bars.close.values
    pos = np.zeros(len(close))
    entry = 0.0
    in_pos = False
    stopped = False  # реарм: после стопа ждём сброса сигнала
    for i in range(len(close)):
        if in_pos:
            # Аудит 2026-07: стоп триггерится по интрадей low (close-
            # проверка прятала внутрибарный пробой), и после стопа вход
            # заблокирован до сброса bull — раньше код перевходил В ТОМ
            # ЖЕ баре, и стоп лишь сбрасывал entry, не закрывая позицию.
            if low[i] <= entry * (1 - stop):
                in_pos = False
                stopped = True
            elif not bull[i]:
                in_pos = False
            else:
                pos[i] = 1.0
        else:
            if stopped and not bull[i]:
                stopped = False  # сигнал сбросился — реарм
            if not stopped and bull[i]:
                in_pos = True
                entry = close[i]
                pos[i] = 1.0
    return pd.Series(pos, index=bars.index)


def sma_trend(bars: Bars, period: int = 200) -> pd.Series:
    """Цена выше SMA(period) — простой медленный трендовый бенчмарк.

    На золоте в трендовые годы БЬЁТ кроссовер (ловит крупное движение).
    Честный факт для отчёта: кроссовер не доминирует над тупой SMA200.

    Args:
        bars: Данные инструмента.
        period: Окно SMA.

    Returns:
        position: 1.0 когда close выше SMA, иначе 0.
    """
    sma = bars.close.rolling(period).mean()
    return (bars.close > sma).astype(float)


def ema_trend(bars: Bars, period: int = 200) -> pd.Series:
    """Цена выше EMA(period) — вариант бенчмарка на EMA вместо SMA.

    Args:
        bars: Данные инструмента.
        period: Окно EMA.

    Returns:
        position: 1.0 когда close выше EMA, иначе 0.
    """
    ema = bars.close.ewm(span=period, adjust=False).mean()
    return (bars.close > ema).astype(float)


def _ensemble_vote(
    close: pd.Series, pairs=CARVER_PAIRS, threshold: float = 0.5
) -> pd.Series:
    """Доля бычьих голосов пар EMA превышает порог.

    Каждая пара голосует «бык» (fast>slow) или «медведь». Входим, когда
    доля быков > threshold. Механизм: требует согласия таймфреймов —
    меньше входов в шум. Устойчив к параметрам (плато 0.5..0.6).

    Args:
        close: Ряд цен закрытия.
        pairs: Набор (fast, slow) пар.
        threshold: Порог доли бычьих голосов.

    Returns:
        position: 1.0 когда голосов достаточно, иначе 0.
    """
    votes = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ef = close.ewm(span=fast, adjust=False).mean()
        es = close.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}_{slow}"] = (ef > es).astype(float)
    bull_frac = votes.mean(axis=1)
    return (bull_frac > threshold).astype(float)


def ema_ensemble(
    bars: Bars, pairs=CARVER_PAIRS, threshold: float = 0.5
) -> pd.Series:
    """Ансамбль EMA: несколько пар голосуют за направление.

    Снижает просадки (Gas 2025 -63%->-43%), но режет и прибыль — меньше
    торгует. Делает убыточное менее убыточным, не прибыльным. Не проходит
    DD<40% на сырье сам по себе (Gas -43%).

    Args:
        bars: Данные инструмента.
        pairs: Набор пар EMA.
        threshold: Порог голосования.

    Returns:
        position: сырой сигнал ансамбля [0, 1].
    """
    return _ensemble_vote(bars.close, pairs, threshold)


def ema_ensemble_voltarget(
    bars: Bars,
    pairs=CARVER_PAIRS,
    threshold: float = 0.5,
    target_vol: float = 0.15,
    max_leverage: float = 2.0,
) -> pd.Series:
    """ЧЕМПИОН EMA-трека: ансамбль + vol targeting.

    Единственный EMA-вариант, проходящий DD<40% на всех инструментах
    (max -21.2%, worst case -23.3% на широкой корзине). Устойчив к
    параметрам. На АКЦИЯХ — сильнейший equity-перформер (Gold +110%,
    акции x2+). vol targeting берёт bars_per_year из Bars — корректен
    на H4.

    Args:
        bars: Данные инструмента.
        pairs: Набор пар EMA.
        threshold: Порог голосования.
        target_vol: Целевая годовая волатильность.
        max_leverage: Потолок плеча.

    Returns:
        position: сигнал ансамбля, масштабированный vol targeting.
    """
    signal = _ensemble_vote(bars.close, pairs, threshold)
    size = vol_target_size(bars, target_vol, max_leverage=max_leverage)
    return signal * size


def ema_ensemble_long_short(
    bars: Bars, pairs=CARVER_PAIRS, threshold: float = 0.5
) -> pd.Series:
    """L/S версия ансамбля — ЗАКРЫТА как неэффективная.

    Симметричный шорт против структурно растущего актива обречён:
    шортит откаты, ловит отскок вверх. Портфель -35%, 1/19 прибыльных.
    Оставлена для честной документации провала (не удаляется).

    Args:
        bars: Данные инструмента.
        pairs: Набор пар EMA.
        threshold: Порог голосования.

    Returns:
        position: +1 при быках, -1 при медведях (зеркально).
    """
    bull = _ensemble_vote(bars.close, pairs, threshold)
    return bull * 2.0 - 1.0  # {0,1} -> {-1,+1}


def ema_ensemble_voltarget_ls(
    bars: Bars,
    pairs=CARVER_PAIRS,
    threshold: float = 0.5,
    target_vol: float = 0.15,
) -> pd.Series:
    """L/S ансамбль + vol targeting — ЗАКРЫТА.

    Формально проходит DD<40% только из-за vol targeting (урезает
    позицию), не из-за качества шорт-сигнала. Недоходность сохраняется.

    Args:
        bars: Данные инструмента.
        pairs: Набор пар EMA.
        threshold: Порог голосования.
        target_vol: Целевая волатильность.

    Returns:
        position: зеркальный сигнал, масштабированный vol targeting.
    """
    ls = ema_ensemble_long_short(bars, pairs, threshold)
    size = vol_target_size(bars, target_vol)
    return ls * size


# Барбелл-вариант: находка из web research — средний горизонт (~125д)
# бесполезен, размывает и convexity коротких, и persistence длинных.
BARBELL_PAIRS = ((5, 20), (64, 256))


def ema_ensemble_barbell_voltarget(
    bars: Bars, target_vol: float = 0.15
) -> pd.Series:
    """Барбелл-ансамбль: только короткая и длинная пары + vol targeting.

    Проверка находки из research: убрать средние пары. Короткая ловит
    convexity, длинная — persistence, среднее только размывает.

    Args:
        bars: Данные инструмента.
        target_vol: Целевая волатильность.

    Returns:
        position: барбелл-сигнал, масштабированный vol targeting.
    """
    return ema_ensemble_voltarget(
        bars, pairs=BARBELL_PAIRS, target_vol=target_vol
    )
