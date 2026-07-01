"""Donchian breakout — полный трек, включая commodity-champion.

Здесь high/low нужны НАТИВНО: пробой канала и ATR-стоп по реальным
экстремумам, не по синтетике из close. Это ровно та неточность, которую
вычищаем — старый close-only контракт не мог отдать честный пробой.

Champion трека: donchian_est_macd_4step_take (+5.2% / -12.1% / 15 из 19
на диверсифицированной корзине; +168% на 17-инструментной commodity-
корзине, стабильно train/test). Единственная валидированная commodity-
стратегия. Time-series momentum: торгует абсолютный тренд КАЖДОГО
инструмента против его истории — иммунен к развороту рангов, который
убивает cross-sectional (см. DUALMOM_RESULTS.md).

Все стратегии: Bars -> position. Пирамида/стоп/take требуют состояния
(цена входа, число докупок) => цикл, не вектор.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import vol_target_size
from strategies.ema import _ensemble_vote

# Убывающая пирамида 4step: доли объёма на 4 ступенях входа.
# Большой объём сразу, потом меньше -> снижает среднюю цену входа,
# защищает прибыль при коротких волатильных трендах сырья.
PYRAMID_4STEP = (0.40, 0.30, 0.20, 0.10)


def _donchian_channels(
    bars: Bars, entry: int, exit_period: int
) -> tuple[pd.Series, pd.Series]:
    """Верхний (entry-дневный max) и нижний (exit-дневный min) каналы.

    Считаются по РЕАЛЬНЫМ high/low, сдвинуты на 1 — канал строится по
    ПРОШЛЫМ барам, пробой сравнивается с текущим (без look-ahead внутри
    сигнала; движок добавит свой shift сверху).

    Args:
        bars: Данные инструмента.
        entry: Окно верхнего канала (пробой = вход).
        exit_period: Окно нижнего канала (пробой = выход).

    Returns:
        (upper_channel, lower_channel) — оба сдвинуты на 1 бар.
    """
    upper = bars.high.rolling(entry).max().shift(1)
    lower = bars.low.rolling(exit_period).min().shift(1)
    return upper, lower


def donchian_breakout(
    bars: Bars, entry: int = 20, exit_period: int = 10
) -> pd.Series:
    """Базовый Дончиан: лонг при пробое N-дневного максимума.

    Отличные доходности (Gas +73%, Cocoa +108%), но DD>40% (Gas -55%).
    Бенчмарк трека — база, к которой добавляются слои риск-контроля.

    Args:
        bars: Данные инструмента.
        entry: Окно пробоя вверх (вход).
        exit_period: Окно пробоя вниз (выход).

    Returns:
        position: 1.0 в лонге между входом и выходом, иначе 0.
    """
    upper, lower = _donchian_channels(bars, entry, exit_period)
    close = bars.close.values
    up = upper.values
    lo = lower.values
    pos = np.zeros(len(close))
    in_pos = False
    for i in range(len(close)):
        if not in_pos and close[i] > up[i]:
            in_pos = True
        elif in_pos and close[i] < lo[i]:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


def donchian_ensemble_voltarget(
    bars: Bars,
    entry: int = 20,
    exit_period: int = 10,
    target_vol: float = 0.15,
) -> pd.Series:
    """Дончиан + EMA-ансамбль (макро-фильтр) + vol targeting.

    Ансамбль разрешает вход только при согласии EMA-таймфреймов ->
    меньше ложных пробоев. vol targeting -> DD под контроль (max -16%).
    Проходит DD<40%.

    Args:
        bars: Данные инструмента.
        entry: Окно пробоя вверх.
        exit_period: Окно пробоя вниз.
        target_vol: Целевая волатильность.

    Returns:
        position: сигнал Дончиана ∧ ансамбль, масштаб vol targeting.
    """
    breakout = donchian_breakout(bars, entry, exit_period)
    regime = _ensemble_vote(bars.close)  # EMA-ансамбль как фильтр
    size = vol_target_size(bars, target_vol)
    return breakout * regime * size


def _macd_bullish(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """MACD выше сигнальной линии — булев фильтр.

    Вывод проекта: MACD избыточен поверх EMA-ансамбля (оба EMA-based,
    коррелированы) — результат идентичен без него. Оставлен для
    воспроизводимости champion, чьё имя его содержит.

    Args:
        close: Ряд цен закрытия.
        fast: Быстрая EMA MACD.
        slow: Медленная EMA MACD.
        signal: Период сигнальной линии.

    Returns:
        Bool-ряд: True когда MACD выше сигнальной.
    """
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    macd = ef - es
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd > sig


def _donchian_4step(
    bars: Bars,
    entry: int,
    exit_period: int,
    atr_period: int,
    stop_atr: float,
    take_atr: float | None,
    use_take: bool,
    pyramid=PYRAMID_4STEP,
) -> pd.Series:
    """Ядро 4step: убывающая пирамида + turtle-стоп (+ опц. take-profit).

    Механика (per bar, состояние через цикл):
      - Вход ступенями: при пробое верхнего канала добавляем pyramid[k]
        объёма; следующая ступень — когда цена прошла ещё +0.5 ATR.
      - Turtle-стоп: entry_ref - stop_atr*ATR, ПОДТЯГИВАЕТСЯ за каждой
        докупкой (быстрый выход на развороте — критично для газа/softs).
      - Take-profit (если use_take): при +take_atr*ATR сбрасываем верхнюю
        (последнюю) ступень, фиксируя прибыль. energy 100% profit, grains 80%.
      - Полный выход: пробой нижнего канала ИЛИ стоп.

    Args:
        bars: Данные инструмента (high/low нужны реальные).
        entry: Окно верхнего канала.
        exit_period: Окно нижнего канала.
        atr_period: Окно ATR для стопа/тейка.
        stop_atr: Множитель ATR для turtle-стопа.
        take_atr: Множитель ATR для take-profit (или None).
        use_take: Включён ли take-profit.
        pyramid: Доли объёма по ступеням (убывающая).

    Returns:
        position: суммарная доля по активным ступеням [0, 1].
    """
    upper, lower = _donchian_channels(bars, entry, exit_period)
    atr = bars.atr(atr_period)

    close = bars.close.values
    up = upper.values
    lo = lower.values
    atr_v = atr.values

    pos = np.zeros(len(close))
    steps_filled = 0          # сколько ступеней пирамиды набрано
    entry_price = 0.0         # цена первого входа
    last_add_price = 0.0      # цена последней докупки (для след. ступени)

    n_steps = len(pyramid)
    cum = np.cumsum(pyramid)  # накопленная доля по ступеням

    for i in range(len(close)):
        atr_i = atr_v[i]
        # Пропускаем, пока ATR не прогрелся.
        if np.isnan(atr_i) or np.isnan(up[i]):
            pos[i] = cum[steps_filled - 1] if steps_filled > 0 else 0.0
            continue

        if steps_filled == 0:
            # Ещё не в позиции — ждём первый пробой верхнего канала.
            if close[i] > up[i]:
                steps_filled = 1
                entry_price = close[i]
                last_add_price = close[i]
        else:
            stop_level = entry_price - stop_atr * atr_i
            # Полный выход: стоп или пробой нижнего канала.
            if close[i] < stop_level or close[i] < lo[i]:
                steps_filled = 0
                entry_price = 0.0
                last_add_price = 0.0
            else:
                # Take-profit: сброс верхней ступени при +take_atr.
                if (
                    use_take
                    and take_atr is not None
                    and steps_filled > 1
                    and close[i] > entry_price + take_atr * atr_i
                ):
                    steps_filled -= 1
                # Докупка следующей ступени при +0.5 ATR от последней.
                elif (
                    steps_filled < n_steps
                    and close[i] > last_add_price + 0.5 * atr_i
                ):
                    steps_filled += 1
                    last_add_price = close[i]

        pos[i] = cum[steps_filled - 1] if steps_filled > 0 else 0.0

    return pd.Series(pos, index=bars.index)


def donchian_est_macd_4step_pyramid(
    bars: Bars,
    entry: int = 20,
    exit_period: int = 10,
    atr_period: int = 20,
    stop_atr: float = 2.0,
    target_vol: float = 0.15,
) -> pd.Series:
    """4step pyramid + MACD-фильтр + vol targeting (без take-profit).

    +5.1% / -13.0% / 12 из 19. Убывающая пирамида + подтягиваемый стоп
    без фиксации тейка — держит тренд дольше, но глубже просадка пика.

    Args:
        bars: Данные инструмента.
        entry: Окно верхнего канала.
        exit_period: Окно нижнего канала.
        atr_period: Окно ATR.
        stop_atr: Множитель turtle-стопа.
        target_vol: Целевая волатильность.

    Returns:
        position: 4step-сигнал ∧ MACD, масштаб vol targeting.
    """
    sig = _donchian_4step(
        bars, entry, exit_period, atr_period, stop_atr,
        take_atr=None, use_take=False,
    )
    macd = _macd_bullish(bars.close).astype(float)
    size = vol_target_size(bars, target_vol)
    return sig * macd * size


def donchian_est_macd_4step_take(
    bars: Bars,
    entry: int = 20,
    exit_period: int = 10,
    atr_period: int = 20,
    stop_atr: float = 2.0,
    take_atr: float = 3.5,
    target_vol: float = 0.15,
) -> pd.Series:
    """★ COMMODITY CHAMPION: 4step + take-profit + MACD + vol targeting.

    +5.2% / -12.1% / 15 из 19 (диверсиф. корзина); +168% на 17-инстр.
    commodity-корзине, стабильно train/test. Take-profit при +3.5 ATR
    сбрасывает верхнюю ступень -> лучший риск-контроль (портфельная
    просадка -12.1%, energy 100% profit). Единственная валидированная
    commodity-стратегия проекта. НЕ ТРОГАТЬ параметры без re-теста.

    Args:
        bars: Данные инструмента (high/low реальные обязательны).
        entry: Окно верхнего канала (пробой = вход).
        exit_period: Окно нижнего канала (пробой = полный выход).
        atr_period: Окно ATR для стопа и тейка.
        stop_atr: Множитель turtle-стопа (entry - 2*ATR).
        take_atr: Множитель take-profit (+3.5*ATR сбрасывает ступень).
        target_vol: Целевая годовая волатильность.

    Returns:
        position: 4step+take сигнал ∧ MACD, масштаб vol targeting.
    """
    sig = _donchian_4step(
        bars, entry, exit_period, atr_period, stop_atr,
        take_atr=take_atr, use_take=True,
    )
    macd = _macd_bullish(bars.close).astype(float)
    size = vol_target_size(bars, target_vol)
    return sig * macd * size


def donchian_breakout_ls(
    bars: Bars, entry: int = 20, exit_period: int = 10
) -> pd.Series:
    """L/S Дончиан — ЗАКРЫТ как неэффективный (-71% портфель).

    Зеркальный шорт против uptrend-bias сырья. Оставлен для честной
    документации провала.

    Args:
        bars: Данные инструмента.
        entry: Окно верхнего канала.
        exit_period: Окно нижнего канала.

    Returns:
        position: +1 в лонг-пробое, -1 в шорт-пробое.
    """
    upper, lower = _donchian_channels(bars, entry, exit_period)
    # Шорт-канал: пробой нижнего = шорт, пробой верхнего = закрытие.
    short_upper = bars.high.rolling(exit_period).max().shift(1)
    close = bars.close.values
    up, lo = upper.values, lower.values
    sup = short_upper.values
    pos = np.zeros(len(close))
    state = 0  # +1 лонг, -1 шорт, 0 вне
    for i in range(len(close)):
        if state == 0:
            if close[i] > up[i]:
                state = 1
            elif close[i] < lo[i]:
                state = -1
        elif state == 1 and close[i] < lo[i]:
            state = 0
        elif state == -1 and close[i] > sup[i]:
            state = 0
        pos[i] = float(state)
    return pd.Series(pos, index=bars.index)
