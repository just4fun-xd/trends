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
    low = bars.low.values
    up = upper.values
    lo = lower.values
    pos = np.zeros(len(close))
    in_pos = False
    for i in range(len(close)):
        # Аудит 3 (Gemini): выход и вход — НЕЗАВИСИМЫЕ блоки, не if/elif.
        # На outside-баре (сильная волатильность) low мог пробить нижний
        # канал, а close закрыться выше верхнего. С elif мы бы вышли и
        # пропустили вход до следующего бара; двумя if — выходим по риску,
        # затем тем же баром переоткрываемся по close-confirmed пробою.
        # Порядок выход->вход даёт приоритет свежему входу.
        if in_pos and not np.isnan(lo[i]) and low[i] < lo[i]:
            in_pos = False
        if not in_pos and close[i] > up[i]:
            in_pos = True
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
      - Вход и докупки — по close (close-confirmed: требуем закрытия за
        уровнем, меньше ложных интрадей-проколов). Следующая ступень —
        когда close прошёл +0.5 ATR от последней докупки.
      - Turtle-стоп: last_add_price − stop_atr*ENTRY_ATR — ПОДТЯГИВАЕТСЯ
        за каждой докупкой (аудит 2026-07: раньше код держал стоп от
        первого входа вопреки собственному докстрингу; набранная
        пирамида ездила со стопом у самого низа).
      - ATR ФИКСИРУЕТСЯ НА ВХОДЕ (аудит 2, Gemini): entry_atr = ATR бара
        пробоя, и стоп/тейк/шаг докупки считаются по нему всю сделку.
        Плавающий текущий atr_i дрейфовал сетку риска: при всплеске
        волатильности стоп уезжал вниз (риск > запланированного — прямое
        нарушение DD-дисциплины), при сжатии — тейк съезжал вниз и мог
        сработать от падения волы, а не достижения цели. Так делали
        оригинальные Черепахи: N фиксируется на входе. Это меняет числа
        на реальных данных (сделки держатся дольше) — ре-валидация.
      - РИСК-ТРИГГЕРЫ ПО ЭКСТРЕМУМАМ (аудит 2026-07): стоп и нижний
        канал проверяются по low бара — интрадей-пробой виден, close-
        проверка его прятала. Тейк — по high (лимит-фиксация исполнилась
        бы внутри бара). Асимметрия осознанная: входы консервативные
        (close-confirmed), выходы быстрые (intraday). Исполнение всех
        сигналов остаётся по close следующего бара (движок, shift(1)).
      - Take-profit (если use_take): при +take_atr*ENTRY_ATR от entry
        сбрасываем верхнюю ступень ОДИН РАЗ на цикл позиции (one-shot).
        После тейка пирамида ЗАМОРОЖЕНА до полного выхода — без заморозки
        докупка через бар аннулировала бы тейк и возникала пила
        тейк<->докупка с оборотом каждый бар (аудит 2026-07, крит. баг).
      - Полный выход: пробой нижнего канала ИЛИ стоп; сбрасывает всё
        состояние, включая one-shot флаг и entry_atr.

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
    high = bars.high.values
    low = bars.low.values
    up = upper.values
    lo = lower.values
    atr_v = atr.values

    pos = np.zeros(len(close))
    steps_filled = 0          # сколько ступеней пирамиды набрано
    entry_price = 0.0         # цена первого входа (уровень тейка)
    last_add_price = 0.0      # цена последней докупки (стоп + ступень)
    entry_atr = 0.0           # ATR бара пробоя — сетка риска сделки
    take_done = False         # one-shot: тейк уже забран в этом цикле

    n_steps = len(pyramid)
    cum = np.cumsum(pyramid)  # накопленная доля по ступеням

    for i in range(len(close)):
        atr_i = atr_v[i]
        # Пропускаем, пока индикаторы не прогрелись.
        if np.isnan(atr_i) or np.isnan(up[i]):
            pos[i] = cum[steps_filled - 1] if steps_filled > 0 else 0.0
            continue

        if steps_filled == 0:
            # Ещё не в позиции — ждём close-confirmed пробой канала.
            if close[i] > up[i]:
                steps_filled = 1
                entry_price = close[i]
                last_add_price = close[i]
                entry_atr = atr_i     # фиксируем волатильность пробоя
                take_done = False
        else:
            # Сетка риска — по зафиксированному на входе entry_atr.
            stop_level = last_add_price - stop_atr * entry_atr
            # Риск-выход по интрадей-минимуму: стоп или нижний канал.
            exit_hit = low[i] < stop_level or (
                not np.isnan(lo[i]) and low[i] < lo[i]
            )
            if exit_hit:
                steps_filled = 0
                entry_price = 0.0
                last_add_price = 0.0
                entry_atr = 0.0
                take_done = False
            elif not take_done:
                # Take-profit по интрадей-максимуму: one-shot сброс
                # верхней ступени, затем пирамида заморожена.
                if (
                    use_take
                    and take_atr is not None
                    and steps_filled > 1
                    and high[i] > entry_price + take_atr * entry_atr
                ):
                    steps_filled -= 1
                    take_done = True
                # Докупка следующей ступени при +0.5 ENTRY_ATR от последней.
                elif (
                    steps_filled < n_steps
                    and close[i] > last_add_price + 0.5 * entry_atr
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


def donchian_champion_raw(
    bars: Bars,
    entry: int = 20,
    exit_period: int = 10,
    atr_period: int = 20,
    stop_atr: float = 2.0,
    take_atr: float = 3.5,
) -> pd.Series:
    """Сырой сигнал champion БЕЗ vol targeting: 4step+take ∧ MACD.

    Выделен для regime-роутера (аудит 2026-07): роутер смешивает СЫРЫЕ
    ноги и накладывает единый vol_target на итог — иначе ноги с разным
    внутренним сайзингом несут несопоставимый риск. Полный champion
    ниже = этот сигнал × vol_target_size, тождественно.

    Args:
        bars: Данные инструмента (high/low реальные обязательны).
        entry: Окно верхнего канала.
        exit_period: Окно нижнего канала.
        atr_period: Окно ATR для стопа и тейка.
        stop_atr: Множитель turtle-стопа.
        take_atr: Множитель take-profit.

    Returns:
        position: сырой сигнал [0, 1] (пирамида ∧ MACD).
    """
    sig = _donchian_4step(
        bars, entry, exit_period, atr_period, stop_atr,
        take_atr=take_atr, use_take=True,
    )
    macd = _macd_bullish(bars.close).astype(float)
    return sig * macd


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

    Эталонные числа старого бэктеста: +5.2% / -12.1% / 15 из 19
    (диверсиф. корзина); +168% на 17-инстр. commodity-корзине.

    ВНИМАНИЕ (аудит 2026-07): семантика УТОЧНЕНА относительно кода,
    дававшего старые числа: (1) take-profit теперь one-shot с заморозкой
    пирамиды (был пулемётный сброс + пила тейк<->докупка); (2) стоп
    подтягивается за докупками (был приколочен к первому входу);
    (3) стоп/выход триггерятся по low, тейк по high (были по close).
    Все три — исправления багов, но числа НА РЕАЛЬНЫХ ДАННЫХ ИЗМЕНЯТСЯ
    и требуют ре-валидации против BENCHMARK_RESULTS.md перед докладом.

    Args:
        bars: Данные инструмента (high/low реальные обязательны).
        entry: Окно верхнего канала (пробой = вход).
        exit_period: Окно нижнего канала (пробой = полный выход).
        atr_period: Окно ATR для стопа и тейка.
        stop_atr: Множитель turtle-стопа (last_add - 2*ATR).
        take_atr: Множитель take-profit (+3.5*ATR сбрасывает ступень).
        target_vol: Целевая годовая волатильность.

    Returns:
        position: 4step+take сигнал ∧ MACD, масштаб vol targeting.
    """
    raw = donchian_champion_raw(
        bars, entry, exit_period, atr_period, stop_atr, take_atr,
    )
    size = vol_target_size(bars, target_vol)
    return raw * size


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


def donchian_ensemble_pyramid(
    bars: Bars,
    pairs: list[tuple[int, int]] | None = None,
    threshold: float = 0.5,
    entry: int = 20,
    exit_period: int = 10,
    target_vol: float = 0.15,
    max_pos: float = 2.0,
    pyramid_atr: float = 1.5,
    atr_period: int = 14
) -> pd.Series:
    """
    Donchian breakout + EMA ensemble filter + pyramiding + vol targeting.
    Адаптировано под инфраструктуру Bars и очищено от look-ahead багов.
    """
    if pairs is None:
        pairs = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]

    close_s = bars.close
    high_s = bars.high
    low_s = bars.low

    votes = pd.DataFrame(index=close_s.index)
    for fast, slow in pairs:
        ema_fast = close_s.ewm(span=fast, adjust=False).mean()
        ema_slow = close_s.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}/{slow}"] = (ema_fast > ema_slow).astype(int)

    macro_bullish = votes.mean(axis=1) > threshold

    up = high_s.rolling(entry).max().shift(1).values
    lo = low_s.rolling(exit_period).min().shift(1).values

    prev_close = close_s.shift(1)
    tr = pd.concat([
        high_s - low_s,
        (high_s - prev_close).abs(),
        (low_s - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().values

    close = close_s.values
    high = high_s.values
    low = low_s.values
    macro = macro_bullish.values

    pos = np.zeros(len(close))

    in_position = False
    pyramid_done = False
    entry_price = 0.0
    entry_atr = 0.0

    for i in range(len(close)):
        if in_position:
            if not np.isnan(lo[i]) and low[i] < lo[i]:
                in_position = False
                pyramid_done = False
                entry_price = 0.0
                entry_atr = 0.0
            else:
                if (not pyramid_done and not np.isnan(entry_atr)
                        and entry_atr > 0):
                    if high[i] >= (entry_price +
                                   pyramid_atr * entry_atr):
                        pyramid_done = True

        if not in_position:
            if not np.isnan(up[i]) and close[i] > up[i] and macro[i]:
                in_position = True
                pyramid_done = False
                entry_price = close[i]
                entry_atr = atr[i]

        if in_position:
            pos[i] = 1.0 if pyramid_done else 0.5

    raw_position = pd.Series(pos, index=close_s.index)

    size = vol_target_size(bars, target_vol=target_vol)
    size = size.clip(upper=max_pos)

    return raw_position * size
