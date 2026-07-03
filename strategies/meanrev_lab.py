"""Лаборатория mean-reversion: 10 вариантов на базе bb_rsi.

Базовый bb_rsi показал сильные числа на 2021-2026 (сырьё 19/19,
медиана +14.5% с VT). Лаборатория выжимает из механизма максимум,
варьируя МАТЕМАТИКУ, а не параметры: вход, выход, сайзинг, фильтр
режима, меру растяжения. Каждый вариант — отдельная гипотеза с
механизмом, документируемая по стандарту проекта.

╔════════════════════════════════════════════════════════════════╗
║ ДИСКЛЕЙМЕР MULTIPLE TESTING (читать до интерпретации чисел):   ║
║ 10 вариантов на ОДНОМ периоде гарантируют, что лучший выглядит ║
║ хорошо ПО СЛУЧАЙНОСТИ. Отбор кандидата — ТОЛЬКО через           ║
║ walk-forward (train/test) + стабильность параметров + механизм. ║
║ Это ЛАБОРАТОРИЯ, не боевые стратегии. И помни два красных флага ║
║ базового bb_rsi: (1) без VT хвост открыт (Tesla DD −53% — RSI-  ║
║ выход НЕ риск-стоп); (2) long-only mean-rev в растущем рынке    ║
║ несёт бету — mr_short здесь как честный контроль на это.        ║
╚════════════════════════════════════════════════════════════════╝

Все стратегии: Bars -> position (посерийный контракт, run_basket).
VT-слой накладывается флагом --vt раннера (position * vol_target_size).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.bollinger import _bollinger_bands, _rsi


def _zscore(close: pd.Series, period: int = 20) -> pd.Series:
    """Z-score цены к своей скользящей: (close - SMA) / std.

    Непрерывная мера растяжения (у полос — бинарная: ниже/выше).

    Args:
        close: Цены закрытия.
        period: Окно SMA и std.

    Returns:
        Ряд z-score (NaN на вырожденном std).
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    std = std.where(std > 1e-12)
    return (close - sma) / std


def mr_atr_stop(
    bars: Bars, bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_buy: float = 30.0, rsi_exit: float = 50.0,
    stop_atr: float = 2.0,
) -> pd.Series:
    """Вариант 1: базовый вход + ЖЁСТКИЙ ATR-стоп на выход.

    Механизм: RSI-выход базового bb_rsi — не риск-стоп (Tesla DD −53%:
    пока RSI не восстановился, позиция тонет без ограничения). Добавляем
    стоп low < entry − stop_atr*ATR(входа): хвост обрезан жёстко.
    Гипотеза: чуть ниже доходность, радикально лучше DD.

    Args:
        bars: Данные инструмента.
        bb_period: Окно полос.
        bb_std: Ширина полос в std.
        rsi_period: Окно RSI.
        rsi_buy: Порог перепроданности.
        rsi_exit: Порог выхода по RSI.
        stop_atr: Множитель ATR-стопа (фиксируется на входе).

    Returns:
        position: 1.0 в позиции, иначе 0.
    """
    _, _, lower = _bollinger_bands(bars.close, bb_period, bb_std)
    rsi = _rsi(bars.close, rsi_period)
    atr = bars.atr(bb_period)
    close, low = bars.close.values, bars.low.values
    lo_b, rsi_v, atr_v = lower.values, rsi.values, atr.values

    pos = np.zeros(len(close))
    in_pos = False
    entry_px, entry_atr = 0.0, 0.0
    for i in range(len(close)):
        if np.isnan(lo_b[i]) or np.isnan(rsi_v[i]) or np.isnan(atr_v[i]):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        if in_pos:
            if low[i] < entry_px - stop_atr * entry_atr:
                in_pos = False           # риск-стоп: хвост обрезан
            elif rsi_v[i] > rsi_exit:
                in_pos = False           # штатный выход по реверсии
        if not in_pos and close[i] < lo_b[i] and rsi_v[i] < rsi_buy:
            in_pos = True
            entry_px, entry_atr = close[i], atr_v[i]
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


def mr_time_stop(
    bars: Bars, bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_buy: float = 30.0, rsi_exit: float = 50.0,
    max_hold: int = 10,
) -> pd.Series:
    """Вариант 2: time-stop — выход через max_hold баров без реверсии.

    Механизм: у mean-reversion есть срок годности (half-life). Если за
    max_hold баров отскок не случился — гипотеза «перепроданность»
    мертва, дальше это падающий тренд. Классика литературы mean-rev.

    Args:
        bars: Данные инструмента.
        bb_period: Окно полос.
        bb_std: Ширина полос.
        rsi_period: Окно RSI.
        rsi_buy: Порог перепроданности.
        rsi_exit: Порог выхода.
        max_hold: Максимум баров в позиции.

    Returns:
        position: 1.0 в позиции, иначе 0.
    """
    _, _, lower = _bollinger_bands(bars.close, bb_period, bb_std)
    rsi = _rsi(bars.close, rsi_period)
    close = bars.close.values
    lo_b, rsi_v = lower.values, rsi.values

    pos = np.zeros(len(close))
    in_pos = False
    held = 0
    for i in range(len(close)):
        if np.isnan(lo_b[i]) or np.isnan(rsi_v[i]):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        if in_pos:
            held += 1
            if rsi_v[i] > rsi_exit or held >= max_hold:
                in_pos = False
        if not in_pos and close[i] < lo_b[i] and rsi_v[i] < rsi_buy:
            in_pos = True
            held = 0
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


def mr_scaled(
    bars: Bars, period: int = 20, entry_z: float = 2.0,
    full_z: float = 3.5, exit_z: float = 0.5,
) -> pd.Series:
    """Вариант 3: непрерывный сайзинг по глубине растяжения.

    Механизм: бинарный вход (0/1) игнорирует ИНФОРМАЦИЮ в глубине
    перепроданности. Размер растёт линейно от entry_z до full_z:
    z=-2 -> 0 (порог), z=-3.5 -> 1.0 (полный). Глубже растяжение —
    сильнее ожидаемая реверсия — больше ставка. Выход при |z|<exit_z.

    Args:
        bars: Данные инструмента.
        period: Окно z-score.
        entry_z: |z| начала набора позиции.
        full_z: |z| полного размера.
        exit_z: |z| выхода.

    Returns:
        position: непрерывная [0, 1].
    """
    z = _zscore(bars.close, period).values
    pos = np.zeros(len(z))
    state = 0.0
    for i in range(len(z)):
        if np.isnan(z[i]):
            pos[i] = state
            continue
        if state > 0 and z[i] > -exit_z:
            state = 0.0                      # реверсия случилась
        elif z[i] < -entry_z:
            depth = (-z[i] - entry_z) / (full_z - entry_z)
            state = float(np.clip(depth, 0.0, 1.0))
        pos[i] = state
    return pd.Series(pos, index=bars.index)


def mr_ladder(
    bars: Bars, period: int = 20, z1: float = 2.0, z2: float = 3.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """Вариант 4: двухступенчатый вход (лесенка с жёстким капом).

    Механизм: половина позиции на z<-z1, добор до полной на z<-z2.
    НЕ мартингейл: кап 1.0 жёсткий, ступеней ровно две. Средняя цена
    входа ниже базовой -> реверсия к среднему выгоднее.

    Args:
        bars: Данные инструмента.
        period: Окно z-score.
        z1: Порог первой ступени (0.5 позиции).
        z2: Порог второй ступени (полная).
        exit_z: Порог выхода.

    Returns:
        position: {0, 0.5, 1.0}.
    """
    z = _zscore(bars.close, period).values
    pos = np.zeros(len(z))
    state = 0.0
    for i in range(len(z)):
        if np.isnan(z[i]):
            pos[i] = state
            continue
        if state > 0 and z[i] > -exit_z:
            state = 0.0
        else:
            if z[i] < -z2:
                state = 1.0
            elif z[i] < -z1 and state == 0.0:
                state = 0.5
        pos[i] = state
    return pd.Series(pos, index=bars.index)


def mr_trend_filter(
    bars: Bars, bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_buy: float = 30.0, rsi_exit: float = 50.0,
    sma_filter: int = 200,
) -> pd.Series:
    """Вариант 5: buy-the-dip — базовый вход ТОЛЬКО выше SMA200.

    Механизм: перепроданность в аптренде — откат (реверсия вероятна);
    перепроданность в даунтренде — падающий нож (реверсии нет). Фильтр
    оставляет только первые. Самая чистая гипотеза для акций со
    структурным дрейфом.

    Args:
        bars: Данные инструмента.
        bb_period: Окно полос.
        bb_std: Ширина полос.
        rsi_period: Окно RSI.
        rsi_buy: Порог перепроданности.
        rsi_exit: Порог выхода.
        sma_filter: Окно трендового фильтра.

    Returns:
        position: 1.0 в позиции, иначе 0.
    """
    from strategies.bollinger import bollinger_rsi
    base = bollinger_rsi(bars, bb_period, bb_std, rsi_period,
                         rsi_buy, rsi_exit)
    sma = bars.close.rolling(sma_filter).mean()
    uptrend = (bars.close > sma).astype(float)
    # Гейт на ВХОД: раз войдя выше SMA, позицию не рвём при её проколе
    # (иначе дёргание на границе). Реализация: вход разрешён только в
    # аптренде; выходы — базовые.
    pos = np.zeros(len(base))
    in_pos = False
    b, u = base.values, uptrend.values
    for i in range(len(b)):
        if not in_pos and b[i] > 0 and u[i] > 0:
            in_pos = True
        elif in_pos and b[i] == 0:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


def mr_lowvol(
    bars: Bars, bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_buy: float = 30.0, rsi_exit: float = 50.0,
    vol_lookback: int = 20, vol_pct: float = 0.7,
) -> pd.Series:
    """Вариант 6: режимный фильтр — входы только при спокойной воле.

    Механизм: mean-reversion живёт в спокойных режимах; высокая вола —
    признак тренда/кризиса, где реверсия ломается (газ-2022). Вход
    разрешён, только если реализованная вола ниже своего expanding-
    перцентиля vol_pct (без look-ahead).

    Args:
        bars: Данные инструмента.
        bb_period: Окно полос.
        bb_std: Ширина полос.
        rsi_period: Окно RSI.
        rsi_buy: Порог перепроданности.
        rsi_exit: Порог выхода.
        vol_lookback: Окно реализованной волатильности.
        vol_pct: Перцентиль-порог (вола ниже -> режим спокоен).

    Returns:
        position: 1.0 в позиции, иначе 0.
    """
    from strategies.bollinger import bollinger_rsi
    base = bollinger_rsi(bars, bb_period, bb_std, rsi_period,
                         rsi_buy, rsi_exit)
    vol = bars.returns().rolling(vol_lookback).std()
    thresh = vol.expanding(min_periods=vol_lookback * 3).quantile(vol_pct)
    calm = (vol < thresh).values
    b = base.values
    pos = np.zeros(len(b))
    in_pos = False
    for i in range(len(b)):
        if not in_pos and b[i] > 0 and calm[i]:
            in_pos = True
        elif in_pos and b[i] == 0:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


def mr_keltner(
    bars: Bars, period: int = 20, atr_mult: float = 2.0,
    rsi_period: int = 14, rsi_buy: float = 30.0, rsi_exit: float = 50.0,
) -> pd.Series:
    """Вариант 7: Keltner-каналы (ATR) вместо Bollinger (std).

    Механизм: std раздувается выбросами (один шоковый бар расширяет
    полосы на всё окно), ATR по Wilder устойчивее. Гипотеза: меньше
    ложных «перепроданностей» после единичных шоков.

    Args:
        bars: Данные инструмента.
        period: Окно EMA и ATR.
        atr_mult: Ширина канала в ATR.
        rsi_period: Окно RSI.
        rsi_buy: Порог перепроданности.
        rsi_exit: Порог выхода.

    Returns:
        position: 1.0 в позиции, иначе 0.
    """
    ema = bars.close.ewm(span=period, adjust=False).mean()
    atr = bars.atr(period)
    lower = (ema - atr_mult * atr).values
    rsi = _rsi(bars.close, rsi_period).values
    close = bars.close.values
    pos = np.zeros(len(close))
    in_pos = False
    for i in range(len(close)):
        if np.isnan(lower[i]) or np.isnan(rsi[i]):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        if in_pos and rsi[i] > rsi_exit:
            in_pos = False
        if not in_pos and close[i] < lower[i] and rsi[i] < rsi_buy:
            in_pos = True
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


def mr_connors_rsi2(
    bars: Bars, rsi_period: int = 2, rsi_buy: float = 10.0,
    sma_trend: int = 200, sma_exit: int = 5,
) -> pd.Series:
    """Вариант 8: классика Connors RSI(2).

    Механизм: сверхкороткий RSI(2) ловит 1-3-дневные микро-откаты
    (другая ЧАСТОТА реверсии, чем RSI(14)). Правила Connors: вход
    RSI(2)<10 И close>SMA200 (только аптренд), выход close>SMA5.
    Литературный эталон для сравнения с нашим bb_rsi.

    Args:
        bars: Данные инструмента.
        rsi_period: Окно RSI (2 — суть варианта).
        rsi_buy: Порог входа RSI(2).
        sma_trend: Трендовый фильтр.
        sma_exit: SMA выхода.

    Returns:
        position: 1.0 в позиции, иначе 0.
    """
    rsi = _rsi(bars.close, rsi_period).values
    sma_t = bars.close.rolling(sma_trend).mean().values
    sma_e = bars.close.rolling(sma_exit).mean().values
    close = bars.close.values
    pos = np.zeros(len(close))
    in_pos = False
    for i in range(len(close)):
        if np.isnan(sma_t[i]) or np.isnan(rsi[i]):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        if in_pos and close[i] > sma_e[i]:
            in_pos = False
        if not in_pos and rsi[i] < rsi_buy and close[i] > sma_t[i]:
            in_pos = True
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


def mr_confirm(
    bars: Bars, bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_buy: float = 30.0, rsi_exit: float = 50.0,
) -> pd.Series:
    """Вариант 9: вход на ПОДТВЕРЖДЕНИИ разворота, не на касании.

    Механизм: базовый bb_rsi ловит падающий нож (входит, пока цена ещё
    валится). Здесь ждём разворотный бар: условия базы вчера/сегодня И
    close > предыдущего close (нож воткнулся, отскок начался). Платим
    худшей ценой входа за меньше «поймал и поехал ниже».

    Args:
        bars: Данные инструмента.
        bb_period: Окно полос.
        bb_std: Ширина полос.
        rsi_period: Окно RSI.
        rsi_buy: Порог перепроданности.
        rsi_exit: Порог выхода.

    Returns:
        position: 1.0 в позиции, иначе 0.
    """
    _, _, lower = _bollinger_bands(bars.close, bb_period, bb_std)
    rsi = _rsi(bars.close, rsi_period).values
    close = bars.close.values
    lo_b = lower.values
    pos = np.zeros(len(close))
    in_pos = False
    for i in range(1, len(close)):
        if np.isnan(lo_b[i]) or np.isnan(rsi[i]):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        if in_pos and rsi[i] > rsi_exit:
            in_pos = False
        oversold = close[i] < lo_b[i] and rsi[i] < rsi_buy
        reversal = close[i] > close[i - 1]
        if not in_pos and oversold and reversal:
            in_pos = True
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


def mr_short(
    bars: Bars, bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_sell: float = 70.0, rsi_exit: float = 50.0,
) -> pd.Series:
    """Вариант 10: симметричный ШОРТ перекупленности — ЧЕСТНЫЙ КОНТРОЛЬ.

    Механизм-вопрос: если лонг перепроданности зарабатывает, шорт
    перекупленности должен зарабатывать симметрично — ЕСЛИ край в
    реверсии, а не в бете. По урокам проекта (L/S-треки) на активах со
    структурным дрейфом шорт сливает. Этот вариант в лаборатории как
    КОНТРОЛЬ: если mr_short стабильно теряет там, где лонг растёт —
    значительная часть «края» bb_rsi — бета растущего рынка.

    Args:
        bars: Данные инструмента.
        bb_period: Окно полос.
        bb_std: Ширина полос.
        rsi_period: Окно RSI.
        rsi_sell: Порог перекупленности (вход в шорт).
        rsi_exit: Порог выхода (возврат RSI к середине).

    Returns:
        position: -1.0 в шорте, иначе 0.
    """
    mid, upper, _ = _bollinger_bands(bars.close, bb_period, bb_std)
    rsi = _rsi(bars.close, rsi_period).values
    close = bars.close.values
    up_b = upper.values
    pos = np.zeros(len(close))
    in_pos = False
    for i in range(len(close)):
        if np.isnan(up_b[i]) or np.isnan(rsi[i]):
            pos[i] = -1.0 if in_pos else 0.0
            continue
        if in_pos and rsi[i] < rsi_exit:
            in_pos = False
        if not in_pos and close[i] > up_b[i] and rsi[i] > rsi_sell:
            in_pos = True
        pos[i] = -1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


# Реестр лаборатории: имя -> функция. Для run_basket и тестов.
MEANREV_LAB = {
    "mr_atr_stop": mr_atr_stop,
    "mr_time_stop": mr_time_stop,
    "mr_scaled": mr_scaled,
    "mr_ladder": mr_ladder,
    "mr_trend": mr_trend_filter,
    "mr_lowvol": mr_lowvol,
    "mr_keltner": mr_keltner,
    "mr_connors": mr_connors_rsi2,
    "mr_confirm": mr_confirm,
    "mr_short": mr_short,
}
