"""Bollinger — mean-reversion и momentum-фильтр треки. Оба закрыты.

Выводы (зафиксированы):
  - BB+RSI: комплементарна трендовым (зарабатывает в боковые годы, когда
    трендовые теряют: Gas 2020 +43%, 2022 +14%). Все DD<40%. Baseline.
  - Don %b: %b>=1.0 как momentum-подтверждение поверх Дончиана НИЧЕГО не
    добавил (+1.8% vs +2.0% Pyr, оба 9/19). Фильтр коррелирован с тем,
    что уже в сигнале — та же история, что MACD. Просадка даже хуже.
  - TTM Squeeze: провалил тест стабильности параметров. Закрыт.

Bollinger по close (полосы вокруг SMA close) — high/low не требуются,
но контракт единый: Bars -> position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import vol_target_size
from strategies.donchian import donchian_breakout
from strategies.ema import _ensemble_vote


def _bollinger_bands(
    close: pd.Series, period: int = 20, n_std: float = 2.0
):
    """Средняя полоса (SMA) и верх/низ на n_std стандартных отклонений.

    Args:
        close: Ряд цен закрытия.
        period: Окно SMA и std.
        n_std: Число стандартных отклонений для полос.

    Returns:
        (mid, upper, lower) — три ряда полос Боллинджера.
    """
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid, mid + n_std * std, mid - n_std * std


def _percent_b(
    close: pd.Series, period: int = 20, n_std: float = 2.0
) -> pd.Series:
    """%b: положение цены внутри полос. 0 = нижняя, 1 = верхняя.

    Args:
        close: Ряд цен закрытия.
        period: Окно полос.
        n_std: Число std.

    Returns:
        Ряд %b (может выходить за [0,1] при пробое полос).
    """
    _, upper, lower = _bollinger_bands(close, period, n_std)
    return (close - lower) / (upper - lower)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI без цикла (Wilder EMA сглаживание gain/loss).

    Args:
        close: Ряд цен закрытия.
        period: Окно RSI.

    Returns:
        Ряд RSI [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_rsi(
    bars: Bars,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_exit: float = 50.0,
) -> pd.Series:
    """BB+RSI mean-reversion: лонг на перепроданности у нижней полосы.

    Вход: цена ниже нижней полосы И RSI < oversold. Выход: RSI > exit
    (возврат к среднему). Комплементарна трендовым — зарабатывает
    именно в боковые годы. Требует состояния -> цикл.

    Args:
        bars: Данные инструмента.
        bb_period: Окно полос Боллинджера.
        bb_std: Число std для полос.
        rsi_period: Окно RSI.
        rsi_oversold: Порог перепроданности для входа.
        rsi_exit: Порог RSI для выхода.

    Returns:
        position: 1.0 в лонге между входом и выходом, иначе 0.
    """
    _, _, lower = _bollinger_bands(bars.close, bb_period, bb_std)
    rsi = _rsi(bars.close, rsi_period)
    close = bars.close.values
    lo = lower.values
    rsi_v = rsi.values
    pos = np.zeros(len(close))
    in_pos = False
    for i in range(len(close)):
        if np.isnan(lo[i]) or np.isnan(rsi_v[i]):
            continue
        if not in_pos and close[i] < lo[i] and rsi_v[i] < rsi_oversold:
            in_pos = True
        elif in_pos and rsi_v[i] > rsi_exit:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


def bollinger_rsi_voltarget(
    bars: Bars, target_vol: float = 0.15
) -> pd.Series:
    """BB+RSI + vol targeting поверх — для портфельного сочетания.

    Args:
        bars: Данные инструмента.
        target_vol: Целевая волатильность.

    Returns:
        position: сигнал BB+RSI, масштабированный vol targeting.
    """
    return bollinger_rsi(bars) * vol_target_size(bars, target_vol)


def donchian_percent_b(
    bars: Bars,
    entry: int = 20,
    exit_period: int = 10,
    bb_period: int = 20,
    target_vol: float = 0.15,
) -> pd.Series:
    """Дончиан-пробой + %b>=1.0 momentum-фильтр + ансамбль + vol targeting.

    ЗАКРЫТА: %b ничего не добавил поверх ансамбля (+1.8% vs +2.0% Pyr).
    Двойной выход (%b<0.5 ИЛИ Donchian low) режет хорошие тренды.
    Оставлена для документации: фильтр, коррелированный с сигналом, — шум.

    Args:
        bars: Данные инструмента.
        entry: Окно верхнего канала Дончиана.
        exit_period: Окно нижнего канала.
        bb_period: Окно полос для %b.
        target_vol: Целевая волатильность.

    Returns:
        position: Дончиан ∧ %b>=1 ∧ ансамбль, масштаб vol targeting.
    """
    breakout = donchian_breakout(bars, entry, exit_period)
    pb = _percent_b(bars.close, bb_period)
    momentum = (pb >= 1.0).astype(float)  # цена пробила верхнюю полосу
    regime = _ensemble_vote(bars.close)
    size = vol_target_size(bars, target_vol)
    return breakout * momentum * regime * size


def _keltner_channels(
    bars: Bars, period: int = 20, atr_mult: float = 1.5
):
    """Каналы Кельтнера (EMA ± ATR) — для TTM Squeeze.

    Args:
        bars: Данные инструмента.
        period: Окно EMA и ATR.
        atr_mult: Множитель ATR для каналов.

    Returns:
        (upper, lower) каналы Кельтнера.
    """
    ema = bars.close.ewm(span=period, adjust=False).mean()
    atr = bars.atr(period)
    return ema + atr_mult * atr, ema - atr_mult * atr


def ttm_squeeze(
    bars: Bars, period: int = 20, target_vol: float = 0.15
) -> pd.Series:
    """TTM Squeeze — ЗАКРЫТА (провал теста стабильности параметров).

    Сжатие = полосы Боллинджера внутри каналов Кельтнера (низкая вола
    перед движением). Вход по направлению выхода из сжатия. Результат
    дико гулял по параметрам -> overfitting. Оставлена для документации.

    Args:
        bars: Данные инструмента.
        period: Общее окно полос/каналов.
        target_vol: Целевая волатильность.

    Returns:
        position: направленный сигнал выхода из сжатия, масштаб vol.
    """
    _, bb_up, bb_lo = _bollinger_bands(bars.close, period)
    kc_up, kc_lo = _keltner_channels(bars, period)
    # Сжатие: полосы BB уже каналов Кельтнера.
    squeeze_on = (bb_up < kc_up) & (bb_lo > kc_lo)
    # Направление: momentum относительно средней.
    mid = bars.close.rolling(period).mean()
    direction = np.sign(bars.close - mid)
    # Торгуем ПОСЛЕ снятия сжатия по накопленному направлению.
    fired = squeeze_on.shift(1).fillna(False) & (~squeeze_on)
    signal = pd.Series(0.0, index=bars.index)
    signal[fired] = direction[fired]
    signal = signal.replace(0.0, np.nan).ffill().fillna(0.0).clip(-1, 1)
    return signal * vol_target_size(bars, target_vol)
