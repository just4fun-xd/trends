"""Сезонные (календарные) стратегии — портированы из EMA1221.

ЦЕННОСТЬ ТРЕКА: единственный сигнал в проекте, ОРТОГОНАЛЬНЫЙ цене —
чистый календарь. Именно поэтому ценен для ансамбля: зарабатывает в
другие моменты, чем Дончиан/EMA, и его P&L слабо коррелирован с
трендовыми. Гипотеза для газа: рост перед зимой (закачка в хранилища),
падение весной (низкий спрос после отопительного сезона).

Портирование под контракт Bars + правки аудита 2026-07:
  - bars_per_year берётся из Bars (было 252**0.5 хардкодом);
  - vol-таргетинг через общий vol_target_size с буфером (было сырое
    дрожание размера каждый бар).

ВНИМАНИЕ: числа в старых докстрингах (+62.9% 2024 и т.п.) получены
старым кодом со старой семантикой издержек — подлежат ре-валидации
на реальных данных (см. docs/AUDIT_2026-07.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import vol_target_size

# Сезонные окна газа: закачка (лонг) и весенний слив (флэт).
GAS_BUY_MONTHS = (8, 9, 10, 11)      # август-ноябрь: пред-зимняя закачка
GAS_SELL_MONTHS = (3, 4, 5, 6)       # весна: слив после отопит. сезона


def seasonal_gas(
    bars: Bars, buy_months: tuple = GAS_BUY_MONTHS
) -> pd.Series:
    """Чистый календарь: лонг только в исторически сильные месяцы.

    Никакого ценового сигнала — только месяц. База трека: если она
    прибыльна в ДРУГИЕ годы, чем трендовые стратегии, — ценна для
    ансамбля. Исключение 2022 (−18.8%): европейский газовый кризис
    перебил сезонность (геополитика > календарь).

    Args:
        bars: Данные инструмента.
        buy_months: Месяцы (1-12) для лонга. Дефолт — закачка газа.

    Returns:
        position: 1.0 в сильные месяцы, иначе 0.
    """
    pos = pd.Series(0.0, index=bars.index)
    pos[bars.index.month.isin(buy_months)] = 1.0
    return pos


def _donchian_channels_close(
    close: pd.Series, high: pd.Series, low: pd.Series,
    entry: int, exit_period: int,
):
    """Каналы Дончиана по реальным high/low, сдвинуты на 1 бар.

    Args:
        close: Цены закрытия (для совместимости индекса).
        high: Максимумы бара.
        low: Минимумы бара.
        entry: Окно верхнего канала.
        exit_period: Окно нижнего канала.

    Returns:
        (upper, lower) — каналы, сдвинутые на 1 (без look-ahead).
    """
    upper = high.rolling(entry).max().shift(1)
    lower = low.rolling(exit_period).min().shift(1)
    return upper, lower


def donchian_seasonal(
    bars: Bars, entry: int = 20, exit_period: int = 10,
    buy_months: tuple = GAS_BUY_MONTHS,
) -> pd.Series:
    """Дончиан-пробой, отфильтрованный сезонным окном.

    Два независимых условия для входа: (1) календарь — только в сильные
    месяцы; (2) пробой — цена выше N-дневного максимума. Оба должны
    держаться. Выход — по пробою нижнего канала, без сезонного условия.

    Риск-выход по интрадей low (аудит 2026-07): нижний канал видит
    внутрибарный пробой. Вход close-confirmed.

    Args:
        bars: Данные инструмента (high/low реальные).
        entry: Окно верхнего канала.
        exit_period: Окно нижнего канала.
        buy_months: Месяцы, когда вход разрешён.

    Returns:
        position: 1.0 в позиции, иначе 0.
    """
    upper, lower = _donchian_channels_close(
        bars.close, bars.high, bars.low, entry, exit_period
    )
    close = bars.close.values
    low = bars.low.values
    up = upper.values
    lo = lower.values
    months = bars.index.month.values
    pos = np.zeros(len(close))
    in_pos = False
    for i in range(len(close)):
        if in_pos:
            if not np.isnan(lo[i]) and low[i] < lo[i]:
                in_pos = False
            else:
                pos[i] = 1.0
        else:
            month_ok = months[i] in buy_months
            if not np.isnan(up[i]) and close[i] > up[i] and month_ok:
                in_pos = True
                pos[i] = 1.0
    return pd.Series(pos, index=bars.index)


def donchian_seasonal_voltarget(
    bars: Bars, entry: int = 20, exit_period: int = 10,
    buy_months: tuple = GAS_BUY_MONTHS, target_vol: float = 0.15,
) -> pd.Series:
    """Сезонный Дончиан + vol-таргетинг (для портфельного сочетания).

    Тот же сигнал входа/выхода, что donchian_seasonal, но размер
    масштабирован target_vol / реализованная вола. vol-таргетинг —
    отдельный слой поверх сигнала (не смешан с логикой входа), через
    общий vol_target_size с буфером и bars_per_year из Bars.

    Старый результат (газ 2019-2025): max DD −25.9% → −7.9%, доходность
    пропорционально ниже. Лучший выбор для портфеля (равный риск-вклад).

    Args:
        bars: Данные инструмента.
        entry: Окно верхнего канала.
        exit_period: Окно нижнего канала.
        buy_months: Месяцы входа.
        target_vol: Целевая годовая волатильность.

    Returns:
        position: сезонный сигнал, масштабированный vol-таргетингом.
    """
    raw = donchian_seasonal(bars, entry, exit_period, buy_months)
    size = vol_target_size(bars, target_vol)
    return raw * size


def seasonal_gas_voltarget(
    bars: Bars, buy_months: tuple = GAS_BUY_MONTHS,
    target_vol: float = 0.15,
) -> pd.Series:
    """Чистый календарь + vol-таргетинг — для портфельного вклада.

    Календарный сигнал масштабируется по волатильности, чтобы вклад в
    портфель был риск-выровнен с трендовыми ногами.

    Args:
        bars: Данные инструмента.
        buy_months: Месяцы лонга.
        target_vol: Целевая волатильность.

    Returns:
        position: календарный сигнал, масштабированный vol-таргетингом.
    """
    raw = seasonal_gas(bars, buy_months)
    size = vol_target_size(bars, target_vol)
    return raw * size
