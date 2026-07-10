"""Carver-MR и мягкий vol-гейт — реверсия «аппаратом Карвера».

Две стратегии из идей Кирилла (2026-07g), доработанные:

1. carver_mr — мульти-оконный z-осциллятор с инверсией и скейлингом
   Карвера. Ансамбль ТЕМПОВ реверсии: короткое окно ловит резкий спад,
   длинное держит затяжной. Ниша по карте — реверсионный кластер
   (PL, PA, SI, NG), где donchian терял на ложных пробоях.
   Отличия от спецификации: позиция клипуется в [0,1]/[−1,1] (без
   внутреннего плеча — плечо только через VT снаружи, иначе двойной
   учёт риска); полный FDM не нужен — z-окна одного семейства сильно
   скоррелированы по построению, корреляционный множитель выродится в
   константу, достаточно равновзвешенной суммы со скейлингом.

2. mr_lowvol_soft — непрерывная версия vol-гейта mr_lowvol. Дешёвая
   альтернатива HMM-идее: вместо бинарного «вола<порога -> торгуем»
   вес позиции = логистическая функция ранга волы (плавное затухание
   с ростом волы). Если мягкость не побьёт жёсткий гейт — HMM тем
   более не окупится (тот же механизм, больше машинерии). Это честный
   тест гипотезы «плавность помогает» без hmmlearn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars


def carver_mr(
    bars: Bars,
    windows: tuple = (5, 10, 20, 40, 80),
    z_cap: float = 3.0,
    long_only: bool = True,
) -> pd.Series:
    """Реверсионный ансамбль z-осцилляторов со скейлингом Карвера.

    Для каждого окна w: z_w = (close − SMA_w)/std_w, сигнал = −z_w
    (растяжение вверх -> шорт, вниз -> лонг). Каждый −z нормируется
    трейлинг-E|z| (скейлинг Карвера: единый масштаб ~1), клип ±z_cap,
    равновзвешенная сумма окон, tanh в [−1,1]. Короткие окна дают
    ранний вход, длинные не дают выйти раньше времени — «резинка»
    натягивается на нескольких таймфреймах сразу.

    Args:
        bars: Данные инструмента.
        windows: Набор окон z-score (темпы реверсии).
        z_cap: Клип нормированного сигнала (единицы Карвера).
        long_only: Только покупка перепроданности (дисциплина проекта;
            шорт — отдельный ре-аудит).

    Returns:
        position: [0,1] при long_only, иначе [−1,1]. Сдвиг 1 бар.
    """
    close = bars.close
    sigs = []
    for w in windows:
        ma = close.rolling(w).mean()
        sd = close.rolling(w).std().replace(0.0, np.nan)
        z = (close - ma) / sd
        inv = -z  # инверсия: перекуплен -> шорт-сигнал
        # Скейлинг Карвера: единый масштаб через трейлинг E|сигнал|.
        abs_mean = inv.abs().rolling(252, min_periods=60).mean()
        scaled = (inv / abs_mean.where(abs_mean > 1e-12)).clip(
            -z_cap, z_cap)
        sigs.append(scaled)
    combined = pd.concat(sigs, axis=1).mean(axis=1)
    pos = pd.Series(
        np.tanh(combined.to_numpy() / 2.0), index=close.index)
    if long_only:
        pos = pos.clip(lower=0.0)
    # ФИКС 2026-07j: двойной shift удалён (движок сдвигает сам).
    return pos.fillna(0.0)


def mr_lowvol_soft(
    bars: Bars, bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_buy: float = 30.0, rsi_exit: float = 50.0,
    vol_lookback: int = 20, mid: float = 0.6, steep: float = 8.0,
) -> pd.Series:
    """mr_lowvol с НЕПРЕРЫВНЫМ vol-гейтом (логистика вместо порога).

    Вес = sigmoid(steep · (mid − vol_rank)): при спокойной воле ~1,
    у mid — 0.5, на всплеске -> 0. Та же идея «реверсия живёт в тиши»,
    но позиция плавно затухает с ростом волы вместо бинарного клика.
    Дешёвая замена HMM: если и это не бьёт жёсткий гейт, HMM не
    окупится (механизм тот же, машинерии больше).

    Args:
        bars: Данные инструмента.
        bb_period, bb_std, rsi_period, rsi_buy, rsi_exit: Параметры
            базового bollinger_rsi (идентичны mr_lowvol).
        vol_lookback: Окно реализованной волы.
        mid: Ранг волы, где вес = 0.5 (центр перехода).
        steep: Крутизна логистики.

    Returns:
        position в [0,1]: базовый сигнал × непрерывный vol-вес.
    """
    from strategies.bollinger import bollinger_rsi
    base = bollinger_rsi(bars, bb_period, bb_std, rsi_period,
                         rsi_buy, rsi_exit)
    vol = bars.returns().rolling(vol_lookback).std()
    rank = vol.expanding(min_periods=vol_lookback * 3).rank(pct=True)
    weight = 1.0 / (1.0 + np.exp(-steep * (mid - rank)))
    weight = pd.Series(weight, index=bars.index).fillna(0.0)
    # Сдвиг веса на 1 бар: ранг волы известен по закрытию.
    return (base * weight.shift(1).fillna(0.0)).clip(0.0, 1.0)


CARVER_MR = {
    "carver_mr": carver_mr,
    "mr_lowvol_soft": mr_lowvol_soft,
}
