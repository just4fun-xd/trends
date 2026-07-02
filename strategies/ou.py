"""Ornstein-Uhlenbeck mean-reversion (трек 2.1).

СТАТУС (из OU_RESULTS.md): математика ВАЛИДНА (half-life, z-score —
корректный детектор растяжения), но НЕ робастна автономно. Провалила
walk-forward на ДВУХ классах (0/17 сырьё, 0/17 акции) по одному
механизму: торгует возврат к среднему ВСЛЕПУЮ, не различая режим. На
тренде шортит растущее и самоуничтожается (CL -1988% на train).

Поэтому OU здесь — рабочая математика для RANGE-ноги будущего роутера,
НЕ автономная стратегия. Прогонять её одну на трендовом активе опасно.
Роутер (regime/router.py) включает OU только когда детектор говорит RANGE.

dX = θ(μ - X)dt + σ dW. Оценка OLS: ΔX = a + b·X_{t-1} + ε,
θ = -b, μ = a/θ, half_life = ln(2)/θ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars


def ou_fit(series: pd.Series, adf_alpha: float = 0.05) -> dict:
    """Оценивает параметры OU по ряду через OLS + ADF-фильтр.

    ΔX_t = a + b·X_{t-1} + ε  =>  θ=-b, μ=a/θ, half_life=ln(2)/θ.

    ФИЛЬТР СТАЦИОНАРНОСТИ (аудит 2, Gemini): перед OLS прогоняем
    Augmented Dickey-Fuller. Причина: OLS-оценка b смещена вниз на
    конечной выборке (само распределение Дики-Фуллера), поэтому на
    ЧИСТОМ random walk (истинное b=0) почти всегда выходит b<0 -> θ>0 ->
    ложное «well_defined». Проверка знака θ случайное блуждание НЕ
    отсекает. ADF даёт формальный тест: p > adf_alpha -> нет статзначимой
    реверсии -> well_defined=False. Это формализует shuffle-control из
    OU_RESULTS.md (отделить реальную реверсию от артефакта).

    ВАЖНО (урок OU_RESULTS.md): применять к ЭКОНОМИЧЕСКИ осмысленному
    ряду (спред пары, лог-цена), НЕ к (цена - короткая SMA) — последнее
    даёт half-life 7-9д как АРТЕФАКТ фильтра (SMA догоняет цену с лагом),
    а не свойство рынка.

    Args:
        series: Ряд для оценки (уровень, не приращения).
        adf_alpha: Порог p-value ADF-теста. p выше -> ряд считается
            нестационарным, well_defined=False.

    Returns:
        dict с ключами theta, mu, sigma, half_life, adf_pvalue,
        well_defined (bool).
    """
    fail = {"theta": np.nan, "mu": np.nan, "sigma": np.nan,
            "half_life": np.nan, "adf_pvalue": np.nan,
            "well_defined": False}
    x = series.dropna()
    if len(x) < 30:
        return fail

    # ADF-фильтр стационарности (мягкий fallback, если нет statsmodels).
    adf_pvalue = np.nan
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_pvalue = float(adfuller(x.values, maxlag=1)[1])
        if adf_pvalue > adf_alpha:
            fail["adf_pvalue"] = adf_pvalue
            return fail
    except ImportError:
        # statsmodels не установлен — ADF пропущен, θ>0 остаётся
        # НЕОБХОДИМЫМ, но НЕ достаточным условием. Помечаем в выводе.
        pass

    x_lag = x.shift(1).dropna()
    dx = (x - x.shift(1)).dropna()
    # Выравниваем.
    x_lag, dx = x_lag.align(dx, join="inner")
    # OLS: dx = a + b*x_lag.
    A = np.vstack([np.ones(len(x_lag)), x_lag.values]).T
    coef, *_ = np.linalg.lstsq(A, dx.values, rcond=None)
    a, b = coef
    theta = -b
    if theta <= 0:
        # Не mean-reverting (θ<=0) — half-life не определён.
        out = dict(fail)
        out["theta"] = theta
        out["adf_pvalue"] = adf_pvalue
        return out
    mu = a / theta
    resid = dx.values - A @ coef
    sigma = np.std(resid)
    half_life = np.log(2) / theta
    return {"theta": theta, "mu": mu, "sigma": sigma,
            "half_life": half_life, "adf_pvalue": adf_pvalue,
            "well_defined": True}


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std.

    Без look-ahead — окно только из прошлых значений.

    Args:
        series: Ряд.
        window: Окно среднего и std.

    Returns:
        Ряд z-score.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    # Аудит 2026-07: вырожденное окно (std~0: плоские склейки, limit-дни)
    # маскируем в NaN. Подстановка малого eps вместо нуля дала бы
    # z ~ 1e8 из float-шума числителя -> ложный сигнал на мёртвом рынке.
    # NaN означает «нет нового сигнала»; цикл ниже держит состояние.
    std = std.where(std > 1e-12)
    return (series - mean) / std


def ou_zscore(
    bars: Bars,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> pd.Series:
    """OU-сигнал по z-score. НЕ автономная — RANGE-нога роутера.

    Логика (требует состояния -> цикл):
      - z < -entry_z (слишком низко) -> лонг (ставка на возврат вверх).
      - z > +entry_z (слишком высоко) -> шорт (ставка на возврат вниз).
      - |z| < exit_z -> закрыть (вернулись к среднему).
      - |z| > stop_z -> стоп (растяжение усиливается, не возврат).

    ПРЕДУПРЕЖДЕНИЕ: на трендовом активе шортит растущее и тонет. Гонять
    ТОЛЬКО под управлением детектора режима (regime=RANGE) или на
    подтверждённо mean-reverting спреде. Проверена на синтетич. OU-ряде.

    Args:
        bars: Данные инструмента.
        window: Окно z-score.
        entry_z: Порог входа по модулю z.
        exit_z: Порог выхода (возврат к среднему).
        stop_z: Порог стопа (растяжение вместо возврата).

    Returns:
        position: +1 лонг / -1 шорт / 0 вне.
    """
    z = _rolling_zscore(bars.close, window).values
    pos = np.zeros(len(z))
    state = 0
    for i in range(len(z)):
        if np.isnan(z[i]):
            # Аудит 2026-07: NaN = «нет нового сигнала», ДЕРЖИМ текущее
            # состояние. Раньше continue пропускал присвоение pos[i] ->
            # позиция проваливалась в 0 на один бар посреди сделки ->
            # фантомный выход-вход и двойные издержки.
            pos[i] = float(state)
            continue
        if state == 0:
            if z[i] < -entry_z:
                state = 1     # слишком низко -> лонг на возврат
            elif z[i] > entry_z:
                state = -1    # слишком высоко -> шорт на возврат
        elif state == 1:
            if z[i] > -exit_z or z[i] < -stop_z:
                state = 0
        elif state == -1:
            if z[i] < exit_z or z[i] > stop_z:
                state = 0
        pos[i] = float(state)
    return pd.Series(pos, index=bars.index)
