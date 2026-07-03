"""GARCH(1,1) прогноз волатильности для vol targeting.

Гипотеза трека: realized vol в vol_target_size — запаздывающая оценка
(rolling std за 30 баров реагирует на вспышку волы через дни). GARCH(1,1)
даёт forward-looking прогноз: вчерашний шок входит в завтрашнюю сигму
немедленно через alpha, и прогноз сходится к долгосрочной воле со
скоростью (alpha + beta). Ожидание: VT режет плечо на 2-5 баров раньше
в начале кризиса -> меньше хвост. Это ГИПОТЕЗА — вердикт только через
A/B против realized-VT на walk-forward.

Модель:
    r_t = eps_t,  eps_t ~ N(0, sigma2_t)
    sigma2_t = omega + alpha * eps2_{t-1} + beta * sigma2_{t-1}

Оценка: MLE (минимизация -loglik) c variance targeting:
    omega = var(r) * (1 - alpha - beta)
т.е. оптимизируем только (alpha, beta) — устойчивее и быстрее, чем
трёхпараметрическая MLE (стандартный приём, Engle-Mezrich 1996).

Честность (без look-ahead):
    - Параметры рефитятся раз в refit_every баров ТОЛЬКО на прошлом
      (trailing window fit_window баров).
    - Между рефитами sigma2 обновляется рекурсией по мере прихода
      новых баров; прогноз на бар t+1 использует только данные <= t.
    - Прогрев: пока данных < min_obs, множитель = NaN -> держим 0
      (как прогрев в vol_target_size).

Чистый numpy + scipy.optimize (minimal stack). Никакого пакета arch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.bars import Bars


def garch_neg_loglik(
    params: np.ndarray, r: np.ndarray, var_target: float
) -> float:
    """Отрицательный лог-лайклихуд GARCH(1,1) с variance targeting.

    Args:
        params: (alpha, beta).
        r: Ряд доходностей (без NaN), центрированный не обязателен —
            для дневных фьючерсов среднее пренебрежимо мало против std.
        var_target: Безусловная дисперсия ряда (задаёт omega).

    Returns:
        -loglik (float). Большое число при нарушении ограничений.
    """
    alpha, beta = params
    if alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10
    omega = var_target * (1.0 - alpha - beta)
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = var_target
    r2 = r * r
    for t in range(1, n):
        sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
    # Защита от численного нуля.
    sigma2 = np.maximum(sigma2, 1e-18)
    return float(0.5 * np.sum(np.log(sigma2) + r2 / sigma2))


def fit_garch(r: np.ndarray) -> tuple[float, float, float]:
    """Оценивает (omega, alpha, beta) GARCH(1,1) по MLE.

    Args:
        r: Доходности (окно фита, без NaN, длина >= ~200).

    Returns:
        (omega, alpha, beta). При провале оптимизации — вырожденный
        случай alpha=0, beta=0: sigma2 = var(r), т.е. честный откат
        к константной воле (эквивалент realized vol на окне).
    """
    var_target = float(np.var(r))
    if var_target <= 0:
        return 1e-12, 0.0, 0.0
    best = None
    # Мультистарт: поверхность loglik с variance targeting обычно
    # унимодальна, но два старта страхуют от плато.
    for x0 in ((0.05, 0.90), (0.10, 0.80)):
        res = minimize(
            garch_neg_loglik, x0=np.array(x0),
            args=(r, var_target), method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 500},
        )
        if best is None or res.fun < best.fun:
            best = res
    alpha, beta = best.x
    if alpha < 0 or beta < 0 or alpha + beta >= 0.999 or not best.success:
        return var_target, 0.0, 0.0
    omega = var_target * (1.0 - alpha - beta)
    return float(omega), float(alpha), float(beta)


def garch_vol_forecast(
    returns: pd.Series,
    fit_window: int = 750,
    refit_every: int = 63,
    min_obs: int = 250,
) -> pd.Series:
    """Побарный one-step-ahead прогноз волы (std за бар), без look-ahead.

    На баре t возвращается прогноз sigma_{t+1}, использующий данные
    <= t. Параметры рефитятся каждые refit_every баров на последних
    fit_window наблюдениях; между рефитами рекурсия sigma2 идёт на
    зафиксированных параметрах (шоки продолжают входить через alpha).

    Args:
        returns: Побарные доходности (NaN в начале допустим).
        fit_window: Trailing-окно MLE (750 ~ 3 года дневных).
        refit_every: Период рефита (63 ~ квартал).
        min_obs: Минимум наблюдений до первого фита (прогрев).

    Returns:
        Ряд прогнозов std за бар; NaN на прогреве.
    """
    r = returns.to_numpy(dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    valid = ~np.isnan(r)

    omega = alpha = beta = None
    sigma2 = None
    obs_seen = 0
    since_fit = refit_every  # форсируем фит на первом валидном пороге

    for t in range(n):
        if not valid[t]:
            continue
        obs_seen += 1
        if obs_seen < min_obs:
            continue
        since_fit += 1
        if since_fit >= refit_every or omega is None:
            hist = r[: t + 1]
            hist = hist[~np.isnan(hist)][-fit_window:]
            omega, alpha, beta = fit_garch(hist)
            # Инициализация рекурсии с безусловной дисперсии окна.
            if sigma2 is None:
                sigma2 = float(np.var(hist))
            since_fit = 0
        # Рекурсия: sigma2 к бару t уже включает eps_{t-1}; обновляем
        # текущим баром и получаем прогноз на t+1.
        sigma2 = omega + alpha * r[t] * r[t] + beta * sigma2
        out[t] = np.sqrt(max(sigma2, 1e-18))
    return pd.Series(out, index=returns.index)


def garch_vol_target_size(
    bars: Bars,
    target_vol: float = 0.15,
    max_leverage: float = 2.0,
    buffer: float = 0.10,
    fit_window: int = 750,
    refit_every: int = 63,
) -> pd.Series:
    """Множитель позиции: target_vol / GARCH-прогноз волы.

    Полный аналог vol_target_size (тот же потолок плеча и буфер
    ребалансировки из аудита 2026-07), но знаменатель — прогноз
    GARCH(1,1) вместо rolling std. Взаимозаменяем в любой стратегии:
        position * garch_vol_target_size(bars)

    Args:
        bars: Данные инструмента.
        target_vol: Целевая годовая волатильность.
        max_leverage: Потолок множителя.
        buffer: Мёртвая зона ребалансировки (см. vol_target_size).
        fit_window: Окно MLE.
        refit_every: Период рефита параметров.

    Returns:
        Ряд множителей размера [0, max_leverage].
    """
    bar_sigma = garch_vol_forecast(
        bars.returns(), fit_window=fit_window, refit_every=refit_every
    )
    annual = bar_sigma * np.sqrt(bars.bars_per_year)
    raw = (target_vol / annual).clip(upper=max_leverage)
    raw_v = raw.to_numpy()

    out = np.zeros(len(raw_v))
    applied = 0.0
    for i in range(len(raw_v)):
        r = raw_v[i]
        if np.isnan(r):
            out[i] = applied
            continue
        if applied == 0.0:
            applied = r
        elif abs(r - applied) > buffer * abs(applied):
            applied = r
        out[i] = applied
    return pd.Series(out, index=bars.index)
