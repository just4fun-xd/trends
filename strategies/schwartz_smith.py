"""Schwartz-Smith two-factor (Roadmap D.1) — НОВАЯ гипотеза, не OU-реванш.

Формулировка для Александра (из roadmap, дословно по смыслу): раздел
OU закрыт честно — 18 форм торговали грубый прокси остатка
(цена - SMA/EWMA), где скользящее среднее в тренде само едет за ценой
и OU воюет с трендом. Schwartz-Smith — единственная НЕтестированная
форма: Kalman-фильтр ЯВНО раскладывает лог-цену на два ненаблюдаемых
фактора,

    ln(S_t) = chi_t + xi_t
    chi_t   — краткосрочное OU-отклонение (спрос/предложение),
              d chi = -kappa*chi dt + sigma_chi dW_chi
    xi_t    — долгосрочный равновесный уровень (GBM с дрейфом),
              d xi  = mu dt + sigma_xi dW_xi

и MR-сигнал запускается СТРОГО на изолированном chi_t — пространстве,
очищенном от тренда (тренд живёт в xi и отдан Дончиану).

Честные ограничения реализации (проговорить в отчёте):
1. Классический Schwartz-Smith оценивается на ПАНЕЛИ фьючерсных
   сроков (term structure идентифицирует факторы). У нас один
   roll-adjusted непрерывный ряд => идентификация только через
   динамику (сумма OU + RW наблюдаемая = ARIMA(1,1,1)): параметры
   идентифицируемы частично. Поэтому rho (корр. шоков) по умолчанию
   ЗАФИКСИРОВАН в 0 — стандартное упрощение при одном ряде; флаг
   rho_free=True включает полную 5-параметрическую MLE.
2. Roll-adjusted ряд может уходить <= 0 (WTI апрель-2020): бары с
   close <= 0 пропускаются фильтром (predict без update).
3. Схема честности — как у core/garch.py: рефит MLE раз в
   refit_every баров на trailing fit_window, между рефитами фильтр
   идёт вперёд на зафиксированных параметрах; chi_t на баре t
   использует только данные <= t. Префикс-тест обязателен.

Барьер приёмки — как всем: bootstrap против mr_lowvol (UNDEFEATED)
на ДВУХ источниках (gross) + corr < 0.6 для ансамбля. Трезвое
ожидание: MR-нишу сырья mr_lowvol уже закрывает; тест one-shot.

Контракт: Bars -> position [0, 1] (long-only MR). Сдвига внутри
НЕТ — движок сдвигает. VT снаружи (--vt).

Стратегии:
  ss_chi_mr   — пороговая MR-нога на z(chi): вход z<-1.5, выход z>=0.
  ss_chi_soft — непрерывная позиция clip(-z/2, 0, 1) (Carver-стиль).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.bars import Bars


# ── фильтр Калмана (скалярный, 2 состояния) ──────────────────────────
def _kalman_pass(
    y: np.ndarray, kappa: float, sig_chi: float, mu: float,
    sig_xi: float, rho: float = 0.0, r_obs: float = 1e-6,
    collect: bool = False,
) -> tuple[float, np.ndarray | None]:
    """Один проход фильтра по ряду лог-цен.

    Возвращает (loglik, chi_filtered | None). NaN-бары: predict без
    update (позиция состояния дрейфует, неопределённость растёт).

    Args:
        y: Лог-цены (NaN допустимы).
        kappa: Скорость реверсии chi (за бар).
        sig_chi: Вола шока chi (за бар, до дискретизации).
        mu: Дрейф xi за бар.
        sig_xi: Вола шока xi за бар.
        rho: Корреляция шоков chi/xi (0 при одном ряде — см. шапку).
        r_obs: Дисперсия шума наблюдения (численная регуляризация).
        collect: Собирать ли отфильтрованный chi.

    Returns:
        (loglik, chi) — chi есть только при collect=True.
    """
    n = len(y)
    phi = float(np.exp(-kappa))
    q11 = sig_chi ** 2 * (1.0 - phi ** 2) / (2.0 * kappa)
    q22 = sig_xi ** 2
    q12 = rho * sig_chi * sig_xi * (1.0 - phi) / kappa
    # инициализация: xi = первый валидный y, chi = 0 (стационарная P)
    first = 0
    while first < n and np.isnan(y[first]):
        first += 1
    if first >= n - 10:
        return -np.inf, None
    chi, xi = 0.0, float(y[first])
    p11 = sig_chi ** 2 / (2.0 * kappa)
    p12, p22 = 0.0, 1.0
    ll = 0.0
    out = np.full(n, np.nan) if collect else None
    log2pi = float(np.log(2.0 * np.pi))
    for t in range(first + 1, n):
        # predict
        chi = phi * chi
        xi = xi + mu
        p11 = phi * phi * p11 + q11
        p12 = phi * p12 + q12
        p22 = p22 + q22
        if np.isnan(y[t]):
            if collect:
                out[t] = chi
            continue
        # update, H = [1, 1]
        a = p11 + p12
        b = p12 + p22
        s = a + b + r_obs
        v = y[t] - (chi + xi)
        chi += (a / s) * v
        xi += (b / s) * v
        p11 -= a * a / s
        p12 -= a * b / s
        p22 -= b * b / s
        ll += -0.5 * (log2pi + np.log(s) + v * v / s)
        if collect:
            out[t] = chi
    return ll, out


def _fit_ss(
    y: np.ndarray, rho_free: bool = False,
) -> tuple[float, float, float, float, float] | None:
    """MLE (kappa, sig_chi, mu, sig_xi[, rho]) на окне лог-цен.

    Nelder-Mead в трансформированном пространстве (log для
    положительных, atanh для rho) — без градиентов, устойчив к
    оврагам частично идентифицируемой поверхности.
    """
    dy = np.diff(y)
    dy = dy[~np.isnan(dy)]
    if len(dy) < 100:
        return None
    sd = float(dy.std())
    if sd <= 0:
        return None
    x0 = [np.log(0.05), np.log(0.7 * sd / np.sqrt(1.0)), float(dy.mean()),
          np.log(0.5 * sd)]
    if rho_free:
        x0.append(0.0)

    def neg_ll(x: np.ndarray) -> float:
        kappa = float(np.exp(np.clip(x[0], -7.0, 1.5)))
        s_chi = float(np.exp(np.clip(x[1], -12.0, 0.0)))
        mu = float(np.clip(x[2], -0.05, 0.05))
        s_xi = float(np.exp(np.clip(x[3], -12.0, 0.0)))
        rho = float(np.tanh(x[4])) if rho_free else 0.0
        ll, _ = _kalman_pass(y, kappa, s_chi, mu, s_xi, rho)
        return -ll if np.isfinite(ll) else 1e12

    res = minimize(neg_ll, np.asarray(x0), method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-4,
                            "fatol": 1e-4})
    x = res.x
    kappa = float(np.exp(np.clip(x[0], -7.0, 1.5)))
    s_chi = float(np.exp(np.clip(x[1], -12.0, 0.0)))
    mu = float(np.clip(x[2], -0.05, 0.05))
    s_xi = float(np.exp(np.clip(x[3], -12.0, 0.0)))
    rho = float(np.tanh(x[4])) if rho_free else 0.0
    return kappa, s_chi, mu, s_xi, rho


def schwartz_smith_z(
    close: pd.Series,
    fit_window: int = 750,
    refit_every: int = 126,
    min_obs: int = 500,
    rho_free: bool = False,
) -> pd.Series:
    """Побарный z(chi_t) без look-ahead: chi / модельная сигма chi.

    Схема честности garch_vol_forecast: на баре рефита t параметры
    оцениваются на y[max(0, t-fit_window):t+1], фильтр перезапускается
    по этому же trailing-окну (только прошлое) и идёт вперёд на
    фиксированных параметрах до следующего рефита. До min_obs — NaN.

    Returns:
        Ряд z = chi_t * sqrt(2*kappa)/sigma_chi (стационарная
        нормировка из ПАРАМЕТРОВ модели, не из rolling-std).
    """
    y_full = np.log(close.where(close > 0)).to_numpy(dtype=float)
    n = len(y_full)
    z = np.full(n, np.nan)
    params = None
    state = None  # (chi, xi, p11, p12, p22) на конец прошлого бара
    for t in range(n):
        need_fit = (t >= min_obs
                    and (t - min_obs) % refit_every == 0)
        if need_fit:
            lo = max(0, t - fit_window + 1)
            fitted = _fit_ss(y_full[lo:t + 1], rho_free)
            if fitted is not None:
                params = fitted
                # перезапуск фильтра по trailing-окну (только прошлое)
                kappa, s_chi, mu, s_xi, rho = params
                _, chi_win = _kalman_pass(
                    y_full[lo:t + 1], kappa, s_chi, mu, s_xi, rho,
                    collect=True)
                state = _refilter_state(
                    y_full[lo:t + 1], kappa, s_chi, mu, s_xi, rho)
                if chi_win is not None and not np.isnan(chi_win[-1]):
                    sd_st = s_chi / np.sqrt(2.0 * kappa)
                    z[t] = chi_win[-1] / sd_st
                continue
        if params is None or state is None:
            continue
        kappa, s_chi, mu, s_xi, rho = params
        state, chi_t = _step(state, y_full[t], kappa, s_chi, mu,
                             s_xi, rho)
        sd_st = s_chi / np.sqrt(2.0 * kappa)
        z[t] = chi_t / sd_st
    return pd.Series(z, index=close.index)


def _refilter_state(
    y: np.ndarray, kappa: float, s_chi: float, mu: float,
    s_xi: float, rho: float,
) -> tuple[float, float, float, float, float] | None:
    """Прогон фильтра по окну, возврат конечного состояния."""
    n = len(y)
    phi = float(np.exp(-kappa))
    q11 = s_chi ** 2 * (1.0 - phi ** 2) / (2.0 * kappa)
    q22 = s_xi ** 2
    q12 = rho * s_chi * s_xi * (1.0 - phi) / kappa
    first = 0
    while first < n and np.isnan(y[first]):
        first += 1
    if first >= n - 2:
        return None
    chi, xi = 0.0, float(y[first])
    p11 = s_chi ** 2 / (2.0 * kappa)
    p12, p22 = 0.0, 1.0
    st = (chi, xi, p11, p12, p22)
    for t in range(first + 1, n):
        st, _ = _step(st, y[t], kappa, s_chi, mu, s_xi, rho)
    return st


def _step(
    state: tuple[float, float, float, float, float], y_t: float,
    kappa: float, s_chi: float, mu: float, s_xi: float, rho: float,
    r_obs: float = 1e-6,
) -> tuple[tuple[float, float, float, float, float], float]:
    """Один шаг фильтра (predict + update); NaN => только predict."""
    chi, xi, p11, p12, p22 = state
    phi = float(np.exp(-kappa))
    q11 = s_chi ** 2 * (1.0 - phi ** 2) / (2.0 * kappa)
    q22 = s_xi ** 2
    q12 = rho * s_chi * s_xi * (1.0 - phi) / kappa
    chi = phi * chi
    xi = xi + mu
    p11 = phi * phi * p11 + q11
    p12 = phi * p12 + q12
    p22 = p22 + q22
    if not np.isnan(y_t):
        a = p11 + p12
        b = p12 + p22
        s = a + b + r_obs
        v = y_t - (chi + xi)
        chi += (a / s) * v
        xi += (b / s) * v
        p11 -= a * a / s
        p12 -= a * b / s
        p22 -= b * b / s
    return (chi, xi, p11, p12, p22), chi


# ── стратегии ────────────────────────────────────────────────────────
def ss_chi_mr(
    bars: Bars, z_in: float = 1.5, z_out: float = 0.0,
    fit_window: int = 750, refit_every: int = 126,
    min_obs: int = 500, rho_free: bool = False,
) -> pd.Series:
    """Пороговая MR-нога на изолированном chi: вход z<-z_in, выход z>=z_out.

    Порог входа 1.5 (не 2.0 полосных): chi нормирован МОДЕЛЬНОЙ
    стационарной сигмой, а не rolling-std, растяжение читается чище.
    """
    z = schwartz_smith_z(bars.close, fit_window, refit_every,
                         min_obs, rho_free)
    zv = z.to_numpy()
    n = len(zv)
    pos = np.zeros(n)
    in_pos = False
    for i in range(n):
        if np.isnan(zv[i]):
            pos[i] = 0.0
            in_pos = False
            continue
        if in_pos and zv[i] >= z_out:
            in_pos = False
        elif not in_pos and zv[i] < -z_in:
            in_pos = True
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


def ss_chi_soft(
    bars: Bars, scale: float = 2.0, cap: float = 1.0,
    fit_window: int = 750, refit_every: int = 126,
    min_obs: int = 500, rho_free: bool = False,
) -> pd.Series:
    """Непрерывная позиция clip(-z/scale, 0, cap) — Carver-стиль.

    Больше времени в рынке на малых растяжениях (план «прибыль без
    плеча»), размер пропорционален глубине. Пара к ss_chi_mr: если
    выживет только одна форма — узнаем, где сидит edge (в порогах
    или в непрерывности).
    """
    z = schwartz_smith_z(bars.close, fit_window, refit_every,
                         min_obs, rho_free)
    return (-z / scale).clip(lower=0.0, upper=cap).fillna(0.0)


SCHWARTZ_SMITH = {
    "ss_chi_mr": ss_chi_mr,
    "ss_chi_soft": ss_chi_soft,
}
