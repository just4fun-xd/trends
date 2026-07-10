"""Variance-ratio тест Ло-МакКинлея — основа Hurst-аллокатора v2.

Зачем не агрегированная дисперсия (hurst_aggvar): VR-тест даёт не только
точечную оценку персистентности, но и СТАТИСТИКУ ЗНАЧИМОСТИ отклонения
от случайного блуждания (z-score с гетероскедастичной поправкой). Это
позволяет отличить «инструмент действительно трендовый» от «шумная
оценка H около 0.5».

Теория (Lo & MacKinlay 1988):
  VR(q) = Var[r_t(q)] / (q · Var[r_t])
где r_t(q) — q-периодная сумма лог-доходностей. Для случайного
блуждания VR(q)=1. VR>1 — положительная автокорреляция (тренд,
персистентность); VR<1 — отрицательная (реверсия, антиперсистентность).

Связь с показателем Хёрста: VR(q) ≈ q^(2H−1), откуда
  H ≈ 0.5 + log2(VR(2)) / 2   (по q=2),
или устойчивее — из наклона log VR(q) по log q через несколько q.

Гетероскедастичная z-статистика (Lo-MacKinlay, robust):
  z(q) = (VR(q) − 1) / sqrt(theta(q))
где theta(q) — асимптотическая дисперсия при гетероскедастичности
(не требует гомоскедастичности доходностей — важно для сырья).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def variance_ratio(rets: np.ndarray, q: int) -> tuple[float, float]:
    """VR(q) и его гетероскедастичный z-score (Lo-MacKinlay 1988).

    Args:
        rets: Лог-доходности (1D, без NaN).
        q: Горизонт агрегирования (>=2).

    Returns:
        (vr, z): отношение дисперсий и robust z-статистика.
        z ~ N(0,1) при H0 (случайное блуждание). |z|>1.64 значимо
        на 90%, >1.96 на 95%.
    """
    n = len(rets)
    if n < q + 1 or q < 2:
        return float("nan"), float("nan")
    mu = rets.mean()
    # Дисперсия 1-периодная (несмещённая).
    var1 = np.sum((rets - mu) ** 2) / (n - 1)
    if var1 <= 0:
        return float("nan"), float("nan")
    # q-периодные перекрывающиеся суммы.
    cumsum = np.cumsum(rets)
    # r_t(q) = sum_{i=0}^{q-1} r_{t-i}; берём перекрывающиеся окна.
    q_sums = cumsum[q - 1:] - np.concatenate(
        ([0.0], cumsum[:-q]))
    m = q * (n - q + 1) * (1.0 - q / n)  # нормировка Lo-MacKinlay
    if m <= 0:
        return float("nan"), float("nan")
    varq = np.sum((q_sums - q * mu) ** 2) / m
    vr = varq / var1

    # Гетероскедастичная асимптотическая дисперсия theta(q).
    theta = 0.0
    dev = (rets - mu) ** 2
    denom = (np.sum(dev)) ** 2
    if denom <= 0:
        return float(vr), float("nan")
    for j in range(1, q):
        # delta_j — вклад лаг-j (robust к гетероскедастичности).
        num = np.sum(dev[j:] * dev[:-j])
        delta_j = num / denom * n
        weight = (2.0 * (q - j) / q) ** 2
        theta += weight * delta_j
    if theta <= 0:
        return float(vr), float("nan")
    z = (vr - 1.0) / np.sqrt(theta)
    return float(vr), float(z)


def hurst_from_vr(
    rets: np.ndarray, qs: tuple = (2, 4, 8, 16),
) -> tuple[float, float]:
    """H и агрегированный z из наклона log VR(q) по log q.

    VR(q) ≈ q^(2H−1)  =>  log VR(q) = (2H−1) log q.
    Наклон b регрессии log VR на log q даёт H = (b+1)/2.

    Args:
        rets: Лог-доходности (без NaN).
        qs: Горизонты (>=2 каждый).

    Returns:
        (H, z_mean): показатель Хёрста и средний |z| по q
        (сила сигнала отклонения от RW).
    """
    log_q, log_vr, zs = [], [], []
    for q in qs:
        vr, z = variance_ratio(rets, q)
        if np.isnan(vr) or vr <= 0:
            continue
        log_q.append(np.log(q))
        log_vr.append(np.log(vr))
        if not np.isnan(z):
            zs.append(z)
    if len(log_q) < 2:
        return float("nan"), float("nan")
    slope = np.polyfit(np.asarray(log_q), np.asarray(log_vr), 1)[0]
    h = (slope + 1.0) / 2.0
    z_mean = float(np.mean(zs)) if zs else float("nan")
    return float(h), z_mean


def rolling_hurst_vr(
    close: pd.Series, window: int = 504,
    qs: tuple = (2, 4, 8, 16), step: int = 21,
) -> pd.Series:
    """Rolling H по variance-ratio (trailing, сдвиг 1 бар).

    Пересчёт раз в `step` баров (H — медленное свойство; ежедневный
    пересчёт даёт шум и оборот). Между пересчётами держится последнее
    значение. Сдвиг 1 бар исключает look-ahead.

    Args:
        close: Цены закрытия.
        window: Trailing-окно (~2 года при 252).
        qs: Горизонты VR.
        step: Период пересчёта (баров).

    Returns:
        Ряд H (NaN на прогреве), сдвинутый на 1 бар.
    """
    # Гвард: WTI (CL) на 20.04.2020 печатал ОТРИЦАТЕЛЬНУЮ цену
    # (историческое событие, не ошибка данных) — log(отрицательное)
    # даёт RuntimeWarning и NaN. Явно гасим через close<=0 -> NaN
    # ДО лога, чтобы не полагаться на неявное поведение numpy.
    safe_close = close.where(close > 0)
    logret = np.log(safe_close / safe_close.shift(1)).to_numpy()
    n = len(logret)
    out = np.full(n, np.nan)
    last = np.nan
    for i in range(n):
        if i % step == 0 and i >= window:
            seg = logret[i - window:i]
            seg = seg[~np.isnan(seg)]
            if len(seg) >= window // 2:
                h, _ = hurst_from_vr(seg, qs)
                if not np.isnan(h):
                    last = h
        out[i] = last
    return pd.Series(out, index=close.index).shift(1)
