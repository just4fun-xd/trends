"""Kalman trend — тренд-следование через state-space (уровень + наклон).

Ответ на «пробой Дончиана слишком просто». Дончиан смотрит на
экстремумы окна — грубая эвристика. Здесь тренд оценивается
математической моделью: локальная линейная модель (local linear trend)
в форме Калмана раздельно отслеживает УРОВЕНЬ и СКОРОСТЬ (наклон),
фильтруя шум оптимально по Калману.

Модель состояния (Harvey local linear trend):
    level_t  = level_{t-1} + slope_{t-1} + w_level
    slope_t  = slope_{t-1}            + w_slope
    y_t      = level_t                + v          (наблюдение = цена)

Скрытый вектор x = [level, slope]. Калман рекурсивно оценивает оба.
Торговый сигнал — по ЗНАКУ И СИЛЕ оценённого наклона: рынок трендит
вверх, когда фильтр уверенно видит положительную скорость. В отличие
от пробоя, наклон:
  - непрерывен (сила сигнала, не 0/1) — меньше кэша, плавный вход;
  - адаптивен (Калман сам решает, как быстро реагировать, по
    соотношению шумов q/r) — не фиксированное окно;
  - фильтрует шум оптимально, а не скользящим средним с лагом.

Параметр `q_slope_ratio` = дисперсия шума наклона / дисперсия
наблюдения. Мал -> гладкий, медленный тренд (крупные движения); велик
-> шустрый, ловит быстрые смены (для интрадей-крипты). Это и есть
«быстрый/медленный тренд» из трёхкластерной модели — один параметр.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars


def _kalman_level_slope(
    y: np.ndarray, q_level: float, q_slope: float, r: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Калман local linear trend: возвращает (level, slope) по ряду.

    Args:
        y: Наблюдения (лог-цена).
        q_level: Дисперсия шума уровня.
        q_slope: Дисперсия шума наклона.
        r: Дисперсия шума наблюдения.

    Returns:
        (level, slope): отфильтрованные оценки скрытого состояния.
    """
    n = len(y)
    level = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    # Матрицы модели: x = [level, slope]; переход F, наблюдение H.
    # F = [[1,1],[0,1]], H = [1,0].
    x = np.array([y[0], 0.0])
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([[q_level, 0.0], [0.0, q_slope]])
    for t in range(n):
        # Предсказание.
        x_pred = np.array([x[0] + x[1], x[1]])
        f = np.array([[1.0, 1.0], [0.0, 1.0]])
        p_pred = f @ p @ f.T + q
        # Обновление по наблюдению y[t] (H = [1,0]).
        resid = y[t] - x_pred[0]
        s = p_pred[0, 0] + r
        k = p_pred[:, 0] / s          # калман-усиление (2-вектор)
        x = x_pred + k * resid
        p = p_pred - np.outer(k, p_pred[0, :])
        level[t] = x[0]
        slope[t] = x[1]
    return level, slope


def kalman_trend(
    bars: Bars,
    q_slope_ratio: float = 1e-3,
    q_level_ratio: float = 1e-2,
    smooth: int = 5,
) -> pd.Series:
    """Позиция по знаку/силе Калман-оценённого наклона (тренда).

    Сигнал = нормированный наклon, раздавленный в [-1, 1]: сильный
    устойчивый тренд -> позиция к ±1, слабый/шумный -> к 0. Непрерывно,
    без бинарного пробоя. Сдвиг 1 бар (наклон на t торгуется на t+1).

    Args:
        bars: Данные инструмента.
        q_slope_ratio: q_slope / r — реактивность наклона (мал=медленный
            тренд, велик=быстрый; «быстрый/медленный» кластер).
        q_level_ratio: q_level / r — гибкость уровня.
        smooth: Окно нормировки наклона (в его же ст.отклонениях).

    Returns:
        position в [-1, 1] (long+short). Для long-only обёртка снаружи.
    """
    close = bars.close.where(bars.close > 0)
    y = np.log(close).to_numpy(dtype=float)
    # Дисперсия наблюдения r из краткосрочной вариации лог-цены.
    diffs = np.diff(y[~np.isnan(y)])
    r = float(np.nanvar(diffs)) if len(diffs) > 2 else 1e-4
    r = max(r, 1e-10)
    _, slope = _kalman_level_slope(
        y, q_level_ratio * r, q_slope_ratio * r, r)
    slope_s = pd.Series(slope, index=bars.close.index)
    # Нормируем наклон на его волатильность -> сигнал в единицах «сигм
    # наклона», давим tanh в [-1,1] (сильный тренд -> насыщение).
    scale = slope_s.rolling(63, min_periods=20).std()
    z = slope_s / scale.replace(0.0, np.nan)
    sig = np.tanh(z / 1.5)
    sig = sig.rolling(smooth, min_periods=1).mean()
    return sig.shift(1).fillna(0.0)


def kalman_trend_long(bars: Bars, **kw) -> pd.Series:
    """Long-only версия: только положительный наклон (крипта-спот)."""
    return kalman_trend(bars, **kw).clip(lower=0.0)


KALMAN_TREND = {
    "kalman_trend": kalman_trend,
    "kalman_trend_long": kalman_trend_long,
}
