"""Trend Lab 2 — десять трендовых моделей на РАЗНОМ матаппарате.

Запрос: «ещё 10 стратегий, отлично работающих с трендами, ищи лучше
и больше». Принцип отбора — не десять вариаций одной идеи, а десять
РАЗНЫХ математик тренда: регрессия, экспоненциальное сглаживание
Хольта, ATR-состояние, производная осциллятора, мера режима VHF,
структура рынка (фракталы), композит Ишимоку, эффективность Кауфмана,
lag-компенсированные средние (ZLEMA, Hull).

Дисциплина multiple testing (10 моделей = 2-3 «сработают» случайно):
walk-forward скрининг -> bootstrap выживших против donchian_vt ->
второй источник -> one-shot. Позиции непрерывные либо {0,1}, сдвиг
1 бар, VT снаружи.

 1. tr_regress   — t-статистика наклона × положение над линией.
 2. tr_holt      — тренд-компонента двойного сглаживания Хольта (ETS).
 3. tr_supertrend— ATR-полоса с перекладкой (стейтфул трейлинг).
 4. tr_macd_hz   — z-score MACD-гистограммы (ускорение момента).
 5. tr_vhf       — Vertical Horizontal Filter: тренд только в
                   трендовом режиме по VHF (не ADX, не Hurst).
 6. tr_fractal   — пробой последнего подтверждённого фрактала (структура).
 7. tr_ichimoku  — Tenkan/Kijun композит (упрощённый Ишимоку).
 8. tr_er        — Efficiency Ratio Кауфмана как непрерывный размер.
 9. tr_zlema     — кросс zero-lag EMA (компенсация лага).
10. tr_hull      — наклон Hull MA (минимальный лаг WMA-каскада).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars


def _shift01(sig: pd.Series) -> pd.Series:
    # ФИКС 2026-07j: раньше здесь был shift(1) — ДВОЙНОЙ лаг
    # (движок run_engine сам сдвигает позицию). Лаборатория
    # торговала на t+2 против t+1 у donchian — фора чемпиону
    # во всех bootstrap-сравнениях. Имя сохранено от диффа.
    return sig.fillna(0.0)


# ── 1. Регрессионный канал ────────────────────────────────────────────
def tr_regress(bars: Bars, window: int = 63) -> pd.Series:
    """t-stat наклона × сторона от линии: значимый тренд И цена с ним.

    Наклон/SE — значимость тренда (пила даёт большой SE -> нет сигнала);
    вход только когда цена НАД регрессионной линией (движение
    подтверждено, не ловим разворот). Непрерывно: tanh(t/2) при
    цене над линией, иначе 0. Long+short симметрично.
    """
    y = np.log(bars.close.where(bars.close > 0))
    x = np.arange(window, dtype=float)
    x -= x.mean()
    sxx = float((x ** 2).sum())
    arr = y.to_numpy()
    n = len(arr)
    t_stat = np.full(n, np.nan)
    above = np.full(n, 0.0)
    for i in range(window, n):
        win = arr[i - window + 1:i + 1]
        if np.isnan(win).any():
            continue
        b = float((x * (win - win.mean())).sum()) / sxx
        fit_last = win.mean() + b * x[-1]
        resid = win - (win.mean() + b * x)
        se2 = float((resid ** 2).sum()) / max(window - 2, 1) / sxx
        t_stat[i] = b / np.sqrt(se2) if se2 > 0 else 0.0
        above[i] = 1.0 if win[-1] >= fit_last else -1.0
    t = pd.Series(t_stat, index=y.index)
    side = pd.Series(above, index=y.index)
    sig = np.tanh(t / 2.0)
    sig = sig.where(np.sign(sig) == side, 0.0)
    return _shift01(sig.clip(-1, 1))


# ── 2. Двойное сглаживание Хольта ────────────────────────────────────
def tr_holt(bars: Bars, alpha: float = 0.1,
            beta: float = 0.05) -> pd.Series:
    """Тренд-компонента Хольта (ETS): классика до-калмановской эпохи.

    level_t = α·y + (1−α)(level+trend); trend_t = β·Δlevel + (1−β)·trend.
    Иная адаптивность, чем Kalman (фиксированные α/β вместо q/r) —
    независимая точка в пространстве сглаживателей. Сигнал = tanh
    нормированного тренда.
    """
    y = np.log(bars.close.where(bars.close > 0)).to_numpy(dtype=float)
    n = len(y)
    trend = np.full(n, np.nan)
    lvl, tr = y[0], 0.0
    for t in range(1, n):
        if np.isnan(y[t]):
            trend[t] = tr
            continue
        prev = lvl
        lvl = alpha * y[t] + (1 - alpha) * (lvl + tr)
        tr = beta * (lvl - prev) + (1 - beta) * tr
        trend[t] = tr
    s = pd.Series(trend, index=bars.close.index)
    z = s / s.rolling(63, min_periods=20).std().replace(0, np.nan)
    return _shift01(pd.Series(np.tanh(z), index=s.index))


# ── 3. SuperTrend ─────────────────────────────────────────────────────
def tr_supertrend(bars: Bars, period: int = 10,
                  mult: float = 3.0) -> pd.Series:
    """ATR-полоса с перекладкой: культовый трейлинг-переключатель.

    Полоса hl2 ± mult·ATR «подтягивается» за ценой и не отступает;
    пробой полосы перекладывает режим лонг<->шорт. Стейтфул, держит
    тренд до фактического слома — профиль «держать до упора».
    """
    hl2 = (bars.high + bars.low) / 2.0
    tr_ = pd.concat([
        bars.high - bars.low,
        (bars.high - bars.close.shift(1)).abs(),
        (bars.low - bars.close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr_.ewm(span=period, adjust=False).mean()
    up = (hl2 + mult * atr).to_numpy()
    dn = (hl2 - mult * atr).to_numpy()
    close = bars.close.to_numpy()
    n = len(close)
    pos = np.zeros(n)
    ub, lb, state = up[0], dn[0], 0
    for i in range(1, n):
        ub = min(up[i], ub) if close[i - 1] <= ub else up[i]
        lb = max(dn[i], lb) if close[i - 1] >= lb else dn[i]
        if close[i] > ub:
            state = 1
        elif close[i] < lb:
            state = -1
        pos[i] = float(state)
    return pd.Series(pos, index=bars.close.index).fillna(0.0)


# ── 4. Z-гистограмма MACD ────────────────────────────────────────────
def tr_macd_hz(bars: Bars, fast: int = 12, slow: int = 26,
               sig: int = 9) -> pd.Series:
    """Гистограмма MACD (момент минус его среднее) в z-шкале.

    EWMAC меряет скорость; гистограмма — её отклонение от своей EMA,
    т.е. УСКОРЕНИЕ момента. Ранний вход в разгон, ранний сброс на
    затухании; z-нормировка даёт сопоставимый масштаб на всех активах.
    """
    c = bars.close
    macd = (c.ewm(span=fast, adjust=False).mean()
            - c.ewm(span=slow, adjust=False).mean())
    hist = macd - macd.ewm(span=sig, adjust=False).mean()
    z = hist / hist.rolling(126, min_periods=30).std().replace(0, np.nan)
    return _shift01(pd.Series(np.tanh(z), index=c.index))


# ── 5. VHF-гейт ───────────────────────────────────────────────────────
def tr_vhf(bars: Bars, window: int = 28, gate: float = 0.4,
           span: int = 20) -> pd.Series:
    """Направление EMA-наклона ТОЛЬКО в трендовом режиме по VHF.

    VHF = |max−min| / Σ|Δclose|: доля чистого хода в суммарном пути.
    Высокий VHF — рынок ИДЁТ, низкий — пилит. Третья, независимая от
    ADX и Hurst, мера трендовости; торгуем только когда путь
    эффективен (VHF>gate).
    """
    c = bars.close
    num = (c.rolling(window).max() - c.rolling(window).min()).abs()
    den = c.diff().abs().rolling(window).sum().replace(0, np.nan)
    vhf = num / den
    slope = c.ewm(span=span, adjust=False).mean().pct_change(5)
    z = slope / slope.rolling(126, min_periods=30).std().replace(0, np.nan)
    sig = pd.Series(np.tanh(z), index=c.index).where(vhf > gate, 0.0)
    return _shift01(sig)


# ── 6. Пробой фрактала ────────────────────────────────────────────────
def tr_fractal(bars: Bars, wing: int = 2) -> pd.Series:
    """Пробой последнего ПОДТВЕРЖДЁННОГО свинг-максимума (структура).

    Фрактал Вильямса: бар, чей high выше wing соседей с обеих сторон
    (подтверждается с лагом wing — без look-ahead). Пробой последнего
    фрактал-хая -> лонг до пробоя фрактал-лоу. Торгует СТРУКТУРУ
    рынка, а не сглаженные средние — иной класс информации.
    """
    h, low = bars.high.to_numpy(), bars.low.to_numpy()
    c = bars.close.to_numpy()
    n = len(c)
    last_hi, last_lo = np.nan, np.nan
    pos = np.zeros(n)
    state = 0
    for i in range(n):
        j = i - wing  # кандидат, подтверждаемый сегодня
        if j >= wing:
            if h[j] == max(h[j - wing:j + wing + 1]):
                last_hi = h[j]
            if low[j] == min(low[j - wing:j + wing + 1]):
                last_lo = low[j]
        if not np.isnan(last_hi) and c[i] > last_hi:
            state = 1
        elif not np.isnan(last_lo) and c[i] < last_lo:
            state = -1
        pos[i] = float(state)
    return pd.Series(pos, index=bars.close.index).fillna(0.0)


# ── 7. Ишимоку-лайт ──────────────────────────────────────────────────
def tr_ichimoku(bars: Bars, tenkan: int = 9,
                kijun: int = 26) -> pd.Series:
    """Tenkan/Kijun композит: двойное подтверждение направления.

    Tenkan (середина 9-барного диапазона) над Kijun (26-барного) И цена
    над Kijun -> лонг; зеркально -> шорт. Средние по ДИАПАЗОНУ (не по
    close) — иная оценка центра, устойчивая к разрывам закрытий.
    """
    tk = (bars.high.rolling(tenkan).max()
          + bars.low.rolling(tenkan).min()) / 2.0
    kj = (bars.high.rolling(kijun).max()
          + bars.low.rolling(kijun).min()) / 2.0
    long_ = (tk > kj) & (bars.close > kj)
    short = (tk < kj) & (bars.close < kj)
    pos = pd.Series(0.0, index=bars.close.index)
    pos[long_] = 1.0
    pos[short] = -1.0
    return _shift01(pos)


# ── 8. Efficiency Ratio как размер ───────────────────────────────────
def tr_er(bars: Bars, window: int = 20) -> pd.Series:
    """Позиция = знак хода × эффективность Кауфмана (чистота тренда).

    ER = |ход за N| / Σ|шагов|: 1 — идеальная прямая, 0 — пила.
    KAMA использует ER внутри сглаживания; здесь ER — непосредственно
    РАЗМЕР позиции: чистый тренд получает полную ставку, пила — ноль.
    Самая прямая математизация «торгуй тренд по его качеству».
    """
    c = bars.close
    move = (c - c.shift(window)).abs()
    path = c.diff().abs().rolling(window).sum().replace(0, np.nan)
    er = (move / path).clip(0, 1)
    sig = np.sign(c - c.shift(window)) * er
    return _shift01(pd.Series(sig, index=c.index))


# ── 9. Zero-lag EMA кросс ────────────────────────────────────────────
def tr_zlema(bars: Bars, fast: int = 20, slow: int = 60) -> pd.Series:
    """Кросс ZLEMA: EMA с компенсацией лага (де-лаг входного ряда).

    ZLEMA подаёт в EMA ряд 2·p − p.shift(lag), выталкивая запаздывание.
    Кросс быстрой/медленной ZLEMA реагирует на разворот на несколько
    баров раньше обычного EMA-кросса — критично на импульсной крипте.
    """
    def zlema(s: pd.Series, span: int) -> pd.Series:
        lag = (span - 1) // 2
        return (2 * s - s.shift(lag)).ewm(span=span, adjust=False).mean()

    c = bars.close
    diff = zlema(c, fast) - zlema(c, slow)
    z = diff / (c * c.pct_change().rolling(63).std()).replace(0, np.nan)
    return _shift01(pd.Series(np.tanh(z * 2), index=c.index))


# ── 10. Наклон Hull MA ───────────────────────────────────────────────
def tr_hull(bars: Bars, window: int = 32) -> pd.Series:
    """Знак и сила наклона Hull MA (WMA-каскад минимального лага).

    HMA = WMA(2·WMA(n/2) − WMA(n), sqrt(n)): гасит лаг почти до нуля,
    сохраняя гладкость. Наклон HMA — быстрый и чистый детектор
    направления; сигнал = tanh нормированного наклона.
    """
    def wma(s: pd.Series, n: int) -> pd.Series:
        w = np.arange(1, n + 1, dtype=float)
        return s.rolling(n).apply(
            lambda x: float(np.dot(x, w) / w.sum()), raw=True)

    c = bars.close
    n2, ns = max(window // 2, 2), max(int(np.sqrt(window)), 2)
    hma = wma(2 * wma(c, n2) - wma(c, window), ns)
    slope = hma.pct_change(3)
    z = slope / slope.rolling(126, min_periods=30).std().replace(0, np.nan)
    return _shift01(pd.Series(np.tanh(z), index=c.index))


TREND_LAB2 = {
    "tr_regress": tr_regress,
    "tr_holt": tr_holt,
    "tr_supertrend": tr_supertrend,
    "tr_macd_hz": tr_macd_hz,
    "tr_vhf": tr_vhf,
    "tr_fractal": tr_fractal,
    "tr_ichimoku": tr_ichimoku,
    "tr_er": tr_er,
    "tr_zlema": tr_zlema,
    "tr_hull": tr_hull,
}
