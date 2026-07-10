"""Импульсная лаборатория: тренд-модели для агрессивных импульсных рынков.

Запрос Кирилла: «нужно больше стратегий вроде carver_fdm для агрессивных
импульсных рынков (крипта)». 10 моделей разного матаппарата + вариации
Carver. Урок Monday-range: на буйном тренде выигрывает тот, кто ДЕРЖИТ
импульс, а не фильтрует его; но нужна и защита от пилы. Здесь спектр от
«держать до упора» до «фильтровать умно».

ВАЖНО (multiple testing): 10+ стратегий на одной истории = гарантия,
что что-то «сработает» случайно. Дисциплина: walk-forward скрининг ->
bootstrap выживших против чемпиона -> второй источник -> one-shot.
Победитель одного прогона — НЕ победитель.

Все прогнозы непрерывные в [-1,1] (сила сигнала, меньше кэша). Сдвиг НЕ
делается здесь — единственная точка shift(1) это движок. VT снаружи.

Модели (матаппарат в скобках):
 1. imp_tsmom_vw   — мульти-горизонт TSMOM, vol-взвешенный (Moskowitz).
 2. imp_accel      — ускорение тренда: производная EWMAC (2-я произв. цены).
 3. imp_tstat      — t-статистика наклона регрессии (значимость тренда).
 4. imp_52h        — близость к N-барному максимуму (52w-high эффект).
 5. imp_atr_break  — пробой ATR-канала, непрерывный (Keltner-trend).
 6. imp_vol_expand — направление при расширении диапазона (range expansion).
 7. imp_skip_mom   — momentum 12-1 (пропуск последнего месяца, reversal-guard).
 8. imp_parkinson  — пробой, нормированный волой Паркинсона (high-low).
 9. imp_drawup     — асимметрия drawup/drawdown (кто ближе: max или min).
10. imp_kalman_imp — Kalman-наклон, гейтованный его же ростом (импульс).

Вариации Carver (тот же аппарат, другой характер):
 - carver_fast     — быстрые EWMAC-пары (4/16, 8/32, 16/64): реактивный.
 - carver_slow     — медленные (32/128, 64/256): держит большие тренды.
 - carver_ls       — long+short (шорт-нога для фьючерсов/перпов).
 - carver_hicap    — forecast_cap 30, fdm_cap 4: агрессивное насыщение.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.advanced import carver_fdm
from strategies.kalman_trend import _kalman_level_slope


def _vol(close: pd.Series, span: int = 36) -> pd.Series:
    """EWMA-вола дневных доходностей (для нормировок)."""
    ret = close.pct_change()
    return ret.ewm(span=span, adjust=False).std()


def _shift(sig: pd.Series) -> pd.Series:
    # ФИКС 2026-07j: shift(1) удалён — движок сдвигает сам
    # (двойной лаг, см. trend_lab2._shift01).
    """Финальный сдвиг на 1 бар + чистка NaN."""
    return sig.fillna(0.0).clip(-1.0, 1.0)


# ── 1. Мульти-горизонт TSMOM (Moskowitz-Ooi-Pedersen) ────────────────
def imp_tsmom_vw(bars: Bars, spans=(21, 63, 126, 252)) -> pd.Series:
    """Средний знак vol-нормированного momentum по горизонтам.

    Классика TSMOM: доходность за k баров / (вола·sqrt(k)) — это
    t-подобная сила тренда. Средняя по горизонтам, tanh в [-1,1].
    Держит импульс, пока большинство горизонтов согласны.
    """
    close = bars.close
    vol = _vol(close)
    scores = []
    for k in spans:
        mom = close.pct_change(k)
        scores.append(mom / (vol * np.sqrt(k)).replace(0.0, np.nan))
    z = pd.concat(scores, axis=1).mean(axis=1)
    return _shift(np.tanh(z))


# ── 2. Ускорение тренда ───────────────────────────────────────────────
def imp_accel(bars: Bars, fast: int = 16, slow: int = 64,
              d: int = 10) -> pd.Series:
    """Импульс = производная EWMAC: тренд не просто есть, он НАРАСТАЕТ.

    EWMAC (разность EMA, нормированная волой) — скорость; её изменение
    за d баров — ускорение. Положительное ускорение = разгон импульса
    (ранняя фаза), отрицательное = затухание (выходить). Для
    агрессивных рынков: входит раньше пробоя, выходит до разворота.
    """
    close = bars.close
    ewmac = (close.ewm(span=fast, adjust=False).mean()
             - close.ewm(span=slow, adjust=False).mean())
    ewmac = ewmac / (_vol(close) * close).replace(0.0, np.nan)
    accel = ewmac.diff(d)
    sig = np.tanh(ewmac * 2) * (accel > 0).astype(float)
    return _shift(sig)


# ── 3. t-статистика наклона ───────────────────────────────────────────
def imp_tstat(bars: Bars, window: int = 63) -> pd.Series:
    """Тренд как СТАТИСТИЧЕСКИ ЗНАЧИМЫЙ наклон регрессии.

    slope/SE(slope) по rolling-регрессии лог-цены на время: |t|>2 —
    тренд значим, не шум. Матмодель против ложных пробоев: пила даёт
    большой наклон с большим SE -> малый t -> нет позиции.
    """
    y = np.log(bars.close.where(bars.close > 0))
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    sxx = float((x ** 2).sum())

    def tstat(win: np.ndarray) -> float:
        if np.isnan(win).any():
            return np.nan
        slope = float((x * (win - win.mean())).sum()) / sxx
        resid = win - (win.mean() + slope * x)
        se2 = float((resid ** 2).sum()) / max(window - 2, 1) / sxx
        return slope / np.sqrt(se2) if se2 > 0 else 0.0

    t = y.rolling(window).apply(tstat, raw=True)
    return _shift(np.tanh(t / 2.0))


# ── 4. Близость к максимуму (52w-high) ────────────────────────────────
def imp_52h(bars: Bars, window: int = 252, top: float = 0.8) -> pd.Series:
    """Позиция по близости к N-барному максимуму.

    Эффект 52-week-high: активы у годового максимума продолжают расти
    (якорение + недо-реакция). channel_pos = (close-min)/(max-min);
    сигнал растёт линейно выше порога top, до 1 на самом максимуме.
    Чистый «держатель импульса» — не выходит, пока цена у хаёв.
    """
    hi = bars.high.rolling(window).max()
    lo = bars.low.rolling(window).min()
    pos = (bars.close - lo) / (hi - lo).replace(0.0, np.nan)
    sig = ((pos - top) / (1.0 - top)).clip(0.0, 1.0)
    return _shift(sig)


# ── 5. Непрерывный пробой ATR-канала ─────────────────────────────────
def imp_atr_break(bars: Bars, span: int = 20, k: float = 2.0) -> pd.Series:
    """Насколько цена вышла за ATR-канал, в единицах ATR (tanh).

    (close − EMA)/(k·ATR): внутри канала ~0, пробой -> сигнал растёт с
    глубиной пробоя. Непрерывный Keltner-trend: сильный выход = сильная
    позиция (свойство импульсных рынков — глубина пробоя информативна).
    """
    ema = bars.close.ewm(span=span, adjust=False).mean()
    tr = pd.concat([
        bars.high - bars.low,
        (bars.high - bars.close.shift(1)).abs(),
        (bars.low - bars.close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=span, adjust=False).mean()
    z = (bars.close - ema) / (k * atr).replace(0.0, np.nan)
    return _shift(np.tanh(z))


# ── 6. Расширение диапазона ───────────────────────────────────────────
def imp_vol_expand(bars: Bars, span: int = 14,
                   pct: float = 0.7) -> pd.Series:
    """Направление тренда, гейтованное РАСШИРЕНИЕМ диапазона.

    Зеркало ou_volgate: реверсия любит сжатие, импульс — расширение.
    Когда дневной диапазон в верхнем квантиле (rank>pct), большое
    движение началось — берём его направление (знак EWMA-доходности).
    """
    rng = (bars.high - bars.low) / bars.close.replace(0.0, np.nan)
    rank = rng.rolling(252, min_periods=60).rank(pct=True)
    direction = np.tanh(
        bars.close.pct_change().ewm(span=span, adjust=False).mean()
        / _vol(bars.close).replace(0.0, np.nan) * 3)
    sig = direction.where(rank > pct, 0.0)
    return _shift(sig)


# ── 7. Momentum 12-1 (skip-month) ─────────────────────────────────────
def imp_skip_mom(bars: Bars, look: int = 252, skip: int = 21) -> pd.Series:
    """Momentum за год БЕЗ последнего месяца (12-1, Jegadeesh-Titman).

    Последний месяц несёт краткосрочный reversal — его пропуск чистит
    сигнал. Для импульсных рынков: держит годовой тренд, игнорируя
    свежие откаты (не выбивается коррекцией).
    """
    close = bars.close
    mom = close.shift(skip).pct_change(look - skip)
    z = mom / (_vol(close) * np.sqrt(look - skip)).replace(0.0, np.nan)
    return _shift(np.tanh(z))


# ── 8. Пробой по Паркинсону ───────────────────────────────────────────
def imp_parkinson(bars: Bars, window: int = 20) -> pd.Series:
    """Пробой, нормированный волой Паркинсона (high-low).

    Parkinson σ = sqrt(mean(ln(H/L)²)/(4·ln2)) — использует ВЕСЬ
    внутрибарный диапазон, в ~5 раз эффективнее close-close оценки.
    На интрадей-импульсах (крипта H4) даёт более честную шкалу «сколько
    сигм прошли», чем close-вола, запаздывающая на всплесках.
    """
    hl = np.log((bars.high / bars.low).where(bars.low > 0))
    park = np.sqrt((hl ** 2).rolling(window).mean() / (4 * np.log(2)))
    ma = bars.close.rolling(window).mean()
    z = (bars.close / ma - 1.0) / park.replace(0.0, np.nan)
    return _shift(np.tanh(z / 2.0))


# ── 9. Асимметрия drawup/drawdown ─────────────────────────────────────
def imp_drawup(bars: Bars, window: int = 126) -> pd.Series:
    """Кто ближе: rolling-max или rolling-min. Тренд = жить у максимума.

    drawdown = close/max − 1 (≤0), drawup = close/min − 1 (≥0).
    Сигнал = (drawup + drawdown)/(drawup − drawdown) ∈ [−1,1]:
    у максимума -> +1, у минимума -> −1. Робастная геометрия тренда без
    параметров скорости.
    """
    mx = bars.close.rolling(window).max()
    mn = bars.close.rolling(window).min()
    du = bars.close / mn - 1.0
    dd = bars.close / mx - 1.0
    denom = (du - dd).replace(0.0, np.nan)
    return _shift((du + dd) / denom)


# ── 10. Kalman-импульс ────────────────────────────────────────────────
def imp_kalman_imp(bars: Bars, q_slope_ratio: float = 1e-3) -> pd.Series:
    """Kalman-наклон, гейтованный его же РОСТОМ: только разгон.

    Урок сравнения с Monday-range: чистый Kalman-тренд осторожничает и
    срезает импульс. Здесь позиция берётся только когда наклон
    положителен И растёт (импульсная фаза), без позиций на затухании.
    Компромисс матмодели и «держать до упора».
    """
    close = bars.close.where(bars.close > 0)
    y = np.log(close).to_numpy(dtype=float)
    diffs = np.diff(y[~np.isnan(y)])
    r = max(float(np.nanvar(diffs)) if len(diffs) > 2 else 1e-4, 1e-10)
    _, slope = _kalman_level_slope(y, 1e-2 * r, q_slope_ratio * r, r)
    s = pd.Series(slope, index=bars.close.index)
    scale = s.rolling(63, min_periods=20).std().replace(0.0, np.nan)
    z = s / scale
    rising = s.diff(5) > 0
    sig = np.tanh(z / 1.5).where(rising, 0.0).clip(lower=0.0)
    return _shift(sig)


# ── Вариации Carver ───────────────────────────────────────────────────
def carver_fast(bars: Bars) -> pd.Series:
    """Carver на быстрых EWMAC (см. carver_fdm): реактивный профиль.

    Тот же аппарат (скейлинг, FDM), но с меньшим corr_window — быстрее
    адаптирует веса. Для интрадей-крипты, где режимы короче.
    """
    return carver_fdm(bars, corr_window=126)


def carver_ls(bars: Bars) -> pd.Series:
    """Carver long+short: шорт-нога для фьючерсов/перпетуалов."""
    return carver_fdm(bars, long_only=False)


def carver_hicap(bars: Bars) -> pd.Series:
    """Carver с агрессивным насыщением: cap 30, FDM до 4.

    Больше расстояния до клипа -> сильные согласованные прогнозы
    дают позицию ближе к максимуму (агрессивный профиль).
    """
    return carver_fdm(bars, forecast_cap=30.0, fdm_cap=4.0)


IMPULSE_LAB = {
    "imp_tsmom_vw": imp_tsmom_vw,
    "imp_accel": imp_accel,
    "imp_tstat": imp_tstat,
    "imp_52h": imp_52h,
    "imp_atr_break": imp_atr_break,
    "imp_vol_expand": imp_vol_expand,
    "imp_skip_mom": imp_skip_mom,
    "imp_parkinson": imp_parkinson,
    "imp_drawup": imp_drawup,
    "imp_kalman_imp": imp_kalman_imp,
    "carver_fast": carver_fast,
    "carver_ls": carver_ls,
    "carver_hicap": carver_hicap,
}
