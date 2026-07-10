"""OU×Trend лаборатория — реверсия, работающая ВМЕСТЕ с трендом.

Кирилл прав, что раздел OU не может быть бесполезен целиком. Диагноз
всех провалов OU одинаков: он ставил ПРОТИВ тренда (ловил ножи,
шортил ракеты). Здесь OU перевёрнут: тренд задаёт НАПРАВЛЕНИЕ,
OU-растяжение задаёт ТОЧКУ ВХОДА. Классика институционалов:
«buy the dip in an uptrend» — реверсия к движущемуся среднему тренда,
а не к плоскому mean.

Использован весь опыт сессий:
  - jump-детектор (ou_lab): не покупаем нож, если это скачок уровня;
  - Kalman level/slope (kalman_trend): движущийся центр реверсии;
  - VR-Hurst (variance_ratio): значимость режима, а не эвристика;
  - урок ou_asym: long-only сам по себе — бета; здесь направление
    даёт ТРЕНД-ФИЛЬТР, а не постоянный лонг.

Дисциплина multiple testing: walk-forward скрининг -> bootstrap
выживших против чемпионов (mr_lowvol / donchian_vt) -> 2-й источник.

Гибриды:
 1. ou_pullback  — тренд-гейт (EMA50>200) + покупка z-провала.
 2. ou_trendline — реверсия к Kalman-ТРЕНДУ (движущийся центр).
 3. ou_residual  — OU на остатках регрессии, направление по наклону.
 4. ou_ride      — MR-вход, ТРЕНД-выход («купи провал, езжай на тренде»).
 5. ou_gap_fade  — откуп контр-трендовых гэпов, кроме скачков уровня.
 6. ou_router    — H-роутер со значимостью: тренд-нога или OU-нога.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.donchian import donchian_champion_raw
from strategies.kalman_trend import _kalman_level_slope
from strategies.ou_lab import _detect_jumps, _z_classic
from strategies.variance_ratio import rolling_hurst_vr


def _uptrend(close: pd.Series, fast: int = 50,
             slow: int = 200) -> pd.Series:
    """Булев тренд-гейт: EMA(fast) выше EMA(slow)."""
    return (close.ewm(span=fast, adjust=False).mean()
            > close.ewm(span=slow, adjust=False).mean())


# ── 1. Pullback в аптренде ────────────────────────────────────────────
def ou_pullback(bars: Bars, window: int = 20, entry: float = 1.5,
                exit_: float = 0.0) -> pd.Series:
    """Покупка OU-провала ТОЛЬКО в аптренде; выход к центру или по тренду.

    Механизм: в аптренде откаты к среднему — это пополнение позиции
    сильных рук, реверсия работает; против тренда тот же |z| — нож.
    Отличие от ou_asym (закрыт как бета): направление даёт ЖИВОЙ
    тренд-фильтр, вне аптренда позиции нет вообще.
    """
    close = bars.close
    z = _z_classic(close, window)
    up = _uptrend(close)
    zv, upv = z.to_numpy(), up.to_numpy()
    pos = np.zeros(len(zv))
    state = 0
    for i in range(len(zv)):
        if not upv[i]:
            state = 0            # тренд умер — позиции нет
        elif state == 0 and zv[i] < -entry:
            state = 1            # провал в аптренде — покупка
        elif state == 1 and zv[i] > exit_:
            state = 0            # вернулись к центру — выход
        pos[i] = float(state)
    return pd.Series(pos, index=close.index).fillna(0.0)


# ── 2. Реверсия к Kalman-тренду ──────────────────────────────────────
def ou_trendline(bars: Bars, entry: float = 1.5,
                 q_slope_ratio: float = 1e-3) -> pd.Series:
    """OU к ДВИЖУЩЕМУСЯ центру: Kalman-уровень вместо плоского mean.

    Главная ошибка плоского OU — mean устаревает в тренде, и «возврат»
    means откат против движения. Здесь центр = Kalman local linear
    trend level (уровень, живущий вместе с трендом); растяжение
    меряется от НЕГО. Покупаем провалы ниже тренда при slope>0.
    """
    close = bars.close.where(bars.close > 0)
    y = np.log(close).to_numpy(dtype=float)
    diffs = np.diff(y[~np.isnan(y)])
    r = max(float(np.nanvar(diffs)) if len(diffs) > 2 else 1e-4, 1e-10)
    level, slope = _kalman_level_slope(y, 1e-2 * r, q_slope_ratio * r, r)
    resid = pd.Series(y - level, index=bars.close.index)
    z = resid / resid.rolling(63, min_periods=20).std().replace(0, np.nan)
    slope_s = pd.Series(slope, index=bars.close.index)
    sig = ((z < -entry) & (slope_s > 0)).astype(float)
    # Держим до возврата растяжения к нулю.
    out = np.zeros(len(sig))
    state = 0
    zv, sv = z.to_numpy(), slope_s.to_numpy()
    for i in range(len(out)):
        if state == 0 and zv[i] < -entry and sv[i] > 0:
            state = 1
        elif state == 1 and (zv[i] > 0 or sv[i] <= 0):
            state = 0
        out[i] = float(state)
    return pd.Series(out, index=bars.close.index).fillna(0.0)


# ── 3. OU на остатках регрессии ──────────────────────────────────────
def ou_residual(bars: Bars, window: int = 126,
                entry: float = 1.5) -> pd.Series:
    """Детренд rolling-регрессией; OU на остатках, направление по наклону.

    Разделение ролей: наклон регрессии = тренд (кто главный), остаток =
    краткосрочное растяжение (когда входить). В ап-наклоне покупаем
    провалы остатка; в даун-наклоне шортим выбросы. Симметричный
    гибрид для фьючерсов.
    """
    y = np.log(bars.close.where(bars.close > 0))
    x = np.arange(window, dtype=float)
    x -= x.mean()
    sxx = float((x ** 2).sum())

    def _fit(win: np.ndarray):
        if np.isnan(win).any():
            return np.nan, np.nan
        b = float((x * (win - win.mean())).sum()) / sxx
        resid_last = win[-1] - (win.mean() + b * x[-1])
        return b, resid_last

    slopes = np.full(len(y), np.nan)
    resids = np.full(len(y), np.nan)
    arr = y.to_numpy()
    for i in range(window, len(arr)):
        b, rl = _fit(arr[i - window:i + 1][-window:])
        slopes[i], resids[i] = b, rl
    resid_s = pd.Series(resids, index=y.index)
    z = resid_s / resid_s.rolling(window, min_periods=30).std()
    slope_s = pd.Series(slopes, index=y.index)
    long_sig = (z < -entry) & (slope_s > 0)
    short_sig = (z > entry) & (slope_s < 0)
    pos = pd.Series(0.0, index=y.index)
    pos[long_sig] = 1.0
    pos[short_sig] = -1.0
    return pos.fillna(0.0)


# ── 4. MR-вход, тренд-выход ──────────────────────────────────────────
def ou_ride(bars: Bars, window: int = 20, entry: float = 2.0,
            fast: int = 20, slow: int = 60) -> pd.Series:
    """Вход по растяжению, выход по СМЕРТИ ТРЕНДА (не по возврату z).

    Асимметрия, которой не было ни в одном OU: реверсия даёт дешёвую
    точку входа, но прибыль отдаёт тренд — держим, пока EMA(fast) >
    EMA(slow), игнорируя возврат z к нулю. «Купи провал, езжай на
    тренде» — конверсия MR-сигнала в трендовое удержание. Ответ на
    урок Monday-range: не срезать импульс ранним выходом.
    """
    close = bars.close
    z = _z_classic(close, window).to_numpy()
    up = (close.ewm(span=fast, adjust=False).mean()
          > close.ewm(span=slow, adjust=False).mean()).to_numpy()
    pos = np.zeros(len(z))
    state = 0
    for i in range(len(z)):
        if state == 0 and z[i] < -entry and up[i]:
            state = 1
        elif state == 1 and not up[i]:
            state = 0            # выходим по тренду, не по z
        pos[i] = float(state)
    return pd.Series(pos, index=close.index).fillna(0.0)


# ── 5. Откуп контр-трендовых гэпов ───────────────────────────────────
def ou_gap_fade(bars: Bars, k: float = 2.0, hold: int = 5,
                jump_k: float = 4.0) -> pd.Series:
    """Откуп резкого провала ПО тренду, если это не скачок уровня.

    Провал < −k·σ в аптренде — перепроданность, откупаем на hold баров.
    Но если jump-детектор (|r|>jump_k·σ, MAD-шкала) говорит «скачок» —
    это смена уровня, не растяжение: не входим (урок ou_jump).
    Разделение «нож vs провал» формальным порогом.
    """
    close = bars.close.where(bars.close > 0)
    ret = np.log(close / close.shift(1))
    sigma = ret.rolling(40).std()
    dip = ret < (-k * sigma)
    jump = _detect_jumps(bars.close, window=40, k=jump_k)
    up = _uptrend(bars.close)
    trigger = (dip & up & ~jump).to_numpy()
    pos = np.zeros(len(trigger))
    left = 0
    for i in range(len(trigger)):
        if trigger[i]:
            left = hold
        if left > 0:
            pos[i] = 1.0
            left -= 1
    return pd.Series(pos, index=bars.close.index).fillna(0.0)


# ── 6. H-роутер со значимостью ───────────────────────────────────────
def ou_router(bars: Bars, h_hi: float = 0.55,
              h_lo: float = 0.45, window: int = 20) -> pd.Series:
    """Роутер: значимый тренд -> donchian; значимая реверсия -> OU-fade.

    Отличие от hurst_alloc (плавная смесь): жёсткая маршрутизация c
    зоной невмешательства. H>h_hi -> вся позиция тренд-ноге; H<h_lo ->
    OU-fade (long-only); между — флэт (не ставим на неопределённость).
    Инструмент сам выбирает своё оружие — механизация карты режимов.
    """
    h = rolling_hurst_vr(bars.close, window=504)
    trend = donchian_champion_raw(bars).fillna(0.0)
    z = _z_classic(bars.close, window)
    fade = (z < -2.0).astype(float)  # простой OU-fade long-only
    pos = pd.Series(0.0, index=bars.close.index)
    pos[h > h_hi] = trend[h > h_hi]
    pos[h < h_lo] = fade[h < h_lo]
    return pos.fillna(0.0)


OU_TREND_LAB = {
    "ou_pullback": ou_pullback,
    "ou_trendline": ou_trendline,
    "ou_residual": ou_residual,
    "ou_ride": ou_ride,
    "ou_gap_fade": ou_gap_fade,
    "ou_router": ou_router,
}
