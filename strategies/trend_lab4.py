"""Trend-лаборатория 4: 10 трендовых моделей на НЕЗАНЯТОМ матаппарате.

Запрос 2026-07k. Дисциплина прежняя: каждый кандидат ЛОВИТ тренд
собственным аппаратом (урок Trend Lab 1: фильтры качества дают ~0
на ровном тренде), ни одного тейк-профита (режет правый хвост),
механизм обязан отличаться от 40+ уже зарегистрированных трендовых
моделей (иначе corr ~1 и в ансамбль не попадает).

Проверено на дубли против trend_lab/2/3, impulse_lab, kalman_trend:
- t-stat наклона занят (imp_tstat), 12-1 momentum занят (imp_skip_mom),
  drawup-геометрия занята (imp_drawup), ER/Hull/VHF закрыты, свинги
  заняты (tr3_hh_hl), ленты заняты (tr3_ribbon).

Здесь ТОЛЬКО новые аппараты: стоп-геометрия с ускорением (PSAR),
событийное время (Renko), ранговая статистика (Mann-Kendall),
эконометрика персистентности (AR(1)>0 — зеркало mr2_halflife),
теория управления (гистерезис Шмитта), DSP (декиклер Элерса),
микроструктура сессии (overnight-дрейф), последовательное обнаружение
разладки (CUSUM Пейджа В ТРЕНД, зеркало mr2_cusum), теория решений
(дробный Келли как непрерывный прогноз), непараметрическая
самокалибровка (перцентиль-ранг momentum).

╔════════════════════════════════════════════════════════════════╗
║ MULTIPLE TESTING: 10 моделей = 1-2 случайных «победителя».     ║
║ Отбор: walk-forward -> bootstrap против donchian_vt -> второй  ║
║ источник -> corr < 0.6 для кандидата в ансамбль. One-shot.     ║
╚════════════════════════════════════════════════════════════════╝

Контракт: Bars -> position. Long-only [0, 1] (сырьё/акции).
Сдвига внутри НЕТ — движок сдвигает. VT снаружи (--vt).

Модели (аппарат в скобках):
 1. tr4_psar      — Parabolic SAR (стоп с фактором ускорения, Wilder).
 2. tr4_renko     — направление Renko-кирпичей (событийное время).
 3. tr4_mk        — rolling Mann-Kendall (ранговый тест тренда).
 4. tr4_ar1       — гейт AR(1)>0 на доходностях (персистентность).
 5. tr4_schmitt   — триггер Шмитта на норм. momentum (гистерезис).
 6. tr4_decycler  — декиклер Элерса: цена над трендом без цикла (DSP).
 7. tr4_overnight — знак overnight-дрейфа (микроструктура сессии).
 8. tr4_page      — CUSUM Пейджа на включение тренда (разладка).
 9. tr4_kelly     — дробный Келли mu/sigma^2 как позиция (теория решений).
10. tr4_mom_pct   — trailing-перцентиль momentum (непараметрика).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars


def _bar_vol(close: pd.Series, span: int = 30) -> pd.Series:
    """EWMA-волатильность побарных доходностей."""
    return close.pct_change().ewm(span=span, adjust=False).std()


# ── 1. Parabolic SAR ─────────────────────────────────────────────────
def tr4_psar(
    bars: Bars, af_start: float = 0.02, af_step: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """Parabolic SAR Уайлдера: лонг, пока цена выше SAR-стопа.

    Механизм: стоп-точка ускоряется К цене с каждым новым экстремумом
    (фактор ускорения растёт 0.02 -> 0.2). В молодом тренде стоп
    далёк (даёт дышать), в зрелом — прижимается (фиксирует).
    Отличие от SuperTrend/Chandelier: адаптация по ВРЕМЕНИ И
    ПРОГРЕССУ тренда (ускорение), а не по воле (ATR).
    Long-only: короткая фаза SAR = кэш.
    """
    hi = bars.high.to_numpy()
    lo = bars.low.to_numpy()
    n = len(hi)
    pos = np.zeros(n)
    if n < 3:
        return pd.Series(pos, index=bars.index)
    up = True
    sar = lo[0]
    ep = hi[0]
    af = af_start
    for i in range(1, n):
        if np.isnan(hi[i]) or np.isnan(lo[i]):
            continue
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, lo[i - 1], lo[max(i - 2, 0)])
            if lo[i] < sar:            # пробой стопа -> разворот вниз
                up, sar, ep, af = False, ep, lo[i], af_start
            elif hi[i] > ep:
                ep, af = hi[i], min(af + af_step, af_max)
        else:
            sar = max(sar, hi[i - 1], hi[max(i - 2, 0)])
            if hi[i] > sar:            # разворот вверх
                up, sar, ep, af = True, ep, hi[i], af_start
            elif lo[i] < ep:
                ep, af = lo[i], min(af + af_step, af_max)
        pos[i] = 1.0 if up else 0.0
    return pd.Series(pos, index=bars.index)


# ── 2. Renko ─────────────────────────────────────────────────────────
def tr4_renko(
    bars: Bars, brick_atr: float = 1.5, atr_win: int = 50,
    confirm: int = 2,
) -> pd.Series:
    """Направление последних Renko-кирпичей (событийное время).

    Механизм: Renko перерисовывает ряд в кирпичи фиксированного
    размера (тут 1.5*ATR на момент закладки) — время исчезает,
    остаётся ЧИСТОЕ движение. Пила меньше кирпича не существует
    вовсе; тренд = серия однонаправленных кирпичей. Лонг после
    confirm подряд восходящих, кэш после первого нисходящего.
    Аппарат событийного времени в реестре не представлен.
    """
    c = bars.close.to_numpy()
    atr = bars.atr(atr_win).to_numpy()
    n = len(c)
    pos = np.zeros(n)
    anchor = np.nan          # уровень последнего кирпича
    streak = 0               # +k подряд вверх / -k подряд вниз
    for i in range(n):
        if np.isnan(c[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            pos[i] = pos[i - 1] if i else 0.0
            continue
        if np.isnan(anchor):
            anchor = c[i]
        brick = brick_atr * atr[i]
        while c[i] >= anchor + brick:
            anchor += brick
            streak = streak + 1 if streak >= 0 else 1
        while c[i] <= anchor - brick:
            anchor -= brick
            streak = streak - 1 if streak <= 0 else -1
        pos[i] = 1.0 if streak >= confirm else 0.0
    return pd.Series(pos, index=bars.index)


# ── 3. Mann-Kendall ──────────────────────────────────────────────────
def tr4_mk(
    bars: Bars, window: int = 42, z_min: float = 1.65,
) -> pd.Series:
    """Rolling-тест Манна-Кендалла: значимость монотонного роста.

    Механизм: S = sum sign(x_j - x_i) по всем парам i<j окна —
    ранговая статистика, не видит величин, только порядок. Устойчив
    к выбросам и не предполагает нормальности (в отличие от
    imp_tstat, где регрессия параметрична). Z = S/sqrt(Var(S));
    лонг при Z > z_min (тренд значим на ~5%).
    """
    x = bars.close.to_numpy()
    n = len(x)
    w = window
    var_s = w * (w - 1) * (2 * w + 5) / 18.0
    denom = np.sqrt(var_s)
    zs = np.full(n, np.nan)
    iu, ju = np.triu_indices(w, k=1)
    for t in range(w - 1, n):
        win = x[t - w + 1:t + 1]
        if np.isnan(win).any():
            continue
        s = np.sign(win[ju] - win[iu]).sum()
        zs[t] = s / denom
    z = pd.Series(zs, index=bars.index)
    return (z > z_min).astype(float).fillna(0.0)


# ── 4. AR(1)-персистентность ─────────────────────────────────────────
def tr4_ar1(
    bars: Bars, corr_win: int = 250, mom_win: int = 63,
    phi_min: float = 0.0,
) -> pd.Series:
    """Гейт положительной автокорреляции доходностей + знак momentum.

    Механизм: зеркало mr2_halflife. Там AR(1)<0 разрешал реверсию;
    тут rolling-phi = corr(r_t, r_{t-1}) > phi_min диагностирует
    персистентность (моментум-режим у самого инструмента), и лишь
    тогда торгуется знак momentum. Эконометрический гейт вместо
    ценового: тренд подтверждается свойством ПРОЦЕССА, не фигурой.
    """
    r = bars.close.pct_change()
    phi = r.rolling(corr_win).corr(r.shift(1))
    mom = bars.close.pct_change(mom_win)
    sig = (phi > phi_min) & (mom > 0)
    return sig.astype(float).fillna(0.0)


# ── 5. Триггер Шмитта ────────────────────────────────────────────────
def tr4_schmitt(
    bars: Bars, mom_win: int = 63, vol_span: int = 30,
    hi: float = 0.75, lo: float = -0.25,
) -> pd.Series:
    """Гистерезис: вход при m > hi, выход только при m < lo.

    Механизм: триггер Шмитта из теории управления. Одиночный порог
    дребезжит, когда сигнал ходит вокруг него (пила сделок у нуля
    momentum); два разнесённых порога создают мёртвую зону — выход
    требует СМЕНЫ знака с запасом, а не касания нуля. Аппарат —
    состояние с памятью, а не мгновенное значение индикатора.
    m = momentum в единицах волы горизонта.
    """
    vol = _bar_vol(bars.close, vol_span)
    m = bars.close.pct_change(mom_win) / (
        vol * np.sqrt(mom_win)).replace(0.0, np.nan)
    mv = m.to_numpy()
    n = len(mv)
    pos = np.zeros(n)
    in_pos = False
    for i in range(n):
        if np.isnan(mv[i]):
            pos[i] = 0.0
            in_pos = False
            continue
        if not in_pos and mv[i] > hi:
            in_pos = True
        elif in_pos and mv[i] < lo:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


# ── 6. Декиклер Элерса ───────────────────────────────────────────────
def tr4_decycler(
    bars: Bars, cutoff: int = 60, slope_win: int = 5,
) -> pd.Series:
    """Декиклер Элерса: цена над бесцикловым трендом, тренд растёт.

    Механизм: DSP-аппарат. High-pass фильтр 2-го порядка выделяет
    циклическую компоненту короче cutoff; декиклер = цена - цикл,
    т.е. тренд, очищенный от колебаний БЕЗ лага сглаживания (у SMA
    лаг ~window/2, у декиклера near-zero в полосе пропускания).
    Лонг: close > decycler И decycler выше, чем slope_win назад.
    """
    p = bars.close.to_numpy(dtype=float)
    n = len(p)
    alpha = (np.cos(np.sqrt(2.0) * np.pi / cutoff)
             + np.sin(np.sqrt(2.0) * np.pi / cutoff) - 1.0) / np.cos(
                 np.sqrt(2.0) * np.pi / cutoff)
    hp = np.zeros(n)
    a1 = (1 - alpha / 2.0) ** 2
    b1 = 2 * (1 - alpha)
    b2 = (1 - alpha) ** 2
    for i in range(2, n):
        if np.isnan(p[i]) or np.isnan(p[i - 1]) or np.isnan(p[i - 2]):
            hp[i] = 0.0
            continue
        hp[i] = (a1 * (p[i] - 2 * p[i - 1] + p[i - 2])
                 + b1 * hp[i - 1] - b2 * hp[i - 2])
    dec = pd.Series(p - hp, index=bars.index)
    sig = (bars.close > dec) & (dec > dec.shift(slope_win))
    return sig.astype(float).fillna(0.0)


# ── 7. Overnight-дрейф ───────────────────────────────────────────────
def tr4_overnight(
    bars: Bars, window: int = 126, z_min: float = 0.3,
) -> pd.Series:
    """Знак накопленного overnight-дрейфа (open против прошлого close).

    Механизм: микроструктура. Побарная доходность распадается на
    overnight (open/close_{t-1} - 1: новости, ребалансы, чужие
    сессии) и intraday. Литература (Lou-Polk-Skouras): overnight
    несёт непропорциональную долю дрейфа. Лонг, когда rolling-сумма
    overnight-компоненты значимо положительна (в z волы компоненты).
    Единственная модель реестра, читающая bars.open.
    """
    on = bars.open / bars.close.shift(1) - 1.0
    mu = on.rolling(window).sum()
    sd = on.rolling(window).std() * np.sqrt(window)
    z = mu / sd.replace(0.0, np.nan)
    return (z > z_min).astype(float).fillna(0.0)


# ── 8. CUSUM Пейджа на включение тренда ──────────────────────────────
def tr4_page(
    bars: Bars, vol_span: int = 30, k: float = 0.15, h: float = 4.0,
) -> pd.Series:
    """Последовательное обнаружение разладки ВВЕРХ (Page, 1954).

    Механизм: g+ = max(0, g+ + r/sigma - k) копит стандартизованный
    дрейф сверх допуска k; g+ > h — «разладка» (режим сменился на
    восходящий) -> лонг. Симметричный g- > h выключает. Зеркало
    mr2_cusum: там накопленное отклонение ГАСИЛИ (реверсия), здесь
    обнаруженный сдвиг уровня СОПРОВОЖДАЕМ. SPC-аппарат гарантирует
    минимальное среднее время до обнаружения при данной частоте
    ложных тревог — то, что пороговым MA-кроссам не обещано.
    """
    r = bars.close.pct_change()
    z = (r / _bar_vol(bars.close, vol_span).replace(0.0, np.nan)
         ).to_numpy()
    n = len(z)
    pos = np.zeros(n)
    gp = gm = 0.0
    in_pos = False
    for i in range(n):
        if np.isnan(z[i]):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        gp = max(0.0, gp + z[i] - k)
        gm = max(0.0, gm - z[i] - k)
        if not in_pos and gp > h:
            in_pos, gm = True, 0.0
        elif in_pos and gm > h:
            in_pos, gp = False, 0.0
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


# ── 9. Дробный Келли ─────────────────────────────────────────────────
def tr4_kelly(
    bars: Bars, window: int = 250, frac: float = 0.2,
    cap: float = 1.0,
) -> pd.Series:
    """Позиция = дробный Келли f* = mu/sigma^2 (теория решений).

    Механизм: не «есть тренд / нет тренда», а СКОЛЬКО брать:
    f*, максимизирующий логарифмический рост при оценённых
    (mu, sigma^2). Непрерывный прогноз (план «прибыль без плеча»,
    п.2 — больше времени в рынке у сильных режимов), дробность
    frac страхует ошибку оценки mu (полный Келли самоубийствен
    при шумном mu). clip в [0, cap] — long-only мандат.
    """
    r = bars.close.pct_change()
    mu = r.rolling(window).mean()
    var = r.rolling(window).var()
    f = frac * mu / var.replace(0.0, np.nan)
    return f.clip(lower=0.0, upper=cap).fillna(0.0)


# ── 10. Перцентиль-ранг momentum ─────────────────────────────────────
def tr4_mom_pct(
    bars: Bars, mom_win: int = 126, rank_window: int = 750,
    pctl: float = 0.7,
) -> pd.Series:
    """Momentum против СОБСТВЕННОЙ trailing-истории (непараметрика).

    Механизм: та же логика, что у принятого vol_percentile_gate,
    но на дрейфе: 126-барная доходность ранжируется в собственном
    trailing-распределении; лонг в верхних (1-pctl) состояниях.
    Самокалибровка: никаких порогов в единицах доходности — у CL
    и у SPY «сильный тренд» свой, ранг делает их сравнимыми.
    """
    mom = bars.close.pct_change(mom_win)
    rank = mom.rolling(rank_window, min_periods=rank_window // 3).rank(
        pct=True)
    return (rank > pctl).astype(float).fillna(0.0)


TREND_LAB4 = {
    "tr4_psar": tr4_psar,
    "tr4_renko": tr4_renko,
    "tr4_mk": tr4_mk,
    "tr4_ar1": tr4_ar1,
    "tr4_schmitt": tr4_schmitt,
    "tr4_decycler": tr4_decycler,
    "tr4_overnight": tr4_overnight,
    "tr4_page": tr4_page,
    "tr4_kelly": tr4_kelly,
    "tr4_mom_pct": tr4_mom_pct,
}
