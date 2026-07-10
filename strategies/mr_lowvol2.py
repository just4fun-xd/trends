"""Доработка mr_lowvol (Roadmap, запрос 2026-07k): 5 вариантов чемпиона.

mr_lowvol UNDEFEATED на сырьевой MR-нише: bb_rsi-ядро + expanding-
перцентильный гейт спокойной волы. Дорабатываем НЕ переоткрывая:
каждый вариант меняет РОВНО ОДИН узел чемпиона (контролируемый
эксперимент — если вариант выигрывает bootstrap, известно ПОЧЕМУ).

Узлы и гипотезы:
  РАЗМЕР   mr_lv2_cont   — бинарный 0/1 -> непрерывный от глубины z
                           (план «прибыль без плеча»: время в рынке).
  НАБОР    mr_lv2_scale  — одна ступень -> две (0.6 @ z<-2, +0.4 @
                           z<-3): добор в глубину эпизода.
  ГЕЙТ     mr_lv2_garch  — realized-вола в гейте -> GARCH-прогноз
                           (урок GARCH-трека: как VT проиграл
                           rolling-30, но гейт - другой узел: тут
                           важна СКОРОСТЬ обнаружения бури, one-shot).
  ВЫХОД    mr_lv2_zexit  — RSI 50 -> касание средней (z >= 0):
                           RSI 50 срабатывает раньше средней,
                           недобирает ход.
  ГЕЙТ x2  mr_lv2_vr     — calm-гейт И variance ratio < 1: реверсия
                           разрешена, только когда процесс И спокоен,
                           И антиперсистентен (двойное подтверждение
                           режима, меньше сделок — чище сделки).

╔════════════════════════════════════════════════════════════════╗
║ Арбитраж: bootstrap КАЖДОГО варианта против mr_lowvol на двух  ║
║ источниках. Выиграл один узел -> собирается v2 из выигравших   ║
║ узлов и СНОВА bootstrap против чемпиона (не молча).            ║
╚════════════════════════════════════════════════════════════════╝

Контракт: Bars -> position [0, 1]. Сдвига внутри НЕТ. VT снаружи.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.bollinger import _rsi, bollinger_rsi
from strategies.meanrev_lab import _zscore


def _calm(bars: Bars, vol_lookback: int = 20,
          vol_pct: float = 0.7) -> pd.Series:
    """Expanding-перцентильный гейт спокойной волы (узел чемпиона)."""
    vol = bars.returns().rolling(vol_lookback).std()
    thresh = vol.expanding(min_periods=vol_lookback * 3).quantile(vol_pct)
    return (vol < thresh).fillna(False)


# ── РАЗМЕР: непрерывная позиция ──────────────────────────────────────
def mr_lv2_cont(
    bars: Bars, z_win: int = 20, z_start: float = 1.0,
    z_full: float = 3.0,
) -> pd.Series:
    """Позиция растёт линейно от z=-1 (0) до z=-3 (1) в calm-режиме.

    Меняется ТОЛЬКО размер: бинарное «вошёл по касанию полосы» ->
    непрерывное «сколько растянуто, столько взял». Мелкие растяжения
    дают частичную позицию (больше времени в рынке), глубокие —
    полную раньше, чем RSI 30 подтвердит.
    """
    z = _zscore(bars.close, z_win)
    depth = ((-z - z_start) / (z_full - z_start)).clip(0.0, 1.0)
    return (depth * _calm(bars).astype(float)).fillna(0.0)


# ── НАБОР: две ступени ───────────────────────────────────────────────
def mr_lv2_scale(
    bars: Bars, z_win: int = 20, z1: float = -2.0, z2: float = -3.0,
    s1: float = 0.6, s2: float = 0.4, rsi_period: int = 14,
    rsi_exit: float = 50.0,
) -> pd.Series:
    """Ступень 0.6 при z<-2, добор 0.4 при z<-3; выход как у чемпиона.

    Меняется ТОЛЬКО набор: чемпион берёт 100% на первом касании и
    глубже усредняться не умеет; здесь резерв 40% на случай второй
    сигмы вниз (средняя цена входа глубже -> больше ход к RSI 50).
    Обе ступени только в calm-режиме, сумма жёстко <= 1.0.
    """
    z = _zscore(bars.close, z_win).to_numpy()
    rsi = _rsi(bars.close, rsi_period).to_numpy()
    calm = _calm(bars).to_numpy()
    n = len(z)
    pos = np.zeros(n)
    step = 0
    for i in range(n):
        if np.isnan(z[i]) or np.isnan(rsi[i]):
            pos[i] = pos[i - 1] if i else 0.0
            continue
        if step > 0 and rsi[i] > rsi_exit:
            step = 0
        if calm[i]:
            if step == 0 and z[i] < z1:
                step = 1
            elif step == 1 and z[i] < z2:
                step = 2
        pos[i] = (0.0, s1, s1 + s2)[step]
    return pd.Series(pos, index=bars.index)


# ── ГЕЙТ: GARCH-прогноз вместо realized ──────────────────────────────
def mr_lv2_garch(
    bars: Bars, bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_buy: float = 30.0,
    rsi_exit: float = 50.0, vol_pct: float = 0.7,
) -> pd.Series:
    """Чемпион с GARCH-прогнозом волы в гейте вместо rolling-std.

    Меняется ТОЛЬКО источник волы в гейте: rolling-20 видит бурю
    через дни, GARCH(1,1) вносит вчерашний шок в сигму немедленно
    (alpha). Гипотеза: гейт закроет вход на 2-5 баров раньше в
    начале кризиса. Помним: для VT GARCH проиграл rolling-30 (рефит
    держал плечо в CL-2020) — но гейт бинарен и узел другой;
    one-shot покажет.
    """
    from core.garch import garch_vol_forecast
    base = bollinger_rsi(bars, bb_period, bb_std, rsi_period,
                         rsi_buy, rsi_exit)
    sigma = garch_vol_forecast(bars.close.pct_change())
    thresh = sigma.expanding(min_periods=60).quantile(vol_pct)
    calm = (sigma < thresh).fillna(False).to_numpy()
    b = base.to_numpy()
    pos = np.zeros(len(b))
    in_pos = False
    for i in range(len(b)):
        if not in_pos and b[i] > 0 and calm[i]:
            in_pos = True
        elif in_pos and b[i] == 0:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


# ── ВЫХОД: касание средней ───────────────────────────────────────────
def mr_lv2_zexit(
    bars: Bars, bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_buy: float = 30.0,
) -> pd.Series:
    """Чемпион с выходом по z>=0 (средняя) вместо RSI 50.

    Меняется ТОЛЬКО выход: RSI 50 достигается раньше касания
    средней (импульс нормализуется быстрее уровня) — чемпион
    систематически отдаёт последний отрезок хода. Вход идентичен
    чемпиону (полоса + RSI 30 + calm).
    """
    z = _zscore(bars.close, bb_period)
    rsi = _rsi(bars.close, rsi_period)
    calm = _calm(bars)
    enter = ((z < -bb_std) & (rsi < rsi_buy) & calm).fillna(False)
    leave = (z >= 0.0).fillna(False)
    ent, lev = enter.to_numpy(), leave.to_numpy()
    pos = np.zeros(len(ent))
    in_pos = False
    for i in range(len(ent)):
        if in_pos and lev[i]:
            in_pos = False
        if not in_pos and ent[i] and not lev[i]:
            in_pos = True
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


# ── ГЕЙТ x2: calm + VR ───────────────────────────────────────────────
def mr_lv2_vr(
    bars: Bars, bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_buy: float = 30.0,
    rsi_exit: float = 50.0, vr_window: int = 250, vr_q: int = 5,
    vr_max: float = 1.0,
) -> pd.Series:
    """Чемпион + второй гейт: variance ratio < 1 (Lo-MacKinlay).

    Меняется ТОЛЬКО добавлением гейта: calm-вола говорит «шторма
    нет», VR(q)<1 говорит «процесс антиперсистентен» — это разные
    утверждения (бывает тихий дрейф с VR>1: реверсию там ловить
    нечем, чемпион входит и ждёт зря). Меньше входов — выше доля
    настоящих эпизодов.
    """
    r = bars.close.pct_change()
    var1 = r.rolling(vr_window).var()
    rq = bars.close.pct_change(vr_q)
    varq = rq.rolling(vr_window).var()
    vr = varq / (vr_q * var1.replace(0.0, np.nan))
    z = _zscore(bars.close, bb_period)
    rsi = _rsi(bars.close, rsi_period)
    gate = (_calm(bars) & (vr < vr_max)).fillna(False)
    enter = ((z < -bb_std) & (rsi < rsi_buy) & gate).fillna(False)
    leave = (rsi > rsi_exit).fillna(False)
    ent, lev = enter.to_numpy(), leave.to_numpy()
    pos = np.zeros(len(ent))
    in_pos = False
    for i in range(len(ent)):
        if in_pos and lev[i]:
            in_pos = False
        if not in_pos and ent[i] and not lev[i]:
            in_pos = True
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


MR_LOWVOL2 = {
    "mr_lv2_cont": mr_lv2_cont,
    "mr_lv2_scale": mr_lv2_scale,
    "mr_lv2_garch": mr_lv2_garch,
    "mr_lv2_zexit": mr_lv2_zexit,
    "mr_lv2_vr": mr_lv2_vr,
}
