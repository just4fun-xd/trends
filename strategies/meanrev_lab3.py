"""MR-лаборатория 3: реверсия с упором на ИЗВЛЕЧЕНИЕ максимума.

Запрос 2026-07k: «MR с максимальной выгодой + мощный матаппарат».
Диагноз по mr_lowvol (чемпион): бинарная позиция 0/1, вход по
касанию полосы, выход по RSI 50 — эпизод реверсии монетизируется
частично. Три направления добора прибыли, каждое своим аппаратом:

  А. РАЗМЕР: сколько брать (Келли-сайзинг от ожидаемой реверсии,
     грид-набор по глубине растяжения, AR(1)-прогноз как позиция).
  Б. ПОРОГИ: где входить/выходить (оптимальные пороги Бертрама из
     OU-параметров, овершут-выход за среднюю, GARCH-шок).
  В. ОТБОР ЭПИЗОДОВ: когда реверсия реальна (DFA-Хёрст, ранговый
     шок Уилкоксона, непараметрический хвост-квантиль, дивергенция).

Про OU (ответ на вопрос Кирилла): раздел OU-КАК-СТРАТЕГИИ закрыт и
не переоткрывается. mr3_bertram использует OU иначе — как ИЗМЕРИТЕЛЬ:
kappa из AR(1)-фита z-ряда задаёт аналитический порог входа
(Bertram 2010), торгует по-прежнему полосная логика с calm-гейтом.
Провал модели закроет и эту форму — one-shot, как всем.

╔════════════════════════════════════════════════════════════════╗
║ MULTIPLE TESTING: 10 моделей. Отбор: walk-forward -> bootstrap ║
║ против mr_lowvol (UNDEFEATED) -> второй источник -> one-shot.  ║
║ Грид/Келли-модели: сразу смотреть DD-мандат (размер > 1 нет,   ║
║ но время в рынке больше).                                      ║
╚════════════════════════════════════════════════════════════════╝

Контракт: Bars -> position [0, 1] (long-only, шорт-нога закрыта
дважды). Сдвига внутри НЕТ — движок сдвигает. VT снаружи (--vt).

Модели (аппарат в скобках):
 1. mr3_bertram   — порог входа из OU-kappa (Bertram 2010).
 2. mr3_garch_z   — шок в GARCH-стандартизованных остатках.
 3. mr3_kelly     — позиция = дробный Келли от ожидаемой реверсии.
 4. mr3_grid      — грид-набор 3 ступеней по глубине z.
 5. mr3_overshoot — выход ЗА средней: капчер овершута.
 6. mr3_dfa       — гейт DFA-Хёрста альфа < 0.5 (антиперсистентность).
 7. mr3_rank      — знако-ранговый шок (аппарат Уилкоксона).
 8. mr3_tail_q    — вход по хвостовому квантилю k-барной доходности.
 9. mr3_ar1_fcst  — прогноз AR(1) с отрицательной автокорреляцией.
10. mr3_div       — бычья дивергенция цена/RSI на свинг-лоу.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.bollinger import _rsi
from strategies.meanrev_lab import _zscore
from strategies.meanrev_lab2 import _hold01


def _calm_gate(bars: Bars, vol_lookback: int = 20,
               vol_pct: float = 0.7) -> pd.Series:
    """Гейт спокойной волы (аппарат чемпиона mr_lowvol): бул-ряд.

    Вола ниже своего expanding-перцентиля vol_pct = режим спокоен.
    """
    vol = bars.returns().rolling(vol_lookback).std()
    thresh = vol.expanding(min_periods=vol_lookback * 3).quantile(vol_pct)
    return (vol < thresh).fillna(False)


def _hold_time(enter: pd.Series, leave: pd.Series,
               max_hold: int) -> pd.Series:
    """0/1-позиция с приоритетным выходом и тайм-стопом."""
    ent = enter.fillna(False).to_numpy()
    lev = leave.fillna(False).to_numpy()
    n = len(ent)
    pos = np.zeros(n)
    in_pos = False
    age = 0
    for i in range(n):
        if in_pos:
            age += 1
            if lev[i] or age >= max_hold:
                in_pos = False
        if not in_pos and ent[i] and not lev[i]:
            in_pos, age = True, 0
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=enter.index)


# ── 1. Пороги Бертрама ───────────────────────────────────────────────
def mr3_bertram(
    bars: Bars, z_win: int = 20, phi_win: int = 250,
    cost: float = 0.0004, a_floor: float = 1.2, a_cap: float = 3.0,
) -> pd.Series:
    """Вход по АНАЛИТИЧЕСКОМУ порогу Бертрама, а не по константе 2.0.

    Механизм: если z-ряд ~ OU со скоростью kappa и стационарной
    дисперсией 1, оптимальный (по скорости прибыли, малые издержки)
    порог входа Бертрама сводится к a* = (1.5 * c_z)^(1/3), где
    c_z — издержки, переведённые в z-единицы через сигму растяжения
    (при var=1 сигма OU есть sqrt(2*kappa) — kappa сокращается
    красиво, но живёт внутри c_z-перевода через размер отклонения).
    Быстрая реверсия/крупные отклонения -> порог ниже (чаще в рынке,
    больше циклов = больше прибыли); вязкая -> порог задирается сам.
    kappa = -ln(phi) из rolling AR(1) фита z. Calm-гейт mr_lowvol.
    OU тут — измеритель порога, не модель цены (см. шапку модуля).
    """
    z = _zscore(bars.close, z_win)
    phi = z.rolling(phi_win).corr(z.shift(1)).clip(0.01, 0.999)
    # сигма растяжения в единицах доходности: std (close/SMA - 1)
    dev = bars.close / bars.close.rolling(z_win).mean() - 1.0
    sigma_dev = dev.rolling(phi_win).std()
    c_z = cost / sigma_dev.replace(0.0, np.nan)
    # скорость реверсии масштабирует эффективную цену цикла:
    # медленный процесс (kappa мал) платит издержки реже за то же
    # растяжение -> c_z делим на kappa, затем кубический корень.
    kappa = -np.log(phi)
    a_star = (1.5 * c_z / kappa.replace(0.0, np.nan)) ** (1.0 / 3.0)
    a_star = a_star.clip(lower=a_floor, upper=a_cap)
    calm = _calm_gate(bars)
    enter = (z < -a_star) & calm
    leave = z >= 0.0
    return _hold01(enter, leave)


# ── 2. GARCH-шок ─────────────────────────────────────────────────────
def mr3_garch_z(
    bars: Bars, z_in: float = 2.5, z_win: int = 20,
    max_hold: int = 15,
) -> pd.Series:
    """Вход на шоке в GARCH-стандартизованных остатках.

    Механизм: урок проекта — сырые доходности сырья неинформативны,
    правильное пространство диагностики = r / sigma_GARCH. Шок
    r_t < -z_in * прогнозной сигмы (прогноз на t сделан на t-1, без
    look-ahead) — событие, редкое ПО МЕРКАМ МОДЕЛИ волы, а не по
    сырой сигме, которая в кризис задрана и «шоков не видит».
    Выход: z цены >= 0 (среднее коснулись) или тайм-стоп.
    """
    from core.garch import garch_vol_forecast
    r = bars.close.pct_change()
    sigma = garch_vol_forecast(r).shift(1)   # прогноз для бара t
    zr = r / sigma.replace(0.0, np.nan)
    z = _zscore(bars.close, z_win)
    enter = zr < -z_in
    leave = z >= 0.0
    return _hold_time(enter, leave, max_hold)


# ── 3. Келли-реверсия ────────────────────────────────────────────────
def mr3_kelly(
    bars: Bars, z_win: int = 20, phi_win: int = 250,
    frac: float = 0.5, cap: float = 1.0,
) -> pd.Series:
    """Позиция = дробный Келли от ОЖИДАЕМОЙ скорости реверсии.

    Механизм: ожидаемый ход к средней за бар = -z*(1-phi)*sigma_dev
    (AR(1)-затухание растяжения в единицах доходности). Ставка
    Келли f = frac * E / var(r): позиция непрерывна и растёт сразу
    по ДВУМ осям — глубине растяжения И скорости реверсии. Мелкое
    растяжение вязкого ряда — почти ноль; глубокое растяжение
    быстрого ряда — полный размер. Монетизация против бинарного
    0/1: размер пропорционален ожидаемой прибыли эпизода.
    """
    z = _zscore(bars.close, z_win)
    phi = z.rolling(phi_win).corr(z.shift(1)).clip(0.0, 0.999)
    dev = bars.close / bars.close.rolling(z_win).mean() - 1.0
    sigma_dev = dev.rolling(phi_win).std()
    exp_ret = (-z).clip(lower=0.0) * (1.0 - phi) * sigma_dev
    var = bars.close.pct_change().rolling(phi_win).var()
    f = frac * exp_ret / var.replace(0.0, np.nan)
    calm = _calm_gate(bars).astype(float)
    return (f.clip(lower=0.0, upper=cap) * calm).fillna(0.0)


# ── 4. Грид-набор ────────────────────────────────────────────────────
def mr3_grid(
    bars: Bars, z_win: int = 20,
    levels: tuple[float, ...] = (-1.5, -2.5, -3.5),
    sizes: tuple[float, ...] = (0.4, 0.3, 0.3),
    z_exit: float = -0.25, panic_pct: float = 0.95,
) -> pd.Series:
    """Лестница из 3 ступеней по глубине z; выход всей пирамиды.

    Механизм: реверсия редко разворачивается на первом касании —
    добор на -1.5/-2.5/-3.5 сдвигает среднюю цену входа глубже в
    растяжение (максимум прибыли на путь эпизода). Против
    мартингейл-риска три предохранителя: суммарный размер жёстко
    <= 1.0 (не удвоение!), новые ступени только в calm-режиме,
    принудительный сброс при воле выше panic-перцентиля (аппарат
    vol_percentile_gate — ответ на CL-2020: падающий нож не
    усредняем). Отличие от mr_scaled/mr_ladder ядра: calm-гейт
    ступеней + паник-сброс, т.е. лестница живёт ТОЛЬКО в режиме,
    где реверсия статистически существует.
    """
    z = _zscore(bars.close, z_win).to_numpy()
    calm = _calm_gate(bars).to_numpy()
    vol = bars.returns().rolling(20).std()
    panic = (vol.rolling(750, min_periods=250).rank(pct=True)
             > panic_pct).fillna(False).to_numpy()
    n = len(z)
    pos = np.zeros(n)
    filled = 0
    for i in range(n):
        if np.isnan(z[i]) or panic[i]:
            filled = 0
            pos[i] = 0.0
            continue
        if filled > 0 and z[i] >= z_exit:
            filled = 0
        if calm[i] and filled < len(levels) and z[i] < levels[filled]:
            filled += 1
        pos[i] = float(sum(sizes[:filled]))
    return pd.Series(pos, index=bars.index)


# ── 5. Овершут-выход ─────────────────────────────────────────────────
def mr3_overshoot(
    bars: Bars, z_win: int = 20, z_in: float = 2.0,
    z_out: float = 0.5, max_hold: int = 25,
) -> pd.Series:
    """Вход как у полос, выход ЗА средней (+0.5 сигмы): овершут.

    Механизм: первопрохождение OU-типа процессов асимметрично —
    импульс возврата по инерции проскакивает среднюю (short-term
    reversal перетекает в микро-моментум). Все MR реестра выходят
    ДО/НА средней (RSI 50, z=0, %B 0.5) и отдают овершут рынку;
    здесь удержание до z >= +0.5 забирает его. Плата — эпизод
    длиннее (тайм-стоп страхует зависание у нуля). Чистый тест
    гипотезы «где лежат недособранные деньги MR: до или после
    средней».
    """
    z = _zscore(bars.close, z_win)
    calm = _calm_gate(bars)
    enter = (z < -z_in) & calm
    leave = z >= z_out
    return _hold_time(enter, leave, max_hold)


# ── 6. DFA-Хёрст ─────────────────────────────────────────────────────
def mr3_dfa(
    bars: Bars, dfa_win: int = 250, recalc: int = 5,
    alpha_max: float = 0.48, z_win: int = 20, z_in: float = 2.0,
) -> pd.Series:
    """Гейт антиперсистентности: DFA-альфа < 0.5, затем полосный вход.

    Механизм: DFA (detrended fluctuation analysis) оценивает Хёрста
    через rms-флуктуации кумулятивного профиля вокруг ЛОКАЛЬНЫХ
    линейных трендов по масштабам {8,16,32,64} — в отличие от
    variance-ratio (аппарат hurst_alloc/mr2_vr), DFA снимает
    нестационарность среднего ДО оценки и не путает дрейф с
    персистентностью. alpha < 0.5 = антиперсистентный режим,
    реверсия статистически существует -> разрешён вход z < -z_in.
    """
    r = bars.close.pct_change().to_numpy()
    n = len(r)
    scales = (8, 16, 32, 64)
    log_s = np.log(scales)
    xs = {s: np.vander(np.arange(s, dtype=float), 2) for s in scales}
    alpha = np.full(n, np.nan)
    last = np.nan
    for t in range(n):
        if t >= dfa_win and t % recalc == 0:
            win = r[t - dfa_win + 1:t + 1]
            if not np.isnan(win).any():
                prof = np.cumsum(win - win.mean())
                fs = []
                for s in scales:
                    nb = dfa_win // s
                    seg = prof[:nb * s].reshape(nb, s)
                    x = xs[s]
                    coef, *_ = np.linalg.lstsq(x, seg.T, rcond=None)
                    resid = seg.T - x @ coef
                    fs.append(np.sqrt((resid ** 2).mean()))
                last = float(np.polyfit(log_s, np.log(fs), 1)[0])
        alpha[t] = last
    gate = pd.Series(alpha, index=bars.index) < alpha_max
    z = _zscore(bars.close, z_win)
    enter = (z < -z_in) & gate.fillna(False)
    leave = z >= 0.0
    return _hold01(enter, leave)


# ── 7. Знако-ранговый шок ────────────────────────────────────────────
def mr3_rank(
    bars: Bars, rank_win: int = 60, k: int = 5,
    w_in: float = -1.8, z_win: int = 20, max_hold: int = 15,
) -> pd.Series:
    """Шок по Уилкоксону: знако-ранговая сумма последних k баров.

    Механизм: W = sum sign(r_i) * rank(|r_i|) по последним k барам,
    ранги |r| в trailing-окне; нормировка sqrt(sum rank^2). Ранги
    вместо величин: устойчив к одиночному выбросу (у mr2_shock
    один -4-сигма бар = сигнал; здесь нужен СОГЛАСОВАННЫЙ по
    рангу нисходящий кластер — другой класс событий). Вход при
    W_norm < w_in, выход по касанию средней или тайм-стопу.
    """
    r = bars.close.pct_change()
    rk = r.abs().rolling(rank_win).rank()      # ранг |r_t| в окне
    signed = np.sign(r) * rk
    num = signed.rolling(k).sum()
    den = np.sqrt((rk ** 2).rolling(k).sum())
    w = num / den.replace(0.0, np.nan)
    z = _zscore(bars.close, z_win)
    enter = w < w_in
    leave = z >= 0.0
    return _hold_time(enter, leave, max_hold)


# ── 8. Хвостовой квантиль ────────────────────────────────────────────
def mr3_tail_q(
    bars: Bars, k: int = 5, q: float = 0.05, dist_win: int = 500,
    z_win: int = 20, max_hold: int = 15,
) -> pd.Series:
    """Вход, когда k-барная доходность в СОБСТВЕННОМ 5%-хвосте.

    Механизм: непараметрический родственник mr2_shock: там шок
    мерился в EWMA-сигмах (параметрика, нормальность неявно),
    здесь — позицией в эмпирическом trailing-распределении k-барных
    доходностей. Толстые хвосты сырья учитываются автоматически:
    5%-квантиль CL и 5%-квантиль ZC — разные величины, ранг делает
    порог самокалибрующимся (та же философия, что vol-гейт).
    """
    ret_k = bars.close.pct_change(k)
    thr = ret_k.rolling(dist_win, min_periods=dist_win // 2).quantile(q)
    z = _zscore(bars.close, z_win)
    enter = ret_k < thr
    leave = z >= 0.0
    return _hold_time(enter, leave, max_hold)


# ── 9. AR(1)-прогноз ─────────────────────────────────────────────────
def mr3_ar1_fcst(
    bars: Bars, phi_win: int = 250, t_min: float = 2.0,
    scale: float = 1.5, cap: float = 1.0,
) -> pd.Series:
    """Прямой эконометрический прогноз: r_hat = phi * r_t при phi<0.

    Механизм: не «растяжение уровня» (это z-семейство), а реверсия
    ПРИРАЩЕНИЙ: значимо отрицательная автокорреляция доходностей
    (t-статистика phi < -t_min) => прогноз следующего бара
    r_hat = phi*r_t. Позиция = clip(r_hat / (scale*сигма), 0, cap):
    в рынке только после красного бара при значимом phi, размер
    пропорционален прогнозу. Зеркало tr4_ar1 (там phi>0 => тренд).
    Внимание в отчёте: горизонт 1 бар => оборот высокий, вердикт
    только NET после издержек.
    """
    r = bars.close.pct_change()
    phi = r.rolling(phi_win).corr(r.shift(1))
    tstat = phi * np.sqrt(phi_win) / np.sqrt(
        (1.0 - phi ** 2).clip(lower=1e-6))
    vol = r.rolling(30).std()
    r_hat = phi * r
    pos = (r_hat / (scale * vol.replace(0.0, np.nan))).clip(
        lower=0.0, upper=cap)
    return pos.where(tstat < -t_min, 0.0).fillna(0.0)


# ── 10. Дивергенция ──────────────────────────────────────────────────
def mr3_div(
    bars: Bars, wing: int = 2, rsi_period: int = 14,
    rsi_exit: float = 55.0, max_hold: int = 20,
) -> pd.Series:
    """Бычья дивергенция: цена делает lower-low, RSI — higher-low.

    Механизм: паттерн второго порядка. Свинг-лоу = фрактал
    (low[t-wing] минимален в окне +-wing, подтверждается через wing
    баров — как tr_fractal, но на минимумах). Цена обновила
    минимум, а импульс (RSI) НЕ обновил => продавцы дожимают цену
    на затухающей силе — классический сетап истощения. Ни одна
    модель реестра не сравнивает ДВА последовательных экстремума;
    все меряют текущее состояние.
    """
    lo = bars.low
    rsi = _rsi(bars.close, rsi_period)
    is_swing = lo.shift(wing) == lo.rolling(2 * wing + 1).min()
    sw = is_swing.fillna(False).to_numpy()
    lo_v = lo.shift(wing).to_numpy()          # цена свинга
    rsi_v = rsi.shift(wing).to_numpy()        # RSI на свинге
    rsi_now = rsi.to_numpy()
    n = len(sw)
    enter = np.zeros(n, dtype=bool)
    prev_lo, prev_rsi = np.nan, np.nan
    for i in range(n):
        if sw[i] and not np.isnan(lo_v[i]) and not np.isnan(rsi_v[i]):
            if (not np.isnan(prev_lo) and lo_v[i] < prev_lo
                    and rsi_v[i] > prev_rsi):
                enter[i] = True
            prev_lo, prev_rsi = lo_v[i], rsi_v[i]
    leave = pd.Series(rsi_now > rsi_exit, index=bars.index)
    return _hold_time(pd.Series(enter, index=bars.index), leave,
                      max_hold)


MEANREV_LAB3 = {
    "mr3_bertram": mr3_bertram,
    "mr3_garch_z": mr3_garch_z,
    "mr3_kelly": mr3_kelly,
    "mr3_grid": mr3_grid,
    "mr3_overshoot": mr3_overshoot,
    "mr3_dfa": mr3_dfa,
    "mr3_rank": mr3_rank,
    "mr3_tail_q": mr3_tail_q,
    "mr3_ar1_fcst": mr3_ar1_fcst,
    "mr3_div": mr3_div,
}
