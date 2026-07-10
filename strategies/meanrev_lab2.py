"""MR-лаборатория 2: 15 моделей реверсии РАЗНОГО матаппарата.

Запрос 2026-07j: расширить пространство MR-гипотез за пределы
bb_rsi-семейства. Каждая модель — другой математический взгляд на
«растяжение и возврат»: state-space, непараметрика, робастная
статистика, теория информации, тесты случайности. НЕ вариации
параметров одной идеи (урок meanrev_lab: варьируем механизм).

Раздел OU закрыт — здесь нет OU-процесса, калибровки theta/mu и
торговли «отклонения от равновесия». Реверсия ловится косвенными
аппаратами без модели равновесной цены.

╔════════════════════════════════════════════════════════════════╗
║ MULTIPLE TESTING: 15 моделей на одной истории гарантируют      ║
║ случайных «победителей». Отбор: walk-forward -> bootstrap      ║
║ против mr_lowvol (чемпион MR) -> второй источник -> one-shot.  ║
╚════════════════════════════════════════════════════════════════╝

Контракт: Bars -> position [0, 1] (long-only, как весь MR-трек;
шорт-нога закрыта дважды). Сдвига внутри НЕТ — единственная точка
shift(1) это движок. VT снаружи (--vt).

Модели (аппарат в скобках):
 1. mr2_kalman_z   — z остатка Kalman-уровня (state-space).
 2. mr2_quantile   — квантильный канал (непараметрика, без нормальности).
 3. mr2_runs       — серия падений подряд (биномиальный тест).
 4. mr2_entropy    — гейт перестановочной энтропии (теория информации).
 5. mr2_halflife   — гейт скорости AR(1)-реверсии (эконометрика).
 6. mr2_vr         — гейт variance ratio < 1 (Lo-MacKinlay).
 7. mr2_mad        — робастный z: медиана/MAD (устойчив к выбросам).
 8. mr2_cusum      — CUSUM-триггер накопленного отклонения (SPC).
 9. mr2_shock      — возврат после шока в EWMA-сигмах (event study).
10. mr2_skew       — отрицательная скошенность + растяжение (моменты).
11. mr2_theil      — отклонение от линии Тейла-Сена (робастная регрессия).
12. mr2_ddband     — полоса просадки от локального максимума в ATR.
13. mr2_tema_dev   — отклонение от TEMA (сглаживание без лага).
14. mr2_percb_bw   — %B ниже нуля при РАСШИРЕННЫХ полосах (не squeeze).
15. mr2_soft_z     — непрерывная логистическая позиция от z (без порогов).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.bollinger import _rsi
from strategies.kalman_trend import _kalman_level_slope
from strategies.meanrev_lab import _zscore


def _hold01(enter: pd.Series, leave: pd.Series) -> pd.Series:
    """Позиция 0/1 из булевых рядов входа/выхода (state machine).

    Args:
        enter: True на баре входа.
        leave: True на баре выхода (приоритет у выхода).

    Returns:
        Ряд позиции {0, 1}.
    """
    ent = enter.fillna(False).to_numpy()
    lev = leave.fillna(False).to_numpy()
    pos = np.zeros(len(ent))
    in_pos = False
    for i in range(len(ent)):
        if in_pos and lev[i]:
            in_pos = False
        if not in_pos and ent[i] and not lev[i]:
            in_pos = True
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=enter.index)


# ── 1. Kalman residual ────────────────────────────────────────────────
def mr2_kalman_z(
    bars: Bars, q_level: float = 1e-5, z_in: float = 2.0,
    z_out: float = 0.25, vol_win: int = 30,
) -> pd.Series:
    """Реверсия к Kalman-уровню: вход при остатке < -z_in сигм.

    Механизм: Kalman-уровень — адаптивное «справедливое» сглаживание
    (state-space, лаг меньше SMA). Остаток цена-минус-уровень,
    нормированный своей волой, — мера растяжения, которая в тренде
    сама смещает уровень и потому не воюет с трендом так, как
    статичная SMA (главный провал одно-факторного OU).
    """
    log_c = np.log(bars.close.where(bars.close > 0)).to_numpy()
    level, _ = _kalman_level_slope(
        log_c, q_level=q_level, q_slope=1e-7, r=1e-3)
    resid = np.log(bars.close.where(bars.close > 0)) - pd.Series(
        level, index=bars.index)
    sd = resid.rolling(vol_win).std().where(lambda s: s > 1e-12)
    z = resid / sd
    return _hold01(z < -z_in, z > -z_out)


# ── 2. Квантильный канал ──────────────────────────────────────────────
def mr2_quantile(
    bars: Bars, window: int = 60, q_in: float = 0.10, q_out: float = 0.50,
) -> pd.Series:
    """Непараметрический канал: вход ниже q10, выход выше медианы.

    Механизм: полосы Боллинджера предполагают нормальность (mean±k*std),
    а сырьё/крипта тяжелохвостые. Скользящие квантили не знают о
    распределении вообще: «нижние 10% за окно» — честная редкость
    без гауссовых допущений.
    """
    lo = bars.close.rolling(window).quantile(q_in).shift(1)
    mid = bars.close.rolling(window).quantile(q_out).shift(1)
    return _hold01(bars.close < lo, bars.close > mid)


# ── 3. Биномиальные серии ─────────────────────────────────────────────
def mr2_runs(
    bars: Bars, run: int = 5, hold: int = 5,
) -> pd.Series:
    """Серия из run падений подряд -> лонг на hold баров.

    Механизм: при p=0.5 серия из 5 минусов имеет вероятность ~3% —
    биномиально редкое событие. Если рынок хоть слабо реверсионен,
    условная доходность после такой серии положительна. Тайм-выход
    вместо ценового: гипотеза про СОБЫТИЕ, не про уровень.
    """
    down = (bars.close.diff() < 0).astype(float)
    streak = down.rolling(run).sum()
    ent = (streak >= run).to_numpy()
    pos = np.zeros(len(ent))
    left = 0
    for i in range(len(ent)):
        if ent[i]:
            left = hold
        if left > 0:
            pos[i] = 1.0
            left -= 1
    return pd.Series(pos, index=bars.index)


# ── 4. Перестановочная энтропия ───────────────────────────────────────
def _perm_entropy(x: np.ndarray, order: int = 4) -> float:
    """Нормированная перестановочная энтропия окна (0..1)."""
    n = len(x) - order + 1
    if n < 8:
        return np.nan
    # Каждый порядковый паттерн кодируется одним целым (упаковка
    # argsort по основанию order) — np.unique по 2D-массиву кортежей
    # молча считал бы отдельные элементы, а не паттерны.
    base = order ** np.arange(order)
    codes = np.empty(n, dtype=np.int64)
    for i in range(n):
        codes[i] = int(np.argsort(x[i:i + order]) @ base)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    h = -(p * np.log(p)).sum()
    hmax = np.log(float(math.factorial(order)))
    return float(h / hmax)


def mr2_entropy(
    bars: Bars, window: int = 100, ent_min: float = 0.93,
    z_win: int = 20, z_in: float = 1.5, z_out: float = 0.25,
) -> pd.Series:
    """Z-вход разрешён только при ВЫСОКОЙ энтропии (шумовой режим).

    Механизм: перестановочная энтропия близка к 1, когда порядковые
    паттерны цены равновероятны (шум/боковик), и падает в тренде
    (паттерн «вверх-вверх-вверх» доминирует). Информационный гейт:
    реверсию торгуем только там, где нет информационной структуры,
    которую ломает MR. Пересчёт раз в 10 баров (дорогая метрика).
    """
    c = bars.close.to_numpy()
    n = len(c)
    ent = np.full(n, np.nan)
    last = np.nan
    for i in range(n):
        if i >= window and i % 10 == 0:
            # По УРОВНЯМ цены: тренд даёт монотонные порядковые
            # паттерны (энтропия -> 0), range — перемешанные
            # (-> 1). На приращениях тренд неотличим от шума.
            last = _perm_entropy(c[i - window:i])
        ent[i] = last
    ent_s = pd.Series(ent, index=bars.index)
    z = _zscore(bars.close, z_win)
    enter = (z < -z_in) & (ent_s > ent_min)
    return _hold01(enter, z > -z_out)


# ── 5. Half-life гейт ─────────────────────────────────────────────────
def mr2_halflife(
    bars: Bars, fit_win: int = 120, hl_max: float = 15.0,
    z_win: int = 20, z_in: float = 1.5, z_out: float = 0.25,
) -> pd.Series:
    """Z-вход только при короткой полужизни AR(1)-отклонений.

    Механизм: rolling AR(1) на уровне цены: dP = a + b*P(-1). Скорость
    kappa = -ln(1+b), half-life = ln2/kappa. Длинная полужизнь =
    отклонения НЕ рассасываются (тренд/блуждание) — реверсию не
    торгуем. Это гейт по скорости возврата без модели равновесия
    (чем OU-трек и грешил: там равновесие торговалось как цель).
    """
    c = bars.close
    dc = c.diff()
    lag = c.shift(1)
    mean_x = lag.rolling(fit_win).mean()
    mean_y = dc.rolling(fit_win).mean()
    cov = (lag * dc).rolling(fit_win).mean() - mean_x * mean_y
    var = lag.rolling(fit_win).var()
    b = cov / var.where(var > 1e-12)
    kappa = -np.log1p(b.clip(-0.999, -1e-6))
    hl = np.log(2.0) / kappa
    z = _zscore(c, z_win)
    enter = (z < -z_in) & (hl < hl_max)
    return _hold01(enter, z > -z_out)


# ── 6. Variance-ratio гейт ────────────────────────────────────────────
def mr2_vr(
    bars: Bars, window: int = 120, q: int = 5, vr_max: float = 0.85,
    z_win: int = 20, z_in: float = 1.5, z_out: float = 0.25,
) -> pd.Series:
    """Z-вход только при VR(q) < vr_max (анти-персистентность).

    Механизм: Lo-MacKinlay: у случайного блуждания Var[r_q] = q*Var[r_1],
    VR=1. VR<1 — приращения антикоррелированы, среда реверсии. Прямой
    статистический тест среды вместо косвенных прокси (низкой волы).
    """
    r = np.log(bars.close / bars.close.shift(1))
    var1 = r.rolling(window).var()
    rq = r.rolling(q).sum()
    varq = rq.rolling(window).var()
    vr = varq / (q * var1.where(var1 > 1e-14))
    z = _zscore(bars.close, z_win)
    enter = (z < -z_in) & (vr < vr_max)
    return _hold01(enter, z > -z_out)


# ── 7. Робастный z (MAD) ─────────────────────────────────────────────
def mr2_mad(
    bars: Bars, window: int = 20, z_in: float = 2.0, z_out: float = 0.3,
) -> pd.Series:
    """Растяжение в единицах MAD от скользящей медианы.

    Механизм: mean/std сами загрязняются выбросом, который ловим:
    паника раздувает std и глушит z ровно в момент сигнала. Медиана и
    MAD (масштаб 1.4826 до сигма-эквивалента) нечувствительны к
    хвосту — z честнее в самый нужный момент.
    """
    med = bars.close.rolling(window).median()
    mad = (bars.close - med).abs().rolling(window).median() * 1.4826
    z = (bars.close - med) / mad.where(mad > 1e-12)
    return _hold01(z < -z_in, z > -z_out)


# ── 8. CUSUM ──────────────────────────────────────────────────────────
def mr2_cusum(
    bars: Bars, k_sigma: float = 0.5, h_sigma: float = 4.0,
    vol_win: int = 30, z_out: float = 0.0,
) -> pd.Series:
    """CUSUM-накопитель отрицательных отклонений как триггер входа.

    Механизм: SPC-аппарат (контроль процессов): S = max(0, S - r - k),
    срабатывание при S > h. В отличие от точечного z, CUSUM ловит
    ДЛИТЕЛЬНОЕ умеренное давление вниз, которое ни один бар по
    отдельности не делает экстремальным. Сброс накопителя после
    входа; выход при возврате z к нулю.
    """
    r = bars.close.pct_change()
    sig = r.rolling(vol_win).std().where(lambda s: s > 1e-12)
    rn = (r / sig).to_numpy()
    z = _zscore(bars.close, vol_win).to_numpy()
    n = len(rn)
    pos = np.zeros(n)
    s = 0.0
    in_pos = False
    for i in range(n):
        x = rn[i]
        if np.isnan(x):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        s = max(0.0, s - x - k_sigma)
        if in_pos and not np.isnan(z[i]) and z[i] > -z_out:
            in_pos = False
        if not in_pos and s > h_sigma:
            in_pos = True
            s = 0.0
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


# ── 9. Шок-реверсия ──────────────────────────────────────────────────
def mr2_shock(
    bars: Bars, span: int = 36, shock: float = 2.5, hold: int = 4,
) -> pd.Series:
    """Однобарный шок вниз < -shock EWMA-сигм -> лонг на hold баров.

    Механизм: event study: аномальный однобарный минус — ликвидация/
    паника, микроструктурный отскок статистически значим на коротком
    горизонте. Жёсткий тайм-выход: торгуем событие, не уровень.
    Отличие от mr2_runs: там серия умеренных минусов, тут один
    экстремальный.
    """
    r = bars.close.pct_change()
    sig = r.ewm(span=span, adjust=False).std()
    std_r = r / sig.where(sig > 1e-12)
    ent = (std_r < -shock).to_numpy()
    pos = np.zeros(len(ent))
    left = 0
    for i in range(len(ent)):
        if ent[i]:
            left = hold
        if left > 0:
            pos[i] = 1.0
            left -= 1
    return pd.Series(pos, index=bars.index)


# ── 10. Скошенность ──────────────────────────────────────────────────
def mr2_skew(
    bars: Bars, skew_win: int = 60, skew_max: float = -0.5,
    z_win: int = 20, z_in: float = 1.5, z_out: float = 0.25,
) -> pd.Series:
    """Z-вход при сильно отрицательной скользящей скошенности.

    Механизм: третий момент как детектор паники: распродажа с
    редкими крупными минусами даёт skew << 0. Комбинация «хвостовое
    давление уже случилось» + «цена растянута» — отскок от
    капитуляции, а не от планомерного тренда вниз (там skew ~ 0).
    """
    r = bars.close.pct_change()
    sk = r.rolling(skew_win).skew()
    z = _zscore(bars.close, z_win)
    enter = (z < -z_in) & (sk < skew_max)
    return _hold01(enter, z > -z_out)


# ── 11. Тейл-Сен ─────────────────────────────────────────────────────
def mr2_theil(
    bars: Bars, window: int = 60, k_in: float = 2.0, k_out: float = 0.3,
    step: int = 5,
) -> pd.Series:
    """Отклонение от робастной линии Тейла-Сена в единицах MAD.

    Механизм: реверсия к ЛОКАЛЬНОМУ ТРЕНДУ, а не к константе: линия
    Тейла-Сена (медиана попарных наклонов) следует за дрейфом и
    нечувствительна к выбросам, поэтому «растяжение» меряется
    поперёк тренда — против самого тренда не торгуем (в отличие от
    SMA-реверсии). Пересчёт раз в step баров (O(w^2) наклонов).
    """
    c = bars.close.to_numpy()
    n = len(c)
    dev = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    xs = np.arange(window, dtype=float)
    ii, jj = np.triu_indices(window, k=1)
    last: tuple[float, float, float, int] | None = None
    for i in range(window, n):
        if i % step == 0 or last is None:
            w = c[i - window + 1:i + 1]
            slopes = (w[jj] - w[ii]) / (xs[jj] - xs[ii])
            slope = float(np.median(slopes))
            inter = float(np.median(w - slope * xs))
            resid = w - (inter + slope * xs)
            mad = float(np.median(np.abs(resid))) * 1.4826
            last = (slope, inter, mad, i)
            dev[i] = resid[-1]
            scale[i] = mad
        else:
            slope, inter, mad, i0 = last
            pred = inter + slope * (window - 1 + (i - i0))
            dev[i] = c[i] - pred
            scale[i] = mad
    dev_s = pd.Series(dev, index=bars.index)
    sc = pd.Series(scale, index=bars.index).where(
        lambda s: s > 1e-12)
    z = dev_s / sc
    return _hold01(z < -k_in, z > -k_out)


# ── 12. Полоса просадки ──────────────────────────────────────────────
def mr2_ddband(
    bars: Bars, high_win: int = 40, k_atr: float = 3.0,
    rsi_max: float = 35.0, recover: float = 0.5,
) -> pd.Series:
    """Просадка от N-барного максимума глубже k*ATR при слабом RSI.

    Механизм: «сколько упали от недавнего пика» — естественная для
    трейдера мера, нормируем её ATR (масштаб инструмента). RSI-гейт
    отфильтровывает случаи, где просадка идёт медленной раздачей без
    перепроданности. Выход при отыгрыше recover доли просадки.
    """
    hi = bars.high.rolling(high_win).max().shift(1)
    atr = bars.atr(20)
    dd = hi - bars.close
    rsi = _rsi(bars.close, 14)
    enter = (dd > k_atr * atr) & (rsi < rsi_max)
    leave = dd < (1.0 - recover) * k_atr * atr
    return _hold01(enter, leave)


# ── 13. TEMA-отклонение ──────────────────────────────────────────────
def _tema(s: pd.Series, span: int) -> pd.Series:
    """Triple EMA (Mulloy): сглаживание с компенсацией лага."""
    e1 = s.ewm(span=span, adjust=False).mean()
    e2 = e1.ewm(span=span, adjust=False).mean()
    e3 = e2.ewm(span=span, adjust=False).mean()
    return 3 * e1 - 3 * e2 + e3


def mr2_tema_dev(
    bars: Bars, span: int = 20, z_in: float = 1.8, z_out: float = 0.25,
    vol_win: int = 30,
) -> pd.Series:
    """Растяжение от TEMA: центр почти без лага.

    Механизм: у SMA/EMA центр отстаёт, и в дрейфе цена «всегда
    растянута» с одной стороны — ложные входы против тренда. TEMA
    компенсирует лаг (3e1-3e2+e3), центр держится у цены, и
    отклонение меряет именно ЛОКАЛЬНЫЙ выброс, а не дрейф.
    """
    center = _tema(bars.close, span)
    resid = bars.close - center
    sd = resid.rolling(vol_win).std().where(lambda s: s > 1e-12)
    z = resid / sd
    return _hold01(z < -z_in, z > -z_out)


# ── 14. %B при широких полосах ───────────────────────────────────────
def mr2_percb_bw(
    bars: Bars, window: int = 20, n_std: float = 2.0,
    bw_win: int = 120, exit_b: float = 0.5,
) -> pd.Series:
    """%B < 0 только когда bandwidth ВЫШЕ своей медианы.

    Механизм: пробой нижней полосы при УЗКИХ полосах — это выход из
    squeeze, начало тренда вниз (среда пробойных стратегий, нож для
    MR). Тот же пробой при ШИРОКИХ полосах — растяжение в уже
    волатильном range. Гейт по ширине разводит эти два мира.
    """
    ma = bars.close.rolling(window).mean()
    sd = bars.close.rolling(window).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    width = (upper - lower) / ma.where(ma.abs() > 1e-12)
    b = (bars.close - lower) / (upper - lower).where(
        lambda s: s > 1e-12)
    wide = width > width.rolling(bw_win).median()
    return _hold01((b < 0.0) & wide, b > exit_b)


# ── 15. Мягкая логистическая позиция ─────────────────────────────────
def mr2_soft_z(
    bars: Bars, z_win: int = 20, steep: float = 1.5,
    z_start: float = 1.0,
) -> pd.Series:
    """Непрерывная позиция: логистическая функция от -z, без порогов.

    Механизм: пороговые MR (вход -2, выход -0.25) страдают
    параметрической хрупкостью (урок Bollinger squeeze). Здесь
    позиция = sigmoid(steep*(-z - z_start)): плавно растёт с
    растяжением, плавно гаснет на возврате. Нет ступеней -> нет
    чувствительности к точному порогу; оборот сглажен.
    """
    z = _zscore(bars.close, z_win)
    pos = 1.0 / (1.0 + np.exp(-steep * (-z - z_start)))
    return pos.fillna(0.0).clip(0.0, 1.0)


MEANREV_LAB2 = {
    "mr2_kalman_z": mr2_kalman_z,
    "mr2_quantile": mr2_quantile,
    "mr2_runs": mr2_runs,
    "mr2_entropy": mr2_entropy,
    "mr2_halflife": mr2_halflife,
    "mr2_vr": mr2_vr,
    "mr2_mad": mr2_mad,
    "mr2_cusum": mr2_cusum,
    "mr2_shock": mr2_shock,
    "mr2_skew": mr2_skew,
    "mr2_theil": mr2_theil,
    "mr2_ddband": mr2_ddband,
    "mr2_tema_dev": mr2_tema_dev,
    "mr2_percb_bw": mr2_percb_bw,
    "mr2_soft_z": mr2_soft_z,
}
