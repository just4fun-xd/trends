"""Trend-лаборатория 3: 15 трендовых моделей разного матаппарата.

Запрос 2026-07j. Уроки предыдущих трендовых треков учтены:
- Trend Lab 1/2: ускорение/качество тренда (~MACD-hist, ER, Hull)
  дают ~0 на ровном тренде — здесь таких фильтров нет, каждый
  кандидат ЛОВИТ тренд своим аппаратом, а не оценивает его качество.
- Take-profit режет правый хвост — ни одной модели с тейком.
- donchian_vt — чемпион обеих ниш; каждый кандидат обязан отличаться
  МЕХАНИЗМОМ (иначе corr ~1 и в ансамбль не попадает).

╔════════════════════════════════════════════════════════════════╗
║ MULTIPLE TESTING: 15 моделей = 2-3 случайных «победителя».     ║
║ Отбор: walk-forward -> bootstrap против donchian_vt -> второй  ║
║ источник -> corr < 0.6 для кандидата в ансамбль.               ║
╚════════════════════════════════════════════════════════════════╝

Контракт: Bars -> position. Long-only по умолчанию (сырьё/акции);
непрерывные сигналы в [0, 1]. Сдвига внутри НЕТ — движок сдвигает.
VT снаружи (--vt).

Модели (аппарат в скобках):
 1. tr3_tsmom      — многогоризонтный знак доходности (Moskowitz TSMOM).
 2. tr3_ribbon     — доля выстроенных EMA-лент (структура сглаживаний).
 3. tr3_adx_di     — ADX-сила + DI-направление (Wilder).
 4. tr3_supertrend — ATR-трейлинг с перекидкой (SuperTrend).
 5. tr3_kama_slope — наклон адаптивной KAMA (Kaufman).
 6. tr3_hh_hl      — структура свингов: выше-максимумы/выше-минимумы.
 7. tr3_fracdiff   — дробное дифференцирование d=0.4 (Lopez de Prado).
 8. tr3_zlema      — кросс zero-lag EMA (компенсация лага).
 9. tr3_persist    — доля растущих баров (биномиальная персистентность).
10. tr3_vr_trend   — гейт variance ratio > 1 (Lo-MacKinlay, зеркало MR).
11. tr3_mid_ride   — езда над серединой Дончиана с растущей серединой.
12. tr3_atr_mom    — momentum в ATR-единицах через tanh (нормировка).
13. tr3_vote3      — большинство знаков на 3 горизонтах (ансамбль голосов).
14. tr3_range_exp  — направление при расширении диапазона.
15. tr3_extreme_t  — свежесть максимума против минимума (время экстремумов).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.trend_lab import _adx


# ── 1. TSMOM ─────────────────────────────────────────────────────────
def tr3_tsmom(
    bars: Bars, horizons: tuple[int, ...] = (63, 126, 252),
) -> pd.Series:
    """Классический time-series momentum: средний знак доходностей.

    Механизм: Moskowitz-Ooi-Pedersen: знак трейлинг-доходности
    предсказывает её продолжение до ~12 месяцев. Три горизонта
    усредняются — не одна удачная длина окна, а консенсус масштабов.
    Позиция = доля положительных горизонтов (0, 1/3, 2/3, 1).
    """
    votes = [
        (bars.close / bars.close.shift(h) - 1.0 > 0).astype(float)
        for h in horizons
    ]
    return sum(votes).div(len(horizons)).fillna(0.0)


# ── 2. EMA-лента ─────────────────────────────────────────────────────
def tr3_ribbon(
    bars: Bars, spans: tuple[int, ...] = (8, 13, 21, 34, 55, 89),
    frac: float = 0.8,
) -> pd.Series:
    """Доля соседних EMA-пар в бычьем порядке (быстрая > медленной).

    Механизм: одна пара EMA — точка; лента из 6 — структура. Полное
    выстраивание всех сглаживаний по убыванию периода бывает только
    в устоявшемся тренде и разрушается постепенно (ранний выход без
    отдельного стопа). Позиция при доле >= frac.
    """
    emas = [bars.close.ewm(span=s, adjust=False).mean() for s in spans]
    aligned = sum(
        (emas[i] > emas[i + 1]).astype(float)
        for i in range(len(emas) - 1)
    ) / (len(emas) - 1)
    return (aligned >= frac).astype(float).fillna(0.0)


# ── 3. ADX + DI ──────────────────────────────────────────────────────
def tr3_adx_di(
    bars: Bars, period: int = 14, adx_min: float = 20.0,
) -> pd.Series:
    """Wilder: тренд есть (ADX > порога) и направлен вверх (+DI > -DI).

    Механизм: ADX меряет СИЛУ движения без направления по системе
    направленного движения (native high/low, не close-производная).
    В отличие от закрытых ER/Hull-фильтров, ADX здесь не фильтр
    поверх пробоя, а самостоятельный вход: сила + направление.
    """
    up = bars.high.diff()
    dn = -bars.low.diff()
    plus_dm = pd.Series(
        np.where((up > dn) & (up > 0), up, 0.0), index=bars.index)
    minus_dm = pd.Series(
        np.where((dn > up) & (dn > 0), dn, 0.0), index=bars.index)
    atr = bars.atr(period)
    alpha = 1.0 / period
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    adx = _adx(bars, period)
    sig = (adx > adx_min) & (plus_di > minus_di)
    return sig.astype(float).fillna(0.0)


# ── 4. SuperTrend ────────────────────────────────────────────────────
def tr3_supertrend(
    bars: Bars, period: int = 10, mult: float = 3.0,
) -> pd.Series:
    """ATR-трейлинг с перекидкой: лонг, пока цена над нижней лентой.

    Механизм: стоп-линия hl2 - mult*ATR подтягивается только вверх,
    пока не пробита; пробой перекидывает режим. Отличие от Дончиана:
    выход адаптивен к воле (ATR), а не к N-барному минимуму —
    в спокойном тренде стоп ближе, в буйном дальше (дышит с рынком).
    """
    hl2 = (bars.high + bars.low) / 2.0
    atr = bars.atr(period)
    up_band = (hl2 - mult * atr).to_numpy()
    dn_band = (hl2 + mult * atr).to_numpy()
    close = bars.close.to_numpy()
    n = len(close)
    pos = np.zeros(n)
    trend_up = False
    lo_line = np.nan
    hi_line = np.nan
    for i in range(n):
        if np.isnan(up_band[i]) or np.isnan(close[i]):
            continue
        if trend_up:
            lo_line = max(lo_line, up_band[i]) \
                if not np.isnan(lo_line) else up_band[i]
            if close[i] < lo_line:
                trend_up = False
                hi_line = dn_band[i]
        else:
            hi_line = min(hi_line, dn_band[i]) \
                if not np.isnan(hi_line) else dn_band[i]
            if close[i] > hi_line:
                trend_up = True
                lo_line = up_band[i]
        pos[i] = 1.0 if trend_up else 0.0
    return pd.Series(pos, index=bars.index)


# ── 5. KAMA-наклон ───────────────────────────────────────────────────
def tr3_kama_slope(
    bars: Bars, er_period: int = 10, fast: int = 2, slow: int = 30,
    slope_win: int = 5,
) -> pd.Series:
    """KAMA растёт устойчиво -> лонг (наклон адаптивного сглаживания).

    Механизм: KAMA сама решает, насколько быстро следовать за ценой
    (efficiency ratio управляет альфой): в пиле почти стоит, в тренде
    ускоряется. Торгуем ЗНАК её наклона за slope_win — сглаживание,
    которое почти не движется в боковике, даёт мало ложных наклонов.
    """
    change = (bars.close - bars.close.shift(er_period)).abs()
    vol_sum = bars.close.diff().abs().rolling(er_period).sum()
    er = (change / vol_sum.where(vol_sum > 1e-12)).fillna(0.0)
    sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1))
          + 2.0 / (slow + 1)) ** 2
    c = bars.close.to_numpy()
    a = sc.to_numpy()
    kama = np.full(len(c), np.nan)
    prev = c[0]
    for i in range(len(c)):
        if np.isnan(c[i]):
            kama[i] = prev
            continue
        prev = prev + a[i] * (c[i] - prev) if not np.isnan(a[i]) else prev
        kama[i] = prev
    ks = pd.Series(kama, index=bars.index)
    return (ks > ks.shift(slope_win)).astype(float).fillna(0.0)


# ── 6. Свинг-структура ───────────────────────────────────────────────
def tr3_hh_hl(
    bars: Bars, pivot: int = 5,
) -> pd.Series:
    """Чартистская структура: последние свинги дают HH и HL.

    Механизм: формализация «восходящей структуры»: pivot-максимум —
    бар, чей high выше pivot соседей с обеих сторон (подтверждение
    задним числом = естественная задержка pivot баров, look-ahead
    нет: свинг признаётся только когда правое плечо закрыто). Лонг,
    пока последний свинг-хай выше предыдущего И свинг-лоу выше
    предыдущего; слом любой половины структуры — выход.
    """
    hi = bars.high.to_numpy()
    lo = bars.low.to_numpy()
    n = len(hi)
    pos = np.zeros(n)
    last_hi = [np.nan, np.nan]
    last_lo = [np.nan, np.nan]
    for i in range(2 * pivot, n):
        j = i - pivot  # кандидат в pivot, подтверждён к бару i
        win_h = hi[j - pivot:j + pivot + 1]
        win_l = lo[j - pivot:j + pivot + 1]
        if not np.isnan(win_h).any() and hi[j] == win_h.max():
            last_hi = [last_hi[1], hi[j]]
        if not np.isnan(win_l).any() and lo[j] == win_l.min():
            last_lo = [last_lo[1], lo[j]]
        hh = last_hi[1] > last_hi[0] if not np.isnan(last_hi[0]) else False
        hl = last_lo[1] > last_lo[0] if not np.isnan(last_lo[0]) else False
        pos[i] = 1.0 if (hh and hl) else 0.0
    return pd.Series(pos, index=bars.index)


# ── 7. Дробное дифференцирование ─────────────────────────────────────
def tr3_fracdiff(
    bars: Bars, d: float = 0.4, window: int = 100, ma_win: int = 16,
) -> pd.Series:
    """Знак сглаженного дробно-дифференцированного лог-ряда.

    Механизм: Lopez de Prado: целое дифференцирование (доходности)
    убивает память ряда, нулевое (цена) нестационарно. d=0.4 —
    компромисс: ряд почти стационарен, но хранит длинную память
    тренда. Сигнал — кросс быстрого/медленного сглаживаний САМОГО
    fd-ряда: momentum на ряде с памятью вместо голой цены.
    """
    weights = np.zeros(window)
    weights[0] = 1.0
    for k in range(1, window):
        weights[k] = -weights[k - 1] * (d - k + 1) / k
    weights = weights[::-1]
    logp = np.log(bars.close.where(bars.close > 0)).to_numpy()
    n = len(logp)
    fd = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = logp[i - window + 1:i + 1]
        if not np.isnan(seg).any():
            fd[i] = float(seg @ weights)
    fds = pd.Series(fd, index=bars.index)
    fast_s = fds.ewm(span=ma_win // 2, adjust=False).mean()
    slow_s = fds.ewm(span=ma_win * 2, adjust=False).mean()
    return (fast_s > slow_s).astype(float).fillna(0.0)


# ── 8. Zero-lag EMA ──────────────────────────────────────────────────
def tr3_zlema(
    bars: Bars, span: int = 40, slope_win: int = 4,
) -> pd.Series:
    """Цена над РАСТУЩЕЙ zero-lag EMA (Ehlers: компенсация лага).

    Механизм: ZLEMA подаёт в EMA цену + (цена - цена(lag)) — грубую
    экстраполяцию, съедающую половину запаздывания сглаживания.
    Кросс ДВУХ ZLEMA в устойчивом тренде вырождается (обе липнут к
    цене, разность — шум), поэтому сигнал: цена над базовой ZLEMA и
    сама база растёт. Тот же аппарат, но роль базы — почти
    безлаговый трейлинг-уровень, а не пара сглаживаний.
    """
    lag = (span - 1) // 2
    zl = (2 * bars.close - bars.close.shift(lag)).ewm(
        span=span, adjust=False).mean()
    sig = (bars.close > zl) & (zl > zl.shift(slope_win))
    return sig.astype(float).fillna(0.0)


# ── 9. Персистентность ───────────────────────────────────────────────
def tr3_persist(
    bars: Bars, window: int = 40, frac: float = 0.58,
) -> pd.Series:
    """Доля растущих баров за окно выше биномиального порога.

    Механизм: у монеты доля плюсов в окне 40 превышает 0.58 с
    вероятностью ~15%; устойчивое превышение — статистическая
    персистентность направления. Знаковая метрика игнорирует
    амплитуду — иммунна к одиночным выбросам, которые ломают
    momentum-меры на сырье (лимитные дни, гэпы).
    """
    up_frac = (bars.close.diff() > 0).astype(float) \
        .rolling(window).mean()
    return (up_frac > frac).astype(float).fillna(0.0)


# ── 10. VR-тренд ─────────────────────────────────────────────────────
def tr3_vr_trend(
    bars: Bars, window: int = 120, q: int = 5, vr_min: float = 1.15,
    ma_win: int = 50,
) -> pd.Series:
    """Гейт VR(q) > vr_min + цена над MA (зеркало mr2_vr).

    Механизм: VR > 1 — приращения положительно автокоррелированы,
    среда персистентна (тот же Lo-MacKinlay, другой хвост). Гейт
    решает «когда тренд-аппарат вообще уместен», направление берёт
    простая MA. Пара mr2_vr/tr3_vr_trend делит время инструмента на
    статистически разные среды одной метрикой.
    """
    r = np.log(bars.close / bars.close.shift(1))
    var1 = r.rolling(window).var()
    varq = r.rolling(q).sum().rolling(window).var()
    vr = varq / (q * var1.where(var1 > 1e-14))
    above = bars.close > bars.close.rolling(ma_win).mean()
    return ((vr > vr_min) & above).astype(float).fillna(0.0)


# ── 11. Езда над серединой канала ────────────────────────────────────
def tr3_mid_ride(
    bars: Bars, window: int = 40, slope_win: int = 10,
) -> pd.Series:
    """Лонг, пока цена над серединой Дончиана И середина растёт.

    Механизм: пробойный Дончиан входит на экстремуме (worst-price
    вход) и ждёт пробоя нижнего канала на выход (широкий стоп).
    Середина канала — медленный устойчивый уровень: вход дешевле
    (на откате к середине), выход раньше (уход под середину).
    Осознанно НЕ пробой — иначе corr с donchian_vt ~1.
    """
    upper = bars.high.rolling(window).max().shift(1)
    lower = bars.low.rolling(window).min().shift(1)
    mid = (upper + lower) / 2.0
    rising = mid > mid.shift(slope_win)
    return ((bars.close > mid) & rising).astype(float).fillna(0.0)


# ── 12. ATR-нормированный momentum ───────────────────────────────────
def tr3_atr_mom(
    bars: Bars, mom_win: int = 60, scale: float = 0.15,
) -> pd.Series:
    """Непрерывная позиция tanh(momentum / (ATR * sqrt(окна))).

    Механизм: сырой momentum несравним между NG и GC (масштабы);
    нормировка ATR*sqrt(t) переводит движение в «сколько типичных
    диапазонов прошли за окно» — безразмерная сила тренда. tanh
    сглаживает позицию: слабый тренд — частичная позиция, не 0/1.
    """
    mom = bars.close - bars.close.shift(mom_win)
    denom = bars.atr(20) * np.sqrt(float(mom_win))
    strength = mom / denom.where(denom > 1e-12)
    return pd.Series(
        np.tanh(strength / scale / 10.0), index=bars.index,
    ).clip(0.0, 1.0).fillna(0.0)


# ── 13. Голосование трёх горизонтов ──────────────────────────────────
def tr3_vote3(
    bars: Bars, horizons: tuple[int, ...] = (20, 60, 120),
) -> pd.Series:
    """Большинство из трёх: цена выше EMA короткого/среднего/длинного.

    Механизм: не консенсус доходностей (tr3_tsmom), а голосование
    положений относительно РАЗНЫХ сглаживаний: 2 из 3 достаточно —
    вход раньше полного выстраивания ленты, выход раньше слома всех
    горизонтов. Дискретные уровни позиции 0/1 по большинству.
    """
    votes = sum(
        (bars.close > bars.close.ewm(span=h, adjust=False).mean())
        .astype(float)
        for h in horizons
    )
    return (votes >= 2).astype(float).fillna(0.0)


# ── 14. Расширение диапазона ─────────────────────────────────────────
def tr3_range_exp(
    bars: Bars, window: int = 20, expand: float = 1.3, hold: int = 10,
) -> pd.Series:
    """Диапазон бара > expand * средний И закрытие в верхней трети.

    Механизм: тренды сырья стартуют с range expansion (выход
    маржинальных участников): широкий бар с сильным закрытием —
    отпечаток агрессора. Вход на hold баров с продлением при
    повторном сигнале: серия расширений = разгон тренда.
    """
    rng = bars.high - bars.low
    avg = rng.rolling(window).mean().shift(1)
    pos_in_bar = (bars.close - bars.low) / rng.where(rng > 1e-12)
    trig = ((rng > expand * avg) & (pos_in_bar > 0.67)).to_numpy()
    pos = np.zeros(len(trig))
    left = 0
    for i in range(len(trig)):
        if trig[i]:
            left = hold
        if left > 0:
            pos[i] = 1.0
            left -= 1
    return pd.Series(pos, index=bars.index)


# ── 15. Свежесть экстремумов ─────────────────────────────────────────
def tr3_extreme_t(
    bars: Bars, window: int = 60, edge: float = 0.25,
) -> pd.Series:
    """Максимум окна свежее минимума с запасом -> лонг.

    Механизм: чисто временнАя мера без амплитуд: в аптренде максимумы
    обновляются (возраст max мал), минимум стареет. Позиция при
    (возраст min - возраст max) / окно > edge. Игнорирует величину
    движений полностью — ортогонален momentum-аппаратам по
    построению, кандидат на низкую корреляцию в ансамбле.
    """
    hi = bars.high.to_numpy()
    lo = bars.low.to_numpy()
    n = len(hi)
    pos = np.zeros(n)
    for i in range(window, n):
        wh = hi[i - window + 1:i + 1]
        wl = lo[i - window + 1:i + 1]
        if np.isnan(wh).any() or np.isnan(wl).any():
            continue
        age_max = window - 1 - int(np.argmax(wh))
        age_min = window - 1 - int(np.argmin(wl))
        pos[i] = 1.0 if (age_min - age_max) / window > edge else 0.0
    return pd.Series(pos, index=bars.index)


TREND_LAB3 = {
    "tr3_tsmom": tr3_tsmom,
    "tr3_ribbon": tr3_ribbon,
    "tr3_adx_di": tr3_adx_di,
    "tr3_supertrend": tr3_supertrend,
    "tr3_kama_slope": tr3_kama_slope,
    "tr3_hh_hl": tr3_hh_hl,
    "tr3_fracdiff": tr3_fracdiff,
    "tr3_zlema": tr3_zlema,
    "tr3_persist": tr3_persist,
    "tr3_vr_trend": tr3_vr_trend,
    "tr3_mid_ride": tr3_mid_ride,
    "tr3_atr_mom": tr3_atr_mom,
    "tr3_vote3": tr3_vote3,
    "tr3_range_exp": tr3_range_exp,
    "tr3_extreme_t": tr3_extreme_t,
}
