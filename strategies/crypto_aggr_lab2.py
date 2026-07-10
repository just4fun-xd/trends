"""Крипто-лаборатория AGGR-2: агрессия + матаппарат защиты в кризис.

Запрос 2026-07k: агрессивные крипто-модели «сливают весь капитал»
в обвал — найти мат. модели, которые режут именно кризисный хвост,
не убивая агрессию на бычьей фазе.

Архитектурный ответ (важно для Александра): волшебной стратегии
«агрессивно И безопасно» не существует — это компромисс. Но у
компромисса ЕСТЬ формальный аппарат: каждая модель ниже = агрессивное
ядро × ЗАЩИТНЫЙ МАТАППАРАТ с собственной теорией. Три семейства защит:

  А. Капитальные (управляют экспозицией от ПУТИ капитала):
     CPPI/TIPP (Black-Perold), Grossman-Zhou drawdown control —
     экспозиция пропорциональна подушке до пола; слить «весь капитал»
     математически невозможно (пол ратчетится с хай-вотермарком).
  Б. Хвостовые (меряют кризис в РАСПРЕДЕЛЕНИИ доходностей):
     bipower-variation jump-тест (Barndorff-Nielsen-Shephard),
     эмпирический CVaR-сайзинг (EVT-семейство), полудисперсия
     (Sortino-сайзинг), skew-гейт, Келли с куртозис-штрафом.
  В. Динамические (меряют кризис в ДИНАМИКЕ процесса):
     vol-of-vol гейт, самовозбуждающаяся интенсивность обвалов
     (Хоукс), circuit-breaker с кулдауном (SPC).

Дополнение к защите на уровне ПОРТФЕЛЯ (не здесь, но часть ответа):
VT-слой, DD-мандат 40%, HRP-веса и шорт-нога ca_short_break —
крипто-обвалы быстрее ралли, кризис у перпетуалов ТОРГУЕТСЯ.

Внутренняя честность: капитальные защиты (ca2_cppi, ca2_gz) ведут
СИНТЕТИЧЕСКУЮ кривую капитала внутри стратегии (gross, без издержек)
через pos.shift(1) — это учёт ПРОШЛОЙ доходности, не сигнальный
сдвиг; look-ahead нет, префикс-тест обязателен, как всем.

╔════════════════════════════════════════════════════════════════╗
║ MULTIPLE TESTING: 10 моделей. Скрининг walk-forward на H4 ->   ║
║ bootstrap против donchian_vt -> corr < 0.6 с donchian и        ║
║ tr_ichimoku. У капитальных защит СРАЗУ смотреть DD-мандат.     ║
╚════════════════════════════════════════════════════════════════╝

Контракт: Bars -> position [0, 2] (пирамиды/Келли) либо [0, 1].
Сдвига сигнала внутри НЕТ — движок сдвигает. VT снаружи (--vt).
Дефолты окон — под H4 (6 баров/день).

Модели (ядро × защита):
 1. ca2_cppi    — Дончиан 12/6 × ратчет-CPPI (Black-Perold).
 2. ca2_gz      — TSMOM × Grossman-Zhou drawdown-множитель.
 3. ca2_bns     — пробой 20 × jump-гейт bipower variation (BNS).
 4. ca2_evt     — momentum × CVaR-сайзинг по хвосту (EVT).
 5. ca2_semi    — momentum × таргет полудисперсии (Sortino-сайзинг).
 6. ca2_vvol    — пробой 15 × гейт vol-of-vol перцентиля.
 7. ca2_hawkes  — momentum × интенсивность обвалов (Хоукс).
 8. ca2_skew    — EMA-кросс × гейт rolling-скошенности.
 9. ca2_kelly_t — Келли с куртозис-штрафом (непрерывная агрессия).
10. ca2_breaker — Дончиан 20/8 × circuit-breaker с кулдауном.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars


# ── общие помощники ──────────────────────────────────────────────────
def _ewma_vol(close: pd.Series, span: int = 36) -> pd.Series:
    """EWMA-волатильность побарных доходностей."""
    return close.pct_change().ewm(span=span, adjust=False).std()


def _donch_pos(bars: Bars, entry: int, exit_period: int) -> pd.Series:
    """Бинарный Дончиан entry/exit (long-only ядро)."""
    upper = bars.high.rolling(entry).max().shift(1)
    lower = bars.low.rolling(exit_period).min().shift(1)
    c = bars.close.to_numpy()
    up, lo = upper.to_numpy(), lower.to_numpy()
    pos = np.zeros(len(c))
    in_pos = False
    for i in range(len(c)):
        if np.isnan(c[i]) or np.isnan(up[i]) or np.isnan(lo[i]):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        if in_pos and c[i] < lo[i]:
            in_pos = False
        elif not in_pos and c[i] > up[i]:
            in_pos = True
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)
    # shift(1) на каналах — стандарт пробойных ядер проекта (уровень
    # вчерашний), это НЕ сигнальный сдвиг: сигнал сдвинет движок.


def _mom_pos(bars: Bars, window: int = 60) -> pd.Series:
    """Бинарное momentum-ядро: close выше, чем window баров назад."""
    return (bars.close.pct_change(window) > 0).astype(float).fillna(0.0)


# ── 1. Ратчет-CPPI ───────────────────────────────────────────────────
def ca2_cppi(
    bars: Bars, entry: int = 12, exit_period: int = 6,
    dd_budget: float = 0.25, mult: float = 4.0, cap: float = 2.0,
) -> pd.Series:
    """Быстрый Дончиан × CPPI: экспозиция = mult * подушка до пола.

    Механизм (Black-Perold): пол = хай-вотермарк * (1 - dd_budget),
    ратчетится только вверх. Подушка = (капитал - пол)/капитал;
    экспозиция = mult * подушка (cap сверху). Пока стратегия
    зарабатывает — торгует в полный рост (агрессия сохранена);
    в обвале подушка тает и экспозиция ГЕОМЕТРИЧЕСКИ гасится ДО
    нуля у пола — просадка глубже dd_budget математически требует
    гэпа сквозь пол, а не серии убыточных баров.
    """
    core = _donch_pos(bars, entry, exit_period).to_numpy()
    r = bars.close.pct_change().fillna(0.0).to_numpy()
    n = len(r)
    pos = np.zeros(n)
    eq, hwm = 1.0, 1.0
    for i in range(n):
        if i > 0:
            eq *= 1.0 + r[i] * pos[i - 1]     # учёт ПРОШЛОЙ позиции
            hwm = max(hwm, eq)
        floor = hwm * (1.0 - dd_budget)
        cushion = max(0.0, (eq - floor) / eq)
        pos[i] = core[i] * min(cap, mult * cushion)
    return pd.Series(pos, index=bars.index)


# ── 2. Grossman-Zhou ─────────────────────────────────────────────────
def ca2_gz(
    bars: Bars, mom_win: int = 60, dd_max: float = 0.30,
    gamma: float = 1.0,
) -> pd.Series:
    """TSMOM × множитель Гроссмана-Жу: (1 - DD/DDmax)^gamma.

    Механизм: Grossman-Zhou (1993) — оптимальная стратегия при
    жёстком ограничении просадки ЛИНЕЙНО гасит риск по мере роста
    DD от хай-вотермарка и обнуляет его на границе. В отличие от
    CPPI, множитель привязан к ОТНОСИТЕЛЬНОЙ просадке (не к полу
    в деньгах) — восстановление после отскока быстрее.
    """
    core = _mom_pos(bars, mom_win).to_numpy()
    r = bars.close.pct_change().fillna(0.0).to_numpy()
    n = len(r)
    pos = np.zeros(n)
    eq, hwm = 1.0, 1.0
    for i in range(n):
        if i > 0:
            eq *= 1.0 + r[i] * pos[i - 1]
            hwm = max(hwm, eq)
        dd = 1.0 - eq / hwm
        pos[i] = core[i] * max(0.0, 1.0 - dd / dd_max) ** gamma
    return pd.Series(pos, index=bars.index)


# ── 3. BNS jump-гейт ─────────────────────────────────────────────────
def ca2_bns(
    bars: Bars, entry: int = 20, exit_period: int = 10,
    var_win: int = 42, j_max: float = 0.35,
) -> pd.Series:
    """Пробой × гейт скачков: RV против bipower variation.

    Механизм (Barndorff-Nielsen-Shephard): realized variance
    RV = sum r^2 ловит и диффузию, и скачки; bipower
    BV = (pi/2) * sum |r_t||r_{t-1}| робастна к скачкам (скачок
    входит в произведение один раз, не квадратом). J = (RV-BV)/RV —
    доля вариации от СКАЧКОВ. Крипто-кризис = скачковый режим
    (каскады ликвидаций); J > j_max -> флет. Диффузионная вола
    (бычьи ралли тоже волатильны!) гейт НЕ трогает — в этом
    отличие от vol-гейта, глушащего любой рост волы.
    """
    r = bars.close.pct_change()
    rv = (r ** 2).rolling(var_win).sum()
    bv = (np.pi / 2.0) * (r.abs() * r.abs().shift(1)).rolling(
        var_win).sum()
    j = ((rv - bv).clip(lower=0.0) / rv.replace(0.0, np.nan))
    gate = (j < j_max).astype(float).where(~j.isna(), 1.0)
    return _donch_pos(bars, entry, exit_period) * gate


# ── 4. CVaR-сайзинг (EVT-семейство) ──────────────────────────────────
def ca2_evt(
    bars: Bars, mom_win: int = 60, tail_win: int = 500,
    q: float = 0.05, target_cvar: float = 0.02, cap: float = 2.0,
) -> pd.Series:
    """Momentum × размер = target_CVaR / эмпирический CVaR хвоста.

    Механизм: сайзинг не по сигме (вола симметрична и слепа к
    асимметрии обвалов), а по ХВОСТУ: CVaR = средний убыток в
    худших q-процентах trailing-окна. Толстеет левый хвост ->
    позиция сжимается пропорционально, тонкий хвост бычьего рынка ->
    позиция до cap (агрессия). Эмпирический CVaR = непараметрический
    родственник EVT/POT: без предположений о форме хвоста.
    """
    core = _mom_pos(bars, mom_win)
    r = bars.close.pct_change()

    def _cvar(win: np.ndarray) -> float:
        thr = np.nanquantile(win, q)
        tail = win[win <= thr]
        return float(-tail.mean()) if len(tail) else np.nan

    cvar = r.rolling(tail_win, min_periods=tail_win // 2).apply(
        _cvar, raw=True)
    size = (target_cvar / cvar.replace(0.0, np.nan)).clip(
        lower=0.0, upper=cap)
    return (core * size).fillna(0.0)


# ── 5. Полудисперсия ─────────────────────────────────────────────────
def ca2_semi(
    bars: Bars, mom_win: int = 60, span: int = 60,
    target_dvol: float = 0.008, cap: float = 2.0,
) -> pd.Series:
    """Momentum × таргетирование НИСХОДЯЩЕЙ полуволатильности.

    Механизм: сайзер Сортино-типа. Обычный VT делит на полную сигму
    и режет позицию в бурном РОСТЕ (у крипты рост волатилен — теряем
    правый хвост). Полудисперсия считает только min(r,0)^2: бурный
    рост позицию НЕ трогает (агрессия), первые же тяжёлые красные
    бары раздувают downside-сигму и валят размер квадратично быстрее
    обычного VT.
    """
    core = _mom_pos(bars, mom_win)
    r = bars.close.pct_change()
    dvar = (r.clip(upper=0.0) ** 2).ewm(span=span, adjust=False).mean()
    dvol = np.sqrt(dvar)
    size = (target_dvol / dvol.replace(0.0, np.nan)).clip(
        lower=0.0, upper=cap)
    return (core * size).fillna(0.0)


# ── 6. Vol-of-vol гейт ───────────────────────────────────────────────
def ca2_vvol(
    bars: Bars, entry: int = 15, exit_period: int = 8,
    vol_span: int = 36, vv_win: int = 90, rank_window: int = 750,
    pctl: float = 0.90,
) -> pd.Series:
    """Пробой × гейт волатильности ВОЛАТИЛЬНОСТИ.

    Механизм: вторая производная риска. Уровень волы у крипты высок
    всегда и кризис им не отличить; предвестник каскада — когда сама
    вола становится нестабильной (vol-of-vol = std приращений
    log-сигмы). Гейт по trailing-перцентилю vv (аппарат принятого
    vol_percentile_gate, но на порядок выше): режим «вола мечется» ->
    флет, «вола высокая, но стабильная» (бычье ралли) -> торгуем.
    """
    v = _ewma_vol(bars.close, vol_span)
    vv = np.log(v.replace(0.0, np.nan)).diff().rolling(vv_win).std()
    rank = vv.rolling(rank_window, min_periods=rank_window // 3).rank(
        pct=True)
    gate = (rank < pctl).astype(float).where(~rank.isna(), 1.0)
    return _donch_pos(bars, entry, exit_period) * gate


# ── 7. Интенсивность Хоукса ──────────────────────────────────────────
def ca2_hawkes(
    bars: Bars, mom_win: int = 60, vol_span: int = 36,
    shock_sigma: float = 2.5, beta: float = 0.10,
    lam_max: float = 1.5,
) -> pd.Series:
    """Momentum × самовозбуждающаяся интенсивность обвалов (Хоукс).

    Механизм: крипто-обвалы кластеризуются (ликвидации порождают
    ликвидации) — это точечный процесс Хоукса, не Пуассон.
    lambda_t = lambda_{t-1} * exp(-beta) + 1{r_t < -shock_sigma*сигма}:
    каждый шок ВОЗБУЖДАЕТ интенсивность, она затухает с полураспадом
    ln2/beta (~7 баров). lambda > lam_max = «идёт кластер» -> флет
    до затухания. Одиночный шок в спокойном фоне позицию не убивает —
    в отличие от простого стопа по бару.
    """
    core = _mom_pos(bars, mom_win).to_numpy()
    r = bars.close.pct_change()
    shock = (r < -shock_sigma * _ewma_vol(bars.close, vol_span)
             ).fillna(False).to_numpy()
    n = len(shock)
    pos = np.zeros(n)
    lam = 0.0
    decay = float(np.exp(-beta))
    for i in range(n):
        lam = lam * decay + (1.0 if shock[i] else 0.0)
        pos[i] = core[i] if lam <= lam_max else 0.0
    return pd.Series(pos, index=bars.index)


# ── 8. Skew-гейт ─────────────────────────────────────────────────────
def ca2_skew(
    bars: Bars, fast: int = 12, slow: int = 48,
    skew_win: int = 90, skew_min: float = -1.0,
) -> pd.Series:
    """EMA-кросс × гейт rolling-скошенности доходностей.

    Механизм: третий момент как кризис-сенсор. Здоровый крипто-тренд
    даёт skew около нуля/положительный; режим «медленно вверх,
    провалы вниз» (созревание пузыря, дистрибуция) — устойчиво
    отрицательный skew ДО главного обвала. Гейт закрывает лонги,
    когда trailing-скошенность ниже skew_min. Дополняет vol/jump
    гейты: skew видит АСИММЕТРИЮ там, где сигма ещё спокойна.
    """
    ef = bars.close.ewm(span=fast, adjust=False).mean()
    es = bars.close.ewm(span=slow, adjust=False).mean()
    core = (ef > es).astype(float)
    sk = bars.close.pct_change().rolling(skew_win).skew()
    gate = (sk > skew_min).astype(float).where(~sk.isna(), 1.0)
    return (core * gate).fillna(0.0)


# ── 9. Келли с куртозис-штрафом ──────────────────────────────────────
def ca2_kelly_t(
    bars: Bars, window: int = 250, frac: float = 0.3,
    kurt_scale: float = 6.0, cap: float = 2.0,
) -> pd.Series:
    """Дробный Келли, деленный на (1 + избыточный куртозис / scale).

    Механизм: гауссов Келли f = mu/sigma^2 ЗАВЫШЕН при толстых
    хвостах (реальная вероятность разорения выше нормальной).
    Поправка: f / (1 + excess_kurt/scale) — измеряем толщину хвостов
    прямо из данных и штрафуем ставку. Бычий рынок с ровными
    доходностями -> куртозис мал -> ставка до cap (агрессия);
    режим редких гигантских свечей -> ставка сжата ещё ДО того,
    как вырастет сигма.
    """
    r = bars.close.pct_change()
    mu = r.rolling(window).mean()
    var = r.rolling(window).var()
    kurt = r.rolling(window).kurt()          # pandas: избыточный
    f = frac * mu / var.replace(0.0, np.nan)
    f = f / (1.0 + kurt.clip(lower=0.0) / kurt_scale)
    return f.clip(lower=0.0, upper=cap).fillna(0.0)


# ── 10. Circuit breaker ──────────────────────────────────────────────
def ca2_breaker(
    bars: Bars, entry: int = 20, exit_period: int = 8,
    vol_span: int = 36, crash_sigma: float = 3.0,
    cooldown: int = 30,
) -> pd.Series:
    """Дончиан × рубильник: N-сигма бар -> флет + кулдаун + ре-вход.

    Механизм: биржевой circuit breaker, перенесённый в стратегию.
    Один бар хуже -crash_sigma*EWMA-сигмы = аварийное закрытие и
    запрет торговли на cooldown баров (каскад не ловим руками);
    после кулдауна ре-вход ТОЛЬКО по свежему пробою entry-максимума
    (рынок обязан ДОКАЗАТЬ восстановление). Простейшая из десяти
    защит — бенчмарк: если изощрённые аппараты её не бьют,
    Александру честно показываем именно это.
    """
    upper = bars.high.rolling(entry).max().shift(1)
    lower = bars.low.rolling(exit_period).min().shift(1)
    crash = (bars.close.pct_change()
             < -crash_sigma * _ewma_vol(bars.close, vol_span)
             ).fillna(False).to_numpy()
    c = bars.close.to_numpy()
    up, lo = upper.to_numpy(), lower.to_numpy()
    n = len(c)
    pos = np.zeros(n)
    in_pos = False
    cd = 0
    for i in range(n):
        if crash[i]:
            in_pos, cd = False, cooldown
        elif cd > 0:
            cd -= 1
        elif np.isnan(c[i]) or np.isnan(up[i]) or np.isnan(lo[i]):
            pass
        elif in_pos and c[i] < lo[i]:
            in_pos = False
        elif not in_pos and c[i] > up[i]:
            in_pos = True
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


CRYPTO_AGGR_LAB2 = {
    "ca2_cppi": ca2_cppi,
    "ca2_gz": ca2_gz,
    "ca2_bns": ca2_bns,
    "ca2_evt": ca2_evt,
    "ca2_semi": ca2_semi,
    "ca2_vvol": ca2_vvol,
    "ca2_hawkes": ca2_hawkes,
    "ca2_skew": ca2_skew,
    "ca2_kelly_t": ca2_kelly_t,
    "ca2_breaker": ca2_breaker,
}
