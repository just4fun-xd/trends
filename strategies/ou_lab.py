"""OU-лаборатория: 10 модификаций OU-реверсии разным матаппаратом.

Контекст (2026-07f): базовый `ou` закрыт на сырье с механизмом
(walk-forward 0/17: слепая реверсия тонет на трендах). Но на крипте-2025
он показал Sharpe +1.05 при LOO-профиле «держатели — альты, балласт —
BNB/ETH». Гипотеза: у OU есть ниша (реверсионный кластер крипты), и
правильная модификация может её расширить. Это ИССЛЕДОВАНИЕ, не
пополнение чемпионов: каждая модификация проходит стандартный конвейер
(walk-forward, два источника, bootstrap против ou-базы, one-shot).

Все 10 — на общей state-machine базового ou_zscore (вход |z|>entry,
выход |z|<exit, стоп |z|>stop), различия — в конструкции z и гейтах:

 1. ou_adf        — торгует только при well_defined ou_fit (ADF-гейт).
 2. ou_halflife   — окно z = f(half-life) инструмента (адаптивное).
 3. ou_log        — z по лог-цене (мультипликативная реверсия).
 4. ou_robust     — z на медиане/MAD (устойчив к выбросам крипты).
 5. ou_hurst_gate — торгует только при VR-H < 0.5 (реверсионный режим).
 6. ou_timestop   — выход по времени = half-life (не ждёт exit_z).
 7. ou_asym       — long-only (для спота/крипты без шорта).
 8. ou_kalman     — z от Kalman local-level (адаптивное среднее).
 9. ou_volgate    — вход только при сжатой воле (растяжение на тонкой
                    воле реверсирует чаще, на всплеске — это тренд).
10. ou_ewma_z     — z на EWMA-среднем/EWMA-std (быстрее реагирует).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.ou import ou_fit
from strategies.variance_ratio import rolling_hurst_vr


def _state_machine(
    z: np.ndarray, entry: float, exit_: float, stop: float,
    long_only: bool = False, max_hold: int | None = None,
) -> np.ndarray:
    """Общая машина состояний OU (вход/выход/стоп/тайм-стоп)."""
    pos = np.zeros(len(z))
    state, held = 0, 0
    for i in range(len(z)):
        if np.isnan(z[i]):
            pos[i] = float(state)
            continue
        if state == 0:
            if z[i] < -entry:
                state, held = 1, 0
            elif z[i] > entry and not long_only:
                state, held = -1, 0
        elif state == 1:
            held += 1
            if (z[i] > -exit_ or z[i] < -stop
                    or (max_hold and held >= max_hold)):
                state = 0
        else:  # state == -1
            held += 1
            if (z[i] < exit_ or z[i] > stop
                    or (max_hold and held >= max_hold)):
                state = 0
        pos[i] = float(state)
    return pos


def _z_classic(close: pd.Series, window: int) -> pd.Series:
    """Классический rolling z-score (mean/std), сдвиг внутри rolling."""
    mu = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (close - mu) / sd.replace(0.0, np.nan)


def _series(bars: Bars, z: pd.Series, entry=2.0, exit_=0.5, stop=4.0,
            long_only=False, max_hold=None) -> pd.Series:
    arr = _state_machine(z.to_numpy(dtype=float), entry, exit_, stop,
                         long_only, max_hold)
    return pd.Series(arr, index=bars.close.index)


# ── 1. ADF-гейт ────────────────────────────────────────────────────────
def ou_adf(bars: Bars, window: int = 20, fit_win: int = 252) -> pd.Series:
    """OU только там, где ou_fit подтверждает стационарность (ADF).

    Раз в месяц ou_fit на trailing fit_win; торговля разрешена только
    при well_defined=True. Механизм: главный провал базового ou —
    реверсия на нестационарном (трендовом) ряде; ADF-гейт отрезает это
    формальным тестом, а не эвристикой.
    """
    z = _z_classic(bars.close, window)
    ok = pd.Series(False, index=bars.close.index)
    last = False
    for i in range(len(bars.close)):
        if i % 21 == 0 and i >= fit_win:
            fit = ou_fit(bars.close.iloc[i - fit_win:i])
            last = bool(fit["well_defined"])
        ok.iloc[i] = last
    return _series(bars, z.where(ok.shift(1).fillna(False)))


# ── 2. Адаптивное окно по half-life ───────────────────────────────────
def ou_halflife(bars: Bars, fit_win: int = 252) -> pd.Series:
    """Окно z-score = clip(half-life, 5, 60) инструмента.

    Механизм: фиксированное окно 20 навязывает всем один темп реверсии;
    настоящий темп даёт θ (half_life=ln2/θ). Пересчёт раз в месяц.
    """
    close = bars.close
    win = pd.Series(20.0, index=close.index)
    last = 20.0
    for i in range(len(close)):
        if i % 21 == 0 and i >= fit_win:
            fit = ou_fit(close.iloc[i - fit_win:i])
            hl = fit["half_life"]
            if fit["well_defined"] and np.isfinite(hl):
                last = float(np.clip(hl, 5, 60))
        win.iloc[i] = last
    win = win.shift(1).fillna(20.0)
    # Пересобираем z с переменным окном (по уникальным значениям).
    z = pd.Series(np.nan, index=close.index)
    for w in win.unique():
        mask = win == w
        zw = _z_classic(close, int(w))
        z[mask] = zw[mask]
    return _series(bars, z)


# ── 3. Лог-цена ───────────────────────────────────────────────────────
def ou_log(bars: Bars, window: int = 20) -> pd.Series:
    """z по лог-цене: реверсия мультипликативная, не аддитивная.

    Механизм: на активах с большим диапазоном (крипта: цена x10 за
    период) аддитивный z раздувается масштабом уровня; лог выравнивает.
    """
    # Защита от неположительных цен (roll-adjusted ряды могут уходить
    # <=0): log даёт NaN/предупреждение. Клипуем к малому положительному.
    safe = bars.close.where(bars.close > 0)
    return _series(bars, _z_classic(np.log(safe), window))


# ── 4. Робастный z (медиана/MAD) ──────────────────────────────────────
def ou_robust(bars: Bars, window: int = 20) -> pd.Series:
    """z = (close − median)/(1.4826·MAD): устойчив к выбросам.

    Механизм: у крипты хвосты тяжёлые; выброс в окне раздувает std и
    глушит классический z. MAD не реагирует на единичный выброс.
    """
    med = bars.close.rolling(window).median()
    mad = (bars.close - med).abs().rolling(window).median()
    z = (bars.close - med) / (1.4826 * mad.replace(0.0, np.nan))
    return _series(bars, z)


# ── 5. Hurst-гейт ─────────────────────────────────────────────────────
def ou_hurst_gate(bars: Bars, window: int = 20,
                  h_max: float = 0.5) -> pd.Series:
    """OU только при VR-H < h_max (инструмент в реверсионном режиме).

    Механизм: соединяет закрытие ou («слепая реверсия») с картой
    2026-07f — реверсию торгуем только там, где ряд антиперсистентен.
    """
    h = rolling_hurst_vr(bars.close, window=504)
    z = _z_classic(bars.close, window)
    return _series(bars, z.where(h < h_max))


# ── 6. Тайм-стоп по half-life ─────────────────────────────────────────
def ou_timestop(bars: Bars, window: int = 20,
                fit_win: int = 252) -> pd.Series:
    """Выход по времени = median half-life: возврат не пришёл — выходим.

    Механизм: OU-теория даёт ожидаемое время возврата; если позиция
    висит дольше half-life, гипотеза возврата опровергнута данными.
    """
    fit = ou_fit(bars.close.iloc[:fit_win]) if len(bars.close) >= fit_win \
        else {"half_life": np.nan, "well_defined": False}
    hl = fit["half_life"]
    max_hold = int(np.clip(hl, 5, 40)) if np.isfinite(hl) else 20
    z = _z_classic(bars.close, window)
    return _series(bars, z, max_hold=max_hold)


# ── 7. Long-only ──────────────────────────────────────────────────────
def ou_asym(bars: Bars, window: int = 20) -> pd.Series:
    """Только лонг перепроданности (для спота/крипты без шорта).

    Механизм: урок equity-шорт-ноги — на структурно растущих активах
    шорт-половина реверсии несёт бету против себя. Отрезаем её.
    """
    return _series(bars, _z_classic(bars.close, window), long_only=True)


# ── 8. Kalman local-level ─────────────────────────────────────────────
def ou_kalman(bars: Bars, q_ratio: float = 1e-4) -> pd.Series:
    """z от Kalman local-level: адаптивное среднее вместо rolling.

    x_t = x_{t-1} + w (уровень),  y_t = x_t + v (наблюдение).
    z = инновация/√S. Механизм: rolling-mean отстаёт на полокна и
    даёт ложные растяжения после сдвига уровня; Калман перецентрируется
    со скоростью, диктуемой данными (q_ratio = q/r).
    """
    y = bars.close.to_numpy(dtype=float)
    n = len(y)
    z = np.full(n, np.nan)
    # Инициализация по первым наблюдениям.
    x, p = y[0], 1.0
    r = np.nanvar(np.diff(y[:100])) if n > 100 else 1.0
    r = max(r, 1e-12)
    q = q_ratio * r
    for t in range(1, n):
        p = p + q
        s = p + r
        innov = y[t] - x
        z[t] = innov / np.sqrt(s)
        k = p / s
        x = x + k * innov
        p = (1 - k) * p
    # Kalman-z уже стандартизован; порог тот же контракт state-machine.
    return _series(bars, pd.Series(z, index=bars.close.index))


# ── 9. Vol-гейт ───────────────────────────────────────────────────────
def ou_volgate(bars: Bars, window: int = 20,
               vol_pct: float = 0.7) -> pd.Series:
    """Вход только при воле ниже перцентиля: тихое растяжение реверсирует.

    Механизм: |z|>2 при всплеске волы — это чаще пробой (тренд), а не
    растяжение; тот же |z| на сжатой воле — растяжение внутри рейнджа.
    Зеркален vol_percentile_gate тренда: тренд берёт всплеск, OU — тишь.
    """
    ret = bars.close.pct_change()
    vol = ret.rolling(window).std()
    rank = vol.rolling(252).rank(pct=True)
    z = _z_classic(bars.close, window)
    return _series(bars, z.where(rank < vol_pct))


# ── 10. EWMA-z ────────────────────────────────────────────────────────
def ou_ewma_z(bars: Bars, span: int = 20) -> pd.Series:
    """z на EWMA-среднем и EWMA-std: быстрее видит смену уровня.

    Механизм: равновесное окно даёт всем барам равный вес — среднее
    тащит хвост истории. EWMA взвешивает свежее, реверсия меряется к
    актуальному центру. Компромисс между rolling и Калманом.
    """
    mu = bars.close.ewm(span=span, adjust=False).mean()
    var = ((bars.close - mu) ** 2).ewm(span=span, adjust=False).mean()
    z = (bars.close - mu) / np.sqrt(var).replace(0.0, np.nan)
    return _series(bars, z)


OU_LAB = {
    "ou_adf": ou_adf,
    "ou_halflife": ou_halflife,
    "ou_log": ou_log,
    "ou_robust": ou_robust,
    "ou_hurst_gate": ou_hurst_gate,
    "ou_timestop": ou_timestop,
    "ou_asym": ou_asym,
    "ou_kalman": ou_kalman,
    "ou_volgate": ou_volgate,
    "ou_ewma_z": ou_ewma_z,
}


# ── 11. Jump-diffusion aware (ответ Александра) ───────────────────────
def _detect_jumps(close: pd.Series, window: int = 40,
                  k: float = 4.0) -> pd.Series:
    """Булев ряд: True там, где дневное движение — СКАЧОК, не диффузия.

    Механизм (Мертон/Ли-Мыкланд): jump-diffusion раскладывает движение
    на диффузию (σ√dt) и редкие скачки (пуассон). Скачок опознаётся
    как приращение, чей модуль сильно превышает локальную диффузионную
    шкалу: |r_t| > k · σ_local. Такое движение — смена УРОВНЯ (μ
    сдвинулся), а не растяжение вокруг μ.

    Args:
        close: Цены.
        window: Окно оценки локальной диффузионной σ.
        k: Порог в единицах σ (4 ≈ хвост, редкое событие).

    Returns:
        Булев ряд: True = бар содержит скачок (сдвинут на 1: скачок
        виден только ПОСЛЕ закрытия бара).
    """
    safe = close.where(close > 0)
    ret = np.log(safe / safe.shift(1))
    # Локальная σ по MAD (устойчива к самим скачкам — не раздувается
    # ими, в отличие от std).
    sigma = (ret.rolling(window).apply(
        lambda x: np.median(np.abs(x - np.median(x))) * 1.4826,
        raw=True))
    is_jump = ret.abs() > (k * sigma)
    return is_jump.fillna(False).shift(1).fillna(False)


def ou_jump(bars: Bars, window: int = 20, jump_win: int = 40,
            jump_k: float = 4.0, cooldown: int = 5) -> pd.Series:
    """OU-реверсия, подавленная при скачках (jump-diffusion фильтр).

    Прямая реализация замечания Александра: «OU can work if you add
    jump-diffusion components». Чистый OU тонет, потому что торгует
    возврат ВСЛЕПУЮ — на скачке (смена уровня) он шортит растущее /
    ловит падающий нож. Здесь скачок детектируется (|r| > k·σ_local) и
    на `cooldown` баров реверсия ОТКЛЮЧАЕТСЯ: новый уровень должен
    устояться, μ пересчитаться, прежде чем снова ставить на возврат.

    Механизм фильтра:
      - диффузионное растяжение (|z|>entry, скачка нет) -> торгуем
        возврат, как классический OU;
      - скачок (level-shift) -> НЕ входим, а если в позиции — выходим
        (гипотеза возврата опровергнута сменой уровня).

    Это отделяет «temporary deviation» (ниша OU) от «permanent
    level-shift» (то, что его убивало). Ожидание: убирает катастрофы
    вроде Solana -99% / Crude -106% из ou/ou_kalman.

    Args:
        bars: Данные инструмента.
        window: Окно z-score.
        jump_win: Окно локальной σ для детекции скачка.
        jump_k: Порог скачка в σ.
        cooldown: Сколько баров глушить реверсию после скачка.

    Returns:
        position +1/-1/0.
    """
    z = _z_classic(bars.close, window).to_numpy(dtype=float)
    jump = _detect_jumps(bars.close, jump_win, jump_k).to_numpy()
    n = len(z)
    pos = np.zeros(n)
    state, cool = 0, 0
    for i in range(n):
        if jump[i]:
            cool = cooldown      # скачок -> глушим реверсию
            state = 0            # и выходим, если были в позиции
        if cool > 0:
            cool -= 1
            pos[i] = float(state)
            continue
        if np.isnan(z[i]):
            pos[i] = float(state)
            continue
        if state == 0:
            if z[i] < -2.0:
                state = 1
            elif z[i] > 2.0:
                state = -1
        elif state == 1:
            if z[i] > -0.5 or z[i] < -4.0:
                state = 0
        else:
            if z[i] < 0.5 or z[i] > 4.0:
                state = 0
        pos[i] = float(state)
    return pd.Series(pos, index=bars.close.index)


def ou_jump_asym(bars: Bars, window: int = 20, jump_win: int = 40,
                 jump_k: float = 4.0, cooldown: int = 5) -> pd.Series:
    """ou_jump, но long-only: jump-фильтр + без шорт-ноги.

    Комбинирует два рабочих механизма: подавление реверсии на скачках
    (ou_jump) и отказ от шорта на структурно растущих активах
    (ou_asym). Для крипты/акций, где обе проблемы бьют одновременно.
    """
    full = ou_jump(bars, window, jump_win, jump_k, cooldown)
    return full.clip(lower=0.0)


OU_LAB["ou_jump"] = ou_jump
OU_LAB["ou_jump_asym"] = ou_jump_asym
