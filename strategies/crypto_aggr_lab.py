"""Крипто-лаборатория AGGR: 10 агрессивных моделей для H4-крипты.

Запрос 2026-07j: стратегии, которые работают «супер эффективно и
агрессивно» на крипте. Агрессия здесь — это НЕ плечо (плечо — дело
VT и портфельного слоя; VT>2-3x = бумажная доходность): агрессия в
СИГНАЛЕ: быстрые входы, пирамидинг, удержание импульса, шорт-нога
(перпетуалы позволяют, и крипто-обвалы быстрее ралли), реакция на
расширение волатильности вместо её избегания.

Уроки, встроенные в дизайн:
- Monday-range: на буйном тренде выигрывает тот, кто ДЕРЖИТ импульс.
- Take-profit режет правый хвост — тейков нет, только трейлинги.
- Крипта живёт с волой: анти-lowvol модели (ca_vol_ride) — гипотеза,
  что расширение волы у крипты сопровождает тренд, а не коллапс
  (на сырье наоборот, там vol_percentile_gate защищает).

╔════════════════════════════════════════════════════════════════╗
║ MULTIPLE TESTING: 10 моделей. Скрининг walk-forward на H4 ->   ║
║ bootstrap выживших против donchian_vt (+1.53 gross, честный    ║
║ bpy!) -> кандидат в ансамбль при corr < 0.6 с donchian и       ║
║ tr_ichimoku. Пирамидные позиции >1 — проверять DD-мандат.      ║
╚════════════════════════════════════════════════════════════════╝

Контракт: Bars -> position. Диапазоны шире обычного: [0, 2] у
пирамид, [-1, 1] у long/short. Сдвига внутри НЕТ — движок сдвигает.
VT снаружи (--vt): помнить, что VT нормирует ВОЛУ, поэтому позиция
2.0 после VT — это удвоенный риск-бюджет, следить за плечом.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars


def _ewma_vol(close: pd.Series, span: int = 36) -> pd.Series:
    """EWMA-волатильность побарных доходностей."""
    return close.pct_change().ewm(span=span, adjust=False).std()


# ── 1. Турбо-Дончиан с пирамидой ─────────────────────────────────────
def ca_turbo_don(
    bars: Bars, entry: int = 10, exit_period: int = 5,
    add_every: int = 5, max_units: int = 3,
) -> pd.Series:
    """Быстрый Дончиан 10/5 + докупка на каждом новом максимуме.

    Механизм: крипто-импульсы на H4 короче и круче сырьевых —
    стандартные окна 20/10 опаздывают. Вход на пробое 10-барного
    максимума, докупка +0.5 на каждом новом entry-максимуме не чаще
    add_every баров (до 2.0), выход всей пирамиды по пробою
    5-барного минимума. Пирамида в тренд = агрессия по Ливермору.
    """
    upper = bars.high.rolling(entry).max().shift(1)
    lower = bars.low.rolling(exit_period).min().shift(1)
    c = bars.close.to_numpy()
    up = upper.to_numpy()
    lo = lower.to_numpy()
    n = len(c)
    pos = np.zeros(n)
    units = 0.0
    cooldown = 0
    for i in range(n):
        if np.isnan(up[i]) or np.isnan(lo[i]) or np.isnan(c[i]):
            continue
        if units > 0 and c[i] < lo[i]:
            units = 0.0
        elif c[i] > up[i] and cooldown == 0 and units < 1 + 0.5 * (
                max_units - 1):
            units = 1.0 if units == 0 else units + 0.5
            cooldown = add_every
        if cooldown > 0:
            cooldown -= 1
        pos[i] = units
    return pd.Series(pos, index=bars.index)


# ── 2. Squeeze pop ───────────────────────────────────────────────────
def ca_squeeze_pop(
    bars: Bars, bb_win: int = 20, bb_std: float = 2.0,
    kelt_mult: float = 1.5, hold: int = 18,
) -> pd.Series:
    """Полосы Боллинджера внутри Кельтнера -> пробой в сторону выхода.

    Механизм: TTM Squeeze: BB уже Кельтнера = сжатие волы (энергия
    накоплена). Направление ПЕРВОГО выхода цены за BB после сжатия —
    сторона разрядки; крипта разряжается резко. Двусторонняя (шорт
    на пробое вниз — обвалы крипты быстрее ралли). Тайм-выход hold
    баров (~3 дня H4): торгуем разрядку, не строим тренд-систему.

    Отличие от закрытого Bollinger squeeze (сырьё, дневки): там
    хрупкость параметров на медленных барах; здесь другой класс
    актива, другой ТФ и симметричный L/S — отдельная гипотеза.
    """
    ma = bars.close.rolling(bb_win).mean()
    sd = bars.close.rolling(bb_win).std()
    bb_up, bb_dn = ma + bb_std * sd, ma - bb_std * sd
    atr = bars.atr(bb_win)
    in_squeeze = ((bb_up < ma + kelt_mult * atr)
                  & (bb_dn > ma - kelt_mult * atr))
    was_squeeze = in_squeeze.shift(1).fillna(False)
    br_up = (bars.close > bb_up) & was_squeeze
    br_dn = (bars.close < bb_dn) & was_squeeze
    up_t = br_up.to_numpy()
    dn_t = br_dn.to_numpy()
    pos = np.zeros(len(up_t))
    left, side = 0, 0.0
    for i in range(len(up_t)):
        if up_t[i]:
            left, side = hold, 1.0
        elif dn_t[i]:
            left, side = hold, -1.0
        if left > 0:
            pos[i] = side
            left -= 1
    return pd.Series(pos, index=bars.index)


# ── 3. Ускорение импульса ────────────────────────────────────────────
def ca_accel(
    bars: Bars, fast: int = 12, slow: int = 48, accel_win: int = 6,
) -> pd.Series:
    """EWMAC растёт И его производная растёт: вторая производная цены.

    Механизм: позиция только в фазе РАЗГОНА (momentum + ускорение
    одного знака), сброс на замедлении — до разворота цены. Урок
    Trend Lab 1/2 (ускорение ~0 на ровных трендах сырья) здесь
    переворачивается в гипотезу: крипто-импульсы паработичны, фаза
    разгона содержит основную доходность. Размер = tanh силы.
    """
    ewmac = (bars.close.ewm(span=fast, adjust=False).mean()
             - bars.close.ewm(span=slow, adjust=False).mean())
    vol = _ewma_vol(bars.close) * bars.close
    strength = ewmac / vol.where(vol > 1e-12)
    accel = strength - strength.shift(accel_win)
    sig = np.tanh(strength.clip(lower=0.0)) * (accel > 0).astype(float)
    return pd.Series(sig, index=bars.index).clip(0.0, 1.0).fillna(0.0)


# ── 4. RSI-thrust ────────────────────────────────────────────────────
def ca_thrust(
    bars: Bars, period: int = 14, up_lvl: float = 62.0,
    dn_lvl: float = 45.0, vol_span: int = 36,
) -> pd.Series:
    """RSI пробивает 62 снизу при растущей воле -> держать до <45.

    Механизм: momentum-прочтение RSI (не реверсионное): выход RSI в
    верхнюю зону при РАСШИРЯЮЩЕЙСЯ воле — толчок (thrust) с топливом.
    Держим, пока импульс жив (RSI > 45), без ценового стопа: сам
    RSI-уровень и есть трейлинг по силе движения.
    """
    delta = bars.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.where(loss > 1e-12))
    vol = _ewma_vol(bars.close, vol_span)
    vol_rising = vol > vol.shift(vol_span // 2)
    cross_up = ((rsi > up_lvl) & (rsi.shift(1) <= up_lvl)
                & vol_rising)
    return _latch(cross_up, rsi < dn_lvl, bars.index)


def _latch(
    enter: pd.Series, leave: pd.Series, index: pd.Index,
) -> pd.Series:
    """Защёлка 0/1: вход по enter, сброс по leave."""
    ent = np.asarray(enter.fillna(False))
    lev = np.asarray(leave.fillna(False))
    pos = np.zeros(len(ent))
    on = False
    for i in range(len(ent)):
        if on and lev[i]:
            on = False
        if not on and ent[i] and not lev[i]:
            on = True
        pos[i] = 1.0 if on else 0.0
    return pd.Series(pos, index=index)


# ── 5. Chandelier re-entry ───────────────────────────────────────────
def ca_chand(
    bars: Bars, fast: int = 9, slow: int = 26,
    atr_mult: float = 3.0, atr_win: int = 22,
) -> pd.Series:
    """EMA-кросс вход + Chandelier ATR-трейлинг + мгновенный re-entry.

    Механизм: Chandelier: стоп = max(high, N) - mult*ATR, только
    подтягивается. Агрессия — в БЫСТРОМ ПЕРЕВХОДЕ: после выбивания
    стопом позиция восстанавливается, как только цена делает новый
    локальный максимум при живом EMA-кроссе. Обычные системы после
    стопа ждут нового кросса и пропускают продолжение — здесь
    выбивание считается шумом, пока структура тренда цела.
    """
    ema_f = bars.close.ewm(span=fast, adjust=False).mean()
    ema_s = bars.close.ewm(span=slow, adjust=False).mean()
    bull = (ema_f > ema_s).to_numpy()
    atr = bars.atr(atr_win).to_numpy()
    hi_n = bars.high.rolling(atr_win).max().to_numpy()
    hi_prev = bars.high.rolling(atr_win).max().shift(1).to_numpy()
    c = bars.close.to_numpy()
    h = bars.high.to_numpy()
    n = len(c)
    pos = np.zeros(n)
    in_pos = False
    stop = np.nan
    for i in range(n):
        if np.isnan(c[i]) or np.isnan(atr[i]) or np.isnan(hi_n[i]):
            continue
        if in_pos:
            stop = max(stop, hi_n[i] - atr_mult * atr[i])
            if c[i] < stop or not bull[i]:
                in_pos = False
        else:
            new_high = (not np.isnan(hi_prev[i])) and h[i] > hi_prev[i]
            if bull[i] and new_high:
                in_pos = True
                stop = hi_n[i] - atr_mult * atr[i]
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


# ── 6. Двойной таймфрейм ─────────────────────────────────────────────
def ca_double_don(
    bars: Bars, fast_entry: int = 10, slow_entry: int = 40,
    exit_period: int = 8,
) -> pd.Series:
    """Пробой быстрого канала ТОЛЬКО при пробитом медленном.

    Механизм: консенсус масштабов: медленный канал (40 баров H4 ~
    неделя) задаёт режим, быстрый (10) — точку входа. Вход только
    когда оба «за» отсекает контртрендовые быстрые пробои, выход
    по быстрому минимуму (8) сохраняет реактивность. Идея
    родственна системе Turtle S1/S2, сжатой под H4.
    """
    up_f = bars.high.rolling(fast_entry).max().shift(1)
    up_s = bars.high.rolling(slow_entry).max().shift(1)
    lo_x = bars.low.rolling(exit_period).min().shift(1)
    near_slow = bars.close > up_s * 0.995
    enter = (bars.close > up_f) & near_slow
    leave = bars.close < lo_x
    return _latch(enter, leave, bars.index)


# ── 7. Burst ─────────────────────────────────────────────────────────
def ca_burst(
    bars: Bars, atr_win: int = 20, k: float = 2.0, hold: int = 8,
) -> pd.Series:
    """Бар размером > k*ATR -> ехать в его сторону hold баров. L/S.

    Механизм: ликвидационные каскады: аномальный H4-бар на крипте —
    каскад стоп-ликвидаций, движение продолжается по инерции ещё
    несколько баров (позиции докрывают). Симметрично в обе стороны,
    жёсткий тайм-выход. Зеркало mr2_shock: там ставка на возврат
    дневного шока сырья, тут — на продолжение крипто-каскада.
    Пусть данные рассудят, какой хвост у кого.
    """
    move = bars.close - bars.close.shift(1)
    atr = bars.atr(atr_win).shift(1)
    up_t = (move > k * atr).to_numpy()
    dn_t = (move < -k * atr).to_numpy()
    pos = np.zeros(len(up_t))
    left, side = 0, 0.0
    for i in range(len(up_t)):
        if up_t[i]:
            left, side = hold, 1.0
        elif dn_t[i]:
            left, side = hold, -1.0
        if left > 0:
            pos[i] = side
            left -= 1
    return pd.Series(pos, index=bars.index)


# ── 8. Vol-ride ──────────────────────────────────────────────────────
def ca_vol_ride(
    bars: Bars, vol_span: int = 36, vol_ratio: float = 1.2,
    ma_win: int = 30,
) -> pd.Series:
    """Анти-lowvol: лонг при РАСТУЩЕЙ воле и цене над MA.

    Механизм: прямая инверсия mr_lowvol (чемпион сырьевого MR):
    на сырье низкая вола = безопасный range, на крипте расширение
    волы исторически сопровождает импульсные ралли (и обвалы —
    отсекаются условием цены над MA). Если гипотеза верна, corr с
    donchian_vt будет умеренной (другой триггер), а с mr-ногой —
    отрицательной: идеальный профиль для ансамбля.
    """
    vol = _ewma_vol(bars.close, vol_span)
    vol_base = vol.rolling(vol_span * 2).median()
    expanding = vol > vol_ratio * vol_base
    above = bars.close > bars.close.rolling(ma_win).mean()
    return (expanding & above).astype(float).fillna(0.0)


# ── 9. Шорт-пробой ───────────────────────────────────────────────────
def ca_short_break(
    bars: Bars, entry: int = 12, exit_period: int = 6,
) -> pd.Series:
    """Зеркальный Дончиан ВНИЗ: шорт на пробое минимума. [-1, 0].

    Механизм: закрытие equity-шорта не переносится на крипту:
    у акций структурный дрейф вверх съедает шорт-альфу, у крипты
    обвалы быстрее и глубже ралли (ликвидационные каскады), а
    перпетуалы делают шорт дешёвым. Отдельная гипотеза для отдельного
    класса. Если и тут шорт мёртв — закрываем шорты и на крипте, с
    механизмом.
    """
    lower = bars.low.rolling(entry).min().shift(1)
    upper = bars.high.rolling(exit_period).max().shift(1)
    enter = bars.close < lower
    leave = bars.close > upper
    return -_latch(enter, leave, bars.index)


# ── 10. Пирамида на максимумах ───────────────────────────────────────
def ca_pyramid_max(
    bars: Bars, entry: int = 20, step_atr: float = 1.0,
    stop_atr: float = 2.5, max_units: int = 4,
) -> pd.Series:
    """Докупка +0.5 на каждом шаге step_atr*ATR выше входа, стоп ATR.

    Механизм: анти-усреднение: добавляем ТОЛЬКО в прибыльную позицию
    (усиление победителя), шаги в ATR — темп докупок дышит с волой.
    Стоп пирамиды — chandelier: close - stop_atr*ATR, только
    подтягивается (прибыль ранних юнитов защищает поздние).
    Профиль: редкие большие выигрыши / частые нули — правый хвост,
    где живёт трендовый P&L.
    """
    upper = bars.high.rolling(entry).max().shift(1)
    up = upper.to_numpy()
    c = bars.close.to_numpy()
    atr = bars.atr(20).to_numpy()
    n = len(c)
    pos = np.zeros(n)
    units = 0.0
    anchor = np.nan
    stop = np.nan
    for i in range(n):
        if np.isnan(c[i]) or np.isnan(atr[i]):
            continue
        if units > 0:
            # Chandelier-трейлинг: стоп подтягивается за ценой и без
            # докупок, иначе в range после тренда позиция висит вечно.
            stop = max(stop, c[i] - stop_atr * atr[i])
            if c[i] < stop:
                units, anchor, stop = 0.0, np.nan, np.nan
            elif (units < 1 + 0.5 * (max_units - 1)
                    and c[i] > anchor + step_atr * atr[i]):
                units += 0.5
                anchor = c[i]
        elif not np.isnan(up[i]) and c[i] > up[i]:
            units = 1.0
            anchor = c[i]
            stop = anchor - stop_atr * atr[i]
        pos[i] = units
    return pd.Series(pos, index=bars.index)


CRYPTO_AGGR_LAB = {
    "ca_turbo_don": ca_turbo_don,
    "ca_squeeze_pop": ca_squeeze_pop,
    "ca_accel": ca_accel,
    "ca_thrust": ca_thrust,
    "ca_chand": ca_chand,
    "ca_double_don": ca_double_don,
    "ca_burst": ca_burst,
    "ca_vol_ride": ca_vol_ride,
    "ca_short_break": ca_short_break,
    "ca_pyramid_max": ca_pyramid_max,
}
