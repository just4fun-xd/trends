"""Лаборатория тренда: пакет мат-моделей для пере-выбора чемпиона.

Контекст: champion (Donchian 4step + take-profit) проиграл простому
donchian_vt на walk-forward (60% окон против 80%, средний +0.8% против
+1.6%). Механизм известен: take-profit обрезает правый хвост, где живёт
P&L трендследования. Прежде чем короновать donchian_vt, честно
прогоняем ПАКЕТ альтернативных тренд-моделей через один и тот же
walk-forward — иначе «новый чемпион» будет выбором из одного кандидата.

╔════════════════════════════════════════════════════════════════╗
║ ДИСКЛЕЙМЕР MULTIPLE TESTING (тот же, что в meanrev_lab):        ║
║ 8 моделей на одном периоде — лучшая выглядит хорошо ПО          ║
║ СЛУЧАЙНОСТИ. Отбор: walk-forward стабильность + механизм +      ║
║ стабильность параметров. ЭТО ЛАБОРАТОРИЯ.                       ║
╚════════════════════════════════════════════════════════════════╝

Все сигналы LONG-ONLY (max(signal, 0)): L/S Дончиан закрыт (-71%),
шорт-нога на растущих активах закрыта с четырёхслойным доказательством.
Все сигналы СЫРЫЕ — VT снаружи (--vt / --sizer). Диапазон [0, 1].

Модели и их математика:
  tsmom           — знак 12-мес. доходности (Moskowitz-Ooi-Pedersen
                    2012). КОНТРОЛЬ-БЕЙЗЛАЙН: если Donchian не бьёт
                    голый TSMOM, сложность каналов не оправдана.
  tsmom_multi     — среднее знаков 1/3/12 мес.: диверсификация по
                    горизонту momentum внутри одного сигнала.
  ewmac_forecast  — Carver: (EMA_f − EMA_s) / ценовая вола, клип,
                    среднее по трём парам (8/32, 16/64, 32/128).
                    НЕПРЕРЫВНАЯ сила тренда вместо бинарного входа —
                    информация в глубине сигнала не выбрасывается.
  donchian_multi  — среднее пробойных сигналов 10/20/40/80 (выход =
                    entry/2). Диверсификация по lookback: убирает
                    зависимость от магического «20».
  channel_pos     — непрерывная позиция цены в канале Дончиана:
                    clip((close − mid)/(upper − mid), 0, 1). Плавный
                    вход/выход вместо ступеньки — меньше пилы на
                    границе канала.
  kama_trend      — адаптивная скользящая Кауфмана: сглаживание
                    подстраивается под efficiency ratio (тренд/шум).
                    В боковике MA замирает — меньше ложных кроссов.
  chandelier      — вход по пробою, выход по трейлинг-стопу
                    high_since_entry − k*ATR (Chandelier exit).
                    Правый хвост открыт (нет тейка!), левый обрезан.
  adx_donchian    — пробой Дончиана, разрешённый только при ADX > 20:
                    фильтр СИЛЫ тренда (не направления — направление
                    фильтровать нечем без лага, урок EMA200).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars


# --- Вспомогательная математика ---

def _ewma(s: pd.Series, span: int) -> pd.Series:
    """EMA через pandas ewm (adjust=False — рекурсивная форма)."""
    return s.ewm(span=span, adjust=False).mean()


def _adx(bars: Bars, period: int = 14) -> pd.Series:
    """ADX Уайлдера: сила тренда без направления.

    Args:
        bars: Данные инструмента.
        period: Окно сглаживания Уайлдера.

    Returns:
        Ряд ADX [0, 100].
    """
    high, low = bars.high, bars.low
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(
        np.where((up > dn) & (up > 0), up, 0.0), index=bars.index
    )
    minus_dm = pd.Series(
        np.where((dn > up) & (dn > 0), dn, 0.0), index=bars.index
    )
    tr = bars.true_range()
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    atr = atr.where(atr > 1e-12)
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    denom = (plus_di + minus_di).where(lambda x: x > 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=alpha, adjust=False).mean()


# --- Модели ---

def tsmom(bars: Bars, lookback: int = 252) -> pd.Series:
    """TSMOM-бейзлайн: лонг, если trailing-доходность положительна.

    Args:
        bars: Данные инструмента.
        lookback: Горизонт momentum (252 = 12 мес.).

    Returns:
        position {0, 1} (long-only знак).
    """
    mom = bars.close.pct_change(lookback)
    return (mom > 0).astype(float)


def tsmom_multi(
    bars: Bars, lookbacks: tuple = (21, 63, 252)
) -> pd.Series:
    """Мульти-горизонтный TSMOM: среднее знаков по горизонтам.

    Args:
        bars: Данные инструмента.
        lookbacks: Горизонты momentum в барах.

    Returns:
        position [0, 1]: доля горизонтов в лонге.
    """
    acc = None
    for lb in lookbacks:
        sig = (bars.close.pct_change(lb) > 0).astype(float)
        acc = sig if acc is None else acc + sig
    return acc / float(len(lookbacks))


def ewmac_forecast(
    bars: Bars,
    pairs: tuple = ((8, 32), (16, 64), (32, 128)),
    vol_lookback: int = 30,
    forecast_cap: float = 2.0,
) -> pd.Series:
    """Carver EWMAC: непрерывный прогноз силы тренда, long-only клип.

    forecast = (EMA_fast − EMA_slow) / (std ценовых изменений) — сырой
    momentum, нормированный волой в ценовых единицах (безразмерный,
    сравним между инструментами). Клип и нормировка в [0, 1], среднее
    по парам span'ов (диверсификация горизонтов, FDM-лайт).

    Args:
        bars: Данные инструмента.
        pairs: Пары (fast, slow) span'ов.
        vol_lookback: Окно волы ценовых изменений.
        forecast_cap: Клип нормированного прогноза (в единицах std).

    Returns:
        position [0, 1] — непрерывная сила тренда.
    """
    price_vol = bars.close.diff().rolling(vol_lookback).std()
    price_vol = price_vol.where(price_vol > 1e-12)
    acc = None
    for fast, slow in pairs:
        raw = (_ewma(bars.close, fast) - _ewma(bars.close, slow))
        fc = (raw / price_vol).clip(-forecast_cap, forecast_cap)
        fc = (fc / forecast_cap).clip(lower=0.0)  # long-only [0,1]
        acc = fc if acc is None else acc + fc
    return (acc / float(len(pairs))).fillna(0.0)


def donchian_multi(
    bars: Bars, entries: tuple = (10, 20, 40, 80)
) -> pd.Series:
    """Ансамбль Дончианов по lookback'ам (выход = entry/2).

    Args:
        bars: Данные инструмента.
        entries: Окна верхнего канала.

    Returns:
        position [0, 1]: доля lookback'ов в позиции.
    """
    from strategies.donchian import donchian_breakout
    acc = None
    for e in entries:
        sig = donchian_breakout(bars, entry=e,
                                exit_period=max(e // 2, 5))
        acc = sig if acc is None else acc + sig
    return acc / float(len(entries))


def channel_pos(
    bars: Bars, entry: int = 40, smooth: int = 5,
) -> pd.Series:
    """Непрерывная позиция цены в канале Дончиана.

    pos = clip((close − mid) / (upper − mid), 0, 1), сглаженная EMA.
    У верхней границы — полный лонг, у середины — кэш, между —
    пропорционально. Каналы берутся со сдвигом на 1 бар (пробой
    считается против ВЧЕРАШНЕГО канала, как в базовом Дончиане).

    Args:
        bars: Данные инструмента.
        entry: Окно канала.
        smooth: EMA-сглаживание позиции (гасит дрожание у границы).

    Returns:
        position [0, 1].
    """
    upper = bars.high.rolling(entry).max().shift(1)
    lower = bars.low.rolling(entry).min().shift(1)
    mid = (upper + lower) / 2.0
    width = (upper - mid).where(lambda x: x > 1e-12)
    raw = ((bars.close - mid) / width).clip(0.0, 1.0)
    return _ewma(raw.fillna(0.0), smooth)


def kama_trend(
    bars: Bars, er_period: int = 10, fast: int = 2, slow: int = 30,
) -> pd.Series:
    """Тренд по адаптивной скользящей Кауфмана (KAMA).

    ER = |Δclose за n| / Σ|Δclose| — доля направленного движения.
    Сглаживание интерполируется между fast (чистый тренд) и slow
    (чистый шум): в боковике KAMA замирает, ложных кроссов меньше.
    Лонг, когда close > KAMA и KAMA растёт.

    Args:
        bars: Данные инструмента.
        er_period: Окно efficiency ratio.
        fast: Быстрая константа сглаживания (span).
        slow: Медленная константа сглаживания (span).

    Returns:
        position {0, 1}.
    """
    close = bars.close.to_numpy(dtype=float)
    n = len(close)
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    vol = pd.Series(np.abs(np.diff(close, prepend=close[0])),
                    index=bars.index).rolling(er_period).sum().to_numpy()
    er = np.where(vol > 1e-12, change / vol, 0.0)
    sc_fast = 2.0 / (fast + 1.0)
    sc_slow = 2.0 / (slow + 1.0)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2

    # Рекурсия устойчива к NaN в sc (был баг: одиночный NaN на границе
    # прогрева отравлял ВСЕ последующие значения KAMA -> позиция вечный
    # 0). NaN-бар: держим предыдущее значение, не обновляя.
    kama = np.full(n, np.nan)
    started = False
    for i in range(n):
        if np.isnan(sc[i]):
            if started:
                kama[i] = kama[i - 1]
            continue
        if not started:
            kama[i] = close[i]
            started = True
            continue
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    kama_s = pd.Series(kama, index=bars.index)
    pos = ((bars.close > kama_s) & (kama_s.diff() > 0)).astype(float)
    return pos


def chandelier(
    bars: Bars, entry: int = 20, atr_period: int = 20,
    trail_atr: float = 3.0,
) -> pd.Series:
    """Пробой Дончиана + Chandelier-выход (трейлинг от максимума).

    Выход: close < (max(high) с момента входа) − trail_atr*ATR.
    В отличие от нижнего канала (фикс. окно) стоп монотонно
    подтягивается за трендом и НИКОГДА не фиксирует прибыль сам —
    правый хвост открыт полностью (анти-take-profit по построению).

    Args:
        bars: Данные инструмента.
        entry: Окно пробойного канала.
        atr_period: Окно ATR.
        trail_atr: Множитель трейлинг-стопа.

    Returns:
        position {0, 1}.
    """
    upper = bars.high.rolling(entry).max().shift(1)
    atr = bars.atr(atr_period)
    close, high = bars.close.values, bars.high.values
    up_v, atr_v = upper.values, atr.values

    pos = np.zeros(len(close))
    in_pos = False
    peak = 0.0
    for i in range(len(close)):
        if np.isnan(up_v[i]) or np.isnan(atr_v[i]):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        if in_pos:
            peak = max(peak, high[i])
            if close[i] < peak - trail_atr * atr_v[i]:
                in_pos = False
        if not in_pos and close[i] > up_v[i]:
            in_pos = True
            peak = high[i]
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


def adx_donchian(
    bars: Bars, entry: int = 20, exit_period: int = 10,
    adx_period: int = 14, adx_min: float = 20.0,
) -> pd.Series:
    """Дончиан с фильтром силы тренда: вход только при ADX > adx_min.

    ADX меряет СИЛУ движения без направления — не повторяет закрытый
    EMA200-фильтр (тот угадывал направление и опаздывал). Слабый ADX =
    боковик = пробой скорее ложный. Выход обычный (нижний канал),
    фильтр только на ВХОД — удержание тренда ADX не прерывает.

    Args:
        bars: Данные инструмента.
        entry: Окно верхнего канала.
        exit_period: Окно нижнего канала.
        adx_period: Окно ADX.
        adx_min: Минимальный ADX для входа.

    Returns:
        position {0, 1}.
    """
    upper = bars.high.rolling(entry).max().shift(1)
    lower = bars.low.rolling(exit_period).min().shift(1)
    adx = _adx(bars, adx_period)
    close, low = bars.close.values, bars.low.values
    up_v, lo_v, adx_v = upper.values, lower.values, adx.values

    pos = np.zeros(len(close))
    in_pos = False
    for i in range(len(close)):
        if np.isnan(up_v[i]) or np.isnan(lo_v[i]):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        if in_pos and low[i] < lo_v[i]:
            in_pos = False
        if (not in_pos and close[i] > up_v[i]
                and not np.isnan(adx_v[i]) and adx_v[i] > adx_min):
            in_pos = True
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


TREND_LAB = {
    "tsmom": tsmom,
    "tsmom_multi": tsmom_multi,
    "ewmac": ewmac_forecast,
    "donch_multi": donchian_multi,
    "channel_pos": channel_pos,
    "kama": kama_trend,
    "chandelier": chandelier,
    "adx_donch": adx_donchian,
}
