"""Продвинутый мат-аппарат: Carver FDM и Hurst-аллокатор.

Оба метода бьют в проблему «тонкой доходности» (вердикт Александра:
3.4%/год < облигаций) БЕЗ плеча — через использование риск-бюджета и
распределение риска туда, где у инструмента статистическое свойство.

═══ 1. carver_fdm — непрерывные прогнозы + FDM ═══

Механизм тонкой доходности №1: бинарные сигналы держат портфель в
кэше большую часть времени (MR ~15% в рынке, тренд ~40%). Carver:
  forecast_i = raw_signal_i / vol,  клип [−cap, +cap], нормировка
  combined = FDM × Σ w_i × forecast_i
  FDM = 1 / sqrt(w' Ρ w)   (Ρ — корреляция прогнозов, trailing)
FDM > 1 по построению при корреляции < 1: комбинация прогнозов имеет
меньшую дисперсию, чем одиночный, и БЕЗ FDM комбинированная позиция
систематически недоиспользует риск-бюджет. FDM возвращает масштаб —
это легитимный «подъём позиции», не займ. Кэп FDM = 2.5 (Carver).

Члены: ewmac (3 пары), kama-прогноз, donch_multi — все двухисточниковые
после реабилитации 2026-07e. Look-ahead нет: Ρ оценивается на trailing
окне со сдвигом 1 бар.

═══ 2. hurst_combo — аллокатор ног по показателю Хёрста ═══

Механизм тонкой доходности №2: обе ноги (тренд+MR) размазаны на все
инструменты поровну, хотя Gold и NG — статистически разные животные.
Показатель Хёрста H различает:
  H > 0.5 — персистентность (тренды продолжаются) -> вес тренд-ноге
  H < 0.5 — антиперсистентность (реверсия)        -> вес MR-ноге
Оценка H — агрегированная дисперсия (устойчивее классического R/S на
коротких окнах): Var[сумма q баров] ~ q^(2H) => H из наклона
log-log регрессии Var(q) по q. Плавное отображение веса:
  w_trend = clip((H − 0.4) / 0.2, 0, 1)   (H=0.4 -> чистый MR,
  H=0.6 -> чистый тренд, между — микс).
Это НЕ закрытый HMM: H — свойство ИНСТРУМЕНТА на окне, не «режим
времени» с лагом переключения; веса непрерывны, переключений нет.
Look-ahead нет: H считается на trailing окне со сдвигом 1 бар,
пересчёт раз в 21 бар (месяц) — H медленная характеристика, частый
пересчёт даёт только шум и оборот.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.ensemble import mr_keltner_confirm
from strategies.trend_lab import _ewma


# ─────────────────────────── Carver FDM ───────────────────────────

def _ewmac_raw_forecast(close: pd.Series, fast: int, slow: int,
                        vol_lookback: int = 30) -> pd.Series:
    """Сырой EWMAC-прогноз: (EMAf − EMAs) / ценовая вола."""
    price_vol = close.diff().rolling(vol_lookback).std()
    price_vol = price_vol.where(price_vol > 1e-12)
    return (_ewma(close, fast) - _ewma(close, slow)) / price_vol


def _kama_forecast(bars: Bars, er_period: int = 10) -> pd.Series:
    """KAMA-подобный прогноз: efficiency ratio со знаком тренда.

    ER (0..1, доля направленного движения) × знак движения — непрерывная
    сила тренда без рекурсии KAMA (для прогноза достаточно ER).
    """
    close = bars.close
    change = close.diff(er_period)
    vol = close.diff().abs().rolling(er_period).sum()
    er = (change.abs() / vol.where(vol > 1e-12)).fillna(0.0)
    return er * np.sign(change.fillna(0.0)) * 10.0  # масштаб к EWMAC


def _donch_pos_forecast(bars: Bars, entry: int = 40) -> pd.Series:
    """Непрерывный Дончиан-прогноз: позиция в канале, центрированная.

    (close − mid)/(upper − mid) в [−1, 1] × 10 — сопоставимый масштаб.
    """
    upper = bars.high.rolling(entry).max().shift(1)
    lower = bars.low.rolling(entry).min().shift(1)
    mid = (upper + lower) / 2.0
    width = (upper - mid).where(lambda x: x > 1e-12)
    return (((bars.close - mid) / width).clip(-1.0, 1.0) * 10.0)


def carver_fdm(
    bars: Bars,
    forecast_cap: float = 20.0,
    fdm_cap: float = 2.5,
    corr_window: int = 252,
    long_only: bool = True,
) -> pd.Series:
    """Комбинированный непрерывный прогноз Carver с FDM.

    Члены: EWMAC 8/32, 16/64, 32/128 + KAMA-ER + Donchian-позиция.
    Каждый прогноз нормируется к abs-среднему 10 (скейлинг Carver),
    клип ±forecast_cap, равновзвешенная комбинация × FDM,
    FDM = 1/sqrt(w'Ρw) на trailing-корреляции прогнозов (сдвиг 1 бар),
    кэп fdm_cap. Итог маппится в позицию [0,1] (long-only) делением
    на forecast_cap.

    Args:
        bars: Данные инструмента.
        forecast_cap: Клип прогноза (единицы Carver, ±20).
        fdm_cap: Потолок diversification multiplier.
        corr_window: Окно оценки корреляции прогнозов.
        long_only: Обрезать шорт-часть (True по дисциплине проекта;
            ре-аудит шорта — отдельный трек 0.4 роадмапа).

    Returns:
        position: [0,1] при long_only, иначе [−1,1].
    """
    raw = {
        "ewmac_8_32": _ewmac_raw_forecast(bars.close, 8, 32),
        "ewmac_16_64": _ewmac_raw_forecast(bars.close, 16, 64),
        "ewmac_32_128": _ewmac_raw_forecast(bars.close, 32, 128),
        "kama_er": _kama_forecast(bars),
        "donch_pos": _donch_pos_forecast(bars),
    }
    fdf = pd.DataFrame(raw)
    # Скейлинг Carver: прогноз × (10 / trailing E|forecast|).
    abs_mean = fdf.abs().rolling(corr_window, min_periods=60).mean()
    scaled = (fdf * (10.0 / abs_mean.where(abs_mean > 1e-12))).clip(
        -forecast_cap, forecast_cap)

    k = scaled.shape[1]
    w = np.full(k, 1.0 / k)
    combined = scaled.mean(axis=1)

    # FDM на trailing-корреляции, сдвиг 1 бар (без look-ahead).
    fdm = pd.Series(1.0, index=bars.index)
    corr = scaled.rolling(corr_window, min_periods=120).corr()
    # rolling().corr() даёт MultiIndex (дата, член) — берём срез по дате
    # раз в 21 бар (корреляции медленные, ежедневный пересчёт — шум).
    dates = bars.index[::21]
    last_val = 1.0
    for dt in bars.index:
        if dt in dates:
            try:
                m = corr.loc[dt].to_numpy()
                if not np.isnan(m).any():
                    denom = float(np.sqrt(w @ m @ w))
                    if denom > 1e-9:
                        last_val = min(1.0 / denom, fdm_cap)
            except KeyError:
                pass
        fdm.loc[dt] = last_val
    fdm = fdm.shift(1).fillna(1.0)

    pos = (combined * fdm / forecast_cap).fillna(0.0)
    if long_only:
        pos = pos.clip(lower=0.0)
    return pos.clip(-1.0, 1.0)


# ─────────────────────── Hurst-аллокатор ───────────────────────

def hurst_aggvar(
    close: pd.Series, window: int = 504,
    qs: tuple = (2, 4, 8, 16, 32),
) -> pd.Series:
    """Показатель Хёрста методом агрегированной дисперсии (rolling).

    Var[sum of q returns] ~ q^(2H): H — наклон/2 регрессии
    log Var(q) на log q. Устойчивее R/S на коротких окнах.
    Пересчёт раз в 21 бар (H — медленное свойство), между
    пересчётами держится последнее значение. Сдвиг 1 бар.

    Args:
        close: Цены закрытия.
        window: Trailing-окно оценки (баров, ~2 года).
        qs: Масштабы агрегирования.

    Returns:
        Ряд H (NaN на прогреве).
    """
    rets = close.pct_change().to_numpy()
    n = len(rets)
    out = np.full(n, np.nan)
    log_q = np.log(np.asarray(qs, dtype=float))
    last = np.nan
    for i in range(n):
        if i % 21 == 0 and i >= window:
            seg = rets[i - window:i]
            log_v = []
            ok = True
            for q in qs:
                m = (len(seg) // q) * q
                agg = seg[:m].reshape(-1, q).sum(axis=1)
                v = agg.var()
                if v <= 0 or len(agg) < 8:
                    ok = False
                    break
                log_v.append(np.log(v))
            if ok:
                slope = np.polyfit(log_q, np.asarray(log_v), 1)[0]
                last = float(slope / 2.0)
        out[i] = last
    return pd.Series(out, index=close.index).shift(1)


def hurst_combo(
    bars: Bars,
    h_lo: float = 0.40,
    h_hi: float = 0.60,
    window: int = 504,
) -> pd.Series:
    """Комбо тренд+MR с весами ног по Хёрсту инструмента.

    w_trend = clip((H − h_lo)/(h_hi − h_lo), 0, 1);
    position = w_trend × donchian_raw + (1 − w_trend) × mr_kelt_confirm.
    На прогреве H (NaN) — 50/50.

    Args:
        bars: Данные инструмента.
        h_lo: H чистого MR (ниже — весь бюджет реверсии).
        h_hi: H чистого тренда.
        window: Окно оценки H.

    Returns:
        position [0, 1] (обе ноги long-only сырые, VT снаружи).
    """
    from strategies.donchian import donchian_champion_raw
    h = hurst_aggvar(bars.close, window=window)
    w_trend = ((h - h_lo) / (h_hi - h_lo)).clip(0.0, 1.0).fillna(0.5)
    trend = donchian_champion_raw(bars).fillna(0.0)
    mr = mr_keltner_confirm(bars).fillna(0.0)
    return w_trend * trend + (1.0 - w_trend) * mr


ADVANCED = {
    "carver_fdm": carver_fdm,
    "hurst_combo": hurst_combo,
}


# ──────────────── Volume z-score подтверждение (№8) ────────────────

def volume_zscore(bars: Bars, lookback: int = 60) -> pd.Series:
    """Z-score объёма к trailing-среднему (сдвиг 1 бар).

    Args:
        bars: Данные инструмента (volume обязателен).
        lookback: Окно среднего/стд объёма.

    Returns:
        Ряд z; если volume отсутствует — ряд NaN (гейт станет
        нейтральным в donch_vol_confirm).
    """
    if bars.volume is None:
        return pd.Series(np.nan, index=bars.index)
    v = bars.volume.astype(float)
    mu = v.rolling(lookback).mean().shift(1)
    sd = v.rolling(lookback).std().shift(1)
    return ((v - mu) / sd.where(sd > 1e-12))


def donch_vol_confirm(
    bars: Bars, entry: int = 20, exit_period: int = 10,
    z_min: float = 1.0, z_lookback: int = 60,
) -> pd.Series:
    """Дончиан-пробой, подтверждённый объёмом: вход при z(volume)>z_min.

    Гипотеза (классика ТА, формализованная): пробой на аномальном
    объёме = участие рынка (информационное событие), пробой на тонком
    объёме = шум/стоп-хантинг. Фильтр ТОЛЬКО на вход; выход обычный
    (нижний канал) — удержание тренда объём не прерывает.

    Честность: если volume в источнике отсутствует (часть
    yfinance-фьючерсов ненадёжна), фильтр НЕЙТРАЛЕН (пропускает все
    входы) — стратегия деградирует до donchian_breakout, а не ломается.
    Настоящий тест — на Databento, где panel_volume.parquet полный.

    Args:
        bars: Данные инструмента.
        entry: Окно верхнего канала.
        exit_period: Окно нижнего канала.
        z_min: Порог z-score объёма для входа.
        z_lookback: Окно z-score.

    Returns:
        position {0, 1}.
    """
    upper = bars.high.rolling(entry).max().shift(1)
    lower = bars.low.rolling(exit_period).min().shift(1)
    z = volume_zscore(bars, z_lookback)
    close, low = bars.close.values, bars.low.values
    up_v, lo_v, z_v = upper.values, lower.values, z.values

    pos = np.zeros(len(close))
    in_pos = False
    for i in range(len(close)):
        if np.isnan(up_v[i]) or np.isnan(lo_v[i]):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        if in_pos and low[i] < lo_v[i]:
            in_pos = False
        vol_ok = np.isnan(z_v[i]) or z_v[i] > z_min  # NaN=нейтрален
        if not in_pos and close[i] > up_v[i] and vol_ok:
            in_pos = True
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=bars.index)


ADVANCED["donch_vol_confirm"] = donch_vol_confirm
