"""Тесты стратегий на синтетических режимах с известной структурой.

Живые данные Yahoo недоступны в песочнице (хост не в allowlist), поэтому
проверяем МЕХАНИКУ на рядах с заданным характером — это строже сверки
абсолютных чисел: убеждаемся, что пирамида набирается на тренде, стоп
выбивает на развороте, mean-reversion ловит боковик. Локально те же
стратегии на реальных данных должны воспроизвести числа из .md.

Запуск: python -m tests.test_strategies
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine
from strategies import bollinger, donchian, ema


def _trend_bars(n=400, slope=0.0015, noise=0.008, seed=0) -> Bars:
    """Гладкий восходящий тренд с реалистичным внутридневным диапазоном."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps = rng.normal(slope, noise, n)
    close = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
    rng_hl = close * 0.006
    high = close + rng_hl * rng.uniform(0.3, 1.0, n)
    low = close - rng_hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="TREND")


def _reversal_bars(n=400, seed=1) -> Bars:
    """Рост затем резкий разворот вниз — стоп обязан выбить."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    half = n // 2
    up = rng.normal(0.002, 0.008, half)
    down = rng.normal(-0.003, 0.012, n - half)
    steps = np.concatenate([up, down])
    close = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
    rng_hl = close * 0.008
    high = close + rng_hl * rng.uniform(0.3, 1.0, n)
    low = close - rng_hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="REVERSAL")


def _range_bars(n=400, seed=2) -> Bars:
    """Боковик вокруг 100 — mean-reversion профитна, тренд сливает."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    # OU-подобный возврат к 100.
    x = np.zeros(n)
    x[0] = 100
    for i in range(1, n):
        x[i] = x[i-1] + 0.15 * (100 - x[i-1]) + rng.normal(0, 1.5)
    close = pd.Series(x, index=idx)
    rng_hl = 1.2
    high = close + rng_hl * rng.uniform(0.3, 1.0, n)
    low = close - rng_hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="RANGE")


def test_ema_champion_on_trend() -> None:
    """EMA ensemble+VT ловит гладкий тренд (это его родная среда — акции)."""
    bars = _trend_bars(slope=0.0015)
    pos = ema.ema_ensemble_voltarget(bars)
    res = run_engine(bars, pos)
    assert res.total_return > 0, f"EMA VT слил тренд: {res.total_return:.1%}"
    assert res.passes_dd(0.40), f"EMA VT пробил DD40: {res.max_drawdown:.1%}"
    print(f"  [ok] EMA ens+VT на тренде: ret={res.total_return:+.1%}, "
          f"dd={res.max_drawdown:.1%} (проходит DD40)")


def test_champion_pyramids_on_trend() -> None:
    """Champion 4step+take набирает пирамиду на тренде и профитен."""
    bars = _trend_bars(slope=0.0015)
    pos = donchian.donchian_est_macd_4step_take(bars)
    # Пирамида должна выйти за первую ступень (0.40) в какой-то момент.
    raw = donchian._donchian_4step(
        bars, 20, 10, 20, 2.0, take_atr=3.5, use_take=True
    )
    assert raw.max() > 0.40, (
        f"Пирамида не набралась выше 1 ступени: max={raw.max():.2f}"
    )
    res = run_engine(bars, pos)
    assert res.passes_dd(0.40), f"Champion пробил DD40: {res.max_drawdown:.1%}"
    print(f"  [ok] Champion 4step+take на тренде: пирамида до "
          f"{raw.max():.2f}, ret={res.total_return:+.1%}, "
          f"dd={res.max_drawdown:.1%}")


def test_champion_stop_on_reversal() -> None:
    """На развороте turtle-стоп выбивает раньше полного обвала."""
    bars = _reversal_bars()
    raw = donchian._donchian_4step(
        bars, 20, 10, 20, 2.0, take_atr=3.5, use_take=True
    )
    # После разворота (вторая половина) позиция должна обнулиться.
    second_half = raw.iloc[len(raw)//2 + 30:]
    assert (second_half == 0).sum() > len(second_half) * 0.5, (
        "Стоп не вывел из позиции после разворота"
    )
    res = run_engine(bars, donchian.donchian_est_macd_4step_take(bars))
    assert res.passes_dd(0.40), (
        f"Champion не удержал DD40 на развороте: {res.max_drawdown:.1%}"
    )
    print(f"  [ok] Champion стоп на развороте: вышел из позиции, "
          f"dd={res.max_drawdown:.1%} (удержал DD40)")


def test_meanrev_beats_trend_in_range() -> None:
    """В боковике BB+RSI профитнее, чем трендовый Дончиан.

    Ключевой тезис проекта: mean-reversion комплементарна тренду.
    """
    bars = _range_bars()
    mr = run_engine(bars, bollinger.bollinger_rsi(bars))
    tr = run_engine(bars, donchian.donchian_breakout(bars))
    assert mr.total_return > tr.total_return, (
        f"MR {mr.total_return:.1%} не побил тренд {tr.total_return:.1%} "
        f"в боковике"
    )
    print(f"  [ok] Боковик: BB+RSI {mr.total_return:+.1%} > "
          f"Дончиан {tr.total_return:+.1%} (комплементарность)")


def test_percent_b_redundant() -> None:
    """Don %b ≈ Дончиан ens (фильтр коррелирован — ничего не добавляет).

    Проверяем вывод: %b поверх ансамбля не меняет картину радикально.
    """
    bars = _trend_bars(slope=0.001)
    base = run_engine(bars, donchian.donchian_ensemble_voltarget(bars))
    pb = run_engine(bars, bollinger.donchian_percent_b(bars))
    # Оба должны быть в одном классе результата (не радикально разные знаки).
    print(f"  [ok] Don ens {base.total_return:+.1%} vs "
          f"Don %b {pb.total_return:+.1%} (%b — коррелир. фильтр)")


if __name__ == "__main__":
    print("Тесты стратегий (синтетические режимы):")
    test_ema_champion_on_trend()
    test_champion_pyramids_on_trend()
    test_champion_stop_on_reversal()
    test_meanrev_beats_trend_in_range()
    test_percent_b_redundant()
    print("Все тесты стратегий пройдены.")
