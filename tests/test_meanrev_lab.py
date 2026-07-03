"""Тесты лаборатории mean-reversion (10 вариантов bb_rsi).

Проверяет МЕХАНИКУ каждого варианта (не доходность — синтетика):
  - time_stop реально ограничивает длительность позиции;
  - atr_stop режет хвост, которого базовый bb_rsi не режет;
  - scaled/ladder дают позиции в [0, 1] с нужной дискретностью;
  - short — зеркальный контроль (позиции <= 0);
  - сравнительная таблица всех 10 на боковике и тренде.

Запуск: python -m tests.test_meanrev_lab
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine
from strategies.bollinger import bollinger_rsi
from strategies.meanrev_lab import MEANREV_LAB


def _range_bars(n=600, seed=2) -> Bars:
    """Боковик (OU) — родная среда mean-reversion.

    theta=0.05 (half-life ~14 баров): медленный Wilder RSI(14) успевает
    уйти в oversold до реверсии. С быстрым OU (hl~5) RSI<30 почти не
    случается и половина вариантов не торгует.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = np.zeros(n)
    x[0] = 100
    for i in range(1, n):
        x[i] = x[i - 1] + 0.05 * (100 - x[i - 1]) + rng.normal(0, 2.2)
    close = pd.Series(x, index=idx)
    hl = 1.2
    high = close + hl * rng.uniform(0.3, 1.0, n)
    low = close - hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="RANGE")


def _crash_bars(n=300, seed=3) -> Bars:
    """Медленный слив без отскока — падающий нож (стресс для лонг-MR)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps = np.concatenate([
        rng.normal(0.0005, 0.008, 100),        # спокойный участок
        rng.normal(-0.004, 0.012, n - 100),     # затяжной слив
    ])
    close = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
    sp = close * 0.008
    high = close + sp * rng.uniform(0.3, 1.0, n)
    low = close - sp * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="CRASH")


def test_all_variants_valid_positions() -> None:
    """Все 10 дают валидные позиции: длина, нет NaN, диапазон."""
    bars = _range_bars()
    for name, fn in MEANREV_LAB.items():
        pos = fn(bars)
        assert len(pos) == len(bars), f"{name}: длина"
        assert not pos.isna().any(), f"{name}: NaN в позициях"
        if name == "mr_short":
            assert (pos <= 0).all(), f"{name}: шорт-контроль дал лонг"
        else:
            assert (pos >= 0).all(), f"{name}: лонг дал шорт"
        assert pos.abs().max() <= 1.0 + 1e-9, f"{name}: |pos| > 1"
    print(f"  [ok] все {len(MEANREV_LAB)} вариантов: позиции валидны")


def test_time_stop_limits_holding() -> None:
    """mr_time_stop: непрерывная позиция не дольше max_hold баров."""
    bars = _crash_bars()   # слив: RSI-выход не срабатывает долго
    pos = MEANREV_LAB["mr_time_stop"](bars)
    # Длины непрерывных блоков позиции 1.0.
    runs, cur = [], 0
    for v in pos.values:
        if v > 0:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    max_run = max(runs) if runs else 0
    assert max_run <= 10, f"time-stop не сработал: холд {max_run} баров"
    print(f"  [ok] mr_time_stop: max холд {max_run} <= 10 баров "
          f"(гипотеза с истёкшим сроком закрывается)")


def test_atr_stop_cuts_tail() -> None:
    """mr_atr_stop на сливе теряет меньше базового bb_rsi (хвост обрезан).

    Регрессия на Tesla-кейс (DD −53% у базы): RSI-выход не риск-стоп,
    ATR-стоп — риск-стоп.
    """
    bars = _crash_bars()
    base = run_engine(bars, bollinger_rsi(bars), cost=0.0)
    stopped = run_engine(bars, MEANREV_LAB["mr_atr_stop"](bars), cost=0.0)
    assert stopped.max_drawdown >= base.max_drawdown - 1e-9, (
        f"ATR-стоп не улучшил DD: {stopped.max_drawdown:.1%} vs "
        f"{base.max_drawdown:.1%}"
    )
    print(f"  [ok] mr_atr_stop на сливе: DD {stopped.max_drawdown:+.1%} "
          f"vs база {base.max_drawdown:+.1%} (хвост обрезан)")


def test_ladder_discrete_levels() -> None:
    """mr_ladder: позиции строго из {0, 0.5, 1.0} (кап жёсткий)."""
    bars = _range_bars()
    pos = MEANREV_LAB["mr_ladder"](bars)
    levels = set(np.round(pos.unique(), 6))
    assert levels <= {0.0, 0.5, 1.0}, f"Лишние уровни: {levels}"
    print(f"  [ok] mr_ladder: уровни {sorted(levels)} (не мартингейл)")


def test_scaled_continuous() -> None:
    """mr_scaled: непрерывный сайзинг в [0, 1], глубже z — больше."""
    bars = _range_bars()
    pos = MEANREV_LAB["mr_scaled"](bars)
    active = pos[pos > 0]
    assert len(active) > 5, "scaled не открыл позиций на боковике"
    assert active.nunique() > 3, "сайзинг не непрерывный"
    print(f"  [ok] mr_scaled: {active.nunique()} уровней размера, "
          f"max {active.max():.2f}")


def test_lab_comparison_table() -> None:
    """Сравнительная таблица 10 вариантов на боковике (родная среда)."""
    bars = _range_bars()
    print("\n  Лаборатория на боковике (синтетика, cost=2bps):")
    print(f"  {'вариант':14s} {'return':>8s} {'maxDD':>8s} {'sharpe':>7s}")
    print("  " + "-" * 40)
    for name, fn in MEANREV_LAB.items():
        res = run_engine(bars, fn(bars))
        print(f"  {name:14s} {res.total_return:>+7.1%} "
              f"{res.max_drawdown:>+7.1%} {res.sharpe:>+7.2f}")
    print("  (числа синтетические — только для проверки механики)")


if __name__ == "__main__":
    print("Тесты лаборатории mean-reversion:")
    test_all_variants_valid_positions()
    test_time_stop_limits_holding()
    test_atr_stop_cuts_tail()
    test_ladder_discrete_levels()
    test_scaled_continuous()
    test_lab_comparison_table()
    print("\nВсе тесты лаборатории пройдены.")
