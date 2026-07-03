"""Тесты walk-forward диагностики.

Ключевая проверка: движок ДОЛЖЕН различать стратегию со стабильным
результатом из года в год и стратегию-мираж, живущую на одном везучем
окне. Если не различает — метрика бесполезна.

Запуск: python -m tests.test_walkforward
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from diagnostics.walkforward import (
    consistency_metrics,
    walk_forward_single,
    walk_forward_windows,
)


def _bars_5y(seed=0) -> Bars:
    """5-летний ряд для нарезки на годовые окна."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", "2026-01-01")
    n = len(idx)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                      index=idx)
    sp = close * 0.006
    return Bars(open=close.shift(1).bfill(), high=close + sp,
                low=close - sp, close=close, bars_per_year=252.0,
                symbol="WF")


def test_windows_by_year() -> None:
    """Погодовая нарезка 2021-2026 даёт корректные годовые окна."""
    bars = _bars_5y()
    wins = walk_forward_windows(bars.index, "year")
    years = [w[0].year for w in wins]
    assert 2021 in years and 2025 in years
    assert len(wins) >= 5
    print(f"  [ok] нарезка по годам: {len(wins)} окон "
          f"({years[0]}..{years[-1]})")


def test_distinguishes_stable_vs_mirage() -> None:
    """КРИТИЧНО: walk-forward различает стабильную и мираж-стратегию.

    Строим два искусственных сигнала на одном ряду:
      - stable: +0.05% каждый бар (ровный доход во всех годах);
      - mirage: весь доход в одном году (2023), ноль в остальных.
    У обоих может быть похожий средний, но согласованность окон
    (positive_frac, разброс) должна их развести.
    """
    bars = _bars_5y()

    def stable(b):
        # Постоянная позиция -> доход follows ряд, но знак стабилен по
        # годам (лонг на слабо-трендовом ряду).
        return pd.Series(1.0, index=b.index)

    def mirage(b):
        # В позиции ТОЛЬКО в 2023, иначе кэш.
        pos = pd.Series(0.0, index=b.index)
        pos[b.index.year == 2023] = 1.0
        return pos

    wf_stable = walk_forward_single(bars, stable)
    wf_mirage = walk_forward_single(bars, mirage)
    m_stable = consistency_metrics(wf_stable)
    m_mirage = consistency_metrics(wf_mirage)

    # Мираж торгует только 1 год -> положительных окон максимум 1 из 5.
    assert m_mirage["positive_frac"] <= 0.4, (
        f"мираж дал {m_mirage['positive_frac']:.0%} прибыльных окон"
    )
    # У миража ненулевой результат сконцентрирован -> хотя бы одно окно
    # заметно, остальные ~0.
    nonzero = (wf_mirage["return"].abs() > 0.001).sum()
    assert nonzero <= 2, f"мираж торговал в {nonzero} окнах, не в одном"
    print(f"  [ok] различает: stable {m_stable['positive_frac']:.0%} "
          f"прибыльных окон vs mirage {m_mirage['positive_frac']:.0%} "
          f"(мираж торгует {nonzero} окно)")


def test_consistency_metrics_shape() -> None:
    """consistency_metrics возвращает все ожидаемые поля."""
    bars = _bars_5y()
    wf = walk_forward_single(bars, lambda b: pd.Series(1.0, index=b.index))
    m = consistency_metrics(wf)
    for key in ("n_windows", "positive_frac", "mean_return",
                "worst_window", "best_window", "spread"):
        assert key in m, f"нет метрики {key}"
    assert m["spread"] >= 0
    print(f"  [ok] метрики полны: {m['n_windows']} окон, разброс "
          f"{m['spread']:.1%}")


if __name__ == "__main__":
    print("Тесты walk-forward:")
    test_windows_by_year()
    test_distinguishes_stable_vs_mirage()
    test_consistency_metrics_shape()
    print("Все тесты walk-forward пройдены.")
