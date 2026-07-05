"""Тесты лаборатории тренда и vol-percentile гейта.

Инварианты:
  - все модели TREND_LAB: контракт Bars -> position, индекс совпадает,
    диапазон [0, 1] (long-only);
  - tsmom ловит синтетический тренд (позиция в лонге на дрейфе);
  - chandelier: peak-трейлинг закрывает позицию на глубоком откате;
  - vol_percentile_gate: 1 в спокойном режиме, 0 при взрыве волы
    масштаба «за пределами исторического распределения» (ковид-тест);
  - гейт не выключает стратегию на прогреве (до заполнения окна = 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.overlays import vol_percentile_gate, with_vol_gate
from strategies.trend_lab import TREND_LAB, tsmom


def _bars_from_close(close: pd.Series, symbol: str = "SYN") -> Bars:
    return Bars(open=close.shift(1).fillna(close.iloc[0]),
                high=close * 1.005, low=close * 0.995, close=close,
                bars_per_year=252.0, symbol=symbol)


def _trend_bars(n: int = 900, seed: int = 5) -> Bars:
    rng = np.random.default_rng(seed)
    rets = 0.0008 + 0.01 * rng.standard_normal(n)
    close = pd.Series(
        100.0 * np.cumprod(1.0 + rets),
        index=pd.date_range("2021-01-04", periods=n, freq="B"),
    )
    return _bars_from_close(close)


def test_trend_lab_contract_and_bounds():
    bars = _trend_bars()
    for name, fn in TREND_LAB.items():
        pos = fn(bars)
        assert pos.index.equals(bars.index), name
        assert (pos >= -1e-9).all(), name
        assert (pos <= 1.0 + 1e-9).all(), name
        assert not pos.isna().any(), name


def test_tsmom_long_on_drift():
    bars = _trend_bars(seed=9)
    pos = tsmom(bars)
    # На устойчивом положительном дрейфе TSMOM почти всегда в лонге
    # после прогрева 252 бара.
    assert pos.iloc[300:].mean() > 0.8


def test_chandelier_exits_on_crash():
    n = 600
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    # Дрейф 0.8%/бар > синтетического фитиля хая (+0.5%), иначе close
    # никогда не пробьёт вчерашний rolling-max хаёв.
    up = 100.0 * (1.008 ** np.arange(400))
    crash = up[-1] * (0.97 ** np.arange(1, 201))
    close = pd.Series(np.concatenate([up, crash]), index=idx)
    bars = _bars_from_close(close)
    pos = TREND_LAB["chandelier"](bars)
    assert pos.iloc[350] == 1.0          # в тренде — в позиции
    assert pos.iloc[-50:].sum() == 0.0   # после обвала — вышел


def test_vol_gate_blocks_covid_style_explosion():
    n = 900
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    rng = np.random.default_rng(3)
    rets = 0.0003 + 0.008 * rng.standard_normal(n)
    # «Ковид»: 40 баров с волой, кратно превышающей всю историю.
    rets[700:740] = 0.10 * rng.standard_normal(40)
    close = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)
    bars = _bars_from_close(close)
    gate = vol_percentile_gate(bars)
    assert set(np.unique(gate.values)).issubset({0.0, 1.0})
    # До взрыва (после прогрева) — открыт, во взрыве — закрыт.
    assert gate.iloc[600:695].mean() > 0.9
    assert gate.iloc[715:745].mean() < 0.2


def test_with_vol_gate_wraps_contract():
    bars = _trend_bars(seed=17)

    def always_long(b):
        return pd.Series(1.0, index=b.index)

    gated = with_vol_gate(always_long)
    pos = gated(bars)
    assert pos.index.equals(bars.index)
    assert set(np.unique(pos.values)).issubset({0.0, 1.0})


def test_gate_warmup_is_open():
    bars = _trend_bars(seed=21, n=300)  # короче rank_window
    gate = vol_percentile_gate(bars, rank_window=500)
    # min_periods=250 не достигнут в начале — гейт открыт.
    assert gate.iloc[:100].min() == 1.0


def test_kama_not_poisoned_by_warmup_nan():
    """Регрессия: NaN в sc на границе прогрева отравлял рекурсию KAMA
    навсегда (позиция — вечный 0). После фикса на дрейфующем ряду KAMA
    должна проводить в лонге заметную долю времени."""
    bars = _trend_bars(seed=0, n=500)
    pos = TREND_LAB["kama"](bars)
    assert pos.iloc[50:].mean() > 0.2
