"""Тесты сайзеров (realized/garch), ансамблей и портфельного sweep.

Проверяемые инварианты (синтетика, без сети):
  - make_sizer: реестр, границы [0, max_leverage], NaN-прогрев
    держит применённое значение;
  - mr_ensemble: позиция в [0, 1], индекс совпадает с bars;
  - trend_mr_combo: позиция в [0, 1], валидация весов;
  - vol_sweep_basket: портфельные колонки присутствуют, портфельный DD
    не хуже (не глубже) worst-case per-instrument (диверсификация не
    может ухудшить equal-weight DD относительно худшего инструмента);
  - оборот sweep'а — drift-aware (совпадает с формулой движка).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.bars import Bars
from core.engine import drift_turnover
from core.sizing import make_sizer
from diagnostics.vol_sweep import vol_sweep_basket, vol_sweep_single
from strategies.ensemble import mr_ensemble, trend_mr_combo


def _make_bars(n: int = 900, seed: int = 7, drift: float = 0.0002,
               symbol: str = "SYN") -> Bars:
    """Синтетический инструмент с кластеризованной волой (для GARCH)."""
    rng = np.random.default_rng(seed)
    # Простая vol-кластеризация: двурежимная сигма.
    regime = (np.sin(np.arange(n) / 60.0) > 0).astype(float)
    sigma = 0.008 + 0.012 * regime
    rets = drift + sigma * rng.standard_normal(n)
    close = pd.Series(
        100.0 * np.cumprod(1.0 + rets),
        index=pd.date_range("2021-01-04", periods=n, freq="B"),
    )
    high = close * 1.005
    low = close * 0.995
    return Bars(open=close.shift(1).fillna(close.iloc[0]), high=high,
                low=low, close=close, bars_per_year=252.0, symbol=symbol)


def test_make_sizer_registry_and_bounds():
    bars = _make_bars()
    for name in ("realized", "garch"):
        sizer = make_sizer(name, target_vol=0.20, max_leverage=2.0)
        mult = sizer(bars)
        assert mult.index.equals(bars.index)
        assert (mult >= 0).all()
        assert (mult <= 2.0 + 1e-12).all()
        # После прогрева сайзер должен быть активен.
        assert float(mult.iloc[-100:].mean()) > 0
    with pytest.raises(KeyError):
        make_sizer("hmm")


def test_mr_ensemble_bounds_and_index():
    bars = _make_bars(seed=11)
    pos = mr_ensemble(bars)
    assert pos.index.equals(bars.index)
    assert (pos >= -1e-12).all()
    assert (pos <= 1.0 + 1e-12).all()
    # Дробность: ансамбль из 4 бинарных ног даёт значения кратные 0.25.
    vals = np.unique(np.round(pos.values, 6))
    assert set(vals).issubset({0.0, 0.25, 0.5, 0.75, 1.0})


def test_trend_mr_combo_bounds_and_weight_validation():
    bars = _make_bars(seed=13)
    pos = trend_mr_combo(bars, w_trend=0.5)
    assert pos.index.equals(bars.index)
    assert (pos >= -1e-12).all()
    assert (pos <= 1.0 + 1e-12).all()
    with pytest.raises(ValueError):
        trend_mr_combo(bars, w_trend=1.5)


def test_vol_sweep_portfolio_columns_and_dd_dominance():
    baskets = {f"S{i}": _make_bars(seed=100 + i) for i in range(4)}

    def sig(bars):
        # Простой всегда-в-лонге сигнал: изолирует эффект сайзера.
        return pd.Series(1.0, index=bars.index)

    df = vol_sweep_basket(baskets, sig, target_vols=(0.15, 0.30),
                          cost=0.0002)
    for col in ("port_return", "port_dd", "port_sharpe",
                "port_passes_dd", "worst_dd"):
        assert col in df.columns
    # Диверсификация: DD equal-weight портфеля не глубже worst-case.
    assert (df["port_dd"] >= df["worst_dd"] - 1e-12).all()


def test_sweep_turnover_matches_engine_formula():
    bars = _make_bars(seed=21)

    def sig(bars_):
        return pd.Series(1.0, index=bars_.index)

    df, _ = vol_sweep_single(bars, sig, target_vols=(0.20,),
                             cost=0.0002)
    # Пересобираем оборот вручную той же формулой движка.
    sizer = make_sizer("realized", target_vol=0.20, max_leverage=6.0)
    pos = sig(bars) * sizer(bars)
    prev = pos.shift(1).fillna(0.0)
    manual = float(drift_turnover(prev, bars.returns()).sum())
    years = len(bars) / bars.bars_per_year
    assert df.loc[0.20, "turnover_ann"] == pytest.approx(
        manual / years, rel=1e-9)


def test_garch_sizer_reacts_to_vol_regimes():
    """GARCH-сайзер должен давать меньший размер в высоковолатильном
    режиме, чем в низковолатильном (иначе он не сайзер)."""
    bars = _make_bars(seed=33, n=1200)
    mult = make_sizer("garch", target_vol=0.20)(bars)
    rv = bars.returns().rolling(30).std()
    hi = mult[rv > rv.quantile(0.8)].mean()
    lo = mult[rv < rv.quantile(0.2)].mean()
    assert hi < lo
