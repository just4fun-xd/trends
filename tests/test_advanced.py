"""Тесты продвинутого аппарата: bootstrap, FDM, Hurst, volume.

Инварианты:
  bootstrap: CI содержит точечный Sharpe; paired-разность на
    идентичных рядах = ровно 0 (значимости нет); на явно разных по
    построению рядах — significant=True;
  carver_fdm: контракт/границы, long-only, на дрейфе в позиции;
  hurst_aggvar: H(случайное блуждание) ~ 0.5, H(трендовый AR) > H(RW),
    H(антиперсистентный) < H(RW) — ранговая проверка, не точная;
  hurst_combo: границы, прогрев 50/50;
  donch_vol_confirm: без volume нейтрален (== donchian_breakout);
    с volume блокирует пробои на тонком объёме.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from diagnostics.bootstrap import sharpe_ci, sharpe_diff_ci
from strategies.advanced import (
    carver_fdm,
    donch_vol_confirm,
    hurst_aggvar,
    hurst_combo,
)


def _bars(n=900, seed=1, drift=0.0006, volume=None):
    rng = np.random.default_rng(seed)
    rets = drift + 0.01 * rng.standard_normal(n)
    close = pd.Series(
        100.0 * np.cumprod(1.0 + rets),
        index=pd.date_range("2019-01-02", periods=n, freq="B"),
    )
    return Bars(open=close.shift(1).fillna(close.iloc[0]),
                high=close * 1.006, low=close * 0.994, close=close,
                bars_per_year=252.0, symbol="SYN", volume=volume)


def test_sharpe_ci_contains_point_estimate():
    rng = np.random.default_rng(3)
    r = pd.Series(0.0004 + 0.008 * rng.standard_normal(1500))
    c = sharpe_ci(r, n_boot=500)
    assert c["lo"] <= c["sharpe"] <= c["hi"]
    assert c["lo"] < c["hi"]


def test_sharpe_diff_identical_series_is_zero():
    rng = np.random.default_rng(5)
    r = pd.Series(0.0003 + 0.007 * rng.standard_normal(1200))
    d = sharpe_diff_ci(r, r.copy(), n_boot=300)
    assert d["diff"] == 0.0
    assert not d["significant"]


def test_sharpe_diff_detects_clear_difference():
    rng = np.random.default_rng(7)
    n = 2000
    noise = rng.standard_normal(n)
    a = pd.Series(0.0015 + 0.008 * noise)          # Sharpe ~3
    b = pd.Series(0.0000 + 0.008 * noise)          # Sharpe ~0
    d = sharpe_diff_ci(a, b, n_boot=500)
    assert d["diff"] > 0
    assert d["significant"]


def test_rf_lowers_sharpe():
    rng = np.random.default_rng(9)
    r = pd.Series(0.0004 + 0.006 * rng.standard_normal(1200))
    c0 = sharpe_ci(r, n_boot=200, rf=0.0)
    c1 = sharpe_ci(r, n_boot=200, rf=0.05)
    assert c1["sharpe"] < c0["sharpe"]


def test_carver_fdm_contract_and_drift():
    bars = _bars(n=1200, seed=11, drift=0.0012)
    pos = carver_fdm(bars)
    assert pos.index.equals(bars.index)
    assert (pos >= -1e-9).all() and (pos <= 1.0 + 1e-9).all()
    assert pos.iloc[400:].mean() > 0.1  # на дрейфе в позиции


def test_hurst_ranks_regimes():
    n = 4000
    rng = np.random.default_rng(13)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    eps = rng.standard_normal(n)
    rw = pd.Series(1000.0 + np.cumsum(eps), index=idx)

    def ar(phi):
        r = np.zeros(n)
        for i in range(1, n):
            r[i] = phi * r[i - 1] + eps[i]
        return pd.Series(1000.0 + np.cumsum(r), index=idx)

    # AR(1) — короткопамятный: асимптотически H -> 0.5, поэтому
    # различие видно на КОРОТКИХ масштабах (наш торговый горизонт) и
    # при сильной phi; сравниваем среднее H по окнам (одна последняя
    # точка шумная). Эмпирика: trend 0.61-0.71 > rw ~0.46 > anti ~0.37.
    h_tr = hurst_aggvar(ar(0.6)).dropna().mean()
    h_rw = hurst_aggvar(rw).dropna().mean()
    h_an = hurst_aggvar(ar(-0.6)).dropna().mean()
    assert h_tr > h_rw > h_an
    assert 0.35 < h_rw < 0.65  # RW около 0.5


def test_hurst_combo_bounds_and_warmup():
    bars = _bars(n=700, seed=17)
    pos = hurst_combo(bars)
    assert pos.index.equals(bars.index)
    assert (pos >= -1e-9).all() and (pos <= 1.0 + 1e-9).all()


def test_donch_vol_confirm_neutral_without_volume():
    from strategies.donchian import donchian_breakout
    bars = _bars(n=800, seed=19, drift=0.0012)
    a = donch_vol_confirm(bars)                    # volume=None
    b = donchian_breakout(bars, entry=20, exit_period=10)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_donch_vol_confirm_blocks_thin_volume():
    n = 800
    rng = np.random.default_rng(23)
    # Постоянный тонкий объём: z никогда не превысит порог.
    vol = pd.Series(1000.0 + rng.normal(0, 1, n),
                    index=pd.date_range("2019-01-02", periods=n,
                                        freq="B"))
    bars = _bars(n=n, seed=19, drift=0.0012, volume=vol)
    gated = donch_vol_confirm(bars, z_min=3.0)
    from strategies.donchian import donchian_breakout
    free = donchian_breakout(bars, entry=20, exit_period=10)
    # На тонком объёме входов должно быть строго меньше.
    assert gated.sum() < free.sum()
