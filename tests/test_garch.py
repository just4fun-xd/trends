"""Тесты GARCH(1,1)-модуля: восстановление параметров, look-ahead,
реакция на шок против rolling std, контракт множителя.

Запуск: python -m tests.test_garch
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.garch import (
    fit_garch, garch_vol_forecast, garch_vol_target_size)


def _sim_garch(n=3000, omega=1e-6, a=0.08, b=0.90, seed=0):
    """Симулирует GARCH(1,1)-доходности с известными параметрами."""
    rng = np.random.default_rng(seed)
    sig2 = omega / (1 - a - b)
    r = np.empty(n)
    for t in range(n):
        r[t] = rng.normal(0, np.sqrt(sig2))
        sig2 = omega + a * r[t] ** 2 + b * sig2
    return r


def test_parameter_recovery() -> None:
    """MLE восстанавливает (alpha, beta) на симуляции."""
    r = _sim_garch()
    _, a, b = fit_garch(r)
    assert abs(a - 0.08) < 0.05, f"alpha {a:.3f}"
    assert abs(b - 0.90) < 0.06, f"beta {b:.3f}"
    print(f"  [ok] recovery: alpha {a:.3f} (~0.08), beta {b:.3f} (~0.90)")


def test_no_lookahead() -> None:
    """Обрезка будущего не меняет прошлые прогнозы (бит-в-бит)."""
    idx = pd.bdate_range("2016-01-01", periods=3000)
    rs = pd.Series(_sim_garch(), index=idx)
    full = garch_vol_forecast(rs)
    cut = garch_vol_forecast(rs.iloc[:2000])
    diff = (full.iloc[:2000] - cut).abs().max()
    assert diff < 1e-15, f"look-ahead: diff {diff:.2e}"
    print("  [ok] look-ahead: прогнозы префикса идентичны")


def test_shock_response_vs_rolling() -> None:
    """После шока GARCH реагирует быстрее И прощает быстрее rolling."""
    idx = pd.bdate_range("2016-01-01", periods=3000)
    r = _sim_garch()
    r[1500] = 0.10  # однодневный шок масштаба 10%
    rs = pd.Series(r, index=idx)
    g = garch_vol_forecast(rs)
    roll = rs.rolling(30).std()
    assert g.iloc[1501] > roll.iloc[1501], "GARCH не отреагировал быстрее"
    assert g.iloc[1520] < roll.iloc[1520], "GARCH не простил быстрее"
    print(f"  [ok] шок: t+1 garch {g.iloc[1501]:.4f} > roll "
          f"{roll.iloc[1501]:.4f}; t+20 garch {g.iloc[1520]:.4f} < "
          f"roll {roll.iloc[1520]:.4f}")


def test_size_contract() -> None:
    """Множитель: [0, max_leverage], NaN нет, буфер гасит дрожание."""
    idx = pd.bdate_range("2016-01-01", periods=1500)
    rng = np.random.default_rng(1)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 1500))),
        index=idx,
    )
    bars = Bars(open=close, high=close * 1.01, low=close * 0.99,
                close=close, bars_per_year=252.0, symbol="SYNTH")
    size = garch_vol_target_size(bars, max_leverage=2.0, buffer=0.10)
    assert not size.isna().any()
    assert (size >= 0).all() and (size <= 2.0 + 1e-9).all()
    raw = garch_vol_target_size(bars, max_leverage=2.0, buffer=0.0)
    ch_buf = (size.diff().abs() > 1e-12).sum()
    ch_raw = (raw.diff().abs() > 1e-12).sum()
    assert ch_buf < ch_raw, "буфер не гасит дрожание"
    print(f"  [ok] контракт множителя: [0,2], смен {ch_raw} -> {ch_buf}")


if __name__ == "__main__":
    print("Тесты GARCH:")
    test_parameter_recovery()
    test_no_lookahead()
    test_shock_response_vs_rolling()
    test_size_contract()
    print("Все тесты GARCH пройдены.")
