"""Тесты ядра: контракт Bars, оба движка, их совпадение.

Запуск: python -m tests.test_core (из корня quantlab).
Ловит фундаментальные поломки ДО написания стратегий.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine, vol_target_size
from core.engine_portfolio import run_portfolio, sanity_check_engines


def _synth_bars(n: int = 500, seed: int = 0) -> Bars:
    """Синтетический трендовый ряд с реалистичными high/low."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps = rng.normal(0.0005, 0.012, n)
    close = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
    # high/low вокруг close с внутридневным разбросом.
    spread = close * 0.008
    high = close + spread * rng.uniform(0.2, 1.0, n)
    low = close - spread * rng.uniform(0.2, 1.0, n)
    open_ = close.shift(1).fillna(close.iloc[0])
    return Bars(
        open=open_, high=high, low=low, close=close,
        bars_per_year=252.0, symbol="SYNTH",
    )


def test_bars_contract() -> None:
    """Bars валидирует выравнивание и считает TR/ATR."""
    bars = _synth_bars()
    assert len(bars) == 500
    tr = bars.true_range()
    assert (tr.dropna() >= 0).all(), "True Range не может быть отрицательным"
    atr = bars.atr(20)
    assert atr.notna().sum() > 400
    # from_close — частный случай.
    cb = Bars.from_close(bars.close, symbol="C")
    assert (cb.high == cb.close).all()
    print("  [ok] Bars контракт: выравнивание, TR>=0, ATR, from_close")


def test_misaligned_rejected() -> None:
    """Bars отвергает несовпадающие индексы."""
    bars = _synth_bars(100)
    bad_high = bars.high.iloc[:-1]
    try:
        Bars(open=bars.open, high=bad_high, low=bars.low,
             close=bars.close, bars_per_year=252.0)
        raise AssertionError("Должно было упасть на невыровненном индексе")
    except ValueError:
        print("  [ok] Bars отвергает невыровненные ряды")


def test_engine_basic() -> None:
    """run_engine: buy-and-hold воспроизводит доходность close."""
    bars = _synth_bars()
    pos = pd.Series(1.0, index=bars.index)  # всегда в лонге
    res = run_engine(bars, pos, cost=0.0)
    # Без издержек и с позицией 1.0 итог ~ доходность close (с точностью
    # до сдвига одного бара).
    close_ret = bars.close.iloc[-1] / bars.close.iloc[1] - 1
    assert abs(res.total_return - close_ret) < 0.01, (
        f"{res.total_return} vs {close_ret}"
    )
    assert res.max_drawdown <= 0
    print(f"  [ok] run_engine buy&hold: ret={res.total_return:+.1%}, "
          f"dd={res.max_drawdown:.1%}, sharpe={res.sharpe:.2f}")


def test_vol_target() -> None:
    """vol_target_size даёт положительный множитель в разумных границах."""
    bars = _synth_bars()
    size = vol_target_size(bars, target_vol=0.15, max_leverage=2.0)
    valid = size[size > 0]
    assert (valid <= 2.0 + 1e-9).all()
    assert len(valid) > 400
    print(f"  [ok] vol_target_size: медиана {valid.median():.2f}, "
          f"макс {valid.max():.2f}")


def test_engines_agree() -> None:
    """КРИТИЧНО: run_portfolio == run_engine на одном инструменте.

    Sanity-чек из SHORT_RESULTS.md — движки обязаны совпасть до 0.0%.
    """
    bars = _synth_bars()
    pos = (bars.close > bars.close.rolling(20).mean()).astype(float)
    diff = sanity_check_engines(bars, pos, cost=0.0002)
    assert diff < 1e-9, f"Движки разъехались: разница {diff:.2e}"
    print(f"  [ok] run_engine == run_portfolio: разница {diff:.2e}")


def test_portfolio_two_assets() -> None:
    """run_portfolio на двух инструментах: equal-weight работает."""
    b1, b2 = _synth_bars(seed=1), _synth_bars(seed=2)
    prices = pd.DataFrame({"A": b1.close, "B": b2.close})
    w = pd.DataFrame(0.5, index=prices.index, columns=["A", "B"])
    res = run_portfolio(prices, w, cost=0.0)
    assert res.gross.iloc[-1] == 1.0  # 0.5 + 0.5
    print(f"  [ok] run_portfolio 2 актива: ret={res.total_return:+.1%}, "
          f"dd={res.max_drawdown:.1%}")


def test_drift_turnover() -> None:
    """Drift-издержки (аудит 2026-07): дробный вес платит, полный — нет.

    Регрессия на «иллюзию нулевого оборота»: формула diff() давала 0
    издержек при постоянном целевом весе; drift-формула берёт плату за
    ежедневный ребаланс дробной доли NAV против движения цены.
    """
    bars = _synth_bars()
    # Полный вес 1.0: buy-and-hold, после входа издержек нет.
    full = pd.Series(1.0, index=bars.index)
    r_full_free = run_engine(bars, full, cost=0.0)
    r_full_cost = run_engine(bars, full, cost=0.0002)
    entry_only = r_full_free.total_return - r_full_cost.total_return
    assert entry_only < 0.001, f"B&H платит лишнее: {entry_only:.4%}"
    # Дробный вес 0.5: дрейф-ребаланс платный (раньше был бесплатным).
    half = pd.Series(0.5, index=bars.index)
    r_half_free = run_engine(bars, half, cost=0.0)
    r_half_cost = run_engine(bars, half, cost=0.0002)
    bleed = r_half_free.total_return - r_half_cost.total_return
    assert bleed > 0, "Дробный вес не платит за дрейф-ребаланс"
    print(f"  [ok] drift-издержки: B&H ~вход только "
          f"({entry_only*1e4:.1f}bps), вес 0.5 платит {bleed*1e4:.1f}bps")


def test_vol_target_buffer() -> None:
    """Буфер ребалансировки гасит дрожание размера (аудит 2026-07)."""
    bars = _synth_bars()
    raw = vol_target_size(bars, buffer=0.0)
    buf = vol_target_size(bars, buffer=0.10)
    ch_raw = (raw.diff().abs() > 1e-12).sum()
    ch_buf = (buf.diff().abs() > 1e-12).sum()
    assert ch_buf < ch_raw * 0.5, (
        f"Буфер не гасит дрожание: {ch_buf} vs {ch_raw}"
    )
    # Размеры остаются в окрестности сырых (буфер — мёртвая зона ±10%).
    diff_rel = ((buf - raw).abs() / raw.replace(0, 1)).max()
    assert diff_rel < 0.15
    print(f"  [ok] буфер vol targeting: смен размера {ch_raw} -> {ch_buf}, "
          f"откл. от сырого <= {diff_rel:.0%}")


def test_result_carries_bars_per_year() -> None:
    """bars_per_year — честное поле результата (не magic-атрибут).

    Регрессия: pandas 3.0 CoW терял атрибут Series, Sharpe тихо считался
    с 252 на H4-данных.
    """
    bars = _synth_bars()
    h4 = Bars(open=bars.open, high=bars.high, low=bars.low,
              close=bars.close, bars_per_year=1512.0, symbol="H4")
    pos = pd.Series(0.5, index=bars.index)
    r_d = run_engine(bars, pos, cost=0.0)
    r_h = run_engine(h4, pos, cost=0.0)
    assert r_d.bars_per_year == 252.0 and r_h.bars_per_year == 1512.0
    # Одинаковые ряды, разный bpy -> Sharpe различается в sqrt(6) раз.
    ratio = r_h.sharpe / r_d.sharpe
    assert abs(ratio - np.sqrt(1512 / 252)) < 1e-6
    print(f"  [ok] bars_per_year в результате: sharpe H4/D1 = {ratio:.3f} "
          f"= sqrt(6)")


if __name__ == "__main__":
    print("Тесты ядра:")
    test_bars_contract()
    test_misaligned_rejected()
    test_engine_basic()
    test_vol_target()
    test_engines_agree()
    test_portfolio_two_assets()
    test_drift_turnover()
    test_vol_target_buffer()
    test_result_carries_bars_per_year()
    print("Все тесты ядра пройдены.")
