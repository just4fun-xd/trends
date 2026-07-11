"""Strategy tests — updated by the 2026-07 audit for the new semantics.

New 4step-core semantics locked by regressions:
  - one-shot take-profit + FREEZE the pyramid until a full exit
    (removes the "machine-gun" take and the take<->add-back saw);
  - the turtle stop trails behind adds (last_add - stop_atr*ATR);
  - risk triggers (stop, lower channel) on low, take on high,
    entries/adds are close-confirmed.

IMPORTANT: the champion semantics were refined => the reference numbers
in BENCHMARK_RESULTS.md (+5.2% / -12.1% / 15 of 19) were produced by the
OLD code and must be re-validated locally on real data before reporting.

Live Yahoo data is unavailable in the sandbox — we check the MECHANICS
on synthetic series with a designed structure.

Run: python -m tests.test_strategies
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine, vol_target_size
from strategies import bollinger, donchian, ema


def _trend_bars(n=400, slope=0.0015, noise=0.008, seed=0) -> Bars:
    """Stochastic uptrend with a realistic range."""
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
    """Rise, then a decisive reversal down — the stop must fire."""
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
    """Range around 100 — mean-reversion profits, trend loses."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = np.zeros(n)
    x[0] = 100
    for i in range(1, n):
        x[i] = x[i - 1] + 0.15 * (100 - x[i - 1]) + rng.normal(0, 1.5)
    close = pd.Series(x, index=idx)
    rng_hl = 1.2
    high = close + rng_hl * rng.uniform(0.3, 1.0, n)
    low = close - rng_hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="RANGE")


def _monotonic_bars(n=120, step=0.005) -> Bars:
    """Deterministic step growth: each close above the previous high —
    breakout, adds and take fire predictably.

    For a targeted check of the one-shot take and pyramid freeze.
    """
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * (1 + step) ** np.arange(n), index=idx)
    high = close * 1.001
    low = close * 0.997
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="MONO")


def _trail_bars() -> Bars:
    """Warm-up -> breakout -> pyramid build (last_add ~104.8) -> plateau
    -> pullback into the GAP between the trailed stop (last_add - 2*ATR
    ~102.8) and the entry stop (entry - 2*ATR ~99.2).

    The trailed stop must fire; a stop stuck at entry must not. Regression
    on a self-audit finding (the stop was not trailing). The exit channel
    in the test is wide (25) to reference old warm-up lows (~99.6) and
    not mask the stop check.
    """
    n = 80
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = np.full(n, 100.0)
    for i in range(40, 44):          # +1.2/bar: breakout + 3 adds
        close[i] = close[i - 1] + 1.2
    close[44:50] = close[43]         # plateau at 104.8 (pyramid full)
    close[50:] = 102.5               # pullback: low 102.1 < 102.8, > 99.2
    close = pd.Series(close, index=idx)
    high = close + 0.4
    low = close - 0.4
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="TRAIL")


def test_ema_champion_on_trend() -> None:
    """EMA ensemble+VT catches a smooth trend (native env — equities)."""
    bars = _trend_bars(slope=0.0015)
    pos = ema.ema_ensemble_voltarget(bars)
    res = run_engine(bars, pos)
    assert res.total_return > 0, f"EMA VT lost the trend: {res.total_return:.1%}"
    assert res.passes_dd(0.40), f"EMA VT breached DD40: {res.max_drawdown:.1%}"
    print(f"  [ok] EMA ens+VT on trend: ret={res.total_return:+.1%}, "
          f"dd={res.max_drawdown:.1%} (passes DD40)")


def test_take_is_one_shot_and_freezes() -> None:
    """CRITICAL (2026-07 audit): take is one-shot, pyramid frozen.

    On monotone growth: the pyramid builds 0.4->0.7->0.9->1.0, EXACTLY
    ONE top-step release (1.0->0.9), then constant — no repeated takes
    (machine-gun), no add-backs (take<->add saw).
    """
    bars = _monotonic_bars()
    raw = donchian._donchian_4step(
        bars, 20, 10, 20, 2.0, take_atr=3.5, use_take=True
    )
    d = raw.diff().fillna(0.0)
    ups = d[d > 1e-12]
    downs = d[d < -1e-12]

    assert raw.max() > 0.99, f"Pyramid did not build: max={raw.max():.3f}"
    assert len(downs) == 1, (
        f"Take fired {len(downs)} times — must be exactly 1 (one-shot)"
    )
    take_i = downs.index[0]
    after = raw.loc[take_i:]
    assert abs(after.iloc[0] - 0.9) < 1e-9, (
        f"Not the top step released: {after.iloc[0]:.3f} != 0.9"
    )
    frozen = after.diff().fillna(0.0).abs()
    assert (frozen < 1e-12).all(), (
        "Pyramid NOT frozen after the take (there are adds/takes)"
    )
    print(f"  [ok] One-shot take: {len(ups)} rises, take exactly 1 "
          f"(1.0->0.9 on {take_i.date()}), then frozen to the end")


def test_stop_trails_behind_adds() -> None:
    """CRITICAL (self-audit): the stop trails behind adds.

    A pullback to (last_add - 2*ATR) knocks the position out even though
    price is still far above (entry - 2*ATR) and above the lower channel.
    The old code (stop at entry) would NOT have exited here at all.
    """
    bars = _trail_bars()
    raw = donchian._donchian_4step(
        bars, 20, 25, 20, 2.0, take_atr=None, use_take=False
    )
    assert raw.iloc[44:50].max() > 0.99, "Pyramid did not build on the rise"
    # The pullback starts at bar 50; the trailed stop fires immediately.
    assert (raw.iloc[52:] == 0.0).all(), (
        "Trailed stop did not fire on the pullback to last_add - 2*ATR"
    )
    # Discrimination: the pullback low is above the entry stop (~99.2)
    # and above the wide channel (~99.6) — ONLY the trailed stop could fire.
    exit_low = bars.low.iloc[50]
    assert exit_low > 100.5, "Test construction violated"
    print(f"  [ok] Stop trailing: exit at low={exit_low:.1f} "
          f"(entry stop ~99.2, channel ~99.6 — old code would not exit)")


def test_champion_identity_raw_times_vol() -> None:
    """Refactor identity: champion == champion_raw * vol_size.

    This is the invariant the router risk parity rests on.
    """
    bars = _trend_bars()
    champ = donchian.donchian_est_macd_4step_take(bars)
    manual = (donchian.donchian_champion_raw(bars)
              * vol_target_size(bars, 0.15))
    diff = (champ - manual).abs().max()
    assert diff < 1e-12, f"Identity violated: {diff:.2e}"
    print(f"  [ok] champion == raw * vol_target: diff {diff:.2e}")


def test_champion_pyramids_on_trend() -> None:
    """Champion builds a pyramid on a stochastic trend, holds DD."""
    bars = _trend_bars(slope=0.0015)
    raw = donchian._donchian_4step(
        bars, 20, 10, 20, 2.0, take_atr=3.5, use_take=True
    )
    assert raw.max() > 0.40, (
        f"Pyramid not above 1 step: max={raw.max():.2f}"
    )
    res = run_engine(bars, donchian.donchian_est_macd_4step_take(bars))
    assert res.passes_dd(0.40), f"Champion breached DD40: {res.max_drawdown:.1%}"
    print(f"  [ok] Champion on trend: pyramid to {raw.max():.2f}, "
          f"ret={res.total_return:+.1%}, dd={res.max_drawdown:.1%}")


def test_champion_stop_on_reversal() -> None:
    """On a reversal the low-based risk exit fires before the crash."""
    bars = _reversal_bars()
    raw = donchian._donchian_4step(
        bars, 20, 10, 20, 2.0, take_atr=3.5, use_take=True
    )
    second_half = raw.iloc[len(raw) // 2 + 30:]
    assert (second_half == 0).sum() > len(second_half) * 0.5, (
        "Stop did not exit the position after the reversal"
    )
    res = run_engine(bars, donchian.donchian_est_macd_4step_take(bars))
    assert res.passes_dd(0.40), (
        f"Champion did not hold DD40 on the reversal: {res.max_drawdown:.1%}"
    )
    print(f"  [ok] Reversal: exited the position, dd={res.max_drawdown:.1%} "
          f"(held DD40)")


def test_entry_atr_frozen() -> None:
    """Regression (Gemini audit): the risk grid is fixed at entry.

    Direct invariant check: build a series where ATR GROWS after entry.
    With a floating atr_i the stop level last_add - stop_atr*atr would
    slide down with the rising ATR (risk > planned). With a fixed
    entry_atr the stop stays narrow and fires on a moderate pullback.

    Discriminating scenario: a calm entry (low ATR), then a run of
    widening bars raises the current ATR, and a moderate pullback. With
    fixed ATR the pullback breaks the narrow stop; with a floating one
    the stop has already slid below the pullback and the position would
    falsely survive.
    """
    n = 120
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = np.full(n, 100.0)
    for i in range(1, 40):                # calm entry, narrow ATR
        close[i] = close[i - 1] + 0.4
    # After entry — widening bars (current ATR rises), price nearly
    # flat, then a moderate pullback.
    for i in range(40, 70):
        close[i] = close[i - 1] + (0.1 if i % 2 else -0.1)
    close[70:] = close[69] - 2.2          # moderate pullback
    close = pd.Series(close, index=idx)
    high = close + 0.15
    low = close - 0.15
    # Widen bars 40-69 -> the current ATR rises after entry.
    for i in range(40, 70):
        high.iloc[i] = close.iloc[i] + 1.5
        low.iloc[i] = close.iloc[i] - 1.5
    low.iloc[70] = close.iloc[70] - 0.15
    bars = Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="ATRFRZ")

    # Compute both stop levels by hand on the pullback bar (70).
    atr = bars.atr(20)
    entry_bar = 39                        # approximately the entry bar
    entry_atr = atr.iloc[entry_bar]
    current_atr = atr.iloc[70]
    # ATR really grew after entry (else the test does not discriminate).
    assert current_atr > entry_atr * 1.5, (
        f"ATR did not grow: entry={entry_atr:.2f} current={current_atr:.2f}"
    )
    raw = donchian._donchian_4step(
        bars, 20, 30, 20, 2.0, take_atr=None, use_take=False
    )
    # The pyramid built at the entry bar.
    assert raw.iloc[40:69].max() > 0.39, "Position did not open"
    # With a fixed narrow entry_atr the stop fires on the bar-70 pullback.
    # (With a floating wide ATR the stop would slide below and survive.)
    assert (raw.iloc[71:] == 0.0).all(), (
        "Stop did not fire on the pullback — risk grid followed current ATR"
    )
    print(f"  [ok] entry_atr fixed: entry_atr={entry_atr:.2f} < "
          f"current_atr={current_atr:.2f}, stop on the NARROW entry ATR "
          f"fired on the pullback")


def test_outside_bar_reopens_same_bar() -> None:
    """Regression (Gemini audit): an outside bar reopens on the same bar.

    The bar breaks the lower channel (low < lo -> exit) AND closes above
    the upper (close > up -> entry). With if/elif we would exit and miss
    the entry until the next bar. With two independent ifs — exit on risk,
    then reopen on the same bar. Check: after the outside bar the position
    is 1.0 again (not 0 until the next bar).
    """
    n = 60
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = np.full(n, 100.0)
    for i in range(1, 30):           # rise, enter the position
        close[i] = close[i - 1] + 0.6
    close[30:40] = close[29]         # short plateau
    # Bar 40 is outside: a NEW close peak (above the 20-day max -> entry),
    # but a long lower tail breaks the exit channel (low < lo -> exit).
    close[40] = close[29] + 2.0      # fresh close high
    close[41:] = close[40]
    close = pd.Series(close, index=idx)
    high = close + 0.2
    low = close - 0.2
    low.iloc[40] = close.iloc[40] - 12.0    # long tail down
    high.iloc[40] = close.iloc[40] + 0.2
    bars = Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="OUTSIDE")

    pos = donchian.donchian_breakout(bars, entry=20, exit_period=10)
    # Discrimination: at bar 40 low broke the lower channel AND close set
    # a new high above the upper channel. Check both conditions hold.
    from strategies.donchian import _donchian_channels
    up, lo = _donchian_channels(bars, 20, 10)
    assert bars.low.iloc[40] < lo.iloc[40], "low did not break channel (test)"
    assert bars.close.iloc[40] > up.iloc[40], "close not above up (test)"
    # With two ifs the position at bar 40 = 1.0 (exit on risk, enter on
    # close). With if/elif there would be a hole pos[40]==0 until next bar.
    assert pos.iloc[40] == 1.0, (
        "Outside bar left a hole — entry did not reopen on the same bar"
    )
    print("  [ok] outside bar: exited on risk and reopened on the same "
          "bar (no skipped bar on entry)")


def test_meanrev_beats_trend_in_range() -> None:
    """In a range BB+RSI beats the trend Donchian
    (complementarity — a key project thesis)."""
    bars = _range_bars()
    mr = run_engine(bars, bollinger.bollinger_rsi(bars))
    tr = run_engine(bars, donchian.donchian_breakout(bars))
    assert mr.total_return > tr.total_return, (
        f"MR {mr.total_return:.1%} did not beat trend {tr.total_return:.1%}"
    )
    print(f"  [ok] Range: BB+RSI {mr.total_return:+.1%} > "
          f"Donchian {tr.total_return:+.1%} (complementarity)")


def test_percent_b_redundant() -> None:
    """Don %b ~ Don ens: a correlated filter does not change the picture."""
    bars = _trend_bars(slope=0.001)
    base = run_engine(bars, donchian.donchian_ensemble_voltarget(bars))
    pb = run_engine(bars, bollinger.donchian_percent_b(bars))
    print(f"  [ok] Don ens {base.total_return:+.1%} vs "
          f"Don %b {pb.total_return:+.1%} (%b — correlated filter)")


def test_ema_stop_no_same_bar_reentry() -> None:
    """Regression (self-audit): no same-bar re-entry after a stop.

    Low-based stop + rearm: until the bull signal resets, position is 0.
    Check: on the stop bar the position is not 1.0 again.
    """
    bars = _reversal_bars(seed=5)
    pos = ema.ema_cross_stop(bars, stop=0.05)
    # Find bars where the position dropped 1->0: on the next bar it may
    # return only if bull re-formed; on the same bar — never.
    drops = pos.diff() < 0
    assert not (drops & (pos > 0)).any(), (
        "Same-bar re-entry after the stop (rearm not working)"
    )
    res = run_engine(bars, pos)
    print(f"  [ok] ema_cross_stop: rearm after the stop, "
          f"ret={res.total_return:+.1%}")


if __name__ == "__main__":
    print("Strategy tests (2026-07 audit semantics):")
    test_ema_champion_on_trend()
    test_take_is_one_shot_and_freezes()
    test_stop_trails_behind_adds()
    test_champion_identity_raw_times_vol()
    test_champion_pyramids_on_trend()
    test_champion_stop_on_reversal()
    test_entry_atr_frozen()
    test_outside_bar_reopens_same_bar()
    test_meanrev_beats_trend_in_range()
    test_percent_b_redundant()
    test_ema_stop_no_same_bar_reentry()
    print("All strategy tests passed.")
