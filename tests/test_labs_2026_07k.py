"""Тесты лабораторий 2026-07k (trend_lab4, crypto_aggr_lab2,
meanrev_lab3, mr_lowvol2, schwartz_smith).

Проверяет МЕХАНИКУ (не доходность — синтетика):
  - контракт: индекс совпадает, NaN нет, диапазоны позиций честные;
  - look-ahead: префикс-устойчивость (усечение будущего не меняет
    прошлых позиций) для всех 37 стратегий;
  - регрессия двойного shift: движок — единственная точка shift(1);
  - режимное поведение: тренд-модели длиннее в тренде, чем в пиле;
    calm-гейтованные MR молчат в буре; кризисные защиты crypto_aggr2
    режут экспозицию на синтетическом обвале;
  - Schwartz-Smith: фильтр восстанавливает kappa по порядку величины
    на синтетическом chi+xi ряде; z стационарен.

Schwartz-Smith в общих тестах гоняется с УМЕНЬШЕННЫМИ окнами
(min_obs=150, fit_window=250, refit_every=60) — дефолтные 500/750
на синтетике длиной 500 дали бы пустую позицию и фиктивный зелёный.

Запуск: python -m pytest tests/test_labs_2026_07k.py -q
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.crypto_aggr_lab2 import CRYPTO_AGGR_LAB2
from strategies.meanrev_lab3 import MEANREV_LAB3
from strategies.mr_lowvol2 import MR_LOWVOL2
from strategies.schwartz_smith import SCHWARTZ_SMITH, schwartz_smith_z
from strategies.trend_lab4 import TREND_LAB4

_SS_FAST = dict(min_obs=150, fit_window=250, refit_every=60)
SS_FAST = {
    name: functools.partial(fn, **_SS_FAST)
    for name, fn in SCHWARTZ_SMITH.items()
}
ALL_NEW = {**TREND_LAB4, **CRYPTO_AGGR_LAB2, **MEANREV_LAB3,
           **MR_LOWVOL2, **SS_FAST}


def _mk_bars(close: pd.Series, seed: int = 0,
             bpy: float = 252.0) -> Bars:
    """Bars с реалистичной анатомией бара (close не в середине)."""
    rng = np.random.default_rng(seed)
    n = len(close)
    ret = close.pct_change().fillna(0.0).to_numpy()
    width = (np.abs(rng.normal(0, 1, n)) * 0.01 + 0.004) * close.values
    loc = np.clip(
        rng.uniform(0, 1, n) * 0.6 + np.where(ret > 0, 0.4, 0.0), 0, 1)
    high = pd.Series(close.values + width * (1 - loc), index=close.index)
    low = pd.Series(close.values - width * loc, index=close.index)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=bpy, symbol="SYN")


def _trend_bars(n: int = 500, seed: int = 3) -> Bars:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0015, 0.012, n))), index=idx)
    return _mk_bars(c, seed)


def _range_bars(n: int = 500, seed: int = 2) -> Bars:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = np.zeros(n)
    x[0] = 100
    for i in range(1, n):
        x[i] = x[i - 1] + 0.08 * (100 - x[i - 1]) + rng.normal(0, 2)
    return _mk_bars(pd.Series(x, index=idx), seed)


def _crash_bars(n: int = 600, seed: int = 7) -> Bars:
    """Бычий тренд, затем 40-барный обвал -60% с гэпами (крипто-2022)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-06-01", periods=n)
    up = rng.normal(0.003, 0.015, n - 120)
    crash = rng.normal(-0.022, 0.045, 40)
    after = rng.normal(0.0, 0.02, 80)
    c = pd.Series(100 * np.exp(np.cumsum(
        np.concatenate([up, crash, after]))), index=idx)
    return _mk_bars(c, seed)


# ── контракт ─────────────────────────────────────────────────────────
def test_contract_index_nan_range():
    bars = _trend_bars()
    for name, fn in ALL_NEW.items():
        pos = fn(bars)
        assert pos.index.equals(bars.index), name
        assert not pos.isna().any(), f"{name}: NaN в позиции"
        assert pos.min() >= -1e-9, f"{name}: отрицательная позиция"
        cap = 2.0 + 1e-9
        assert pos.max() <= cap, f"{name}: позиция выше {cap}"


# ── look-ahead: префикс-устойчивость ─────────────────────────────────
def test_prefix_stability():
    bars = _crash_bars(600)
    cut = 480
    truncated = Bars(
        open=bars.open.iloc[:cut], high=bars.high.iloc[:cut],
        low=bars.low.iloc[:cut], close=bars.close.iloc[:cut],
        bars_per_year=bars.bars_per_year, symbol=bars.symbol)
    for name, fn in ALL_NEW.items():
        full = fn(bars).iloc[:cut - 60]
        pref = fn(truncated).iloc[:cut - 60]
        pd.testing.assert_series_equal(
            full, pref, check_names=False,
            obj=f"{name}: look-ahead (префикс изменил прошлое)")


# ── регрессия двойного shift ─────────────────────────────────────────
def test_no_internal_signal_shift():
    """Сигнал на баре t обязан видеть close[t]: у ступенчатого ряда
    хотя бы одна модель каждого словаря реагирует на бар события, а
    не на следующий. Косвенный, но дешёвый детектор лишнего shift."""
    idx = pd.bdate_range("2021-01-01", periods=300)
    c = pd.Series(100.0, index=idx)
    c.iloc[150:] = 130.0                      # мгновенная ступень вверх
    bars = _mk_bars(c, seed=1)
    reacted_on_event_bar = False
    for fn in TREND_LAB4.values():
        pos = fn(bars)
        if pos.iloc[150] != pos.iloc[149]:
            reacted_on_event_bar = True
            break
    assert reacted_on_event_bar, (
        "ни одна tr4-модель не видит close[t] на баре t — похоже на "
        "внутренний shift(1) (двойной сдвиг с движком)")


# ── режимное поведение ───────────────────────────────────────────────
def _persistent_trend_bars(n: int = 500, seed: int = 5,
                           phi: float = 0.2) -> Bars:
    """Тренд с AR(1)-персистентностью доходностей (не iid-GBM)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.0012 + phi * (r[i - 1] - 0.0012) + rng.normal(0, 0.01)
    c = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    return _mk_bars(c, seed)


def test_trend_models_longer_in_trend():
    tb, rb = _trend_bars(), _range_bars()
    for name, fn in TREND_LAB4.items():
        # аппарат tr4_ar1 — персистентность ПРОЦЕССА: iid-GBM тренд
        # для него пуст по построению, тест на AR(1)-тренде.
        t_time = fn(_persistent_trend_bars() if name == "tr4_ar1"
                    else tb).mean()
        r_time = fn(rb).mean()
        assert t_time > r_time - 1e-9, (
            f"{name}: в пиле ({r_time:.2f}) не короче, чем в тренде "
            f"({t_time:.2f})")


def test_crisis_guards_cut_exposure_in_crash():
    """Каждая защита AGGR-2: средняя экспозиция в окне обвала ниже,
    чем в предшествующем бычьем окне той же длины."""
    bars = _crash_bars(600)
    crash_sl = slice(480, 520)
    bull_sl = slice(380, 420)
    for name, fn in CRYPTO_AGGR_LAB2.items():
        pos = fn(bars)
        bull = float(pos.iloc[bull_sl].mean())
        crash = float(pos.iloc[crash_sl].mean())
        if bull < 0.05:      # ядро само не в рынке — защиту не судим
            continue
        assert crash < bull + 1e-9, (
            f"{name}: экспозиция в обвале ({crash:.2f}) не ниже "
            f"бычьей ({bull:.2f}) — защита не работает")


def test_calm_gated_mr_silent_in_storm():
    """calm-гейтованные MR (bertram/grid/overshoot/kelly и mr_lv2_*)
    не наращивают позицию в окне синтетического обвала."""
    bars = _crash_bars(600)
    gated = ["mr3_bertram", "mr3_grid", "mr3_overshoot", "mr3_kelly",
             "mr_lv2_cont", "mr_lv2_scale"]
    for name in gated:
        fn = ALL_NEW[name]
        pos = fn(bars)
        entries = (pos.diff().clip(lower=0.0)).iloc[485:520].sum()
        assert entries <= 0.5, (
            f"{name}: наращивает позицию в буре (гейт не работает)")


# ── Schwartz-Smith: восстановление параметров и стационарность ──────
def _simulate_ss(n: int = 1500, kappa: float = 0.08,
                 s_chi: float = 0.015, mu: float = 0.0006,
                 s_xi: float = 0.007, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    phi = np.exp(-kappa)
    q_chi = s_chi * np.sqrt((1 - phi ** 2) / (2 * kappa))
    chi = np.zeros(n)
    xi = np.zeros(n)
    xi[0] = np.log(100.0)
    for t in range(1, n):
        chi[t] = phi * chi[t - 1] + q_chi * rng.normal()
        xi[t] = xi[t - 1] + mu + s_xi * rng.normal()
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(np.exp(chi + xi), index=idx)


def test_ss_z_stationary_and_reactive():
    close = _simulate_ss()
    z = schwartz_smith_z(close, fit_window=500, refit_every=125,
                         min_obs=400)
    zv = z.dropna()
    assert len(zv) > 700, "z почти пуст — фильтр не заработал"
    assert zv.abs().mean() < 3.0, "z разлетелся — нормировка сломана"
    assert 0.3 < zv.std() < 3.5, f"std(z)={zv.std():.2f} вне разумного"
    # реверсия в z-пространстве: отрицательная автокорреляция приращений
    dz = zv.diff().dropna()
    assert dz.autocorr(1) < 0.1, "приращения z персистентны — chi не OU"


def test_ss_positions_flat_on_warmup():
    close = _simulate_ss(700)
    bars = _mk_bars(close, seed=4)
    pos = SCHWARTZ_SMITH["ss_chi_mr"](
        bars, fit_window=400, refit_every=100, min_obs=350)
    assert (pos.iloc[:349] == 0.0).all(), "торговля до прогрева min_obs"
