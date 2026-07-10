"""Тесты лабораторий 2026-07j (meanrev_lab2, trend_lab3, crypto_aggr).

Проверяет МЕХАНИКУ (не доходность — синтетика):
  - контракт: индекс совпадает, NaN нет, диапазоны позиций честные;
  - look-ahead: префикс-устойчивость (усечение будущего не меняет
    прошлых позиций) для ВСЕХ 40 стратегий;
  - регрессия двойного shift: движок — единственная точка shift(1),
    лаборатории не сдвигают сигнал сами (фикс 2026-07j);
  - режимное поведение: тренд-модели длиннее в тренде, чем в пиле;
    гейтованные MR молчат в тренде.

Запуск: python -m pytest tests/test_labs_2026_07j.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from strategies.crypto_aggr_lab import CRYPTO_AGGR_LAB
from strategies.meanrev_lab2 import MEANREV_LAB2
from strategies.trend_lab2 import _shift01
from strategies.trend_lab3 import TREND_LAB3

ALL_NEW = {**MEANREV_LAB2, **TREND_LAB3, **CRYPTO_AGGR_LAB}


def _mk_bars(close: pd.Series, seed: int = 0) -> Bars:
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
                close=close, bars_per_year=252.0, symbol="SYN")


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


def test_contract_all_new() -> None:
    """Индекс, NaN, диапазоны позиций для всех 40 новых стратегий."""
    bars = _trend_bars()
    for name, fn in ALL_NEW.items():
        pos = fn(bars)
        assert pos.index.equals(bars.index), name
        assert not pos.isna().any(), f"{name}: NaN в позиции"
        assert pos.max() <= 2.5 + 1e-9, f"{name}: позиция > 2.5"
        assert pos.min() >= -1.0 - 1e-9, f"{name}: позиция < -1"
        if name not in ("ca_squeeze_pop", "ca_burst", "ca_short_break"):
            assert pos.min() >= -1e-9, f"{name}: лонг-стратегия шортит"


def test_no_lookahead_prefix_stability() -> None:
    """Усечение будущих баров не меняет прошлых позиций.

    Стандартный детектор look-ahead: считаем позицию на полном ряду
    и на префиксе; если стратегия честная, позиции на общем участке
    совпадают (небольшой хвост допускается только у pivot-моделей,
    где подтверждение свинга задним числом — часть механизма, но и
    там прошлое не переписывается дальше pivot-окна).
    """
    bars = _trend_bars(400)
    cut = 320
    bars_cut = Bars(
        open=bars.open.iloc[:cut], high=bars.high.iloc[:cut],
        low=bars.low.iloc[:cut], close=bars.close.iloc[:cut],
        bars_per_year=252.0, symbol="SYN")
    for name, fn in ALL_NEW.items():
        full = fn(bars).iloc[:cut]
        pref = fn(bars_cut)
        pd.testing.assert_series_equal(
            full, pref, check_names=False,
            obj=f"{name}: look-ahead (префикс изменился)")


def test_double_shift_removed() -> None:
    """Регрессия фикса 2026-07j: хелперы лабораторий НЕ сдвигают.

    Движок run_engine — единственная точка shift(1). До фикса
    trend_lab2/_impulse_lab/kalman/carver_mr/monday_range/ou_trend
    сдвигали сигнал сами -> лаг t+2 против t+1 у donchian во всех
    bootstrap-сравнениях (фора чемпиону).
    """
    idx = pd.bdate_range("2020-01-01", periods=5)
    sig = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0], index=idx)
    pd.testing.assert_series_equal(_shift01(sig), sig)


def test_trend_models_prefer_trend() -> None:
    """Каждая тренд-модель дольше в рынке на тренде, чем в пиле."""
    bt, br = _trend_bars(), _range_bars()
    for name, fn in TREND_LAB3.items():
        if name == "tr3_vr_trend":
            continue  # у iid-синтетики VR=1: гейт закрыт всюду — норма
        in_tr = float((fn(bt) > 0).mean())
        in_rg = float((fn(br) > 0).mean())
        assert in_tr > in_rg, (
            f"{name}: в тренде {in_tr:.2f} <= в пиле {in_rg:.2f}")


def test_gated_mr_silent_in_trend() -> None:
    """Режимно-гейтованные MR почти молчат в устойчивом тренде."""
    bt = _trend_bars()
    for name in ("mr2_entropy", "mr2_vr", "mr2_percb_bw"):
        frac = float((MEANREV_LAB2[name](bt) > 0).mean())
        assert frac < 0.15, f"{name}: {frac:.2f} в рынке на тренде"


def test_pyramids_capped_and_exit() -> None:
    """Пирамиды ограничены потолком и реально выходят из позиции."""
    bt = _trend_bars(700)
    for name in ("ca_turbo_don", "ca_pyramid_max"):
        pos = CRYPTO_AGGR_LAB[name](bt)
        assert pos.max() <= 2.5 + 1e-9, name
        assert (pos == 0).any(), f"{name}: никогда не выходит"


def test_short_side_only_short() -> None:
    """ca_short_break не открывает лонгов."""
    pos = CRYPTO_AGGR_LAB["ca_short_break"](_range_bars())
    assert pos.max() <= 1e-9
    assert pos.min() >= -1.0 - 1e-9
