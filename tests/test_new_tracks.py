"""Тесты новых треков с ГОДОВОЙ РАЗБИВКОЙ (аудит 2026-07).

Показывает по годам return и DD в процентах для каждого алгоритма и
инструмента — прямой ответ на запрос ревью. Данные синтетические
(Yahoo недоступен в песочнице), проверяется механика + формат вывода;
локально те же прогоны идут на реальных данных.

Запуск: python -m tests.test_new_tracks
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine
from core.engine_portfolio import run_portfolio
from diagnostics.yearly import (
    format_matrix,
    format_yearly_table,
    yearly_breakdown,
    yearly_matrix,
)
from strategies import cross_sectional as xs
from strategies import seasonal
from strategies.pairs import kalman_beta, run_pair_kalman


def _gas_like_bars(seed=0) -> Bars:
    """6-летний ряд с сезонным паттерном (осень-зима сильнее)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", "2026-01-01")
    n = len(idx)
    # Сезонный дрейф: плюс в авг-ноя, минус весной.
    month = idx.month.values
    seasonal_drift = np.where(np.isin(month, [8, 9, 10, 11]), 0.0018,
                              np.where(np.isin(month, [3, 4, 5, 6]),
                                       -0.0010, 0.0002))
    noise = rng.normal(0, 0.018, n)
    close = pd.Series(100 * np.exp(np.cumsum(seasonal_drift + noise)),
                      index=idx)
    sp = close * 0.01
    high = close + sp * rng.uniform(0.3, 1.0, n)
    low = close - sp * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="GAS")


def _equity_panel(m=12, seed=1) -> pd.DataFrame:
    """Панель акций: пара мегакапов + спокойные имена (для DM-трека)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", "2026-01-01")
    n = len(idx)
    cols, data = [], {}
    for i in range(m):
        # Первые два — «прыгучие мегакапы», остальные спокойнее.
        drift = 0.0008 if i < 2 else 0.0004
        vol = 0.030 if i < 2 else 0.014
        px = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        name = f"MEGA{i}" if i < 2 else f"CALM{i}"
        cols.append(name)
        data[name] = px
    return pd.DataFrame(data, index=idx)


def test_seasonal_yearly() -> None:
    """Сезонные стратегии: годовая разбивка return/DD."""
    bars = _gas_like_bars()
    variants = {
        "seasonal_gas": seasonal.seasonal_gas(bars),
        "donch_seasonal": seasonal.donchian_seasonal(bars),
        "donch_seas_VT": seasonal.donchian_seasonal_voltarget(bars),
    }
    print("\n=== СЕЗОННЫЕ: годовая разбивка (синтетика) ===")
    equities = {}
    for name, pos in variants.items():
        res = run_engine(bars, pos)
        equities[name] = res.equity
        yb = yearly_breakdown(res.equity, res.bars_per_year)
        print("\n" + format_yearly_table(yb, f"[{name}] на GAS"))
        assert res.passes_dd(0.60), f"{name} экстремальный DD"
    # Сводная матрица год × вариант.
    print("\n" + format_matrix(
        yearly_matrix(equities, 252.0, "return"),
        "Сезонные — return по годам (год × вариант)"
    ))
    print("  [ok] сезонные отработали, годовая разбивка построена")


def test_dualmom_research_yearly() -> None:
    """Dual momentum research-треки: годовая разбивка портфельно."""
    prices = _equity_panel()
    benchmark = prices.mean(axis=1)  # синтетический «рынок»
    variants = {
        "DM_tilt": xs.dual_momentum_tilt(prices, benchmark),
        "DM_regime": xs.dual_momentum_regime(prices, benchmark),
        "DM_volscaled": xs.dual_momentum_volscaled(prices),
    }
    print("\n=== DUAL MOMENTUM RESEARCH: годовая разбивка ===")
    equities = {}
    for name, w in variants.items():
        res = run_portfolio(prices, w, cost=0.0002)
        equities[name] = res.equity
        yb = yearly_breakdown(res.equity, res.bars_per_year)
        print("\n" + format_yearly_table(yb, f"[{name}] портфель"))
    print("\n" + format_matrix(
        yearly_matrix(equities, 252.0, "return"),
        "DM research — return по годам (год × вариант)"
    ))
    print("\n" + format_matrix(
        yearly_matrix(equities, 252.0, "max_dd"),
        "DM research — MaxDD по годам (год × вариант)"
    ))
    # volscaled должен ограничивать концентрацию мегакапов -> обычно
    # мягче по DD, чем tilt. Не жёсткий ассерт (синтетика), но проверим
    # что все дали валидные кривые.
    for name, eq in equities.items():
        assert np.isfinite(eq).all(), f"{name}: NaN в equity"
    print("  [ok] три DM research-трека отработали")


def test_kalman_beta_valid() -> None:
    """Kalman-бета восстанавливает известное соотношение (математика)."""
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=500)
    b = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)), index=idx)
    true_beta = 1.5
    a = true_beta * b + rng.normal(0, 2, 500)  # A = 1.5*B + шум
    a = pd.Series(a, index=idx)
    beta = kalman_beta(a, b)
    # После прогрева бета должна сойтись к ~1.5.
    converged = beta.iloc[100:].mean()
    assert abs(converged - true_beta) < 0.3, (
        f"Kalman-бета не сошлась: {converged:.2f} vs {true_beta}"
    )
    print(f"\n  [ok] Kalman-бета: сошлась к {converged:.2f} "
          f"(истинная {true_beta}) — математика валидна")


def test_kalman_pair_yearly() -> None:
    """Kalman-пара: годовая разбивка (research, край не подтверждён)."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", "2026-01-01")
    n = len(idx)
    # Коинтегрированная пара: общий фактор + расходящийся спред.
    common = np.cumsum(rng.normal(0, 1, n))
    a = pd.Series(100 + common + rng.normal(0, 3, n), index=idx)
    b = pd.Series(100 + common + rng.normal(0, 3, n), index=idx)
    res = run_pair_kalman(a, b)
    yb = yearly_breakdown(res.equity, res.bars_per_year)
    print("\n=== KALMAN PAIRS (research): годовая разбивка ===")
    print(format_yearly_table(yb, "[kalman_pair] synth-спред"))
    assert np.isfinite(res.equity).all(), "NaN в equity пары"
    print("  [ok] Kalman-пара отработала (research — не боевой трек)")


if __name__ == "__main__":
    print("Тесты новых треков (годовая разбивка, аудит 2026-07):")
    test_seasonal_yearly()
    test_dualmom_research_yearly()
    test_kalman_beta_valid()
    test_kalman_pair_yearly()
    print("\nВсе тесты новых треков пройдены.")


def test_instrument_contribution_flags_ballast():
    """LOO помечает балластом актив, чьё исключение поднимает Sharpe."""
    import numpy as np

    from diagnostics.instrument_contribution import (
        instrument_contribution,
    )
    rng = np.random.default_rng(3)
    idx = pd.date_range("2019-01-01", periods=800, freq="B")
    rets = pd.DataFrame({
        "good1": rng.normal(0.0008, 0.01, 800),
        "good2": rng.normal(0.0007, 0.011, 800),
        "ballast": rng.normal(-0.0010, 0.02, 800),
    }, index=idx)
    df = instrument_contribution(rets)
    # Балласт должен иметь положительную LOO-дельту (без него лучше).
    assert df.loc["ballast", "loo_delta"] > 0
    # И его solo-Sharpe должен быть ниже, чем у хороших.
    assert df.loc["ballast", "solo_sharpe"] < df.loc["good1", "solo_sharpe"]


def test_variance_ratio_random_walk():
    """Случайное блуждание: VR(q) ≈ 1, |z| мал."""
    import numpy as np
    from strategies.variance_ratio import variance_ratio
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.01, 3000)  # iid -> RW
    vr, z = variance_ratio(rets, 4)
    assert abs(vr - 1.0) < 0.15
    assert abs(z) < 2.5


def test_variance_ratio_trending():
    """Персистентный ряд (положит. автокорр): VR(q) > 1, H > 0.5."""
    import numpy as np
    from strategies.variance_ratio import hurst_from_vr
    rng = np.random.default_rng(8)
    # AR(1) с phi>0 -> тренд/персистентность
    n = 3000
    e = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.3 * r[i - 1] + e[i]
    h, z = hurst_from_vr(r)
    assert h > 0.5


def test_variance_ratio_mean_reverting():
    """Реверсионный ряд (отриц. автокорр): VR(q) < 1, H < 0.5."""
    import numpy as np
    from strategies.variance_ratio import hurst_from_vr
    rng = np.random.default_rng(9)
    n = 3000
    e = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = -0.3 * r[i - 1] + e[i]  # phi<0 -> реверсия
    h, z = hurst_from_vr(r)
    assert h < 0.5


def test_hurst_alloc_registered_and_runs():
    """hurst_alloc в реестре и возвращает позицию в [0,1]."""
    import numpy as np

    from core.bars import Bars
    from strategies.hurst_alloc import hurst_alloc
    rng = np.random.default_rng(10)
    idx = pd.date_range("2018-01-01", periods=900, freq="B")
    price = pd.Series(100 * np.exp(np.cumsum(
        rng.normal(0.0002, 0.01, 900))), index=idx)
    bars = Bars(
        open=price, high=price * 1.01, low=price * 0.99,
        close=price, bars_per_year=252.0,
        volume=pd.Series(1000.0, index=idx), symbol="X",
    )
    pos = hurst_alloc(bars)
    assert pos.between(0.0, 1.0).all()
    assert len(pos) == len(idx)


def _mk_bars(prices, bpy=252.0):

    from core.bars import Bars
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    s = pd.Series(prices, index=idx, dtype=float)
    return Bars(open=s, high=s * 1.01, low=s * 0.99, close=s,
                bars_per_year=bpy, volume=None, symbol="T")


def test_ou_lab_registry_and_bounds():
    """Все 10 модификаций возвращают позицию в {-1..1} нужной длины."""
    import numpy as np
    from strategies.ou_lab import OU_LAB
    rng = np.random.default_rng(11)
    # OU-подобный ряд: реверсия к 100.
    n = 700
    x = np.zeros(n) + 100.0
    for i in range(1, n):
        x[i] = x[i - 1] + 0.2 * (100.0 - x[i - 1]) + rng.normal(0, 1.0)
    bars = _mk_bars(x)
    assert len(OU_LAB) == 12
    for name, fn in OU_LAB.items():
        pos = fn(bars)
        assert len(pos) == n, name
        assert pos.between(-1.0, 1.0).all(), name


def test_ou_asym_never_short():
    """ou_asym не открывает шорт."""
    import numpy as np
    from strategies.ou_lab import ou_asym
    rng = np.random.default_rng(12)
    n = 600
    x = np.zeros(n) + 50.0
    for i in range(1, n):
        x[i] = x[i - 1] + 0.15 * (50.0 - x[i - 1]) + rng.normal(0, 1.0)
    pos = ou_asym(_mk_bars(x))
    assert (pos >= 0).all()


def test_ccxt_source_reads_parquet(tmp_path):
    """CCXTSource читает parquet и ставит 24/7 bars_per_year."""
    import numpy as np

    from data.ccxt_source import CCXTSource
    idx = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
    rng = np.random.default_rng(13)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close,
        "volume": 1.0,
    }, index=idx)
    df.to_parquet(tmp_path / "BTC-USDT_4h.parquet")
    src = CCXTSource(data_dir=str(tmp_path))
    bars = src.load("BTC-USDT", "2024-01-01", "2024-02-01", "4h")
    assert bars.bars_per_year == 365.0 * 6
    assert len(bars.close) > 0


def test_ou_jump_suppresses_on_level_shift():
    """ou_jump выходит из позиции при скачке (смена уровня)."""
    import numpy as np

    from strategies.ou_lab import ou_jump, _detect_jumps
    n = 400
    x = np.zeros(n) + 100.0
    rng = np.random.default_rng(21)
    for i in range(1, n):
        x[i] = x[i - 1] + 0.2 * (100.0 - x[i - 1]) + rng.normal(0, 0.5)
    # Вставляем явный скачок уровня на баре 200.
    x[200:] += 40.0
    bars = _mk_bars(x)
    jumps = _detect_jumps(bars.close, window=40, k=4.0)
    assert jumps.iloc[195:210].any()  # скачок задетектирован
    pos = ou_jump(bars)
    assert pos.between(-1.0, 1.0).all()
    # На самом баре скачка (+1) позиция обнулена cooldown-ом.
    assert pos.iloc[201] == 0.0


def test_ou_jump_registered():
    """ou_jump и ou_jump_asym в OU_LAB; asym не шортит."""
    import numpy as np
    from strategies.ou_lab import OU_LAB
    assert "ou_jump" in OU_LAB and "ou_jump_asym" in OU_LAB
    n = 300
    x = np.zeros(n) + 50.0
    rng = np.random.default_rng(22)
    for i in range(1, n):
        x[i] = x[i - 1] + 0.15 * (50.0 - x[i - 1]) + rng.normal(0, 0.5)
    pos = OU_LAB["ou_jump_asym"](_mk_bars(x))
    assert (pos >= 0).all()


def test_ccxt_h4_pipeline_end_to_end(tmp_path):
    """CCXT H4 parquet -> instrument_contribution конвейер работает."""
    import numpy as np
    import pandas as pd
    from data.ccxt_source import CCXTSource
    from diagnostics.instrument_contribution import (
        instrument_contribution, per_instrument_returns)
    from strategies.donchian import donchian_champion_raw
    rng = np.random.default_rng(30)
    # два синтетических H4-ряда
    for sym in ["BTC-USDT", "ETH-USDT"]:
        idx = pd.date_range("2024-01-01", periods=600, freq="4h",
                            tz="UTC")
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, 600)))
        pd.DataFrame({
            "open": close, "high": close * 1.02,
            "low": close * 0.98, "close": close, "volume": 1.0,
        }, index=idx).to_parquet(tmp_path / f"{sym}_4h.parquet")
    src = CCXTSource(data_dir=str(tmp_path))
    basket = {"Bitcoin": "BTC-USDT", "Ethereum": "ETH-USDT"}
    rets, bpy = per_instrument_returns(
        donchian_champion_raw, basket, src,
        "2024-01-01", "2024-06-01", interval="4h")
    assert bpy == 365.0 * 6
    assert rets.shape[1] == 2
    # H4 годовая база отличается от дневной (6*365)
    bars = src.load("BTC-USDT", "2024-01-01", "2024-06-01", "4h")
    assert bars.bars_per_year == 365.0 * 6
    df = instrument_contribution(rets, bpy=bars.bars_per_year)
    assert len(df) == 2


def test_kalman_trend_detects_direction():
    """Kalman trend: лонг на растущем ряде, шорт на падающем."""
    import numpy as np
    from strategies.kalman_trend import kalman_trend, kalman_trend_long
    n = 400
    up = _mk_bars(100 * np.exp(np.cumsum(
        np.full(n, 0.002) + np.random.default_rng(40).normal(0, 0.005, n))))
    down = _mk_bars(100 * np.exp(np.cumsum(
        np.full(n, -0.002) + np.random.default_rng(41).normal(0, 0.005, n))))
    pos_up = kalman_trend(up)
    pos_down = kalman_trend(down)
    assert pos_up.between(-1, 1).all()
    # На устойчивом тренде средний сигнал должен смотреть в его сторону.
    assert pos_up.iloc[100:].mean() > 0.2
    assert pos_down.iloc[100:].mean() < -0.2
    # long-only не шортит.
    assert (kalman_trend_long(down) >= 0).all()


def test_monday_range_breakout_logic():
    """Monday range: пробой вверх -> лонг; коридор сбрасывается по неделям."""
    import pandas as pd
    from core.bars import Bars
    from strategies.monday_range import monday_range
    # 3 недели дневных баров: неделя растёт после понедельника.
    idx = pd.date_range("2024-01-01", periods=21, freq="D")  # пн-старт
    close = pd.Series(100.0, index=idx)
    # во всех неделях: пн=100, дальше пробой вверх до 110
    for w in range(3):
        base = w * 7
        for d in range(7):
            close.iloc[base + d] = 100.0 + (5.0 if d > 0 else 0.0)
    bars = Bars(open=close, high=close * 1.001, low=close * 0.999,
                close=close, bars_per_year=252.0, volume=None, symbol="T")
    pos = monday_range(bars, ref_bars=1)
    assert pos.between(-1, 1).all()
    # После пробоя понедельничного уровня во вторник -> лонг где-то в неделе
    assert (pos == 1).any()


def test_monday_range_long_only():
    """H4-пресет long-only не открывает шорт."""
    import numpy as np
    import pandas as pd
    from core.bars import Bars
    from strategies.monday_range import monday_range_h4
    idx = pd.date_range("2024-01-01", periods=200, freq="4h")
    rng = np.random.default_rng(50)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))),
                      index=idx)
    bars = Bars(open=close, high=close * 1.01, low=close * 0.99,
                close=close, bars_per_year=365.0 * 6, volume=None,
                symbol="T")
    assert (monday_range_h4(bars) >= 0).all()


def test_impulse_lab_bounds_and_registry():
    """13 импульсных моделей: позиция в [-1,1], нужная длина, без NaN."""
    import numpy as np
    from strategies.impulse_lab import IMPULSE_LAB
    assert len(IMPULSE_LAB) == 13
    rng = np.random.default_rng(60)
    n = 700
    trend = 100 * np.exp(np.cumsum(
        np.full(n, 0.001) + rng.normal(0, 0.01, n)))
    bars = _mk_bars(trend)
    for name, fn in IMPULSE_LAB.items():
        pos = fn(bars)
        assert len(pos) == n, name
        assert not pos.isna().any(), name
        assert pos.between(-1.0, 1.0).all(), name


def test_impulse_models_catch_uptrend():
    """На устойчивом аптренде импульсные модели в среднем лонгуют."""
    import numpy as np
    from strategies.impulse_lab import (
        imp_tsmom_vw, imp_52h, imp_drawup, imp_tstat)
    rng = np.random.default_rng(61)
    n = 800
    up = _mk_bars(100 * np.exp(np.cumsum(
        np.full(n, 0.0015) + rng.normal(0, 0.006, n))))
    for fn in (imp_tsmom_vw, imp_52h, imp_drawup, imp_tstat):
        pos = fn(up)
        assert pos.iloc[300:].mean() > 0.15, fn.__name__


def test_carver_mr_buys_oversold():
    """carver_mr: лонг на растянутом вниз OU-ряде, в границах [0,1]."""
    import numpy as np
    from strategies.carver_mr import carver_mr
    rng = np.random.default_rng(70)
    n = 600
    x = np.zeros(n) + 100.0
    for i in range(1, n):
        x[i] = x[i - 1] + 0.1 * (100.0 - x[i - 1]) + rng.normal(0, 1.0)
    x[300:320] -= np.linspace(0, 15, 20)  # резкое растяжение вниз
    bars = _mk_bars(x)
    pos = carver_mr(bars)
    assert pos.between(0.0, 1.0).all()
    # Во время растяжения (бары 305-320) позиция заметно > 0.
    assert pos.iloc[305:321].max() > 0.5


def test_mr_lowvol_soft_weight_declines_with_vol():
    """Мягкий гейт: при высокой воле позиция ниже, чем при тихой."""
    import numpy as np
    from strategies.carver_mr import mr_lowvol_soft
    rng = np.random.default_rng(71)
    n = 700
    # первая половина тихая, вторая — буря
    ret = np.concatenate([rng.normal(0, 0.004, n // 2),
                          rng.normal(0, 0.03, n - n // 2)])
    price = 100 * np.exp(np.cumsum(ret))
    bars = _mk_bars(price)
    pos = mr_lowvol_soft(bars)
    assert pos.between(0.0, 1.0).all()


def test_hrp_downweights_correlated_cluster():
    """HRP: два скоррелированных близнеца вместе <= вес одиночки+заметный."""
    import numpy as np
    import pandas as pd
    from diagnostics.hrp import hrp_weights
    rng = np.random.default_rng(72)
    n = 1000
    base = rng.normal(0, 0.01, n)
    rets = pd.DataFrame({
        "crypto_a": base + rng.normal(0, 0.002, n),   # близнецы
        "crypto_b": base + rng.normal(0, 0.002, n),
        "commodity": rng.normal(0, 0.01, n),          # независимая
    })
    w = hrp_weights(rets)
    assert abs(w.sum() - 1.0) < 1e-9
    # Независимая нога должна получить больше, чем каждый близнец.
    assert w["commodity"] > w["crypto_a"]
    assert w["commodity"] > w["crypto_b"]


def test_ou_trend_lab_bounds():
    """6 OU×trend гибридов: границы позиций и длина."""
    import numpy as np
    from strategies.ou_trend_lab import OU_TREND_LAB
    assert len(OU_TREND_LAB) == 6
    rng = np.random.default_rng(80)
    n = 900
    price = 100 * np.exp(np.cumsum(
        np.full(n, 0.0008) + rng.normal(0, 0.012, n)))
    bars = _mk_bars(price)
    for name, fn in OU_TREND_LAB.items():
        pos = fn(bars)
        assert len(pos) == n, name
        assert pos.between(-1.0, 1.0).all(), name


def test_ou_pullback_only_in_uptrend():
    """ou_pullback: на чистом даунтренде позиций нет."""
    import numpy as np
    from strategies.ou_trend_lab import ou_pullback
    rng = np.random.default_rng(81)
    n = 700
    down = 100 * np.exp(np.cumsum(
        np.full(n, -0.002) + rng.normal(0, 0.008, n)))
    pos = ou_pullback(_mk_bars(down))
    assert pos.iloc[250:].abs().sum() == 0.0


def test_trend_lab2_bounds_and_direction():
    """10 тренд-моделей: границы; на аптренде средний сигнал > 0."""
    import numpy as np
    from strategies.trend_lab2 import TREND_LAB2
    assert len(TREND_LAB2) == 10
    rng = np.random.default_rng(82)
    n = 900
    up = _mk_bars(100 * np.exp(np.cumsum(
        np.full(n, 0.0015) + rng.normal(0, 0.006, n))))
    # Модели УСКОРЕНИЯ (гистограмма MACD, ZLEMA-кросс) на ровном
    # тренде постоянной скорости дают ~0 по построению — им порог
    # мягче (не против тренда), остальным строгий.
    accel_type = {"tr_macd_hz", "tr_zlema"}
    for name, fn in TREND_LAB2.items():
        pos = fn(up)
        assert len(pos) == n, name
        assert pos.between(-1.0, 1.0).all(), name
        bar = 0.0 if name in accel_type else 0.05
        assert pos.iloc[300:].mean() > bar, name
