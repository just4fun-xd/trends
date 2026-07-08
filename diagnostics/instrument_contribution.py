"""Вклад инструментов в портфель: индивидуальные метрики + leave-one-out.

Отвечает на вопрос «не тянет ли какой-то актив портфель вниз?» —
которого НЕ видно в усреднённой комбо-строке. Портфельная строка
+16.8% / Sharpe +1.44 усредняет всё в кашу; PL с -15% и GC с +149%
неразличимы в агрегате.

Метод для каждого инструмента i:
  - solo: его собственные return / DD / Sharpe в equal-weight портфеле;
  - LOO delta: Sharpe(портфель без i) − Sharpe(полный портфель).
      delta > 0  — БЕЗ инструмента портфель ЛУЧШЕ  -> балласт/вред;
      delta < 0  — инструмент ПОДДЕРЖИВАЕТ портфель -> держать;
      delta ≈ 0  — нейтрален (разбавляет, не мешает).

Важно: Sharpe здесь GROSS (rf=0) — он инвариантен к масштабу позиции,
поэтому сравнение инструментов с разной волой честное (в отличие от
excess-Sharpe, где низковольный актив штрафуется искусственно).

Это диагностика для формирования гипотезы. Решение об исключении
инструмента — только после подтверждения на втором источнике
(правило двух источников) и проверки механизма (почему актив не
торгуется этой стратегией: не трендовый? тонкий? структурный сдвиг?).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.engine import run_engine


def _sharpe(rets: pd.Series, bpy: float = 252.0) -> float:
    """Gross annualized Sharpe (rf=0, ddof=1 — как в core.engine)."""
    std = rets.std(ddof=1)
    if std <= 0 or len(rets) < 2:
        return 0.0
    return float(rets.mean() / std * np.sqrt(bpy))


def _max_dd(rets: pd.Series) -> float:
    """Максимальная просадка по ряду доходностей."""
    eq = (1.0 + rets).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def per_instrument_returns(
    strategy_fn, basket: dict, source, start: str, end: str,
    sizer=None, cost: float = 0.0002, interval: str = "1d",
) -> pd.DataFrame:
    """Дневные доходности КАЖДОГО инструмента (колонки), выровнены.

    Args:
        strategy_fn: Bars -> position.
        basket: dict {название: тикер}.
        source: DataSource.
        start, end, interval: период/таймфрейм.
        sizer: callable(bars)->множитель позиции или None.
        cost: издержки движка.

    Returns:
        (DataFrame, bars_per_year): индекс — даты, колонки — инструменты;
        побарный P&L (после издержек). NaN там, где инструмент не
        торговался в этот день.
    """
    per_inst = {}
    bpy = 252.0
    for name, ticker in basket.items():
        try:
            bars = source.load(ticker, start, end, interval)
        except Exception as exc:  # noqa: BLE001
            print(f"  пропуск {name} ({ticker}): {exc}")
            continue
        bpy = bars.bars_per_year
        pos = strategy_fn(bars)
        if sizer is not None:
            pos = pos * sizer(bars)
        res = run_engine(bars, pos, cost=cost)
        per_inst[name] = res.equity.pct_change()
    if not per_inst:
        raise RuntimeError("нет данных ни по одному инструменту")
    return pd.DataFrame(per_inst), bpy


def instrument_contribution(
    rets: pd.DataFrame, bpy: float = 252.0,
) -> pd.DataFrame:
    """Индивидуальные метрики + leave-one-out вклад по инструментам.

    Портфель = equal-weight по доступным в день инструментам (как в
    sleeve_returns). LOO: пересобираем портфель без инструмента i,
    меряем сдвиг портфельного Sharpe.

    Args:
        rets: DataFrame дневных P&L (колонки — инструменты).
        bpy: баров в году.

    Returns:
        DataFrame по инструментам, отсортированный по loo_delta по
        убыванию (сверху — те, чьё исключение сильнее всего УЛУЧШАЕТ
        портфель, т.е. главные кандидаты в балласт). Колонки:
        solo_ret, solo_dd, solo_sharpe, loo_delta, verdict.
    """
    full = rets.mean(axis=1, skipna=True).fillna(0.0)
    full_sharpe = _sharpe(full, bpy)

    rows = []
    for col in rets.columns:
        solo = rets[col].fillna(0.0)
        rest = rets.drop(columns=[col])
        loo = rest.mean(axis=1, skipna=True).fillna(0.0)
        loo_sharpe = _sharpe(loo, bpy)
        delta = loo_sharpe - full_sharpe  # >0: без него лучше
        if delta > 0.05:
            verdict = "БАЛЛАСТ (без него лучше)"
        elif delta < -0.05:
            verdict = "держать (поддерживает)"
        else:
            verdict = "нейтрален"
        rows.append({
            "instrument": col,
            "solo_ret": float((1.0 + solo).prod() - 1.0),
            "solo_dd": _max_dd(solo),
            "solo_sharpe": _sharpe(solo, bpy),
            "loo_delta": float(delta),
            "verdict": verdict,
        })
    df = pd.DataFrame(rows).set_index("instrument")
    return df.sort_values("loo_delta", ascending=False)


def format_contribution(
    df: pd.DataFrame, full_sharpe: float, title: str = "",
) -> str:
    """Человекочитаемая таблица вклада инструментов."""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"  Полный портфель Sharpe (gross): {full_sharpe:+.2f}")
    lines.append("  " + "-" * 68)
    lines.append(f"  {'инструмент':12s} {'solo_ret':>9s} "
                 f"{'solo_DD':>8s} {'solo_Sh':>8s} {'LOO_Δ':>8s}  вердикт")
    lines.append("  " + "-" * 68)
    for name, r in df.iterrows():
        lines.append(
            f"  {name:12s} {r['solo_ret']:+8.1%} "
            f"{r['solo_dd']:+7.1%} {r['solo_sharpe']:+7.2f} "
            f"{r['loo_delta']:+7.2f}  {r['verdict']}"
        )
    return "\n".join(lines)
