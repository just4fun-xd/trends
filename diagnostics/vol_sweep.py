"""Sweep target_vol: сколько доходности можно выжать плечом до DD<40%.

Прямой тест «есть ли запас по риску». Логика: если сигнал прибыльный, а
DD далёк от лимита 40%, поднятие target_vol масштабирует доход, пока не
упрёшься в потолок DD. Где упрёшься — рабочий уровень риска.

Измеряет ОДНОВРЕМЕННО три вещи (иначе результат обманчив):
  1. Масштаб дохода: линейно ли растёт с target_vol (чистое плечо) или
     затухает (издержки едят).
  2. Издержки: оборот растёт с плечом, доход — нет обязательно. Печатаем
     доход при cost=0 и cost=2bps рядом: зазор = плата за оборот.
  3. DD<40%: жёсткий лимит. Маркер, где пробит.

Важно: max_leverage тоже поднимается вместе с target_vol, иначе клип по
плечу упрётся раньше DD и скроет реальную картину.
"""

from __future__ import annotations

import pandas as pd

from core.bars import Bars
from core.engine import run_engine, vol_target_size


def vol_sweep_single(
    bars: Bars, signal_fn, target_vols=(0.15, 0.25, 0.35, 0.50),
    max_lev_cap: float = 6.0, cost: float = 0.0002,
) -> pd.DataFrame:
    """Sweep target_vol для одной стратегии на одном инструменте.

    Args:
        bars: Данные инструмента.
        signal_fn: Функция СЫРОГО сигнала (Bars -> position 0/1 или ±1),
            БЕЗ встроенного vol-targeting.
        target_vols: Уровни целевой волатильности для перебора.
        max_lev_cap: Общий потолок плеча (поднят, чтобы клип не упирался
            раньше DD).
        cost: Издержки на оборот.

    Returns:
        DataFrame по target_vol: return, return_free (cost=0), max_dd,
        turnover_ann (годовой оборот), avg_lev (ср. плечо в рынке).
    """
    raw = signal_fn(bars)
    rows = {}
    for tv in target_vols:
        size = vol_target_size(bars, target_vol=tv, max_leverage=max_lev_cap)
        pos = raw * size
        res = run_engine(bars, pos, cost=cost)
        res_free = run_engine(bars, pos, cost=0.0)
        # Годовой оборот = сумма |Δpos| нормированная на годы.
        turnover = pos.shift(1).fillna(0.0).diff().abs().sum()
        years = len(bars) / bars.bars_per_year
        active = pos[pos.abs() > 1e-9].abs()
        rows[tv] = {
            "return": res.total_return,
            "return_free": res_free.total_return,
            "max_dd": res.max_drawdown,
            "turnover_ann": turnover / years if years > 0 else 0.0,
            "avg_lev": float(active.mean()) if len(active) else 0.0,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def vol_sweep_basket(
    bars_by_symbol: dict[str, Bars], signal_fn,
    target_vols=(0.15, 0.25, 0.35, 0.50), max_lev_cap: float = 6.0,
    cost: float = 0.0002,
) -> pd.DataFrame:
    """Sweep по корзине: портфельные метрики на каждом target_vol.

    Портфель — equal-weight по инструментам. Return усредняется, DD
    берётся как worst-case по инструментам (консервативно, как в проекте).

    Args:
        bars_by_symbol: dict {инструмент: Bars}.
        signal_fn: Сырой сигнал (Bars -> position).
        target_vols: Уровни target_vol.
        max_lev_cap: Потолок плеча.
        cost: Издержки.

    Returns:
        DataFrame по target_vol: mean_return, median_return,
        return_free, worst_dd, cost_drag (return_free - return),
        turnover_ann, passes_dd (worst_dd > -40%).
    """
    per_tv = {tv: [] for tv in target_vols}
    for bars in bars_by_symbol.values():
        wf = vol_sweep_single(bars, signal_fn, target_vols, max_lev_cap,
                              cost)
        for tv in target_vols:
            per_tv[tv].append(wf.loc[tv])

    rows = {}
    for tv, recs in per_tv.items():
        df = pd.DataFrame(recs)
        rows[tv] = {
            "mean_return": df["return"].mean(),
            "median_return": df["return"].median(),
            "return_free": df["return_free"].mean(),
            "cost_drag": df["return_free"].mean() - df["return"].mean(),
            "worst_dd": df["max_dd"].min(),
            "turnover_ann": df["turnover_ann"].mean(),
            "avg_lev": df["avg_lev"].mean(),
            "passes_dd": bool(df["max_dd"].min() > -0.40),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def format_sweep(df: pd.DataFrame, name: str = "", color: bool = True) -> str:
    """Форматирует sweep-таблицу с маркером пробоя DD и затухания.

    Args:
        df: Результат vol_sweep_basket.
        name: Имя стратегии.
        color: ANSI-цвет.

    Returns:
        Многострочная таблица для консоли.
    """
    green, red, yellow, reset, bold = (
        ("\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m")
        if color else ("", "", "", "", "")
    )
    lines = []
    if name:
        lines.append(f"{bold}{name} — sweep target_vol{reset}")
    lines.append(f"  {'tgVol':>6} {'return':>8} {'medРет':>8} "
                 f"{'worstDD':>8} {'издержк':>8} {'оборот':>7} "
                 f"{'плечо':>6} {'DD<40':>6}")
    lines.append("  " + "-" * 62)
    prev_ret = None
    for tv, r in df.iterrows():
        ret_c = green if r["mean_return"] > 0 else red
        dd_c = (green if r["worst_dd"] > -0.25
                else yellow if r["worst_dd"] > -0.40 else red)
        pass_mark = (f"{green}да{reset}" if r["passes_dd"]
                     else f"{red}НЕТ{reset}")
        # Маркер затухания: если доход растёт медленнее, чем target_vol.
        scale_note = ""
        if prev_ret is not None and prev_ret > 0:
            growth = r["mean_return"] / prev_ret
            if growth < 1.1:
                scale_note = f" {yellow}<-затухание{reset}"
        prev_ret = r["mean_return"]
        lines.append(
            f"  {tv:>6.0%} {ret_c}{r['mean_return']:>+7.1%}{reset} "
            f"{r['median_return']:>+7.1%} "
            f"{dd_c}{r['worst_dd']:>+7.1%}{reset} "
            f"{r['cost_drag']:>+7.1%} {r['turnover_ann']:>6.1f} "
            f"{r['avg_lev']:>6.2f} {pass_mark:>6}{scale_note}"
        )
    lines.append("  " + "-" * 62)
    lines.append(f"  {yellow}издержк{reset}=потеря на обороте "
                 f"(free-net); {yellow}затухание{reset}=доход растёт "
                 f"медленнее плеча")
    return "\n".join(lines)
