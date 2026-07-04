"""Sweep target_vol: сколько доходности можно выжать плечом до DD<40%.

Прямой тест «есть ли запас по риску». Логика: если сигнал прибыльный, а
DD далёк от лимита 40%, поднятие target_vol масштабирует доход, пока не
упрёшься в потолок DD. Где упрёшься — рабочий уровень риска.

Измеряет ОДНОВРЕМЕННО (иначе результат обманчив):
  1. Масштаб дохода: линейно ли растёт с target_vol (чистое плечо) или
     затухает (издержки едят).
  2. Издержки: печатаем доход при cost=0 и cost=2bps рядом: зазор =
     плата за оборот. Оборот считается ТОЙ ЖЕ drift-aware формулой,
     которой движок начисляет издержки (drift_turnover) — старая
     diff()-формула занижала оборот на дробных весах и расходилась с
     фактическими списаниями движка (унификация 2026-07).
  3. DD<40% ДВАЖДЫ: worst-case по инструментам (консервативно, как в
     BENCHMARK_RESULTS) и ПОРТФЕЛЬНЫЙ DD equal-weight дневного P&L.
     Портфельный — то, что реально переживает счёт: диверсификация
     по 19 инструментам гасит DD, и рабочий target_vol портфеля
     заметно выше, чем показывает worst-case критерий. Это главный
     источник «бесплатной» доходности без изменения сигнала.

Важно: max_leverage тоже поднимается вместе с target_vol, иначе клип по
плечу упрётся раньше DD и скроет реальную картину.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import drift_turnover, run_engine
from core.sizing import make_sizer


def vol_sweep_single(
    bars: Bars, signal_fn, target_vols=(0.15, 0.25, 0.35, 0.50),
    max_lev_cap: float = 6.0, cost: float = 0.0002,
    sizer_name: str = "realized",
) -> tuple[pd.DataFrame, dict[float, pd.Series]]:
    """Sweep target_vol для одной стратегии на одном инструменте.

    Args:
        bars: Данные инструмента.
        signal_fn: Функция СЫРОГО сигнала (Bars -> position 0/1 или ±1),
            БЕЗ встроенного vol-targeting.
        target_vols: Уровни целевой волатильности для перебора.
        max_lev_cap: Общий потолок плеча (поднят, чтобы клип не упирался
            раньше DD).
        cost: Издержки на оборот.
        sizer_name: 'realized' или 'garch' (core.sizing.SIZERS).

    Returns:
        (df, pnl): df — DataFrame по target_vol (return, return_free,
        max_dd, turnover_ann, avg_lev); pnl — dict {target_vol:
        побарные доходности после издержек} для портфельной агрегации.
    """
    raw = signal_fn(bars)
    rows = {}
    pnl: dict[float, pd.Series] = {}
    for tv in target_vols:
        sizer = make_sizer(
            sizer_name, target_vol=tv, max_leverage=max_lev_cap,
        )
        pos = raw * sizer(bars)
        res = run_engine(bars, pos, cost=cost)
        res_free = run_engine(bars, pos, cost=0.0)
        # Оборот той же формулой, что издержки движка (drift-aware).
        prev_pos = pos.shift(1).fillna(0.0)
        turnover = float(drift_turnover(prev_pos, bars.returns()).sum())
        years = len(bars) / bars.bars_per_year
        active = pos[pos.abs() > 1e-9].abs()
        rows[tv] = {
            "return": res.total_return,
            "return_free": res_free.total_return,
            "max_dd": res.max_drawdown,
            "turnover_ann": turnover / years if years > 0 else 0.0,
            "avg_lev": float(active.mean()) if len(active) else 0.0,
        }
        pnl[tv] = res.equity.pct_change()
    return pd.DataFrame.from_dict(rows, orient="index"), pnl


def vol_sweep_basket(
    bars_by_symbol: dict[str, Bars], signal_fn,
    target_vols=(0.15, 0.25, 0.35, 0.50), max_lev_cap: float = 6.0,
    cost: float = 0.0002, sizer_name: str = "realized",
) -> pd.DataFrame:
    """Sweep по корзине: per-instrument И портфельные метрики.

    Портфель — equal-weight дневного P&L по инструментам. Печатаются
    оба критерия DD: worst-case по инструментам (старый, консервативный)
    и портфельный (то, что реально видит счёт).

    Args:
        bars_by_symbol: dict {инструмент: Bars}.
        signal_fn: Сырой сигнал (Bars -> position).
        target_vols: Уровни target_vol.
        max_lev_cap: Потолок плеча.
        cost: Издержки.
        sizer_name: 'realized' или 'garch'.

    Returns:
        DataFrame по target_vol: mean_return, median_return,
        return_free, cost_drag, worst_dd, port_return, port_dd,
        port_sharpe, turnover_ann, avg_lev, passes_dd (worst-case),
        port_passes_dd (портфельный).
    """
    per_tv = {tv: [] for tv in target_vols}
    port_pnl = {tv: {} for tv in target_vols}
    bpy = 252.0
    for name, bars in bars_by_symbol.items():
        bpy = bars.bars_per_year
        wf, pnl = vol_sweep_single(
            bars, signal_fn, target_vols, max_lev_cap, cost, sizer_name,
        )
        for tv in target_vols:
            per_tv[tv].append(wf.loc[tv])
            port_pnl[tv][name] = pnl[tv]

    rows = {}
    for tv, recs in per_tv.items():
        df = pd.DataFrame(recs)
        # Портфельный P&L: средний по доступным инструментам в день.
        pp = (pd.DataFrame(port_pnl[tv])
              .mean(axis=1, skipna=True).fillna(0.0))
        eq = (1.0 + pp).cumprod()
        port_dd = float((eq / eq.cummax() - 1.0).min())
        std = pp.std()
        port_sharpe = (float(pp.mean() / std * np.sqrt(bpy))
                       if std > 0 else 0.0)
        rows[tv] = {
            "mean_return": df["return"].mean(),
            "median_return": df["return"].median(),
            "return_free": df["return_free"].mean(),
            "cost_drag": df["return_free"].mean() - df["return"].mean(),
            "worst_dd": df["max_dd"].min(),
            "port_return": float(eq.iloc[-1] - 1.0),
            "port_dd": port_dd,
            "port_sharpe": port_sharpe,
            "turnover_ann": df["turnover_ann"].mean(),
            "avg_lev": df["avg_lev"].mean(),
            "passes_dd": bool(df["max_dd"].min() > -0.40),
            "port_passes_dd": bool(port_dd > -0.40),
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
                 f"{'worstDD':>8} {'портРет':>8} {'портDD':>7} "
                 f"{'Sharpe':>6} {'издержк':>8} {'оборот':>7} "
                 f"{'плечо':>6} {'DD<40':>6} {'портф':>6}")
    lines.append("  " + "-" * 96)
    prev_ret = None
    for tv, r in df.iterrows():
        ret_c = green if r["mean_return"] > 0 else red
        dd_c = (green if r["worst_dd"] > -0.25
                else yellow if r["worst_dd"] > -0.40 else red)
        pdd_c = (green if r["port_dd"] > -0.25
                 else yellow if r["port_dd"] > -0.40 else red)
        pass_mark = (f"{green}да{reset}" if r["passes_dd"]
                     else f"{red}НЕТ{reset}")
        ppass_mark = (f"{green}да{reset}" if r["port_passes_dd"]
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
            f"{r['port_return']:>+7.1%} "
            f"{pdd_c}{r['port_dd']:>+6.1%}{reset} "
            f"{r['port_sharpe']:>+6.2f} "
            f"{r['cost_drag']:>+7.1%} {r['turnover_ann']:>6.1f} "
            f"{r['avg_lev']:>6.2f} {pass_mark:>6} {ppass_mark:>6}"
            f"{scale_note}"
        )
    lines.append("  " + "-" * 96)
    lines.append(f"  {yellow}издержк{reset}=потеря на обороте "
                 f"(free-net, drift-aware); {yellow}портDD{reset}="
                 f"просадка equal-weight портфеля — рабочий критерий; "
                 f"{yellow}затухание{reset}=доход растёт медленнее плеча")
    return "\n".join(lines)
