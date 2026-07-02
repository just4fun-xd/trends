"""Годовая разбивка результатов бэктеста (диагностика).

Отвечает на главный вопрос ревью: «по годам, на каком алгоритме какая
прибыль и какая DD в процентах». Режет кривую капитала по календарным
годам и для каждого считает:
  - return: рост капитала внутри года (equity_end/equity_start − 1);
  - max_dd: худшая просадка ОТ ВНУТРИГОДОВОГО пика (не от общего) —
    иначе годы после большого пика показывали бы ложно-глубокий DD;
  - sharpe: годовой Sharpe по дневным барам этого года.

Работает и с BacktestResult (один инструмент), и с PortfolioResult
(портфель) — обоим нужна только equity + bars_per_year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def yearly_breakdown(
    equity: pd.Series, bars_per_year: float = 252.0
) -> pd.DataFrame:
    """Годовая разбивка одной кривой капитала.

    Args:
        equity: Кривая капитала (мультипликативная, старт ~1.0).
        bars_per_year: Баров в году для аннуализации Sharpe.

    Returns:
        DataFrame, индекс — год (int), колонки:
            return: доходность за год (доля, 0.12 = +12%);
            max_dd: макс. просадка от внутригодового пика (<= 0);
            sharpe: годовой Sharpe по барам года;
            bars: число баров в году (для контроля неполных лет).
    """
    if equity.empty:
        return pd.DataFrame(
            columns=["return", "max_dd", "sharpe", "bars"]
        )
    # Побарная доходность кривой капитала.
    bar_ret = equity.pct_change().fillna(0.0)
    rows = {}
    for year, grp in bar_ret.groupby(bar_ret.index.year):
        # Реконструкция внутригодовой кривой от 1.0.
        curve = (1.0 + grp).cumprod()
        year_ret = float(curve.iloc[-1] - 1.0)
        dd = float((curve / curve.cummax() - 1.0).min())
        if grp.std() > 0 and len(grp) > 1:
            sharpe = float(grp.mean() / grp.std() * np.sqrt(bars_per_year))
        else:
            sharpe = 0.0
        rows[int(year)] = {
            "return": year_ret, "max_dd": dd,
            "sharpe": sharpe, "bars": int(len(grp)),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def format_yearly_table(
    df: pd.DataFrame, title: str = "", color: bool = True
) -> str:
    """Форматирует годовую разбивку в читаемую таблицу для консоли.

    Args:
        df: Результат yearly_breakdown.
        title: Заголовок над таблицей (напр. имя стратегии/инструмента).
        color: Красить ли return/DD ANSI-цветом (зелёный/жёлтый/красный).

    Returns:
        Готовая к печати многострочная строка.
    """
    green, red, yellow, reset, bold = (
        ("\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m")
        if color else ("", "", "", "", "")
    )
    lines = []
    if title:
        lines.append(f"{bold}{title}{reset}")
    lines.append(f"  {'Год':>4}  {'Return':>9}  {'MaxDD':>8}  "
                 f"{'Sharpe':>7}  {'Бары':>5}")
    lines.append("  " + "-" * 42)
    for year, r in df.iterrows():
        ret_c = green if r["return"] > 0 else red
        dd_c = (green if r["max_dd"] > -0.20
                else yellow if r["max_dd"] > -0.40 else red)
        lines.append(
            f"  {year:>4}  {ret_c}{r['return']:>+8.1%}{reset}  "
            f"{dd_c}{r['max_dd']:>+7.1%}{reset}  "
            f"{r['sharpe']:>+7.2f}  {int(r['bars']):>5}"
        )
    # Итоговая строка: компаунд всех лет, худший годовой DD.
    if not df.empty:
        compound = float((1.0 + df["return"]).prod() - 1.0)
        worst_dd = float(df["max_dd"].min())
        avg_sharpe = float(df["sharpe"].mean())
        lines.append("  " + "-" * 42)
        comp_c = green if compound > 0 else red
        lines.append(
            f"  {bold}ИТОГ{reset}  {comp_c}{compound:>+8.1%}{reset}  "
            f"{worst_dd:>+7.1%}  {avg_sharpe:>+7.2f}  "
            f"{'':>5}  (компаунд / худший год / ср.Sharpe)"
        )
    return "\n".join(lines)


def yearly_matrix(
    results: dict[str, pd.Series],
    bars_per_year: float = 252.0,
    metric: str = "return",
) -> pd.DataFrame:
    """Матрица год × инструмент по одной метрике (для сводных таблиц).

    Args:
        results: dict {имя_инструмента: equity_series}.
        bars_per_year: Баров в году.
        metric: Какую метрику класть в ячейки ('return' или 'max_dd').

    Returns:
        DataFrame: индекс — год, колонки — инструменты, значения —
        выбранная метрика. Удобно печатать или писать в CSV.
    """
    cols = {}
    for name, equity in results.items():
        yb = yearly_breakdown(equity, bars_per_year)
        cols[name] = yb[metric]
    return pd.DataFrame(cols).sort_index()


def format_matrix(
    matrix: pd.DataFrame, title: str = "", pct: bool = True,
    color: bool = True,
) -> str:
    """Форматирует матрицу год × инструмент в консольную таблицу.

    Args:
        matrix: Результат yearly_matrix.
        title: Заголовок.
        pct: Форматировать ли значения как проценты.
        color: Красить ли по знаку.

    Returns:
        Многострочная строка для печати.
    """
    green, red, reset, bold = (
        ("\033[92m", "\033[91m", "\033[0m", "\033[1m")
        if color else ("", "", "", "")
    )
    lines = []
    if title:
        lines.append(f"{bold}{title}{reset}")
    header = f"  {'Год':>4}  " + "  ".join(
        f"{c[:8]:>8}" for c in matrix.columns
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for year, row in matrix.iterrows():
        cells = []
        for v in row:
            if pd.isna(v):
                cells.append(f"{'—':>8}")
            else:
                c = green if v > 0 else red
                s = f"{v:>+7.1%}" if pct else f"{v:>+8.2f}"
                cells.append(f"{c}{s}{reset}")
        lines.append(f"  {int(year):>4}  " + "  ".join(cells))
    return "\n".join(lines)
