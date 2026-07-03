"""Walk-forward анализ (anchored) — тест стабильности по времени.

Отвечает на вопрос, который обычный бэктест на всём периоде спрятать:
результат стабилен из года в год, или держится на одном везучем режиме?
Прогон 10 вариантов на одном периоде гарантирует, что лучший выглядит
хорошо ПО СЛУЧАЙНОСТИ (multiple testing). Walk-forward отделяет
«работает» от «повезло на этом отрезке».

ANCHORED (не optimizing): параметры фиксированы, НЕ переоптимизируются.
Твои bb_rsi/mr_* используют стандартные параметры (RSI 30/50, BB 20/2),
поэтому вопрос — робастность во времени, а не перенос подгонки.

Ключевые метрики (принцип проекта: нестабильность = переобучение):
  - согласованность знака: доля окон с положительным return;
  - разброс: return лучшего окна vs худшего (мираж на одном годе виден);
  - худшее окно: если стратегия в одном году проваливает DD<40% —
    красный флаг даже при отличном среднем.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine
from diagnostics.yearly import yearly_breakdown


def walk_forward_windows(
    index: pd.DatetimeIndex, by: str = "year"
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Нарезает индекс на последовательные окна.

    Args:
        index: Временной индекс данных.
        by: 'year' — календарные годы (5 окон на 5 лет); 'half' —
            полугодия; иначе трактуется как число дней катящегося окна.

    Returns:
        Список (start, end) границ окон, включительно с обеих сторон.
    """
    if by == "year":
        years = sorted(index.year.unique())
        return [
            (pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"))
            for y in years
        ]
    if by == "half":
        out = []
        for y in sorted(index.year.unique()):
            out.append((pd.Timestamp(f"{y}-01-01"),
                        pd.Timestamp(f"{y}-06-30")))
            out.append((pd.Timestamp(f"{y}-07-01"),
                        pd.Timestamp(f"{y}-12-31")))
        return out
    # Число дней -> катящееся окно.
    n = int(by)
    starts = index[::n]
    return [(s, s + pd.Timedelta(days=n - 1)) for s in starts]


def walk_forward_single(
    bars: Bars, strategy_fn, cost: float = 0.0002, by: str = "year",
) -> pd.DataFrame:
    """Anchored walk-forward одной стратегии на одном инструменте.

    Прогоняет стратегию на всём ряду (сигнал видит полную историю для
    прогрева индикаторов), затем режет получившуюся кривую капитала по
    окнам и считает per-window метрики. Это честный anchored-подход:
    параметры не меняются, оценивается стабильность результата.

    Args:
        bars: Данные инструмента.
        strategy_fn: Функция стратегии (Bars -> position).
        cost: Издержки.
        by: Нарезка окон ('year'/'half'/число дней).

    Returns:
        DataFrame по окнам: return, max_dd, sharpe, bars.
    """
    pos = strategy_fn(bars)
    res = run_engine(bars, pos, cost=cost)
    # yearly_breakdown уже режет по годам с внутригодовым DD.
    if by == "year":
        return yearly_breakdown(res.equity, res.bars_per_year)
    # Для не-годовой нарезки режем вручную по окнам.
    rows = {}
    bar_ret = res.equity.pct_change().fillna(0.0)
    for start, end in walk_forward_windows(bars.index, by):
        seg = bar_ret[(bar_ret.index >= start) & (bar_ret.index <= end)]
        if len(seg) < 2:
            continue
        curve = (1.0 + seg).cumprod()
        rows[start.date()] = {
            "return": float(curve.iloc[-1] - 1.0),
            "max_dd": float((curve / curve.cummax() - 1.0).min()),
            "sharpe": float(seg.mean() / seg.std()
                            * np.sqrt(res.bars_per_year))
            if seg.std() > 0 else 0.0,
            "bars": int(len(seg)),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def consistency_metrics(window_df: pd.DataFrame) -> dict:
    """Метрики согласованности из per-window таблицы.

    Args:
        window_df: Результат walk_forward_single (окна × метрики).

    Returns:
        dict: n_windows, positive_frac (доля прибыльных окон),
        mean_return, worst_window, best_window, spread (best-worst),
        worst_dd, all_pass_dd (все окна прошли DD<40%).
    """
    if window_df.empty:
        return {}
    rets = window_df["return"]
    return {
        "n_windows": int(len(rets)),
        "positive_frac": float((rets > 0).mean()),
        "mean_return": float(rets.mean()),
        "median_return": float(rets.median()),
        "worst_window": float(rets.min()),
        "best_window": float(rets.max()),
        "spread": float(rets.max() - rets.min()),
        "worst_dd": float(window_df["max_dd"].min()),
        "all_pass_dd": bool((window_df["max_dd"] > -0.40).all()),
    }


def walk_forward_basket(
    bars_by_symbol: dict[str, Bars], strategy_fn,
    cost: float = 0.0002, by: str = "year",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward стратегии по всей корзине.

    Args:
        bars_by_symbol: dict {инструмент: Bars}.
        strategy_fn: Стратегия (Bars -> position).
        cost: Издержки.
        by: Нарезка окон.

    Returns:
        (matrix, consistency): matrix — return по (окно × инструмент);
        consistency — метрики согласованности на портфельном
        (равновзвешенном по инструментам) return каждого окна.
    """
    per_symbol = {}
    for sym, bars in bars_by_symbol.items():
        wf = walk_forward_single(bars, strategy_fn, cost, by)
        if not wf.empty:
            per_symbol[sym] = wf["return"]
    matrix = pd.DataFrame(per_symbol).sort_index()

    # Портфель окна = среднее по инструментам (equal-weight).
    consistency = pd.DataFrame({
        "port_return": matrix.mean(axis=1),
        "median_instrument": matrix.median(axis=1),
        "positive_instruments": (matrix > 0).sum(axis=1),
        "total_instruments": matrix.notna().sum(axis=1),
    })
    return matrix, consistency


def format_consistency(
    name: str, metrics: dict, color: bool = True
) -> str:
    """Форматирует метрики согласованности в вердикт-строку.

    Args:
        name: Имя стратегии.
        metrics: Результат consistency_metrics.
        color: ANSI-цвет.

    Returns:
        Многострочная строка вердикта.
    """
    green, red, yellow, reset, bold = (
        ("\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m")
        if color else ("", "", "", "", "")
    )
    if not metrics:
        return f"{name}: нет данных"
    pf = metrics["positive_frac"]
    pf_c = green if pf >= 0.8 else yellow if pf >= 0.6 else red
    # Вердикт робастности: согласованность знака + ограниченный худший
    # результат (не проваливается в один год). DD оценивается отдельно
    # через --yearly на полном прогоне.
    robust = pf >= 0.8 and metrics["worst_window"] > -0.15
    verdict = (f"{green}РОБАСТНА{reset}" if robust
               else f"{yellow}ПОД ВОПРОСОМ{reset}")
    lines = [
        f"{bold}{name}{reset} — {verdict}",
        f"  Окон прибыльных:  {pf_c}{pf:.0%}{reset} "
        f"({metrics['n_windows']} окон)",
        f"  Средний / медиана: {metrics['mean_return']:+.1%} / "
        f"{metrics['median_return']:+.1%}",
        f"  Лучшее / худшее:   {metrics['best_window']:+.1%} / "
        f"{red if metrics['worst_window'] < 0 else ''}"
        f"{metrics['worst_window']:+.1%}{reset}",
        f"  Разброс окон:      {metrics['spread']:.1%} "
        f"(мираж на 1 годе, если велик)",
    ]
    return "\n".join(lines)
