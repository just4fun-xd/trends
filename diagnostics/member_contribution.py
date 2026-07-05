"""Вклад членов ансамбля: leave-one-out анализ + корреляции.

Отвечает на вопрос «а что если не-лучшие варианты тоже выстрелят?»
измерением вместо мнения. Ключевая идея портфельной теории: ценность
члена ансамбля — НЕ его одиночный Sharpe, а ПРЕДЕЛЬНЫЙ ВКЛАД в
комбинацию. Посредственный, но декоррелированный вариант может
улучшать ансамбль сильнее, чем сильный, но дублирующий лидера
(marginal contribution ~ Sharpe_i − corr(i, rest) × Sharpe_rest).

Метод leave-one-out (LOO): пересобираем ансамбль без члена i и
сравниваем Sharpe. delta = Sharpe(полный) − Sharpe(без i):
  delta > 0  — член ПОМОГАЕТ (без него хуже), даже если сам слабый;
  delta < 0  — член ВРЕДИТ (без него лучше), даже если сам сильный;
  delta ≈ 0  — балласт (разбавляет позицию, не добавляя сигнала).

Это НЕ «выбор лучшего постфактум» (multiple testing): решение о
составе принимается по вкладу в комбинацию на walk-forward и двух
источниках, а не по одиночной доходности на одной выборке. LOO на
одном периоде — диагностика для формирования гипотезы; вердикт о
составе — только после подтверждения на втором источнике данных.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine


def _sleeve_returns(
    bars_by_symbol: dict[str, Bars],
    position_fn,
    sizer,
    cost: float,
) -> pd.Series:
    """Equal-weight sleeve-доходности для функции позиции.

    Args:
        bars_by_symbol: dict {имя: Bars}.
        position_fn: Bars -> position (сырой сигнал).
        sizer: Bars -> множитель размера (или None).
        cost: Издержки движка.

    Returns:
        Побарные доходности sleeve'а (после издержек).
    """
    cols = {}
    for name, bars in bars_by_symbol.items():
        pos = position_fn(bars)
        if sizer is not None:
            pos = pos * sizer(bars)
        cols[name] = run_engine(bars, pos, cost=cost).equity.pct_change()
    return pd.DataFrame(cols).mean(axis=1, skipna=True).fillna(0.0)


def _sharpe(rets: pd.Series, bars_per_year: float = 252.0) -> float:
    """Годовой Sharpe ряда доходностей."""
    std = rets.std()
    if std <= 0:
        return 0.0
    return float(rets.mean() / std * np.sqrt(bars_per_year))


def member_contribution(
    bars_by_symbol: dict[str, Bars],
    members: dict[str, callable],
    sizer=None,
    cost: float = 0.0002,
    bars_per_year: float = 252.0,
) -> dict:
    """LOO-анализ вклада каждого члена в ансамбль позиций.

    Ансамбль = равновзвешенное среднее позиций членов (как mr_ens /
    trend_ens). Для каждого члена считаем sleeve соло, sleeve полного
    ансамбля и sleeve ансамбля БЕЗ этого члена.

    Args:
        bars_by_symbol: dict {имя: Bars}.
        members: dict {имя_члена: fn(Bars) -> position}.
        sizer: Опциональный сайзер (Bars -> множитель) поверх позиций.
        cost: Издержки движка.
        bars_per_year: Баров в году.

    Returns:
        dict с ключами:
          'solo'      — DataFrame по членам: sharpe, return, max_dd;
          'loo'       — DataFrame по членам: sharpe_without,
                        delta (= полный − без члена);
          'full'      — dict метрик полного ансамбля;
          'corr'      — корреляционная матрица sleeve-доходностей
                        членов (+ колонка 'ENSEMBLE').
    """
    names = list(members)
    fns = [members[n] for n in names]

    def ens_fn(subset):
        def fn(bars, _subset=tuple(subset)):
            acc = None
            for f in _subset:
                p = f(bars).fillna(0.0)
                acc = p if acc is None else acc + p
            return acc / float(len(_subset))
        return fn

    member_rets = {
        n: _sleeve_returns(bars_by_symbol, members[n], sizer, cost)
        for n in names
    }
    full_rets = _sleeve_returns(
        bars_by_symbol, ens_fn(fns), sizer, cost)
    full_sharpe = _sharpe(full_rets, bars_per_year)

    solo_rows, loo_rows = {}, {}
    for i, n in enumerate(names):
        r = member_rets[n]
        eq = (1.0 + r).cumprod()
        solo_rows[n] = {
            "sharpe": _sharpe(r, bars_per_year),
            "return": float(eq.iloc[-1] - 1.0),
            "max_dd": float((eq / eq.cummax() - 1.0).min()),
        }
        rest = [f for j, f in enumerate(fns) if j != i]
        loo_rets = _sleeve_returns(
            bars_by_symbol, ens_fn(rest), sizer, cost)
        s_wo = _sharpe(loo_rets, bars_per_year)
        loo_rows[n] = {
            "sharpe_without": s_wo,
            "delta": full_sharpe - s_wo,
        }

    corr_df = pd.DataFrame(member_rets)
    corr_df["ENSEMBLE"] = full_rets
    eq_full = (1.0 + full_rets).cumprod()
    return {
        "solo": pd.DataFrame.from_dict(solo_rows, orient="index"),
        "loo": pd.DataFrame.from_dict(loo_rows, orient="index"),
        "full": {
            "sharpe": full_sharpe,
            "return": float(eq_full.iloc[-1] - 1.0),
            "max_dd": float((eq_full / eq_full.cummax() - 1.0).min()),
        },
        "corr": corr_df.corr(),
    }


def format_contribution(res: dict, title: str = "",
                        color: bool = True) -> str:
    """Форматирует результат member_contribution для консоли.

    Args:
        res: Результат member_contribution.
        title: Заголовок (имя ансамбля).
        color: ANSI-цвет.

    Returns:
        Многострочный отчёт.
    """
    green, red, yellow, reset, bold = (
        ("\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m")
        if color else ("", "", "", "", "")
    )
    lines = []
    if title:
        lines.append(f"{bold}Вклад членов ансамбля: {title}{reset}")
    f = res["full"]
    lines.append(f"  Полный ансамбль: Sharpe {f['sharpe']:+.2f}, "
                 f"ret {f['return']:+.1%}, DD {f['max_dd']:+.1%}")
    lines.append(f"\n  {'член':<14}{'солоShp':>8}{'ret':>8}"
                 f"{'DD':>8}{'безНего':>9}{'delta':>8}  вердикт")
    lines.append("  " + "-" * 66)
    for n in res["solo"].index:
        s = res["solo"].loc[n]
        lo = res["loo"].loc[n]
        d = lo["delta"]
        if d > 0.02:
            verdict, c = "помогает", green
        elif d < -0.02:
            verdict, c = "ВРЕДИТ", red
        else:
            verdict, c = "балласт", yellow
        lines.append(
            f"  {n:<14}{s['sharpe']:>+7.2f} {s['return']:>+7.1%} "
            f"{s['max_dd']:>+7.1%} {lo['sharpe_without']:>+8.2f} "
            f"{c}{d:>+7.2f}{reset}  {c}{verdict}{reset}"
        )
    lines.append("  " + "-" * 66)
    lines.append("  delta = Sharpe(полный) − Sharpe(без члена): "
                 ">0 член нужен, <0 мешает, ~0 балласт")
    lines.append("\n  Корреляции sleeve-доходностей:")
    corr_str = res["corr"].round(2).to_string()
    lines.extend("  " + row for row in corr_str.split("\n"))
    lines.append(
        f"\n  {yellow}Диагностика на одном периоде — гипотеза, не "
        f"вердикт. Изменение состава ансамбля — только после "
        f"подтверждения на втором источнике данных.{reset}"
    )
    return "\n".join(lines)
