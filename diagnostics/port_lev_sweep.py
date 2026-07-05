"""Sweep потолка портфельного плеча: найти максимум в рамках DD<40%.

Прямой ответ на «увеличивай плечо, лишь бы DD<40%» — вместо ручного
подбора target_vol/max_leverage перебираем сетку и берём точку с
максимальной доходностью среди прошедших лимит. Тот же принцип, что
diagnostics/vol_sweep.py для одиночной стратегии, только на уровне
УЖЕ СКОМБИНИРОВАННОГО портфеля (после vol-parity).

Зачем нужен отдельный sweep, а не один вызов portfolio_vol_target:
рост target_vol не линеен по DD после того, как плечо перестаёт быть
маленьким — окно оценки волы (63 бара) запаздывает за резкими
скачками, и при большом плече эта задержка бьёт по факту сильнее.
Sweep находит эмпирическую точку излома, а не экстраполирует линейно.

Издержки НЕ пересчитываются (они уже внутри combo_returns — из
движков sleeve'ов); sweep меняет только ВНЕШНИЙ портфельный
множитель, поэтому результат ниже честен ровно настолько, насколько
честна входная кривая combo_returns. Стоимость фондирования при
большом плече сюда не входит (см. предупреждение в выводе) — это
единственное, чего sweep специально НЕ считает, и об этом сказано
прямым текстом в каждом вызове, а не мелким шрифтом.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.sizing import portfolio_vol_target


def leverage_sweep(
    combo_returns: pd.Series,
    target_vols: tuple = (0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80),
    max_leverage_grid: tuple = (2.0, 4.0, 6.0, 8.0, 12.0),
    window: int = 63,
    bars_per_year: float = 252.0,
    funding_rate: float = 0.0,
) -> pd.DataFrame:
    """Полная сетка (target_vol × кэп плеча) на реальной кривой комбо.

    Args:
        combo_returns: Побарные доходности УЖЕ СКОМБИНИРОВАННОГО
            портфеля (после vol-parity, издержки внутри).
        target_vols: Сетка целевых волатильностей.
        max_leverage_grid: Сетка потолков плеча.
        window: Окно оценки реализованной волы (баров).
        bars_per_year: Баров в году.
        funding_rate: Годовая ставка фондирования заёмной части
            плеча (см. core.sizing.portfolio_vol_target). 0.0 =
            верхняя граница без costs (старое поведение).

    Returns:
        DataFrame, индекс — (target_vol, max_lev), колонки: return,
        max_dd, sharpe, avg_lev, lev_hit_cap (доля дней у потолка —
        сигнал, что кэп связывает и target_vol не достигнут),
        passes_dd (bool).
    """
    rows = {}
    for tv in target_vols:
        for cap in max_leverage_grid:
            scaled, lev = portfolio_vol_target(
                combo_returns, target_vol=tv, window=window,
                max_leverage=cap, bars_per_year=bars_per_year,
                funding_rate=funding_rate,
            )
            eq = (1.0 + scaled).cumprod()
            dd = float((eq / eq.cummax() - 1.0).min())
            std = scaled.std()
            sharpe = (float(scaled.mean() / std
                            * np.sqrt(bars_per_year))
                      if std > 0 else 0.0)
            active = lev[lev > 0]
            avg_lev = float(active.mean()) if len(active) else 0.0
            hit_cap = (float((active >= cap - 1e-9).mean())
                       if len(active) else 0.0)
            rows[(tv, cap)] = {
                "return": float(eq.iloc[-1] - 1.0),
                "max_dd": dd,
                "sharpe": sharpe,
                "avg_lev": avg_lev,
                "lev_hit_cap": hit_cap,
                "passes_dd": bool(dd > -0.40),
            }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.MultiIndex.from_tuples(
        df.index, names=["target_vol", "max_leverage"])
    return df


def best_leverage(df: pd.DataFrame) -> tuple | None:
    """Точка максимальной доходности среди прошедших DD<40%.

    Args:
        df: Результат leverage_sweep.

    Returns:
        (target_vol, max_leverage) с наибольшим return среди
        passes_dd, либо None если ни одна точка не прошла.
    """
    passing = df[df["passes_dd"]]
    if passing.empty:
        return None
    return passing["return"].idxmax()


def format_leverage_sweep(
    df: pd.DataFrame, color: bool = True, funding_rate: float = 0.0,
) -> str:
    """Печатает сетку с маркером прохождения DD и потолка плеча.

    Args:
        df: Результат leverage_sweep.
        color: ANSI-цвет.
        funding_rate: Ставка фондирования, применённая при построении
            df (только для текста предупреждения внизу — сам расчёт
            уже сделан в leverage_sweep).

    Returns:
        Многострочная таблица.
    """
    green, red, yellow, reset, bold = (
        ("\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m")
        if color else ("", "", "", "", "")
    )
    lines = [f"{bold}Sweep потолка плеча (реальная кривая "
             f"комбо){reset}"]
    lines.append(f"  {'tgVol':>6} {'кэп':>5} {'return':>8} "
                 f"{'DD':>7} {'Sharpe':>7} {'плечо':>6} "
                 f"{'уКэпа':>6} {'DD<40':>6}")
    lines.append("  " + "-" * 62)
    for (tv, cap), r in df.iterrows():
        dd_c = (green if r["max_dd"] > -0.25
                else yellow if r["max_dd"] > -0.40 else red)
        pass_mark = (f"{green}да{reset}" if r["passes_dd"]
                     else f"{red}НЕТ{reset}")
        cap_note = (f"{yellow}{r['lev_hit_cap']:>5.0%}{reset}"
                    if r["lev_hit_cap"] > 0.3 else
                    f"{r['lev_hit_cap']:>5.0%}")
        lines.append(
            f"  {tv:>5.0%} {cap:>5.1f} {r['return']:>+7.1%} "
            f"{dd_c}{r['max_dd']:>+6.1%}{reset} "
            f"{r['sharpe']:>+6.2f} {r['avg_lev']:>6.2f} "
            f"{cap_note} {pass_mark:>6}"
        )
    lines.append("  " + "-" * 62)
    best = best_leverage(df)
    if best is not None:
        tv, cap = best
        r = df.loc[(tv, cap)]
        lines.append(
            f"\n{bold}Максимум в рамках DD<40%:{reset} "
            f"target_vol={tv:.0%}, кэп плеча={cap:.1f} -> "
            f"доход {r['return']:+.1%}, DD {r['max_dd']:+.1%}, "
            f"среднее плечо {r['avg_lev']:.2f}"
        )
        if r["lev_hit_cap"] > 0.3:
            lines.append(
                f"{yellow}Плечо у потолка {r['lev_hit_cap']:.0%} "
                f"дней — target_vol НЕ достигнут, реальный лимит "
                f"здесь кэп, а не цель. Подними кэп и перегони "
                f"sweep, а не поднимай только target_vol.{reset}"
            )
    else:
        lines.append(f"\n{red}Ни одна точка сетки не прошла "
                     f"DD<40%.{reset}")
    if funding_rate > 0:
        lines.append(
            f"\n{green}✓ Стоимость фондирования УЧТЕНА: "
            f"{funding_rate:.1%}/год на заёмную часть плеча "
            f"(leverage−1), пропорционально дням. Показанная "
            f"доходность — уже ПОСЛЕ вычета этих costs.{reset}"
        )
    else:
        lines.append(
            f"\n{yellow}⚠ Стоимость фондирования плеча НЕ включена "
            f"(funding_rate=0). Показанная доходность — верхняя "
            f"граница до вычета costs финансирования маржи при "
            f"плече >2-3x. Задай --funding-rate для честной "
            f"оценки.{reset}"
        )
    return "\n".join(lines)
