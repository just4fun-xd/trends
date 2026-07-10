"""Раннер: карта «актив × стратегия» одним прогоном.

Собирает то, что в сессии 2026-07f строилось руками из 10 отдельных
прогонов: матрицу leave-one-out вкладов (актив × стратегия) и для
каждого актива — вердикт «какая нога ему подходит».

Идея: балласт — свойство ПАРЫ (актив × стратегия), не актива. Один
инструмент под тренд-ногой может быть балластом, а под MR-ногой —
держателем. Карта делает это распределение видимым в одной таблице.

Для каждой (стратегия, актив):
  - LOO-дельта = Sharpe(портфель стратегии без актива) − Sharpe(полный).
    LOO>0 — балласт (без него лучше); LOO<0 — держатель.
  - «solo Sharpe» актива под стратегией (gross, масштабо-инвариантен).

Вердикт по активу: нога с самой ОТРИЦАТЕЛЬНОЙ суммарной LOO-дельтой
(сильнее всего поддерживает) в своём классе (тренд / MR). Если актив
балласт во всех — кандидат на исключение (проверить на 2 источниках).

Запуск:
    python -m runners.run_regime_map \\
        --trend donchian_vt kama ewmac \\
        --mr mr_kelt_confirm mr_keltner mr_lowvol \\
        --basket commodity --source databento \\
        --panel-dir data/panels/futures --start 2019-01-01 \\
        --vt --target-vol 0.20
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from core.config import (
    COMMODITY_DATABENTO, COMMODITY_YF, CRYPTO_CCXT, CRYPTO_YF,
    EQUITY_BASKET, filter_basket)
from core.sizing import make_sizer
from data.databento_source import DatabentoSource
from data.ccxt_source import CCXTSource
from data.yfinance_source import YFinanceSource
from diagnostics.instrument_contribution import (
    instrument_contribution,
    per_instrument_returns,
)
from runners.run_basket import STRATEGIES

BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"


def _collect(strats, basket, source, args):
    """Матрицы LOO-дельт и solo-Sharpe: индекс актив, колонки стратегии."""
    loo = {}
    solo = {}
    for name in strats:
        if name not in STRATEGIES:
            print(f"{YELLOW}пропуск {name}: нет в реестре{RESET}")
            continue
        sizer = make_sizer(args.sizer, target_vol=args.target_vol) \
            if args.vt else None
        rets, bpy = per_instrument_returns(
            STRATEGIES[name], basket, source, args.start, args.end,
            sizer=sizer, cost=args.cost, interval=args.interval,
        )
        df = instrument_contribution(rets, bpy)
        loo[name] = df["loo_delta"]
        solo[name] = df["solo_sharpe"]
    return pd.DataFrame(loo), pd.DataFrame(solo)


def _fmt_matrix(mat: pd.DataFrame, title: str) -> str:
    """Печать матрицы LOO-дельт со знаком-цветом."""
    lines = [title, "  " + "-" * (14 + 10 * len(mat.columns))]
    hdr = f"  {'актив':12s}"
    for c in mat.columns:
        hdr += f"{c[:9]:>10s}"
    lines.append(hdr)
    lines.append("  " + "-" * (14 + 10 * len(mat.columns)))
    for inst, row in mat.iterrows():
        line = f"  {inst:12s}"
        for c in mat.columns:
            v = row[c]
            if pd.isna(v):
                line += f"{'—':>10s}"
            else:
                # балласт (>0) красным, держатель (<-0.05) зелёным
                col = (RED if v > 0.05 else
                       GREEN if v < -0.05 else "")
                end = RESET if col else ""
                line += f"{col}{v:+10.2f}{end}"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    """CLI: карта актив×стратегия + вердикт лучшей ноги на актив."""
    p = argparse.ArgumentParser(
        description="Карта актив×стратегия (LOO-матрица + вердикты)")
    p.add_argument("--trend", nargs="*", default=[],
                   help="тренд-стратегии из реестра")
    p.add_argument("--mr", nargs="*", default=[],
                   help="MR-стратегии из реестра")
    p.add_argument("--source", default="yf",
                   choices=["yf", "databento", "ccxt"])
    p.add_argument("--basket", default="commodity",
                   choices=["commodity", "equity", "crypto"])
    p.add_argument("--panel-dir", default=None)
    p.add_argument("--crypto-dir", default="data/crypto",
                   help="каталог parquet-свечей для --source ccxt")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--interval", default="1d")
    p.add_argument("--cost", type=float, default=0.0002)
    p.add_argument("--vt", action="store_true")
    p.add_argument("--sizer", default="realized",
                   choices=["realized", "garch"])
    p.add_argument("--target-vol", type=float, default=0.20)
    p.add_argument("--include", default=None,
                   help="активы или @КОРЗИНА (например @DONCH_CORE_COMM,"
                        " CL,GC); см. core.config.NAMED_BASKETS")
    p.add_argument("--exclude", default=None,
                   help="активы или @КОРЗИНА для исключения")
    args = p.parse_args()

    if not args.trend and not args.mr:
        raise SystemExit("укажи хотя бы --trend или --mr стратегии")

    panel_dir = args.panel_dir
    if panel_dir is None:
        panel_dir = ("data/panels/equities" if args.basket == "equity"
                     else "data/panels/futures")
    if args.source == "yf":
        source = YFinanceSource()
    elif args.source == "ccxt":
        source = CCXTSource(data_dir=getattr(
            args, "crypto_dir", "data/crypto"))
    else:
        source = DatabentoSource(panel_dir=panel_dir)
    if args.basket == "equity":
        basket = EQUITY_BASKET
    elif args.basket == "crypto":
        basket = (CRYPTO_CCXT if args.source == "ccxt" else CRYPTO_YF)
    elif args.source == "databento":
        basket = {s: s for s in COMMODITY_DATABENTO}
    else:
        basket = COMMODITY_YF
    basket = filter_basket(
        basket, include=args.include, exclude=args.exclude)

    print(f"{BOLD}Карта актив×стратегия | {args.basket} | "
          f"{args.source} | vt@{args.target_vol:.0%} | "
          f"{args.start}..{args.end}{RESET}")

    loo_t, solo_t = _collect(args.trend, basket, source, args) \
        if args.trend else (pd.DataFrame(), pd.DataFrame())
    loo_m, solo_m = _collect(args.mr, basket, source, args) \
        if args.mr else (pd.DataFrame(), pd.DataFrame())

    if not loo_t.empty:
        print("\n" + _fmt_matrix(
            loo_t, f"{BOLD}ТРЕНД-ноги — LOO-дельта "
                   f"(красный=балласт, зелёный=держать){RESET}"))
    if not loo_m.empty:
        print("\n" + _fmt_matrix(
            loo_m, f"{BOLD}MR-ноги — LOO-дельта "
                   f"(красный=балласт, зелёный=держать){RESET}"))

    # Вердикт по активу: средняя LOO-дельта по классу (чем ниже —
    # тем сильнее класс поддерживает актив).
    all_inst = sorted(set(loo_t.index) | set(loo_m.index))
    print(f"\n{BOLD}=== ВЕРДИКТ ПО АКТИВУ "
          f"(куда направлять) ==={RESET}")
    print(f"  {'актив':12s} {'тренд μLOO':>11s} {'MR μLOO':>10s}  "
          f"рекомендация")
    print("  " + "-" * 56)
    for inst in all_inst:
        t = loo_t.loc[inst].mean() if inst in loo_t.index else np.nan
        m = loo_m.loc[inst].mean() if inst in loo_m.index else np.nan
        # Ниже дельта = лучше поддерживает. Отрицательная = держать.
        if pd.isna(t):
            rec = "MR (нет тренд-данных)"
        elif pd.isna(m):
            rec = "тренд (нет MR-данных)"
        elif t < -0.03 and t < m:
            rec = f"{GREEN}ТРЕНД{RESET}"
        elif m < -0.03 and m < t:
            rec = f"{GREEN}MR{RESET}"
        elif t > 0.03 and m > 0.03:
            rec = f"{RED}балласт в обеих (кандидат на вылет){RESET}"
        else:
            rec = "нейтрален"
        ts = f"{t:+.2f}" if not pd.isna(t) else "—"
        ms = f"{m:+.2f}" if not pd.isna(m) else "—"
        print(f"  {inst:12s} {ts:>11s} {ms:>10s}  {rec}")

    print(f"\n{YELLOW}Напоминание: одноисточниковый LOO — гипотеза. "
          f"Исключать/перемещать актив только при совпадении на двух "
          f"источниках И наличии механизма (H-тест){RESET}")


if __name__ == "__main__":
    main()
