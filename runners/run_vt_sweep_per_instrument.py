"""Пер-инструментный VT-sweep: максимальный target_vol под DD-лимитом.

Отвечает на вопрос Кирилла: «не душить VT ради балласта, а поднимать
для прибыльных — какие до 60, какие до 40». Общий --target-vol давит
всех одним числом; здесь для КАЖДОГО инструмента ищется наибольший
target_vol, при котором его DD ещё в пределах лимита (40%).

Для каждого (актив × target_vol из сетки):
  - solo return / DD / Sharpe этого инструмента под стратегией;
  - помечается максимальный tv, где DD > -limit (проходит).

Итог — таблица «инструмент -> рекомендуемый target_vol»: балласт
(отрицательный Sharpe на всех tv) в вылет, прибыльные — на свой
потолок. Это вход для будущего пер-инструментного сайзинга.

Sharpe gross (масштабо-инвариантен). Одноисточниковый — гипотеза;
подтверждать на втором источнике.
"""

from __future__ import annotations

import argparse

import numpy as np

from core.config import (
    COMMODITY_DATABENTO, COMMODITY_YF, CRYPTO_CCXT, CRYPTO_YF, EQUITY_BASKET)
from core.engine import run_engine
from core.sizing import make_sizer
from data.databento_source import DatabentoSource
from data.ccxt_source import CCXTSource
from data.yfinance_source import YFinanceSource
from runners.run_basket import STRATEGIES

BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"


def main() -> None:
    """CLI: пер-инструментный VT-sweep под DD-лимитом."""
    p = argparse.ArgumentParser(
        description="Макс target_vol на инструмент под DD-лимитом")
    p.add_argument("--strategy", default="donchian_vt")
    p.add_argument("--source", default="yf",
                   choices=["yf", "databento", "ccxt"])
    p.add_argument("--basket", default="commodity",
                   choices=["commodity", "equity", "crypto"])
    p.add_argument("--panel-dir", default=None)
    p.add_argument("--crypto-dir", default="data/crypto",
                   help="каталог parquet-свечей для --source ccxt")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--interval", default="1d")
    p.add_argument("--cost", type=float, default=0.0002)
    p.add_argument("--sizer", default="realized",
                   choices=["realized", "garch"])
    p.add_argument("--dd-limit", type=float, default=0.40,
                   help="Лимит просадки (0.40 = 40%%)")
    p.add_argument("--grid", default="0.15,0.20,0.30,0.40,0.50,0.60",
                   help="Сетка target_vol через запятую")
    p.add_argument("--max-leverage", type=float, default=2.0,
                   help="Потолок множителя VT (по умолчанию 2x). "
                        "target_vol выше realized_vol -> плечо; при "
                        "упоре в этот кэп Sharpe перестаёт расти "
                        "(плоский хвост = кэп, не рост edge).")
    args = p.parse_args()

    if args.strategy not in STRATEGIES:
        raise SystemExit(f"нет стратегии '{args.strategy}'")
    strat = STRATEGIES[args.strategy]
    grid = [float(x) for x in args.grid.split(",")]

    panel_dir = args.panel_dir or (
        "data/panels/equities" if args.basket == "equity"
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

    print(f"{BOLD}Пер-инструментный VT-sweep | {args.strategy} | "
          f"{args.basket} | {args.source} | DD<{args.dd_limit:.0%} | "
          f"{args.start}..{args.end}{RESET}")
    print(f"  сетка target_vol: "
          f"{', '.join(f'{g:.0%}' for g in grid)}\n")

    hdr = f"  {'инструмент':13s}"
    for g in grid:
        hdr += f"{g:>7.0%}"
    hdr += f"  {'рек.tv':>7s}  вердикт"
    print(hdr)
    print("  " + "-" * (15 + 7 * len(grid) + 22))

    keep, drop = [], []
    for name, ticker in basket.items():
        try:
            bars = source.load(ticker, args.start, args.end,
                               args.interval)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:13s} пропуск: {exc}")
            continue
        base = strat(bars)
        best_tv, best_ret, best_lev = None, None, None
        line = f"  {name:13s}"
        for g in grid:
            sizer = make_sizer(args.sizer, target_vol=g,
                               max_leverage=args.max_leverage)
            mult = sizer(bars)
            res = run_engine(bars, base * mult, cost=args.cost)
            dd_ok = res.max_drawdown > -args.dd_limit
            ret = res.total_return
            mean_lev = float(mult.abs().mean())
            rets = res.equity.pct_change().dropna()
            sh = (float(rets.mean() / rets.std() * np.sqrt(
                res.bars_per_year)) if rets.std() > 0 else 0.0)
            mark = GREEN if (dd_ok and sh > 0) else (
                YELLOW if dd_ok else RED)
            line += f"{mark}{sh:>+7.2f}{RESET}"
            if dd_ok and ret > 0:
                best_tv, best_ret, best_lev = g, ret, mean_lev
        # Вердикт: макс tv с положительным return под DD-лимитом.
        if best_tv is None:
            verdict = f"{RED}балласт (нет прибыльного tv){RESET}"
            rec = "—"
            drop.append(name)
        else:
            lev_warn = (f" {RED}⚠{best_lev:.1f}x{RESET}"
                        if best_lev and best_lev > 3.0
                        else f" ({best_lev:.1f}x)")
            verdict = f"{GREEN}макс tv={best_tv:.0%} "\
                      f"(+{best_ret:.0%}){RESET}{lev_warn}"
            rec = f"{best_tv:.0%}"
            keep.append((name, best_tv, best_lev))
        line += f"  {rec:>7s}  {verdict}"
        print(line)

    print(f"\n  {BOLD}ИТОГ (max_leverage кэп = "
          f"{args.max_leverage:.1f}x){RESET}")
    if keep:
        print(f"  {GREEN}Держать ({len(keep)}):{RESET} " + ", ".join(
            f"{n}@{tv:.0%}({lev:.1f}x)" for n, tv, lev in
            sorted(keep, key=lambda x: -x[1])))
    if drop:
        print(f"  {RED}Балласт ({len(drop)}, в вылет):{RESET} "
              + ", ".join(drop))
    print(f"\n  {YELLOW}Sharpe gross по столбцам target_vol. "
          f"Красный = DD пробил лимит, жёлтый = DD ок но Sharpe<0, "
          f"зелёный = проходит и прибылен.{RESET}")
    print(f"  {YELLOW}Плоский Sharpe в хвосте = упор в кэп плеча "
          f"(edge не растёт, только плечо). ⚠ = среднее плечо >3x: "
          f"бэктестовый DD не ловит маржин-колл на гэпе.{RESET}")


if __name__ == "__main__":
    main()
