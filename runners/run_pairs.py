"""Раннер парного трейдинга (Kalman-бета) — research-инструмент.

Пары НЕЛЬЗЯ гонять через runners.run_basket: посерийный контракт
Bars -> position работает с ОДНИМ инструментом, а паре нужно два ряда
цен одновременно (лонг-нога + хедж-нога). Отдельный раннер.

ПОМЕТКА RESEARCH: OU_RESULTS.md закрыл парный трейдинг (сигнал без края
даже при нулевых издержках на дневных commodity-данных). Kalman чинит
look-ahead беты, но не создаёт край. Инструмент для проверки гипотез на
парах с ПОДТВЕРЖДЁННОЙ коинтеграцией, не боевой.

Запуск (примеры):
    # классическая пара нефтей через yfinance:
    python -m runners.run_pairs --a CL=F --b BZ=F \\
        --start 2021-01-01 --end 2026-01-01 --yearly

    # из Databento futures-панелей:
    python -m runners.run_pairs --a CL --b BZ --source databento
"""

from __future__ import annotations

import argparse

from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from diagnostics.yearly import format_yearly_table, yearly_breakdown
from strategies.pairs import run_pair_kalman

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def main() -> None:
    """CLI-точка входа парного раннера."""
    parser = argparse.ArgumentParser(
        description="Kalman-пара (research): спред A против B"
    )
    parser.add_argument("--a", required=True,
                        help="Тикер лонг-ноги (CL=F для yf, CL для db)")
    parser.add_argument("--b", required=True,
                        help="Тикер хедж-ноги")
    parser.add_argument("--source", default="yf",
                        choices=["yf", "databento"])
    parser.add_argument("--panel-dir", default="data/panels/futures")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--window", type=int, default=20,
                        help="Окно z-score спреда")
    parser.add_argument("--entry", type=float, default=2.0)
    parser.add_argument("--exit", type=float, default=0.5, dest="exit_z")
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--yearly", action="store_true")
    args = parser.parse_args()

    source = (YFinanceSource() if args.source == "yf"
              else DatabentoSource(panel_dir=args.panel_dir))

    print(f"{BOLD}Kalman-пара {args.a} / {args.b} | {args.source} | "
          f"{args.start}..{args.end} (RESEARCH){RESET}")

    bars_a = source.load(args.a, args.start, args.end, args.interval)
    bars_b = source.load(args.b, args.start, args.end, args.interval)

    res = run_pair_kalman(
        bars_a.close, bars_b.close,
        window=args.window, entry=args.entry, exit_z=args.exit_z,
        cost=args.cost,
    )

    ret_c = GREEN if res.total_return > 0 else RED
    print(f"\n  Доходность спреда: {ret_c}{res.total_return:+.1%}{RESET}")
    print(f"  Max DD:            {res.max_drawdown:+.1%}")
    print(f"  Sharpe:            {res.sharpe:+.2f}")
    print(f"  {YELLOW}Research-трек: край парного трейдинга не "
          f"подтверждён (OU_RESULTS.md).{RESET}")

    if args.yearly:
        yb = yearly_breakdown(res.equity, res.bars_per_year)
        print("\n" + format_yearly_table(
            yb, f"Пара {args.a}/{args.b} по годам"
        ))


if __name__ == "__main__":
    main()
