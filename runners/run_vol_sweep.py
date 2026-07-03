"""Раннер vol-sweep: сколько доходности выжать плечом до DD<40%.

Прогоняет СЫРОЙ сигнал стратегии (без встроенного vol-targeting) при
разных target_vol и печатает таблицу доход/DD/издержки/плечо. Показывает
рабочий уровень риска — где доход максимален, а DD ещё под лимитом.

Запуск:
    python -m runners.run_vol_sweep --strategy mr_atr_stop \\
        --basket commodity --start 2021-01-01 --end 2026-01-01

    # свои уровни vol:
    python -m runners.run_vol_sweep --strategy mr_atr_stop \\
        --vols 0.15 0.30 0.45 0.60
"""

from __future__ import annotations

import argparse

from core.config import COMMODITY_DATABENTO, COMMODITY_YF, EQUITY_BASKET
from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from diagnostics.vol_sweep import format_sweep, vol_sweep_basket
from runners.run_basket import STRATEGIES

YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main() -> None:
    """CLI-точка входа vol-sweep."""
    parser = argparse.ArgumentParser(
        description="Sweep target_vol: запас по риску до DD<40%"
    )
    parser.add_argument("--strategy", default="mr_atr_stop",
                        choices=list(STRATEGIES.keys()))
    parser.add_argument("--source", default="yf",
                        choices=["yf", "databento"])
    parser.add_argument("--basket", default="commodity",
                        choices=["commodity", "equity"])
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--vols", nargs="+", type=float,
                        default=[0.15, 0.25, 0.35, 0.50],
                        help="Уровни target_vol для перебора")
    parser.add_argument("--max-lev", type=float, default=6.0,
                        help="Потолок плеча (поднят, чтобы не упереться "
                             "раньше DD)")
    parser.add_argument("--panel-dir", default=None)
    args = parser.parse_args()

    panel_dir = args.panel_dir
    if panel_dir is None:
        panel_dir = ("data/panels/equities" if args.basket == "equity"
                     else "data/panels/futures")
    if args.source == "databento":
        source = DatabentoSource(panel_dir=panel_dir)
        basket = (EQUITY_BASKET if args.basket == "equity"
                  else {s: s for s in COMMODITY_DATABENTO})
    else:
        source = YFinanceSource()
        basket = (EQUITY_BASKET if args.basket == "equity"
                  else COMMODITY_YF)

    signal_fn = STRATEGIES[args.strategy]

    print(f"{BOLD}Vol-sweep {args.strategy} | {args.basket} | "
          f"{args.source} | {args.start}..{args.end}{RESET}")
    bars_by_symbol = {}
    for name, ticker in basket.items():
        try:
            bars_by_symbol[name] = source.load(
                ticker, args.start, args.end, args.interval
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {YELLOW}пропуск {name} ({ticker}): {exc}{RESET}")
    if not bars_by_symbol:
        print(f"{RED}Нет данных.{RESET}")
        return

    df = vol_sweep_basket(
        bars_by_symbol, signal_fn, tuple(args.vols),
        max_lev_cap=args.max_lev, cost=args.cost,
    )
    print("\n" + format_sweep(df, args.strategy))

    # Рабочий уровень: макс. доход среди прошедших DD<40%.
    passing = df[df["passes_dd"]]
    if not passing.empty:
        best_tv = passing["mean_return"].idxmax()
        best = passing.loc[best_tv]
        print(f"\n{BOLD}Рекомендация:{RESET} target_vol ≈ {best_tv:.0%} "
              f"-> доход {best['mean_return']:+.1%}, DD "
              f"{best['worst_dd']:+.1%} (макс. доход в пределах DD<40%)")
    else:
        print(f"\n{RED}Ни один уровень не прошёл DD<40%.{RESET}")


if __name__ == "__main__":
    main()
