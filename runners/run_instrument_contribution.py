"""Раннер: вклад ИНСТРУМЕНТОВ в портфель (LOO — детектор балласта).

Отвечает на «не тянет ли какой-то актив портфель вниз?» — то, чего
не видно в усреднённой комбо-строке. Для каждого инструмента:
индивидуальные return/DD/Sharpe И leave-one-out дельта портфельного
Sharpe (выкинули актив — стало лучше/хуже). Механика — в
diagnostics/instrument_contribution.py.

Sharpe — GROSS (rf=0): инвариантен к масштабу позиции, поэтому
инструменты с разной волой сравниваются честно.

Запуск:
    python -m runners.run_instrument_contribution \\
        --strategy donchian_vt --basket commodity --source databento \\
        --panel-dir data/panels/futures --start 2019-01-01 \\
        --vt --target-vol 0.20
"""

from __future__ import annotations

import argparse

from core.config import (
    COMMODITY_DATABENTO, COMMODITY_YF, CRYPTO_CCXT, CRYPTO_YF, EQUITY_BASKET)
from core.sizing import make_sizer
from data.databento_source import DatabentoSource
from data.ccxt_source import CCXTSource
from data.yfinance_source import YFinanceSource
from diagnostics.instrument_contribution import (
    format_contribution,
    instrument_contribution,
    per_instrument_returns,
)
from diagnostics.instrument_contribution import _sharpe  # noqa: F401
from runners.run_basket import STRATEGIES

BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[91m"


def main() -> None:
    """CLI: LOO-вклад инструментов выбранной стратегии на корзине."""
    p = argparse.ArgumentParser(
        description="LOO-вклад инструментов (детектор балласта)")
    p.add_argument("--strategy", default="donchian_vt",
                   help="имя стратегии из реестра run_basket")
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
    p.add_argument("--vt", action="store_true",
                   help="обернуть стратегию vol-таргетингом")
    p.add_argument("--sizer", default="realized",
                   choices=["realized", "garch"])
    p.add_argument("--target-vol", type=float, default=0.15)
    args = p.parse_args()

    if args.strategy not in STRATEGIES:
        raise SystemExit(f"нет стратегии '{args.strategy}' в реестре")
    strategy_fn = STRATEGIES[args.strategy]

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

    sizer = make_sizer(args.sizer, target_vol=args.target_vol) \
        if args.vt else None

    vt_note = (f" | vt:{args.sizer}@{args.target_vol:.0%}"
               if args.vt else "")
    print(f"{BOLD}Вклад инструментов | {args.strategy} | "
          f"{args.basket} | {args.source}{vt_note} | "
          f"{args.start}..{args.end}{RESET}")

    rets, bpy = per_instrument_returns(
        strategy_fn, basket, source, args.start, args.end,
        sizer=sizer, cost=args.cost, interval=args.interval,
    )
    full = rets.mean(axis=1, skipna=True).fillna(0.0)
    full_sharpe = _sharpe(full, bpy)
    df = instrument_contribution(rets, bpy)
    print("\n" + format_contribution(
        df, full_sharpe,
        f"{args.strategy} — вклад по инструментам "
        f"(сверху вниз: главные кандидаты в балласт)"))

    ballast = df[df["verdict"].str.startswith("БАЛЛАСТ")]
    if not ballast.empty:
        names = ", ".join(ballast.index)
        print(f"\n{RED}Кандидаты в балласт: {names}{RESET}")
        print("  Проверить перед исключением: (1) второй источник, "
              "(2) механизм — почему актив не торгуется этой "
              "стратегией.")


if __name__ == "__main__":
    main()
