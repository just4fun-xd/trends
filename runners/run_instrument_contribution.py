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
    COMMODITY_DATABENTO, COMMODITY_YF, CRYPTO_CCXT, CRYPTO_YF,
    EQUITY_BASKET, filter_basket, instrument_name_map)
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
YELLOW = "\033[93m"
RED = "\033[91m"


def main() -> None:
    """CLI: LOO-вклад инструментов выбранной стратегии на корзине."""
    p = argparse.ArgumentParser(
        description="LOO-вклад инструментов (детектор балласта)")
    p.add_argument("--strategy", default=["donchian_vt"], nargs="+",
                   help="одно или несколько имён стратегий из реестра "
                        "(для пакетного сбора карты актив×стратегия)")
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
    p.add_argument("--include", default=None,
                   help="активы или @КОРЗИНА (например @DONCH_CORE_COMM,"
                        " CL,GC); см. core.config.NAMED_BASKETS")
    p.add_argument("--exclude", default=None,
                   help="активы или @КОРЗИНА для исключения")
    p.add_argument("--csv", default=None,
                   help="дописать вклад по инструментам в CSV "
                        "(strategy,basket,source,asset,solo_ret,solo_dd,"
                        "solo_sharpe,loo_delta,verdict) для карт")
    args = p.parse_args()

    unknown = [s for s in args.strategy if s not in STRATEGIES]
    if unknown:
        raise SystemExit(f"нет стратегий в реестре: {unknown}")

    # Путь панели выводится ИЗ ИНТЕРВАЛА, если явно не задан --panel-dir.
    # 4h -> panels_4h, 1h -> panels_1h, 1d -> panels. Раньше требовалось
    # дублировать --panel-dir data/panels_4h/futures к --interval 4h;
    # теперь достаточно --interval (фикс UX 10.07.26). --panel-dir
    # остаётся как override для нестандартных путей.
    panel_dir = args.panel_dir
    if panel_dir is None:
        sleeve = "equities" if args.basket == "equity" else "futures"
        suffix = {"1d": "panels", "4h": "panels_4h",
                  "1h": "panels_1h"}.get(args.interval, "panels")
        panel_dir = f"data/{suffix}/{sleeve}"
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

    sizer = make_sizer(args.sizer, target_vol=args.target_vol) \
        if args.vt else None

    vt_note = (f" | vt:{args.sizer}@{args.target_vol:.0%}"
               if args.vt else "")

    csv_rows = []
    # Guard 10.07.26: интервал и путь панели должны совпадать. Теперь
    # путь авто-выводится из интервала, но если юзер задал --panel-dir
    # вручную и он противоречит интервалу — предупреждаем.
    if (args.source == "databento" and args.interval != "1d"
            and args.interval not in panel_dir):
        print(f"{YELLOW}  ВНИМАНИЕ: --interval {args.interval}, но путь "
              f"панели '{panel_dir}' не содержит '{args.interval}'. "
              f"bars_per_year может быть неверным.{RESET}")
    for strat in args.strategy:
        strategy_fn = STRATEGIES[strat]
        print(f"{BOLD}Вклад инструментов | {strat} | "
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
            f"{strat} — вклад по инструментам "
            f"(сверху вниз: главные кандидаты в балласт)",
            name_map=instrument_name_map()))

        ballast = df[df["verdict"].str.startswith("БАЛЛАСТ")]
        if not ballast.empty:
            names = ", ".join(ballast.index)
            print(f"\n{RED}Кандидаты в балласт: {names}{RESET}")
            print("  Проверить перед исключением: (1) второй "
                  "источник, (2) механизм — почему актив не "
                  "торгуется этой стратегией.")
        print()

        if args.csv:
            for asset, row in df.iterrows():
                csv_rows.append({
                    "strategy": strat, "basket": args.basket,
                    "source": args.source, "asset": asset,
                    "solo_ret": round(float(row["solo_ret"]), 4),
                    "solo_dd": round(float(row["solo_dd"]), 4),
                    "solo_sharpe": round(
                        float(row["solo_sharpe"]), 4),
                    "corr_rest": round(float(row["corr_rest"]), 4)
                    if row["corr_rest"] == row["corr_rest"] else None,
                    "loo_delta": round(float(row["loo_delta"]), 4),
                    "verdict": row["verdict"],
                })

    if args.csv and csv_rows:
        import csv as _csv
        import os
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            if new:
                w.writeheader()
            w.writerows(csv_rows)
        print(f"{BOLD}CSV дописан: {args.csv} "
              f"(+{len(csv_rows)} строк){RESET}")


if __name__ == "__main__":
    main()
