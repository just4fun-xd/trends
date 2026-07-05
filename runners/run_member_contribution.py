"""Раннер LOO-анализа вклада членов ансамбля (mr / trend).

Отвечает на «а вдруг не-лучшие варианты тоже выстрелят»: ценность
члена — предельный вклад в комбинацию, не одиночный Sharpe. Слабый,
но декоррелированный вариант может помогать; сильный, но дублирующий
— быть балластом. Подробности механики — в
diagnostics/member_contribution.py.

Запуск:
    python -m runners.run_member_contribution --ensemble mr \\
        --basket commodity --start 2019-01-01 --vt --target-vol 0.35
    python -m runners.run_member_contribution --ensemble trend \\
        --basket commodity --source databento \\
        --panel-dir data/panels/futures --start 2019-01-01 --vt
"""

from __future__ import annotations

import argparse

from core.config import (
    COMMODITY_DATABENTO,
    COMMODITY_YF,
    EQUITY_BASKET,
)
from core.sizing import make_sizer
from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from diagnostics.member_contribution import (
    format_contribution,
    member_contribution,
)
from strategies.ensemble import (
    MR_ENSEMBLE_MEMBERS,
    TREND_ENSEMBLE_MEMBERS,
)

YELLOW = "\033[93m"
RESET = "\033[0m"

ENSEMBLE_MEMBERS = {
    "mr": {fn.__name__: fn for fn in MR_ENSEMBLE_MEMBERS},
    "trend": {fn.__name__: fn for fn in TREND_ENSEMBLE_MEMBERS},
}


def main() -> None:
    """CLI: вклад членов выбранного ансамбля на корзине."""
    p = argparse.ArgumentParser(
        description="LOO-вклад членов ансамбля (mr / trend)")
    p.add_argument("--ensemble", choices=list(ENSEMBLE_MEMBERS),
                   default="mr")
    p.add_argument("--basket", default="commodity")
    p.add_argument("--source", default="yf")
    p.add_argument("--panel-dir", default=None)
    p.add_argument("--interval", default="1d")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--vt", action="store_true",
                   help="наложить vol-targeting на позиции членов")
    p.add_argument("--sizer", default="realized",
                   choices=["realized", "garch"])
    p.add_argument("--target-vol", type=float, default=0.15)
    p.add_argument("--cost", type=float, default=0.0002)
    args = p.parse_args()

    panel_dir = args.panel_dir
    if panel_dir is None:
        panel_dir = ("data/panels/equities" if args.basket == "equity"
                     else "data/panels/futures")
    source = (YFinanceSource() if args.source == "yf"
              else DatabentoSource(panel_dir=panel_dir))
    if args.basket == "equity":
        basket = EQUITY_BASKET
    elif args.source == "databento":
        basket = {s: s for s in COMMODITY_DATABENTO}
    else:
        basket = COMMODITY_YF

    bars_by_symbol = {}
    skipped = []
    for name, ticker in basket.items():
        try:
            bars_by_symbol[name] = source.load(
                ticker, args.start, args.end, args.interval)
        except Exception as exc:  # noqa: BLE001
            skipped.append(name)
            print(f"  {YELLOW}пропуск {name} ({ticker}): {exc}{RESET}")
    if not bars_by_symbol:
        raise SystemExit("Нет данных.")
    if skipped:
        print(f"{YELLOW}ВНИМАНИЕ: корзина неполная "
              f"({len(skipped)} пропущено).{RESET}")

    sizer = (make_sizer(args.sizer, target_vol=args.target_vol)
             if args.vt else None)
    members = ENSEMBLE_MEMBERS[args.ensemble]
    res = member_contribution(
        bars_by_symbol, members, sizer=sizer, cost=args.cost)
    vt_tag = (f" | vt:{args.sizer}@{args.target_vol:.0%}"
              if args.vt else "")
    print(format_contribution(
        res,
        title=(f"{args.ensemble} | {args.basket} | {args.source}"
               f"{vt_tag} | {args.start}..{args.end}"),
    ))


if __name__ == "__main__":
    main()
