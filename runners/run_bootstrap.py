"""Раннер bootstrap-сравнения: значимо ли A лучше B по Sharpe.

Арбитр для решений о смене чемпиона: «kama +4.4% против donchian_vt
+3.4%» — точечные числа на 7 окнах; paired stationary bootstrap строит
CI на РАЗНОСТЬ Sharpe. CI накрывает ноль -> неразличимы, корона не
переходит. Подробности метода — diagnostics/bootstrap.py.

Запуск:
    python -m runners.run_bootstrap --a kama --b donchian_vt \\
        --basket commodity --source databento \\
        --panel-dir data/panels/futures --start 2019-01-01 \\
        --vt --rf 0.045
"""

from __future__ import annotations

import argparse

import pandas as pd

from core.config import (
    COMMODITY_DATABENTO,
    COMMODITY_YF,
    CRYPTO_CCXT,
    CRYPTO_YF,
    EQUITY_BASKET,
)
from core.engine import run_engine
from core.sizing import make_sizer
from data.ccxt_source import CCXTSource
from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from diagnostics.bootstrap import sharpe_ci, sharpe_diff_ci

BOLD, GREEN, RED, YELLOW, RESET = (
    "\033[1m", "\033[92m", "\033[91m", "\033[93m", "\033[0m")


def _sleeve(strategy_fn, basket, source, start, end, interval,
            sizer, cost):
    """Equal-weight sleeve-доходности стратегии по корзине.

    Returns:
        (Series P&L, bars_per_year первого инструмента) — bpy нужен
        для честной аннуализации на интрадей (4h=2190, не 252).
    """
    cols = {}
    bpy = 252.0
    for name, ticker in basket.items():
        try:
            bars = source.load(ticker, start, end, interval)
        except Exception as exc:  # noqa: BLE001
            print(f"  {YELLOW}пропуск {name}: {exc}{RESET}")
            continue
        bpy = bars.bars_per_year
        pos = strategy_fn(bars)
        if sizer is not None:
            pos = pos * sizer(bars)
        cols[name] = run_engine(bars, pos, cost=cost).equity.pct_change()
    rets = pd.DataFrame(cols).mean(axis=1, skipna=True).fillna(0.0)
    return rets, bpy


def main() -> None:
    """CLI: CI Sharpe обеих стратегий + CI разности."""
    from runners.run_basket import STRATEGIES

    p = argparse.ArgumentParser(
        description="Bootstrap-сравнение Sharpe двух стратегий")
    p.add_argument("--a", required=True, help="стратегия A")
    p.add_argument("--b", required=True, help="стратегия B")
    p.add_argument("--basket", default="commodity")
    p.add_argument("--source", default="yf")
    p.add_argument("--panel-dir", default=None)
    p.add_argument("--crypto-dir", default="data/crypto")
    p.add_argument("--interval", default="1d")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--vt", action="store_true")
    p.add_argument("--sizer", default="realized",
                   choices=["realized", "garch"])
    p.add_argument("--target-vol", type=float, default=0.15)
    p.add_argument("--cost", type=float, default=0.0002)
    p.add_argument("--rf", type=float, default=0.0,
                   help="годовая безрисковая ставка: excess Sharpe "
                        "(дыра, найденная Александром; напр. 0.045)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--ci", type=float, default=0.90)
    args = p.parse_args()

    panel_dir = args.panel_dir or (
        "data/panels/equities" if args.basket == "equity"
        else "data/panels/futures")
    if args.source == "yf":
        source = YFinanceSource()
    elif args.source == "ccxt":
        source = CCXTSource(data_dir=args.crypto_dir)
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

    sizer = (make_sizer(args.sizer, target_vol=args.target_vol)
             if args.vt else None)
    print(f"{BOLD}Bootstrap {args.a} vs {args.b} | {args.basket} | "
          f"{args.source} | rf={args.rf:.1%} | CI {args.ci:.0%}{RESET}")
    ra, bpy = _sleeve(STRATEGIES[args.a], basket, source, args.start,
                      args.end, args.interval, sizer, args.cost)
    rb, _ = _sleeve(STRATEGIES[args.b], basket, source, args.start,
                    args.end, args.interval, sizer, args.cost)

    for name, r in ((args.a, ra), (args.b, rb)):
        c = sharpe_ci(r, bars_per_year=bpy, n_boot=args.n_boot,
                      ci=args.ci, rf=args.rf)
        print(f"  {name:<18} Sharpe {c['sharpe']:+.2f}  "
              f"CI [{c['lo']:+.2f}, {c['hi']:+.2f}]")

    d = sharpe_diff_ci(ra, rb, bars_per_year=bpy,
                       n_boot=args.n_boot, ci=args.ci,
                       rf=args.rf)
    verdict = (f"{GREEN}ЗНАЧИМО{RESET}" if d["significant"]
               else f"{YELLOW}НЕРАЗЛИЧИМЫ{RESET}")
    print(f"\n  Разность Sharpe ({args.a} − {args.b}): "
          f"{d['diff']:+.2f}  CI [{d['lo']:+.2f}, {d['hi']:+.2f}] "
          f"-> {verdict}")
    print(f"  (paired stationary bootstrap, средний блок "
          f"{d['avg_block']:.0f} баров, {args.n_boot} ресемплов)")
    if not d["significant"]:
        print(f"  {YELLOW}CI разности накрывает 0: смена чемпиона по "
              f"этим данным НЕ обоснована.{RESET}")


if __name__ == "__main__":
    main()
