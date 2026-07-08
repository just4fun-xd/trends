"""Рейтинг стратегий внутри категорий на классе активов.

Отвечает на запрос Кирилла: «составить категории активов и к каждой
сопоставить алгоритм, причём рейтинг алгоритмов внутри категорий».

Гоняет список стратегий по корзине, ранжирует по портфельному gross
Sharpe (равновзвешенный портфель дневных P&L). Стратегии сгруппированы
по семействам (trend / mr / specialist), внутри семейства — рейтинг.

Это НЕ замена walk-forward/bootstrap: это быстрый скрининг «кто вообще
в игре на этом классе». Финальный выбор — через bootstrap разности
(run_bootstrap) и walk-forward на двух источниках.

Sharpe gross (rf=0, масштабо-инвариантен).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from core.config import (
    COMMODITY_DATABENTO, COMMODITY_YF, CRYPTO_CCXT, CRYPTO_YF, EQUITY_BASKET)
from core.sizing import make_sizer
from data.databento_source import DatabentoSource
from data.ccxt_source import CCXTSource
from data.yfinance_source import YFinanceSource
from diagnostics.instrument_contribution import per_instrument_returns
from runners.run_basket import STRATEGIES

BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"

# Семейства стратегий (для группировки рейтинга).
FAMILIES = {
    "trend": ["donchian_vt", "kama", "ewmac", "trend_ens",
              "donch_multi", "chandelier", "adx_donch", "channel_pos",
              "tsmom_multi"],
    "mr": ["mr_keltner", "mr_kelt_confirm", "mr_ens", "mr_ens_gate",
           "mr_atr_stop", "mr_time_stop", "mr_confirm", "mr_lowvol"],
    "specialist": ["ou", "ou_hurst_gate", "ou_jump_asym", "ou_asym",
                   "hurst_alloc"],
}


def _port_sharpe(rets: pd.DataFrame, bpy: float = 252.0) -> float:
    """Gross Sharpe равновзвешенного портфеля дневных P&L."""
    port = rets.mean(axis=1, skipna=True).fillna(0.0)
    std = port.std(ddof=1)
    if std <= 0:
        return 0.0
    return float(port.mean() / std * np.sqrt(bpy))


def main() -> None:
    """CLI: рейтинг стратегий по семействам на классе активов."""
    p = argparse.ArgumentParser(
        description="Рейтинг стратегий внутри категорий")
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
    p.add_argument("--vt", action="store_true")
    p.add_argument("--target-vol", type=float, default=0.20)
    p.add_argument("--families", nargs="*",
                   default=["trend", "mr", "specialist"])
    args = p.parse_args()

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

    sizer = make_sizer(args.sizer if hasattr(args, "sizer")
                       else "realized", target_vol=args.target_vol) \
        if args.vt else None

    vt_note = f" | vt@{args.target_vol:.0%}" if args.vt else ""
    print(f"{BOLD}Рейтинг стратегий по категориям | {args.basket} | "
          f"{args.source}{vt_note} | {args.start}..{args.end}{RESET}\n")

    for fam in args.families:
        names = FAMILIES.get(fam, [])
        scored = []
        for name in names:
            if name not in STRATEGIES:
                continue
            try:
                rets, bpy = per_instrument_returns(
                    STRATEGIES[name], basket, source, args.start,
                    args.end, sizer=sizer, cost=args.cost,
                    interval=args.interval)
            except Exception:  # noqa: BLE001
                continue
            scored.append((name, _port_sharpe(rets, bpy)))
        scored.sort(key=lambda x: -x[1])
        print(f"{BOLD}=== {fam.upper()} — рейтинг по портф. Sharpe "
              f"==={RESET}")
        if not scored:
            print("  (нет доступных стратегий)\n")
            continue
        for rank, (name, sh) in enumerate(scored, 1):
            col = GREEN if sh > 0.5 else (
                YELLOW if sh > 0 else "")
            end = RESET if col else ""
            medal = ("🥇" if rank == 1 else "🥈" if rank == 2
                     else "🥉" if rank == 3 else f" {rank}")
            print(f"  {medal} {col}{name:20s} Sharpe {sh:+.2f}{end}")
        print()

    print(f"{YELLOW}Рейтинг — скрининг «кто в игре», не финальный "
          f"вердикт. Подтверждать через run_bootstrap (разность "
          f"Sharpe) и walk-forward на двух источниках.{RESET}")


if __name__ == "__main__":
    main()
