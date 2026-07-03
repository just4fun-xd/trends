"""Раннер кросс-секционных стратегий (портфельный движок).

Кросс-секционные стратегии (dual momentum и варианты, carry) НЕЛЬЗЯ
гонять через runners.run_basket — тот работает с посерийным контрактом
Bars -> position (один инструмент). Кросс-секция берёт МАТРИЦУ цен всех
инструментов и возвращает МАТРИЦУ весов для run_portfolio. Это жёсткая
архитектурная граница проекта (см. engine_portfolio.py).

Запуск (примеры):
    # research-треки dual momentum на акциях (yfinance):
    python -m runners.run_xs --strategy dual_tilt --basket equity \\
        --start 2021-01-01 --end 2026-01-01 --yearly

    # из Databento-панелей:
    python -m runners.run_xs --strategy dual_regime --basket equity \\
        --source databento --start 2020-01-01 --end 2026-01-01

    # carry на фьючерсах (нужна rollyield-панель -> только databento):
    python -m runners.run_xs --strategy carry --basket commodity \\
        --source databento
"""

from __future__ import annotations

import argparse

import pandas as pd

from core.config import COMMODITY_DATABENTO, COMMODITY_YF, EQUITY_BASKET
from core.engine_portfolio import run_portfolio
from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from diagnostics.yearly import format_yearly_table, yearly_breakdown
from strategies import cross_sectional as xs

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Имя CLI -> (функция, нужен ли benchmark, нужен ли carry).
STRATEGIES_XS = {
    "dual_mom": (xs.dual_momentum, False, False),
    "dual_tilt": (xs.dual_momentum_tilt, True, False),
    "dual_regime": (xs.dual_momentum_regime, True, False),
    "dual_volscaled": (xs.dual_momentum_volscaled, False, False),
    "carry": (xs.carry_rank, False, True),
}


def load_close_panel(
    source, basket: dict, start: str, end: str, interval: str = "1d",
) -> pd.DataFrame:
    """Собирает матрицу close по корзине через любой DataSource.

    Для DatabentoSource быстрее прямой срез панели; для yfinance —
    цикл load + конкатенация close. Инструменты без данных пропускаются
    с предупреждением.

    Args:
        source: DataSource-бэкенд.
        basket: dict {название: тикер}.
        start: Дата начала.
        end: Дата конца.
        interval: Таймфрейм.

    Returns:
        DataFrame close: даты × названия инструментов.
    """
    cols = {}
    for name, ticker in basket.items():
        try:
            bars = source.load(ticker, start, end, interval)
            cols[name] = bars.close
        except Exception as exc:  # noqa: BLE001
            print(f"  {YELLOW}пропуск {name} ({ticker}): {exc}{RESET}")
    return pd.DataFrame(cols)


def main() -> None:
    """CLI-точка входа кросс-секционного раннера."""
    parser = argparse.ArgumentParser(
        description="Прогон кросс-секционной стратегии (run_portfolio)"
    )
    parser.add_argument("--strategy", default="dual_regime",
                        choices=list(STRATEGIES_XS.keys()))
    parser.add_argument("--source", default="yf",
                        choices=["yf", "databento"])
    parser.add_argument("--basket", default="equity",
                        choices=["equity", "commodity"])
    parser.add_argument("--benchmark", default="SPY",
                        help="Тикер прокси рынка (yfinance). 'mean' -> "
                             "среднее корзины (для databento без SPY).")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--panel-dir", default=None)
    parser.add_argument("--yearly", action="store_true")
    args = parser.parse_args()

    fn, needs_bench, needs_carry = STRATEGIES_XS[args.strategy]

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

    print(f"{BOLD}XS-стратегия {args.strategy} | {args.basket} | "
          f"{args.source} | {args.start}..{args.end}{RESET}")

    if needs_carry:
        if args.source != "databento":
            print(f"{RED}carry требует rollyield-панель — только "
                  f"--source databento (фьючерсы M1/M2).{RESET}")
            return
        symbols = list(basket.values())
        carry = source.load_carry(symbols)
        carry = carry[(carry.index >= pd.Timestamp(args.start))
                      & (carry.index <= pd.Timestamp(args.end))]
        prices = source.load_panel_close(symbols).reindex(carry.index)
        weights = fn(carry)
    else:
        prices = load_close_panel(source, basket, args.start, args.end,
                                  args.interval)
        if prices.shape[1] < 5:
            print(f"{RED}Мало инструментов ({prices.shape[1]}) — "
                  f"ранжированию не с чем работать.{RESET}")
            return
        if needs_bench:
            if args.benchmark == "mean" or args.source == "databento":
                bench = prices.mean(axis=1)
                print(f"  {YELLOW}benchmark = среднее корзины "
                      f"(прокси){RESET}")
            else:
                bench = YFinanceSource().load(
                    args.benchmark, args.start, args.end, args.interval
                ).close
            weights = fn(prices, bench)
        else:
            weights = fn(prices)

    res = run_portfolio(prices, weights, cost=args.cost)

    print(f"\n{BOLD}=== {args.strategy} — портфель ==={RESET}")
    ret_c = GREEN if res.total_return > 0 else RED
    print(f"  Доходность:   {ret_c}{res.total_return:+.1%}{RESET}")
    print(f"  Max DD:       {res.max_drawdown:+.1%}")
    print(f"  Sharpe:       {res.sharpe:+.2f}")
    print(f"  Gross (мед.): {res.gross[res.gross > 0].median():.2f}")
    print(f"  Проходит DD<40%: "
          f"{'да' if res.passes_dd() else RED + 'НЕТ' + RESET}")

    if args.yearly:
        yb = yearly_breakdown(res.equity, res.bars_per_year)
        print("\n" + format_yearly_table(yb, f"{args.strategy} по годам"))


if __name__ == "__main__":
    main()
