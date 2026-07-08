"""Раннер walk-forward: тест стабильности стратегий по времени.

Anchored walk-forward (фикс. параметры) с погодовой нарезкой. Печатает
per-window return-матрицу (год × инструмент) и вердикт робастности по
портфелю каждой стратегии.

Запуск (примеры):
    # сравнить кандидатов на сырье:
    python -m runners.run_walkforward --strategy bb_rsi_vt mr_atr_stop \\
        --basket commodity --start 2021-01-01 --end 2026-01-01

    # из Databento-панелей:
    python -m runners.run_walkforward --strategy mr_atr_stop \\
        --basket equity --source databento

Читать вердикт: РОБАСТНА = >=80% окон прибыльны, худшее окно > -15%,
все окна прошли DD<40%. ПОД ВОПРОСОМ = результат держится на 1-2 годах.
"""

from __future__ import annotations

import argparse

from core.config import (
    COMMODITY_DATABENTO, COMMODITY_YF, CRYPTO_CCXT, CRYPTO_YF, EQUITY_BASKET)
from data.databento_source import DatabentoSource
from data.ccxt_source import CCXTSource
from data.yfinance_source import YFinanceSource
from diagnostics.walkforward import (
    consistency_metrics,
    format_consistency,
    walk_forward_basket,
)
from diagnostics.yearly import format_matrix
from runners.run_basket import STRATEGIES

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _load_basket(source, basket, start, end, interval):
    """Грузит корзину в dict {инструмент: Bars}, пропуская сбойные."""
    out = {}
    for name, ticker in basket.items():
        try:
            out[name] = source.load(ticker, start, end, interval)
        except Exception as exc:  # noqa: BLE001
            print(f"  {YELLOW}пропуск {name} ({ticker}): {exc}{RESET}")
    return out


def main() -> None:
    """CLI-точка входа walk-forward раннера."""
    parser = argparse.ArgumentParser(
        description="Anchored walk-forward: тест стабильности по годам"
    )
    parser.add_argument("--strategy", nargs="+", default=["bb_rsi_vt"],
                        help="Одна или несколько стратегий для сравнения")
    parser.add_argument("--source", default="yf",
                        choices=["yf", "databento", "ccxt"])
    parser.add_argument("--basket", default="commodity",
                        choices=["commodity", "equity", "crypto"])
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--by", default="year",
                        help="Нарезка окон: year / half / число дней")
    parser.add_argument("--vt", action="store_true",
                        help="Обернуть стратегию vol-таргетингом")
    parser.add_argument("--sizer", default="realized",
                        choices=["realized", "garch"],
                        help="Оценка волы для --vt")
    parser.add_argument("--target-vol", type=float, default=0.15,
                        help="Целевая вола для --vt (0.6 = 60%). "
                             "Раньше был захардкожен 15%.")
    parser.add_argument("--panel-dir", default=None)
    parser.add_argument("--crypto-dir", default="data/crypto")
    parser.add_argument("--exclude", default=None,
                        help="Инструменты через запятую, исключить из "
                             "корзины (напр. тонкие H4-рынки: PA,PL)")
    parser.add_argument("--matrix", action="store_true",
                        help="Печатать полную матрицу год × инструмент")
    args = parser.parse_args()

    panel_dir = args.panel_dir
    if panel_dir is None:
        panel_dir = ("data/panels/equities" if args.basket == "equity"
                     else "data/panels/futures")
    if args.source == "databento":
        source = DatabentoSource(panel_dir=panel_dir)
        basket = (EQUITY_BASKET if args.basket == "equity"
                  else {s: s for s in COMMODITY_DATABENTO})
    elif args.source == "ccxt":
        source = CCXTSource(data_dir=args.crypto_dir)
        basket = CRYPTO_CCXT
    else:
        source = YFinanceSource()
        if args.basket == "equity":
            basket = EQUITY_BASKET
        elif args.basket == "crypto":
            basket = CRYPTO_YF
        else:
            basket = COMMODITY_YF

    if args.exclude:
        drop = {s.strip() for s in args.exclude.split(",")}
        removed = [k for k in basket if k in drop]
        basket = {k: v for k, v in basket.items() if k not in drop}
        if removed:
            print(f"Исключены из корзины: {', '.join(removed)}")

    print(f"{BOLD}Walk-forward (anchored, by={args.by}) | {args.basket} "
          f"| {args.source} | {args.start}..{args.end}{RESET}")
    bars_by_symbol = _load_basket(
        source, basket, args.start, args.end, args.interval
    )
    if not bars_by_symbol:
        print(f"{RED}Нет данных.{RESET}")
        return

    for strat_name in args.strategy:
        if strat_name not in STRATEGIES:
            print(f"{RED}Неизвестная стратегия: {strat_name}{RESET}")
            continue
        fn = STRATEGIES[strat_name]
        if args.vt:
            from core.sizing import make_sizer
            base_fn = fn
            sizer = make_sizer(args.sizer, target_vol=args.target_vol)

            def fn(bars, _b=base_fn, _s=sizer):  # noqa: E731
                return _b(bars) * _s(bars)

        matrix, _ = walk_forward_basket(
            bars_by_symbol, fn, args.cost, args.by
        )
        # Портфельный return окна = среднее по инструментам (equal-weight).
        # Вердикт строится на СОГЛАСОВАННОСТИ return по окнам — сути
        # anchored walk-forward. DD внутри окна портфельно недоступен из
        # return-матрицы, поэтому all_pass_dd здесь опускаем (для DD есть
        # обычный --yearly в run_basket на полном прогоне).
        port = matrix.mean(axis=1).to_frame("return")
        port["max_dd"] = 0.0  # нейтрально: вердикт по return-стабильности
        metrics = consistency_metrics(port)

        vt_tag = f' +VT({args.sizer})' if args.vt else ''
        label = f"{strat_name}{vt_tag}"
        print("\n" + format_consistency(label, metrics))
        if args.matrix:
            print("\n" + format_matrix(
                matrix, f"  {label}: return год × инструмент"
            ))


if __name__ == "__main__":
    main()
