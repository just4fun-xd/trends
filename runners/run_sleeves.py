"""Портфельная комбинация sleeve'ов: trend + MR на уровне P&L.

Идея (самый дешёвый источник «альфы» в проекте): у champion
(Donchian 4step, тренд) и bb_rsi (mean-reversion) по построению
противоположные режимы работы — нулевые годы одного = рабочие годы
другого (Gold 2024-25: bb_rsi в кэше, Donchian в тренде). Комбинация
на уровне ДНЕВНЫХ P&L диверсифицирует по ТИПУ СИГНАЛА: ожидаемо DD
ниже любого из sleeve'ов при доходности между ними. Это не новая
математика — это правильная архитектура (портфель узких стратегий,
тезис Александра).

Sleeve = (стратегия из реестра run_basket) x (корзина) [x vt].
P&L sleeve'а = equal-weight среднее побарных доходностей стратегии
по инструментам корзины (после издержек движка).

Комбинация:
    --weights w1,w2,...  — фиксированные веса (default: равные);
    --parity             — inverse-vol по trailing-окну 63 бара со
                           сдвигом на 1 бар (БЕЗ look-ahead: вес дня t
                           считается по волам до t-1).

Печатает: метрики каждого sleeve'а, корреляционную матрицу дневных
P&L (главное число прогона!), метрики комбо, годовую разбивку комбо.

Запуск:
    python -m runners.run_sleeves \\
        --sleeve champion:commodity --sleeve bb_rsi:commodity:vt \\
        --start 2021-01-01 --end 2026-01-01 --parity
    python -m runners.run_sleeves \\
        --sleeve champion:commodity --sleeve bb_rsi:commodity:vt \\
        --sleeve ema_vt:equity --weights 0.4,0.3,0.3
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from core.config import (
    COMMODITY_DATABENTO, COMMODITY_YF, EQUITY_BASKET)
from core.engine import run_engine
from core.sizing import (
    breakeven_funding_rate,
    make_sizer,
    portfolio_vol_target,
)
from diagnostics.port_lev_sweep import (
    format_leverage_sweep,
    leverage_sweep,
)
from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from diagnostics.yearly import format_yearly_table, yearly_breakdown
from runners.run_basket import STRATEGIES

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

BASKETS = {"commodity": COMMODITY_YF, "equity": EQUITY_BASKET}


def sleeve_returns(
    strategy_fn, basket: dict, source, start: str, end: str,
    sizer_name: str | None, cost: float = 0.0002,
    interval: str = "1d",
) -> pd.Series:
    """Дневные доходности sleeve'а: equal-weight по корзине.

    Args:
        strategy_fn: Bars -> position.
        basket: dict {название: тикер}.
        source: DataSource.
        start: Начало периода.
        end: Конец периода.
        sizer_name: None (сырой сигнал), 'realized' или 'garch' —
            сайзер из core.sizing, накладываемый на позицию.
        cost: Издержки движка.

    Returns:
        Ряд побарных доходностей sleeve'а (после издержек).
    """
    sizer = make_sizer(sizer_name) if sizer_name else None
    per_inst = {}
    for name, ticker in basket.items():
        try:
            bars = source.load(ticker, start, end, interval)
        except Exception as exc:  # noqa: BLE001
            print(f"  {YELLOW}пропуск {name} ({ticker}): {exc}{RESET}")
            continue
        pos = strategy_fn(bars)
        if sizer is not None:
            pos = pos * sizer(bars)
        res = run_engine(bars, pos, cost=cost)
        per_inst[name] = res.equity.pct_change()
    if not per_inst:
        raise RuntimeError("sleeve пуст: данные не загрузились")
    df = pd.DataFrame(per_inst)
    # Пропуски отдельных инструментов (разные торговые календари) —
    # средний P&L по доступным в этот день, не по всем.
    return df.mean(axis=1, skipna=True).fillna(0.0)


def metrics(returns: pd.Series, bpy: float = 252.0) -> dict:
    """Итоговые метрики P&L-ряда.

    Args:
        returns: Побарные доходности.
        bpy: Баров в году.

    Returns:
        dict(ret, dd, sharpe).
    """
    eq = (1.0 + returns).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    std = returns.std()
    sharpe = (float(returns.mean() / std * np.sqrt(bpy))
              if std > 0 else 0.0)
    return {"ret": float(eq.iloc[-1] - 1.0), "dd": dd, "sharpe": sharpe}


def parity_weights(
    sleeves: pd.DataFrame, lookback: int = 63
) -> pd.DataFrame:
    """Веса inverse-vol, trailing, без look-ahead.

    Вес sleeve'а обратен его реализованной воле за lookback баров;
    ряд волы сдвинут на 1 бар (вес дня t знает только до t-1). До
    прогрева — равные веса.

    Args:
        sleeves: DataFrame дневных P&L (колонка на sleeve).
        lookback: Окно оценки волы.

    Returns:
        DataFrame весов, сумма по строке = 1.
    """
    vol = sleeves.rolling(lookback).std().shift(1)
    inv = 1.0 / vol.where(vol > 1e-12)
    w = inv.div(inv.sum(axis=1), axis=0)
    equal = 1.0 / sleeves.shape[1]
    return w.fillna(equal)


def main() -> None:
    """CLI sleeve-комбинатора."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sleeve", action="append", required=True,
                   help="strategy:basket[:vt|:garch], например "
                        "champion:commodity, bb_rsi:commodity:vt "
                        "или mr_ens:commodity:garch")
    p.add_argument("--source", default="yf",
                   choices=["yf", "databento"])
    p.add_argument("--panel-dir", default=None,
                   help="каталог parquet-панелей для databento "
                        "(commodity). equity-панели ищутся в "
                        "data/panels/equities.")
    p.add_argument("--interval", default="1d")
    p.add_argument("--exclude", default=None,
                   help="инструменты через запятую, исключить из "
                        "корзины (напр. тонкие H4-рынки: PA,PL)")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--weights", default=None,
                   help="фиксированные веса через запятую")
    p.add_argument("--parity", action="store_true",
                   help="trailing inverse-vol веса (без look-ahead)")
    p.add_argument("--port-vol", type=float, default=None,
                   help="портфельный vol-таргетинг комбо: довести "
                        "волу КОМБО до цели плечом (напр. 0.20). "
                        "Конвертирует Sharpe в доходность; без него "
                        "комбо работает на ~5%% риск-бюджета.")
    p.add_argument("--max-port-lev", type=float, default=4.0,
                   help="потолок портфельного плеча для --port-vol")
    p.add_argument("--lev-sweep", action="store_true",
                   help="перебрать сетку target_vol x кэп плеча на "
                        "реальной кривой комбо и найти максимум в "
                        "рамках DD<40%% (вместо одной точки --port-vol)")
    p.add_argument("--funding-rate", type=float, default=0.0,
                   help="годовая ставка фондирования ЗАЁМНОЙ части "
                        "плеча (напр. 0.05 = 5%%/год), применяется в "
                        "--port-vol и --lev-sweep. Типично 0.04-0.08. "
                        "0.0 (default) = верхняя граница без costs.")
    p.add_argument("--cost", type=float, default=0.0002)
    args = p.parse_args()

    def make_source(basket_name: str):
        """Источник + корзина под нотацию тикеров данного бэкенда."""
        if args.source == "yf":
            src = YFinanceSource()
            basket = dict(BASKETS[basket_name])
        else:
            # databento: корневые символы, панели по корзине.
            if basket_name == "equity":
                panel_dir = "data/panels/equities"
                basket = dict(EQUITY_BASKET)
            else:
                panel_dir = args.panel_dir or "data/panels/futures"
                basket = {s: s for s in COMMODITY_DATABENTO}
            src = DatabentoSource(panel_dir=panel_dir)
        if args.exclude:
            drop = {s.strip() for s in args.exclude.split(",")}
            removed = [k for k in basket if k in drop]
            basket = {k: v for k, v in basket.items() if k not in drop}
            if removed:
                print(f"  Исключены из корзины: {', '.join(removed)}")
        return src, basket

    cols = {}
    for spec in args.sleeve:
        parts = spec.split(":")
        strat, basket_name = parts[0], parts[1]
        # Третье поле: vt (realized) или garch — сайзер sleeve'а.
        sizer_name = None
        if len(parts) > 2:
            sizer_name = {"vt": "realized", "garch": "garch"}.get(
                parts[2])
            if sizer_name is None:
                raise SystemExit(
                    f"неизвестный сайзер {parts[2]!r} в {spec!r} "
                    f"(допустимо: vt, garch)")
        fn = STRATEGIES.get(strat)
        if fn is None:
            raise SystemExit(f"нет стратегии '{strat}' в реестре")
        if basket_name not in BASKETS:
            raise SystemExit(f"нет корзины '{basket_name}'")
        src, basket = make_source(basket_name)
        label = spec.replace(":", "_")
        print(f"Sleeve {label} ({args.source}, {args.interval}) ...")
        cols[label] = sleeve_returns(
            fn, basket, src, args.start, args.end, sizer_name,
            args.cost, interval=args.interval,
        )
    sleeves = pd.DataFrame(cols).fillna(0.0)

    print(f"\n{BOLD}=== Sleeve'ы по отдельности ==={RESET}")
    for label in sleeves.columns:
        m = metrics(sleeves[label])
        print(f"  {label:28s} ret {m['ret']:+7.1%}  "
              f"DD {m['dd']:6.1%}  Sharpe {m['sharpe']:+.2f}")

    print(f"\n{BOLD}Корреляция дневных P&L "
          f"(главное число прогона):{RESET}")
    corr = sleeves.corr()
    print(corr.round(2).to_string())

    if args.weights:
        w_fix = [float(x) for x in args.weights.split(",")]
        if len(w_fix) != sleeves.shape[1]:
            raise SystemExit("число весов != числу sleeve'ов")
        w = pd.DataFrame(
            [w_fix], index=sleeves.index[:1], columns=sleeves.columns
        ).reindex(sleeves.index, method="ffill")
        mode = f"фиксированные {w_fix}"
    elif args.parity:
        w = parity_weights(sleeves)
        mode = "vol-parity (trailing 63, shift 1)"
    else:
        w = pd.DataFrame(1.0 / sleeves.shape[1],
                         index=sleeves.index, columns=sleeves.columns)
        mode = "равные"

    combo = (sleeves * w).sum(axis=1)
    m = metrics(combo)
    print(f"\n{BOLD}=== КОМБО ({mode}) ==={RESET}")
    print(f"  Доходность: {m['ret']:+.1%}")
    print(f"  Max DD:     {m['dd']:.1%}")
    print(f"  Sharpe:     {m['sharpe']:+.2f}")
    realized_vol = float(combo.std() * (252.0 ** 0.5))
    print(f"  Годовая вола комбо: {realized_vol:.1%} "
          f"(доля лимита DD<40%: ~{realized_vol / 0.40:.0%})")
    dd_ok = abs(m["dd"]) < 0.40
    mark = f"{GREEN}да{RESET}" if dd_ok else f"{RED}НЕТ{RESET}"
    print(f"  Проходит DD<40%: {mark}")

    eq = (1.0 + combo).cumprod()
    yb = yearly_breakdown(eq, 252.0)
    print("\n" + format_yearly_table(yb, "Комбо по годам"))

    if args.port_vol:
        scaled, lev = portfolio_vol_target(
            combo, target_vol=args.port_vol,
            max_leverage=args.max_port_lev,
            funding_rate=args.funding_rate,
        )
        ms = metrics(scaled)
        print(f"\n{BOLD}=== КОМБО × портфельный VT@"
              f"{args.port_vol:.0%} (плечо кэп "
              f"{args.max_port_lev:.1f}) ==={RESET}")
        print(f"  Доходность: {ms['ret']:+.1%}")
        print(f"  Max DD:     {ms['dd']:.1%}")
        print(f"  Sharpe:     {ms['sharpe']:+.2f}")
        active = lev[lev > 0]
        print(f"  Среднее плечо: "
              f"{float(active.mean()) if len(active) else 0.0:.2f}")
        dd_ok = abs(ms["dd"]) < 0.40
        mark = f"{GREEN}да{RESET}" if dd_ok else f"{RED}НЕТ{RESET}"
        print(f"  Проходит DD<40%: {mark}")
        if args.funding_rate > 0:
            print(f"  {GREEN}фондирование учтено: "
                  f"{args.funding_rate:.1%}/год на заёмную часть "
                  f"плеча{RESET}")
        else:
            print(f"  {YELLOW}издержки ног учтены их движками; "
                  f"фондирование плеча НЕ учтено "
                  f"(--funding-rate 0){RESET}")
        eq_s = (1.0 + scaled).cumprod()
        yb_s = yearly_breakdown(eq_s, 252.0)
        print("\n" + format_yearly_table(
            yb_s, f"Комбо VT@{args.port_vol:.0%} по годам"))

    if args.lev_sweep:
        print(f"\n{BOLD}=== Sweep потолка плеча: максимум в рамках "
              f"DD<40% ==={RESET}")
        grid = leverage_sweep(combo, funding_rate=args.funding_rate)
        print(format_leverage_sweep(
            grid, funding_rate=args.funding_rate))

        is_futures = args.source == "databento"
        be = breakeven_funding_rate(
            combo, target_vol=0.20, max_leverage=args.max_port_lev,
        )
        print(f"\n{BOLD}Breakeven-ставка фондирования{RESET} "
              f"(target_vol=20%, кэп={args.max_port_lev:.1f}): "
              f"{be:.1%}/год")
        print(f"  Выше этой ставки плечо {args.max_port_lev:.0f}x "
              f"уже НЕ окупается на этих данных.")
        if is_futures:
            print(f"  {YELLOW}⚠ Это sleeve на ФЬЮЧЕРСАХ "
                  f"(--source databento). Модель "
                  f"funding_rate=займ-под-ставку НЕ описывает "
                  f"механику фьючерсов (плечо там — маржа/гарантийное "
                  f"обеспечение, не займ наличных; cost-of-carry уже "
                  f"в цене контракта и учтён в P&L стратегии). Эта "
                  f"breakeven-ставка отвечает на гипотетический "
                  f"вопрос «как если бы это были акции на марже» — "
                  f"не применяй её буквально к сырьевой ноге.{RESET}")
        else:
            print("  Модель margin-loan уместна для акций; "
                  "институциональные ставки финансирования обычно "
                  "ниже розничного брокерского маржин-рейта.")


if __name__ == "__main__":
    main()
