"""Прогонщик ОДНОЙ стратегии: выбор корзины, активов, периода, VT.

Назначение: быстро прогнать любую одну стратегию из реестра на любом
классе активов (или на всех сразу), на любом наборе инструментов
(или на всех), за любой период, с VT или без — и получить полный
набор метрик за back-test И за out-of-sample forward-test.

Что считается (на equal-weight портфеле выбранной корзины):
  ROI (компаунд), CAGR (годовая), Max DD, Sharpe, Sortino, Calmar,
  годовая вола, win-rate по барам, доля времени в рынке, проход
  DD<40%. Всё gross (rf=0 по умолчанию, --rf для excess).

Back/Forward: период делится в точке --split (доля, по умолч. 0.70).
Первые 70% — in-sample (back), последние 30% — out-of-sample
(forward). Здоровая стратегия: forward-Sharpe не рушится против
back (деградация <~40% — грубый ориентир, не гарантия).

Примеры:
  # tsmom на сырье, конкретные активы, VT 20%, split 70/30
  python -m runners.run_strategy_test --strategy tsmom \\
      --basket commodity --source databento \\
      --panel-dir data/panels/futures \\
      --assets GC,CL,SI,HG --start 2020-01-01 --vt --target-vol 0.20

  # на ВСЕХ трёх классах разом (крипта на H1/H4/D1), вся вселенная
  python -m runners.run_strategy_test --strategy tsmom --basket all \\
      --panel-dir data/panels/futures --crypto-dir data/crypto \\
      --start 2020-01-01 --vt --target-vol 0.20

  # только крипта, все таймфреймы
  python -m runners.run_strategy_test --strategy donchian_vt \\
      --basket crypto --source ccxt --crypto-dir data/crypto \\
      --interval h1,h4,1d --start 2021-01-01 --vt --target-vol 0.40
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from core.config import (
    COMMODITY_DATABENTO, COMMODITY_YF, CRYPTO_CCXT, CRYPTO_YF,
    EQUITY_BASKET, filter_basket)
from core.sizing import make_sizer
from data.ccxt_source import CCXTSource
from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from diagnostics.instrument_contribution import per_instrument_returns
from runners.run_basket import STRATEGIES, STRATEGY_FAMILY

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def _metrics(rets: pd.Series, bpy: float, rf: float = 0.0) -> dict:
    """Полный набор метрик по ряду побарных доходностей портфеля.

    Args:
        rets: Побарный P&L equal-weight портфеля.
        bpy: Баров в году (честный, из Bars).
        rf: Безрисковая ставка (годовая); 0 = gross.

    Returns:
        dict метрик (roi, cagr, max_dd, sharpe, sortino, calmar,
        vol, win_rate, exposure, n_bars).
    """
    r = rets.fillna(0.0)
    n = len(r)
    if n < 2 or r.std() == 0:
        return {k: 0.0 for k in (
            "roi", "cagr", "max_dd", "sharpe", "sortino", "calmar",
            "vol", "win_rate", "n_bars")}
    eq = (1.0 + r).cumprod()
    roi = float(eq.iloc[-1] - 1.0)
    years = n / bpy
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    dd = float((eq / eq.cummax() - 1.0).min())
    excess = r.mean() - rf / bpy
    vol = float(r.std() * np.sqrt(bpy))
    sharpe = float(excess / r.std() * np.sqrt(bpy))
    downside = r[r < 0].std()
    sortino = (float(excess / downside * np.sqrt(bpy))
               if downside and downside > 0 else float("inf"))
    calmar = float(cagr / abs(dd)) if dd != 0 else float("inf")
    win = float((r > 0).sum() / (r != 0).sum()) if (r != 0).any() else 0.0
    return {
        "roi": roi, "cagr": cagr, "max_dd": dd, "sharpe": sharpe,
        "sortino": sortino, "calmar": calmar, "vol": vol,
        "win_rate": win, "n_bars": n,
    }


def _print_block(label: str, m: dict, dd_limit: float = 0.40) -> None:
    """Печатает блок метрик с цветовой подсветкой ключевых."""
    sh = m["sharpe"]
    shc = GREEN if sh > 1.0 else (YELLOW if sh > 0 else RED)
    ddc = RED if abs(m["max_dd"]) > dd_limit else GREEN
    passed = "да" if abs(m["max_dd"]) <= dd_limit else "НЕТ"
    sortino = ("inf" if m["sortino"] == float("inf")
               else f"{m['sortino']:+.2f}")
    calmar = ("inf" if m["calmar"] == float("inf")
              else f"{m['calmar']:.2f}")
    print(f"{BOLD}{label}{RESET}  ({int(m['n_bars'])} баров)")
    print(f"  ROI (компаунд):  {m['roi']:+.1%}")
    print(f"  CAGR (годовая):  {m['cagr']:+.1%}")
    print(f"  Max DD:          {ddc}{m['max_dd']:+.1%}{RESET}  "
          f"(лимит<{dd_limit:.0%}: {passed})")
    print(f"  Sharpe:          {shc}{sh:+.2f}{RESET}")
    print(f"  Sortino:         {sortino}")
    print(f"  Calmar:          {calmar}")
    print(f"  Годовая вола:    {m['vol']:.1%}")
    print(f"  Win-rate (бары): {m['win_rate']:.1%}")


def _resolve(basket_kind: str, source_name: str) -> dict:
    """Корзина {имя: тикер} по классу активов и источнику."""
    if basket_kind == "equity":
        return EQUITY_BASKET
    if basket_kind == "crypto":
        return CRYPTO_CCXT if source_name == "ccxt" else CRYPTO_YF
    if source_name == "databento":
        return {s: s for s in COMMODITY_DATABENTO}
    return COMMODITY_YF


def _make_source(basket_kind, source_name, panel_dir, crypto_dir):
    """Создаёт нужный DataSource под класс/источник."""
    if source_name == "yf":
        return YFinanceSource()
    if source_name == "ccxt":
        return CCXTSource(data_dir=crypto_dir)
    pd_ = panel_dir or ("data/panels/equities"
                        if basket_kind == "equity"
                        else "data/panels/futures")
    return DatabentoSource(panel_dir=pd_)


def _run_one(strat_fn, basket, source, start, end, interval,
             sizer, cost, split, rf, header) -> None:
    """Прогон одной (корзина × таймфрейм): back + forward + full."""
    print(f"\n{BOLD}{'=' * 66}{RESET}")
    print(f"{BOLD}{header}{RESET}")
    print(f"{BOLD}{'=' * 66}{RESET}")
    try:
        rets_df, bpy = per_instrument_returns(
            strat_fn, basket, source, start, end, sizer=sizer,
            cost=cost, interval=interval)
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}пропуск: {exc}{RESET}")
        return
    port = rets_df.mean(axis=1, skipna=True).fillna(0.0)
    n = len(port)
    cut = int(n * split)
    back, fwd = port.iloc[:cut], port.iloc[cut:]
    print(f"инструментов: {rets_df.shape[1]} | баров: {n} | "
          f"bpy: {bpy:.0f} | split: {split:.0%} "
          f"(back {cut} / forward {n - cut})\n")

    mb = _metrics(back, bpy, rf)
    mf = _metrics(fwd, bpy, rf)
    mall = _metrics(port, bpy, rf)
    _print_block("BACK-TEST (in-sample)", mb)
    print()
    _print_block("FORWARD-TEST (out-of-sample)", mf)
    print()
    _print_block("ВЕСЬ ПЕРИОД", mall)

    # Вердикт устойчивости: forward не должен рушиться против back.
    if mb["sharpe"] > 0.1:
        ratio = mf["sharpe"] / mb["sharpe"]
        if ratio > 0.6:
            v, col = "УСТОЙЧИВА (forward держит back)", GREEN
        elif ratio > 0.3:
            v, col = "ЧАСТИЧНАЯ деградация forward", YELLOW
        else:
            v, col = "ПЕРЕОБУЧЕНИЕ (forward рушится)", RED
        print(f"\n{col}Forward/Back Sharpe = {ratio:.2f} -> {v}{RESET}")
    else:
        print(f"\n{YELLOW}Back-Sharpe≈0: сравнение forward "
              f"неинформативно{RESET}")


def main() -> None:
    """CLI: прогон одной стратегии с выбором всех параметров."""
    p = argparse.ArgumentParser(
        description="Прогонщик одной стратегии (back+forward, метрики)")
    p.add_argument("--strategy", required=True,
                   help="имя стратегии из реестра (напр. tsmom)")
    p.add_argument("--basket", default="commodity",
                   choices=["commodity", "equity", "crypto", "all"],
                   help="класс активов или 'all' (все три)")
    p.add_argument("--source", default=None,
                   choices=["yf", "databento", "ccxt"],
                   help="источник; по умолчанию авто по классу")
    p.add_argument("--panel-dir", default=None)
    p.add_argument("--crypto-dir", default="data/crypto")
    p.add_argument("--assets", default=None,
                   help="активы через запятую или @КОРЗИНА "
                        "(напр. GC,CL,SI,HG); по умолч. вся вселенная")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--interval", default=None,
                   help="таймфрейм(ы) через запятую. Крипта по умолч. "
                        "h1,h4,1d; сырьё/акции 1d")
    p.add_argument("--vt", action="store_true",
                   help="обернуть vol-таргетингом")
    p.add_argument("--target-vol", type=float, default=0.20)
    p.add_argument("--sizer", default="realized",
                   choices=["realized", "garch"])
    p.add_argument("--cost", type=float, default=0.0002)
    p.add_argument("--split", type=float, default=0.70,
                   help="доля in-sample (back); остаток forward")
    p.add_argument("--rf", type=float, default=0.0,
                   help="безрисковая ставка (годовая) для excess-метрик")
    args = p.parse_args()

    if args.strategy not in STRATEGIES:
        raise SystemExit(f"нет стратегии '{args.strategy}' в реестре. "
                         f"Всего в реестре: {len(STRATEGIES)}")
    strat_fn = STRATEGIES[args.strategy]
    fam = STRATEGY_FAMILY.get(args.strategy, "?")

    if not 0.1 <= args.split <= 0.9:
        raise SystemExit("--split должен быть в диапазоне 0.1..0.9")

    sizer = (make_sizer(args.sizer, target_vol=args.target_vol)
             if args.vt else None)
    vt_note = (f"VT:{args.sizer}@{args.target_vol:.0%}"
               if args.vt else "без VT")

    # какие классы активов гоняем
    if args.basket == "all":
        plan = [("commodity", args.source or "databento"),
                ("equity", "yf"),
                ("crypto", "ccxt")]
    else:
        default_src = {"commodity": "databento", "equity": "yf",
                       "crypto": "ccxt"}[args.basket]
        plan = [(args.basket, args.source or default_src)]

    print(f"{BOLD}Прогонщик стратегии: {args.strategy} "
          f"[{fam}] | {vt_note} | {args.start}..{args.end}{RESET}")

    for basket_kind, source_name in plan:
        source = _make_source(basket_kind, source_name,
                              args.panel_dir, args.crypto_dir)
        full = _resolve(basket_kind, source_name)
        basket = filter_basket(full, include=args.assets)

        # таймфреймы: крипта по умолчанию три, остальное 1d
        if args.interval:
            intervals = [s.strip() for s in args.interval.split(",")
                         if s.strip()]
        elif basket_kind == "crypto":
            intervals = ["h1", "h4", "1d"]
        else:
            intervals = ["1d"]

        for interval in intervals:
            header = (f"{args.strategy} | {basket_kind} | "
                      f"{source_name} | {interval} | "
                      f"{len(basket)} инстр")
            _run_one(strat_fn, basket, source, args.start, args.end,
                     interval, sizer, args.cost, args.split, args.rf,
                     header)


if __name__ == "__main__":
    main()
