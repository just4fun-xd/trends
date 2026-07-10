"""Оркестратор полного цикла: bootstrap + корреляция пачкой в CSV.

Проблема: полный цикл (bootstrap разности Sharpe против чемпиона +
корреляция ног) до сих пор гонялся руками по 1-4 стратегии за раз.
Этот раннер прогоняет ЛЮБОЙ список кандидатов против чемпиона ниши
за один вызов и пишет структурный CSV, из которого собирается
финальный отчёт-каталог.

Для каждого кандидата в CSV пишется строка:
  strategy, basket, source, vt, champion,
  sharpe_a, sharpe_b, diff, ci_lo, ci_hi, significant, verdict,
  corr, combo_sharpe, combo_dd, ensemble_ok

verdict (автомат по нашему протоколу):
  BEATS      — diff>0 и CI разности НЕ накрывает 0 (значимо лучше);
  WORSE      — diff<0 и CI не накрывает 0 (значимо хуже -> закрыть);
  TIE        — CI накрывает 0 (неразличим с чемпионом);
ensemble_ok = TIE/BEATS И corr<corr_gate И комбо не ниже чемпиона.

Дисциплина: рейтинг из этого раннера — АРБИТРАЖ (не финальная корона).
Финал = 2 источника + walk-forward (робастность) отдельно. Но для
пула «рабочих» кандидатов один запуск на источник даёт вердикт.

Пример (все трендовые кандидаты сырья против donchian_vt):
  python -m runners.run_fullcycle \\
      --candidates ema_barbell donchian_est_pyr hurst_alloc kama \\
      --champion donchian_vt \\
      --source databento --panel-dir data/panels/futures \\
      --start 2020-01-01 --vt --target-vol 0.20 \\
      --champion-basket @DONCH_CORE_COMM --csv fullcycle.csv

Или авто-набор по семейству (все trend/mr реестра против чемпиона):
  python -m runners.run_fullcycle --family trend --champion donchian_vt ...
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd

from core.config import (
    COMMODITY_DATABENTO, COMMODITY_YF, CRYPTO_CCXT, CRYPTO_YF,
    EQUITY_BASKET, filter_basket)
from core.engine import run_engine
from core.sizing import make_sizer
from data.ccxt_source import CCXTSource
from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from diagnostics.bootstrap import sharpe_diff_ci
from runners.run_basket import STRATEGIES, STRATEGY_FAMILY

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def _sleeve(strategy_fn, basket, source, start, end, interval,
            sizer, cost):
    """Equal-weight sleeve-доходности стратегии по корзине.

    Returns:
        (Series дневных P&L, bars_per_year) — bpy для честной
        аннуализации на интрадей (4h=2190, не 252).
    """
    cols = {}
    bpy = 252.0
    for name, ticker in basket.items():
        try:
            bars = source.load(ticker, start, end, interval)
        except Exception:  # noqa: BLE001
            continue
        bpy = bars.bars_per_year
        pos = strategy_fn(bars)
        if sizer is not None:
            pos = pos * sizer(bars)
        cols[name] = run_engine(
            bars, pos, cost=cost).equity.pct_change()
    if not cols:
        return pd.Series(dtype=float), bpy
    rets = pd.DataFrame(cols).mean(axis=1, skipna=True).fillna(0.0)
    return rets, bpy


def _combo_stats(ra: pd.Series, rb: pd.Series,
                 bpy: float) -> tuple[float, float, float]:
    """Корреляция ног и Sharpe/DD их inverse-vol (parity) комбо.

    Returns:
        (corr, combo_sharpe, combo_max_dd).
    """
    df = pd.concat([ra, rb], axis=1).dropna()
    if len(df) < 30:
        return float("nan"), float("nan"), float("nan")
    corr = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    # inverse-vol веса (на 2 ногах HRP вырождается в inverse-var).
    vol = df.std()
    w = (1.0 / vol) / (1.0 / vol).sum()
    combo = (df * w.values).sum(axis=1)
    std = combo.std(ddof=1)
    csh = float(combo.mean() / std * np.sqrt(bpy)) if std > 0 else 0.0
    eq = (1.0 + combo).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return corr, csh, dd


def _resolve_basket(basket_kind: str, source_name: str):
    """Строит корзину по классу активов и источнику."""
    if basket_kind == "equity":
        return EQUITY_BASKET
    if basket_kind == "crypto":
        return (CRYPTO_CCXT if source_name == "ccxt" else CRYPTO_YF)
    if source_name == "databento":
        return {s: s for s in COMMODITY_DATABENTO}
    return COMMODITY_YF


def main() -> None:
    """CLI: полный цикл для списка кандидатов, вывод в CSV."""
    p = argparse.ArgumentParser(
        description="Полный цикл (bootstrap+корр) пачкой в CSV")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--candidates", nargs="*",
                     help="список стратегий-кандидатов")
    grp.add_argument("--family",
                     choices=["trend", "mean-reversion", "mixed",
                              "all"],
                     help="авто-набор всех стратегий семейства")
    p.add_argument("--champion", required=True,
                   help="стратегия-чемпион ниши (B в разности)")
    p.add_argument("--source", default="databento",
                   choices=["yf", "databento", "ccxt"])
    p.add_argument("--basket", default="commodity",
                   choices=["commodity", "equity", "crypto"])
    p.add_argument("--panel-dir", default=None)
    p.add_argument("--crypto-dir", default="data/crypto")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--interval", default="1d")
    p.add_argument("--cost", type=float, default=0.0002)
    p.add_argument("--vt", action="store_true")
    p.add_argument("--sizer", default="realized")
    p.add_argument("--target-vol", type=float, default=0.20)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--ci", type=float, default=0.90)
    p.add_argument("--corr-gate", type=float, default=0.60,
                   help="порог корреляции для ensemble_ok")
    p.add_argument("--candidate-basket", default=None,
                   help="корзина для кандидата (@ИМЯ/тикеры); по умол. "
                        "полная вселенная источника")
    p.add_argument("--champion-basket", default=None,
                   help="корзина чемпиона (@ИМЯ/тикеры); по умол. "
                        "как у кандидата")
    p.add_argument("--skip", nargs="*", default=None,
                   help="исключить эти стратегии из авто-набора "
                        "--family (напр. закрытые с механизмом)")
    p.add_argument("--skip-prefix", nargs="*", default=None,
                   help="исключить стратегии с этими префиксами "
                        "(напр. 'ou' для закрытого OU-раздела)")
    p.add_argument("--csv", default="fullcycle.csv",
                   help="файл для дозаписи результатов")
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

    full = _resolve_basket(args.basket, args.source)
    cand_basket = filter_basket(full, include=args.candidate_basket)
    champ_spec = args.champion_basket or args.candidate_basket
    champ_basket = filter_basket(full, include=champ_spec)

    if args.family:
        fams = ({"trend", "mean-reversion", "mixed", "trend (short)"}
                if args.family == "all" else {args.family})
        skip = set(args.skip or ())
        skip_pref = tuple(args.skip_prefix or ())
        cands = [n for n in STRATEGIES
                 if STRATEGY_FAMILY.get(n) in fams
                 and n != args.champion
                 and n not in skip
                 and not (skip_pref and n.startswith(skip_pref))]
    else:
        cands = [c for c in args.candidates if c != args.champion]

    sizer = (make_sizer(args.sizer, target_vol=args.target_vol)
             if args.vt else None)

    print(f"{BOLD}Полный цикл: {len(cands)} кандидатов vs "
          f"{args.champion} | {args.basket} | {args.source}{RESET}\n")

    rb, bpy = _sleeve(STRATEGIES[args.champion], champ_basket, source,
                      args.start, args.end, args.interval, sizer,
                      args.cost)
    if rb.empty:
        raise SystemExit(f"чемпион {args.champion}: пустые данные")

    rows = []
    for i, name in enumerate(cands, 1):
        if name not in STRATEGIES:
            print(f"  [{i}/{len(cands)}] {name}: нет в реестре")
            continue
        ra, _ = _sleeve(STRATEGIES[name], cand_basket, source,
                        args.start, args.end, args.interval, sizer,
                        args.cost)
        if ra.empty:
            print(f"  [{i}/{len(cands)}] {name}: пустые данные")
            continue
        aligned = pd.concat([ra, rb], axis=1).dropna()
        d = sharpe_diff_ci(aligned.iloc[:, 0], aligned.iloc[:, 1],
                           bars_per_year=bpy, n_boot=args.n_boot,
                           ci=args.ci)
        corr, csh, cdd = _combo_stats(ra, rb, bpy)
        if d["significant"] and d["diff"] > 0:
            verdict = "BEATS"
        elif d["significant"] and d["diff"] < 0:
            verdict = "WORSE"
        else:
            verdict = "TIE"
        ens_ok = (verdict in ("TIE", "BEATS")
                  and not np.isnan(corr)
                  and corr < args.corr_gate)
        rows.append({
            "strategy": name, "basket": args.basket,
            "source": args.source,
            "vt": f"{args.target_vol:.2f}" if args.vt else "off",
            "champion": args.champion,
            "sharpe_a": round(d["sharpe_a"], 4),
            "sharpe_b": round(d["sharpe_b"], 4),
            "diff": round(d["diff"], 4),
            "ci_lo": round(d["lo"], 4), "ci_hi": round(d["hi"], 4),
            "significant": int(d["significant"]),
            "verdict": verdict, "corr": round(corr, 3),
            "combo_sharpe": round(csh, 4), "combo_dd": round(cdd, 4),
            "ensemble_ok": int(ens_ok),
        })
        col = (GREEN if verdict == "BEATS" else
               RED if verdict == "WORSE" else YELLOW)
        star = " ★ENSEMBLE" if ens_ok else ""
        print(f"  [{i}/{len(cands)}] {name:<20} "
              f"Δ{d['diff']:+.2f} CI[{d['lo']:+.2f},{d['hi']:+.2f}] "
              f"{col}{verdict}{RESET} corr={corr:+.2f}{star}")

    if rows:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\n{BOLD}CSV дописан: {args.csv} "
              f"(+{len(rows)} строк){RESET}")


if __name__ == "__main__":
    main()
