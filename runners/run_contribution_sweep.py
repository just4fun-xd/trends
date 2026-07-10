"""Оркестратор вклада инструментов по РАБОЧИМ парам (стратегия × актив).

Запрос 2026-07k: пройтись по всем работающим стратегиям и по каждому
активу увидеть прибыль и просадку — но каждую стратегию гонять ТОЛЬКО
на том активе-классе, где она работает (не крутить крипто-защиту на
сырье и MR-реверсию на крипте, где ниша закрыта).

Это НЕ новый диагностический раннер — механика вклада уже в
diagnostics/instrument_contribution.py и в
runners/run_instrument_contribution.py. Здесь оркестратор: держит
карту WORKING_PAIRS (стратегия -> список (класс, источник, интервал,
vt) сред, где она прошла как минимум TIE-барьер против чемпиона на
двух источниках либо является действующим чемпионом ниши), и для
каждой пары вызывает per_instrument_returns + instrument_contribution,
сводя всё в ЕДИНЫЙ CSV с колонкой asset_class.

Карта собрана по данным сессии 2026-07k (fullcycle.csv, fc_*.csv):
никаких «мертвых» пар (значимый WORSE на обоих источниках) — только
то, что реально торгуемо, пусть даже средне.

Запуск (полный обход):
    python -m runners.run_contribution_sweep --vt --csv contrib_sweep.csv

Отдельный класс:
    python -m runners.run_contribution_sweep --only equity --vt \\
        --csv contrib_equity.csv

Печатает по каждой паре ту же таблицу, что и одиночный раннер, плюс
финальную сводку «актив × стратегия -> Sharpe / MaxDD / LOO-дельта».
"""

from __future__ import annotations

import argparse
import csv as csvmod

from core.config import (
    COMMODITY_DATABENTO, COMMODITY_YF, CRYPTO_CCXT, CRYPTO_YF,
    EQUITY_BASKET, filter_basket)
from core.sizing import make_sizer
from data.ccxt_source import CCXTSource
from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from diagnostics.instrument_contribution import (
    _sharpe, format_contribution, instrument_contribution,
    per_instrument_returns)
from runners.run_basket import STRATEGIES

BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"


# ── Карта рабочих сред (класс, источник, интервал, exclude) ──────────
# Каждый кортеж — среда, где стратегию ИМЕЕТ смысл смотреть по
# инструментам. Источник в паре — тот, где стратегия НЕ провалила
# bootstrap значимо (см. fullcycle.csv / fc_*.csv сессии 2026-07k).
# Интервал: сырьё/акции — 1d, крипта — 4h (ccxt) и 1d (yf-контроль).
# exclude: PA,PL вырезаются на H4-сырье (низкое native-покрытие) —
# здесь не встречаются, но параметр проброшен для единообразия.

# Действующие чемпионы ниш — обязательны как якоря сравнения.
_COMM = ("commodity", "databento", "1d", "")
_COMM_YF = ("commodity", "yf", "1d", "")
_EQ = ("equity", "yf", "1d", "")
_CR4 = ("crypto", "ccxt", "4h", "")
_CR1 = ("crypto", "yf", "1d", "")

WORKING_PAIRS: dict[str, list[tuple[str, str, str, str]]] = {
    # ── Сырьё: тренд-нога и MR-нога чемпионов ────────────────────────
    "donchian_vt": [_COMM, _COMM_YF, _CR4, _CR1],   # чемпион тренда везде
    "mr_lowvol": [_COMM, _COMM_YF],                 # чемпион MR сырья
    "mr_kelt_confirm": [_COMM, _COMM_YF],           # MR-нога боевого комбо
    # трендовые, прошедшие TIE на обоих источниках сырья:
    "ema_barbell": [_COMM, _COMM_YF],
    "donchian_est_pyr": [_COMM, _COMM_YF],
    "kama": [_COMM, _COMM_YF],
    "tr3_hh_hl": [_COMM, _COMM_YF],                 # 3-й leg сырья (память)
    "tr4_mk": [_COMM],                              # TIE db, WORSE-грань yf
    # специалисты сырья с ensemble_ok:
    "hurst_alloc": [_COMM, _COMM_YF],
    "hurst_combo": [_COMM, _COMM_YF],

    # ── Акции: чемпионы equity (память проекта) + TIE-кандидаты ───────
    # НЕ прогонялись в fullcycle сессии 2026-07k — это и есть п.2.
    "ema_vt": [_EQ],                                # чемпион тренда акций
    "mr_ens": [_EQ],                                # чемпион MR акций
    "ema_ensemble": [_EQ],
    "mr_ens_gate": [_EQ],

    # ── Крипта: чемпион + подтверждённый 3-й leg + защита-ансамблист ──
    "tr_ichimoku": [_CR4, _CR1],                    # 2-й leg крипто-тренда
    "ca_vol_ride": [_CR4],                          # 3-й leg, ensemble_ok
    "ca2_cppi": [_CR4, _CR1],                       # защита, corr<0.6, ens_ok
}


def _make_source(source: str, panel_dir: str, crypto_dir: str):
    if source == "yf":
        return YFinanceSource()
    if source == "ccxt":
        return CCXTSource(data_dir=crypto_dir)
    return DatabentoSource(panel_dir=panel_dir)


def _basket_for(asset_class: str, source: str) -> dict[str, str]:
    if asset_class == "equity":
        return EQUITY_BASKET
    if asset_class == "crypto":
        return CRYPTO_CCXT if source == "ccxt" else CRYPTO_YF
    if source == "databento":
        return {s: s for s in COMMODITY_DATABENTO}
    return COMMODITY_YF


def main() -> None:
    """CLI: обход рабочих пар (стратегия × актив-класс)."""
    p = argparse.ArgumentParser(
        description="Свод вклада инструментов по рабочим парам")
    p.add_argument("--only", default=None,
                   choices=["commodity", "equity", "crypto"],
                   help="ограничить одним классом активов")
    p.add_argument("--strategies", nargs="+", default=None,
                   help="ограничить подмножеством стратегий из карты")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--cost", type=float, default=0.0002)
    p.add_argument("--vt", action="store_true",
                   help="vol-targeting (сырьё/акции 0.20, крипта 0.40)")
    p.add_argument("--sizer", default="realized")
    p.add_argument("--target-vol-comm", type=float, default=0.20)
    p.add_argument("--target-vol-crypto", type=float, default=0.40)
    p.add_argument("--panel-dir-futures", default="data/panels/futures")
    p.add_argument("--panel-dir-equities",
                   default="data/panels/equities")
    p.add_argument("--crypto-dir", default="data/crypto")
    p.add_argument("--csv", default=None, help="сводный CSV всех пар")
    args = p.parse_args()

    pairs = WORKING_PAIRS
    if args.strategies:
        pairs = {k: v for k, v in pairs.items() if k in args.strategies}

    rows: list[dict] = []
    for strat, envs in pairs.items():
        if strat not in STRATEGIES:
            print(f"{YELLOW}пропуск {strat}: нет в реестре{RESET}")
            continue
        for asset_class, source, interval, exclude in envs:
            if args.only and asset_class != args.only:
                continue
            panel_dir = (args.panel_dir_equities
                         if asset_class == "equity"
                         else args.panel_dir_futures)
            src = _make_source(source, panel_dir, args.crypto_dir)
            basket = filter_basket(
                _basket_for(asset_class, source), exclude=exclude or None)
            tvol = (args.target_vol_crypto if asset_class == "crypto"
                    else args.target_vol_comm)
            sizer = make_sizer(args.sizer, target_vol=tvol) \
                if args.vt else None

            vt_note = f" | vt@{tvol:.0%}" if args.vt else ""
            print(f"\n{BOLD}{strat} | {asset_class} | {source}"
                  f"{'/' + interval if source == 'ccxt' else ''}"
                  f"{vt_note}{RESET}")
            try:
                rets, bpy = per_instrument_returns(
                    STRATEGIES[strat], basket, src, args.start, args.end,
                    interval=interval, cost=args.cost, sizer=sizer)
            except Exception as exc:                 # noqa: BLE001
                print(f"  {YELLOW}пропуск: {exc}{RESET}")
                continue
            if rets is None or rets.empty:
                print(f"  {YELLOW}пропуск: нет данных корзины{RESET}")
                continue
            full = rets.mean(axis=1, skipna=True).fillna(0.0)
            full_sharpe = _sharpe(full, bpy)
            df = instrument_contribution(rets, bpy)
            print(format_contribution(
                df, full_sharpe,
                title=f"{strat} · {asset_class} · {source}"))
            for name, r in df.iterrows():
                rows.append({
                    "strategy": strat, "asset_class": asset_class,
                    "source": source, "interval": interval,
                    "instrument": name,
                    "solo_ret": round(float(r["solo_ret"]), 4),
                    "solo_dd": round(float(r["solo_dd"]), 4),
                    "solo_sharpe": round(float(r["solo_sharpe"]), 4),
                    "loo_delta": round(float(r["loo_delta"]), 4),
                    "verdict": r["verdict"],
                })

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{GREEN}CSV: {args.csv} ({len(rows)} строк, "
              f"{len(pairs)} стратегий){RESET}")


if __name__ == "__main__":
    main()
