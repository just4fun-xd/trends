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
    COMMODITY_DATABENTO, COMMODITY_YF, CRYPTO_CCXT, CRYPTO_YF,
    EQUITY_BASKET, filter_basket)
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
    # Лаборатории 2026-07j. Чемпион ниши включён якорем, чтобы рейтинг
    # сразу показывал, кто из новичков хотя бы дотягивается до него
    # (это НЕ арбитраж — арбитраж только bootstrap; тут груба сортировка
    # на полной вселенной перед тем, как тратить bootstrap на выживших).
    "trend3": ["donchian_vt", "tr3_tsmom", "tr3_ribbon", "tr3_adx_di",
               "tr3_supertrend", "tr3_kama_slope", "tr3_hh_hl",
               "tr3_fracdiff", "tr3_zlema", "tr3_persist",
               "tr3_vr_trend", "tr3_mid_ride", "tr3_atr_mom",
               "tr3_vote3", "tr3_range_exp", "tr3_extreme_t"],
    "mr2": ["mr_lowvol", "mr2_kalman_z", "mr2_quantile", "mr2_runs",
            "mr2_entropy", "mr2_halflife", "mr2_vr", "mr2_mad",
            "mr2_cusum", "mr2_shock", "mr2_skew", "mr2_theil",
            "mr2_ddband", "mr2_tema_dev", "mr2_percb_bw", "mr2_soft_z"],
    "crypto_aggr": ["donchian_vt", "tr_ichimoku", "ca_turbo_don",
                    "ca_squeeze_pop", "ca_accel", "ca_thrust",
                    "ca_chand", "ca_double_don", "ca_burst",
                    "ca_vol_ride", "ca_short_break", "ca_pyramid_max"],
}

# --- Полные каталожные наборы (2026-07j): покрывают ВСЕ 132 стратегии
# реестра, сгруппированы по модулю-происхождению, с чемпионом-якорем
# в каждом. Для единого каталога-отчёта: одна серия --families
# cat_* по всем классам активов ранжирует всё без пропусков.
FAMILIES.update({
    "cat_trend_core": [
        "donchian_vt", "donchian", "champion", "4step_pyr",
        "donchian_est_pyr", "ema_cross", "ema_ensemble", "ema_vt",
        "ema_barbell", "kalman_trend", "kalman_trend_long", "tsmom",
        "tsmom_multi", "ewmac", "donch_multi", "channel_pos", "kama",
        "chandelier", "adx_donch"],
    "cat_trend_lab2": [
        "donchian_vt", "tr_regress", "tr_holt", "tr_supertrend",
        "tr_macd_hz", "tr_vhf", "tr_fractal", "tr_ichimoku", "tr_er",
        "tr_zlema", "tr_hull"],
    "cat_trend_lab3": [
        "donchian_vt", "tr3_tsmom", "tr3_ribbon", "tr3_adx_di",
        "tr3_supertrend", "tr3_kama_slope", "tr3_hh_hl", "tr3_fracdiff",
        "tr3_zlema", "tr3_persist", "tr3_vr_trend", "tr3_mid_ride",
        "tr3_atr_mom", "tr3_vote3", "tr3_range_exp", "tr3_extreme_t"],
    "cat_impulse": [
        "donchian_vt", "imp_tsmom_vw", "imp_accel", "imp_tstat",
        "imp_52h", "imp_atr_break", "imp_vol_expand", "imp_skip_mom",
        "imp_parkinson", "imp_drawup", "imp_kalman_imp", "carver_fast",
        "carver_ls", "carver_hicap"],
    "cat_monday": [
        "donchian_vt", "brk_monday_range", "brk_monday_h4",
        "brk_monday_h1"],
    "cat_crypto_aggr": [
        "donchian_vt", "tr_ichimoku", "ca_turbo_don", "ca_squeeze_pop",
        "ca_accel", "ca_thrust", "ca_chand", "ca_double_don",
        "ca_burst", "ca_vol_ride", "ca_short_break", "ca_pyramid_max"],
    "cat_mr_core": [
        "mr_lowvol", "mr_atr_stop", "mr_time_stop", "mr_scaled",
        "mr_ladder", "mr_trend", "mr_keltner", "mr_connors",
        "mr_confirm", "mr_short", "bb_rsi", "bb_rsi_vt", "mr_ens_gate",
        "mr_atr_gate", "carver_mr", "mr_lowvol_soft"],
    "cat_mr_lab2": [
        "mr_lowvol", "mr2_kalman_z", "mr2_quantile", "mr2_runs",
        "mr2_entropy", "mr2_halflife", "mr2_vr", "mr2_mad", "mr2_cusum",
        "mr2_shock", "mr2_skew", "mr2_theil", "mr2_ddband",
        "mr2_tema_dev", "mr2_percb_bw", "mr2_soft_z"],
    "cat_ou": [
        "mr_lowvol", "ou", "ou_adf", "ou_halflife", "ou_log",
        "ou_robust", "ou_hurst_gate", "ou_timestop", "ou_asym",
        "ou_kalman", "ou_volgate", "ou_ewma_z", "ou_jump",
        "ou_jump_asym"],
    "cat_ou_trend": [
        "donchian_vt", "mr_lowvol", "ou_pullback", "ou_trendline",
        "ou_residual", "ou_ride", "ou_gap_fade", "ou_router"],
    "cat_mixed": [
        "donchian_vt", "mr_lowvol", "mr_ens", "combo_tmr", "trend_ens",
        "mr_kelt_confirm", "seasonal", "seasonal_vt", "donch_seasonal",
        "donch_seasonal_vt", "carver_fdm", "hurst_combo",
        "donch_vol_confirm", "hurst_alloc"],
    # Лаборатории 2026-07k (реестр 132 -> 169).
    "cat_trend_lab4": [
        "donchian_vt", "tr4_psar", "tr4_renko", "tr4_mk", "tr4_ar1",
        "tr4_schmitt", "tr4_decycler", "tr4_overnight", "tr4_page",
        "tr4_kelly", "tr4_mom_pct"],
    "cat_crypto_aggr2": [
        "donchian_vt", "tr_ichimoku", "ca2_cppi", "ca2_gz", "ca2_bns",
        "ca2_evt", "ca2_semi", "ca2_vvol", "ca2_hawkes", "ca2_skew",
        "ca2_kelly_t", "ca2_breaker"],
    "cat_mr_lab3": [
        "mr_lowvol", "mr3_bertram", "mr3_garch_z", "mr3_kelly",
        "mr3_grid", "mr3_overshoot", "mr3_dfa", "mr3_rank",
        "mr3_tail_q", "mr3_ar1_fcst", "mr3_div"],
    "cat_ss": [
        "mr_lowvol", "ss_chi_mr", "ss_chi_soft"],
    "cat_mr_lowvol2": [
        "mr_lowvol", "mr_lv2_cont", "mr_lv2_scale", "mr_lv2_garch",
        "mr_lv2_zexit", "mr_lv2_vr"],
})

# Удобный алиас: все каталожные наборы одним аргументом.
ALL_CATALOG = [k for k in FAMILIES if k.startswith("cat_")]


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
                   default=["trend", "mr", "specialist"],
                   help="наборы семейств; 'all' = все cat_* "
                        "(полный каталог 132 стратегий)")
    p.add_argument("--csv", default=None,
                   help="дописать результаты в CSV для сводного "
                        "каталога (колонки: set,strategy,basket,source,"
                        "vt,rank,sharpe)")
    p.add_argument("--include", default=None,
                   help="активы или @КОРЗИНА (например @DONCH_CORE_COMM,"
                        " CL,GC); см. core.config.NAMED_BASKETS")
    p.add_argument("--exclude", default=None,
                   help="активы или @КОРЗИНА для исключения")
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
    if args.source == "ccxt" and args.basket != "crypto":
        # CCXT читает data/crypto/<символ>_<tf>.parquet. Сырьевые/
        # акционные тикеры туда не выгружаются -> раннее падение
        # вместо 150 строк «пропуск ... не найден» и пустого рейтинга
        # (баг 2026-07k: --source ccxt молча брал COMMODITY_YF).
        p.error(
            "--source ccxt требует --basket crypto (крипто-каталог "
            f"data/crypto/), получено --basket {args.basket}. "
            "Для крипто-лабораторий: --source ccxt --basket crypto.")
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

    sizer = make_sizer(args.sizer if hasattr(args, "sizer")
                       else "realized", target_vol=args.target_vol) \
        if args.vt else None

    vt_note = f" | vt@{args.target_vol:.0%}" if args.vt else ""
    print(f"{BOLD}Рейтинг стратегий по категориям | {args.basket} | "
          f"{args.source}{vt_note} | {args.start}..{args.end}{RESET}\n")

    families = args.families
    if families == ["all"] or "all" in families:
        families = ALL_CATALOG

    csv_rows = []
    for fam in families:
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
            csv_rows.append((
                fam, name, args.basket, args.source,
                f"{args.target_vol:.2f}" if args.vt else "off",
                rank, f"{sh:.4f}"))
        print()

    if args.csv and csv_rows:
        import csv
        import os
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["set", "strategy", "basket", "source",
                            "vt", "rank", "sharpe"])
            w.writerows(csv_rows)
        print(f"CSV дописан: {args.csv} (+{len(csv_rows)} строк)")

    print(f"{YELLOW}Рейтинг — скрининг «кто в игре», не финальный "
          f"вердикт. Подтверждать через run_bootstrap (разность "
          f"Sharpe) и walk-forward на двух источниках.{RESET}")


if __name__ == "__main__":
    main()
