"""Раннер: прогон посерийных стратегий по корзине инструментов.

Загружает данные через DataSource (yfinance или Databento — сменные),
прогоняет стратегию на каждом инструменте, печатает verdict-таблицу
с цветным выводом (✅/⚠️/❌) как в отчётах Александру.

Запуск (пример):
    python -m runners.run_basket --strategy champion --source yf \\
        --start 2021-01-01 --end 2026-01-01

Живые данные Yahoo требуют сети; в песочнице недоступны — локально ОК.
"""

from __future__ import annotations

import argparse

import pandas as pd

from core.config import (
    COMMODITY_DATABENTO,
    COMMODITY_YF,
    EQUITY_BASKET,
)
from core.engine import BacktestResult, run_engine
from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from diagnostics.yearly import format_yearly_table, yearly_breakdown
from strategies import bollinger, donchian, ema, seasonal
from strategies.ensemble import ENSEMBLES, mr_ensemble
from strategies.overlays import with_vol_gate
from strategies.trend_lab import TREND_LAB
from strategies.meanrev_lab import MEANREV_LAB
from strategies.ou import ou_zscore

# ANSI-цвета. Выравнивание встраивается ДО escape-кодов (иначе f-string
# считает ширину неверно из-за невидимых символов) — урок проекта.
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Реестр боевых и закрытых стратегий (имя CLI -> функция).
STRATEGIES = {
    # EMA
    "ema_cross": ema.ema_cross,
    "ema_ensemble": ema.ema_ensemble,
    "ema_vt": ema.ema_ensemble_voltarget,          # champion equity
    "ema_barbell": ema.ema_ensemble_barbell_voltarget,
    # Donchian
    "donchian": donchian.donchian_breakout,
    "donchian_vt": donchian.donchian_ensemble_voltarget,
    # 2026-07b: "champion" РАЗЖАЛОВАН — проигрывает donchian_vt на
    # walk-forward (60% окон vs 80%; take-profit режет правый хвост).
    # Имя сохранено для сравнимости со старыми отчётами.
    "champion": donchian.donchian_est_macd_4step_take,
    "4step_pyr": donchian.donchian_est_macd_4step_pyramid,
    "donchian_est_pyr": donchian.donchian_ensemble_pyramid,
    # Bollinger / mean-rev
    "bb_rsi": bollinger.bollinger_rsi,
    "bb_rsi_vt": bollinger.bollinger_rsi_voltarget,
    # Seasonal (ортогональный календарный сигнал)
    "seasonal": seasonal.seasonal_gas,
    "seasonal_vt": seasonal.seasonal_gas_voltarget,
    "donch_seasonal": seasonal.donchian_seasonal,
    "donch_seasonal_vt": seasonal.donchian_seasonal_voltarget,
    # OU (осторожно: не автономна, для сравнения)
    "ou": ou_zscore,
}
# Лаборатория mean-reversion (10 вариантов bb_rsi, см. дисклеймер
# в strategies/meanrev_lab.py про multiple testing).
STRATEGIES.update(MEANREV_LAB)
# Ансамбли уровня сигнала: MR-ансамбль и trend+MR комбо
# (strategies/ensemble.py). Сырые сигналы — VT снаружи (--vt).
STRATEGIES.update(ENSEMBLES)
# Лаборатория тренда: пакет мат-моделей для пере-выбора чемпиона
# (strategies/trend_lab.py, дисклеймер multiple testing там же).
STRATEGIES.update(TREND_LAB)
# Гейтованные версии: vol-percentile gate против структурных
# коллапсов (ответ на CL апрель-2020, см. strategies/overlays.py).
STRATEGIES["mr_ens_gate"] = with_vol_gate(mr_ensemble)
STRATEGIES["mr_atr_gate"] = with_vol_gate(
    MEANREV_LAB["mr_atr_stop"])


def _verdict(res: BacktestResult, dd_limit: float = 0.40) -> str:
    """Цветной вердикт по результату (проходит DD / прибыльна).

    Args:
        res: Результат бэктеста.
        dd_limit: Лимит просадки.

    Returns:
        Строка вердикта с ANSI-цветом.
    """
    if not res.passes_dd(dd_limit):
        return f"{RED}❌ DD{res.max_drawdown:.0%}{RESET}"
    if res.total_return > 0:
        return f"{GREEN}✅ +{res.total_return:.0%}{RESET}"
    return f"{YELLOW}⚠️ {res.total_return:.0%}{RESET}"


def run_strategy_on_basket(
    strategy_fn,
    basket: dict,
    source,
    start: str,
    end: str,
    interval: str = "1d",
    cost: float = 0.0002,
    yearly: bool = False,
) -> pd.DataFrame:
    """Прогоняет стратегию по всей корзине, возвращает сводную таблицу.

    Args:
        strategy_fn: Функция стратегии (Bars -> position).
        basket: dict {название: тикер}.
        source: DataSource бэкенд.
        start: Дата начала.
        end: Дата конца.
        interval: Таймфрейм ('1d', '4h', ...).
        cost: Издержки.
        yearly: Печатать ли годовую разбивку return/DD по каждому
            инструменту (флаг --yearly).

    Returns:
        DataFrame сводки: инструмент, доходность, DD, Sharpe, проходит DD.
    """
    rows = []
    skipped = []
    for name, ticker in basket.items():
        try:
            bars = source.load(ticker, start, end, interval)
        except Exception as exc:  # noqa: BLE001
            skipped.append(name)
            print(f"  {YELLOW}пропуск {name} ({ticker}): {exc}{RESET}")
            continue
        pos = strategy_fn(bars)
        res = run_engine(bars, pos, cost=cost)
        rows.append({
            "instrument": name,
            "return": res.total_return,
            "max_dd": res.max_drawdown,
            "sharpe": res.sharpe,
            "passes_dd": res.passes_dd(),
        })
        if yearly:
            yb = yearly_breakdown(res.equity, res.bars_per_year)
            print("\n" + format_yearly_table(yb, f"{name} ({ticker})"))
        else:
            print(f"  {name:14s} {_verdict(res)}  "
                  f"Sharpe {res.sharpe:+.2f}")
    if skipped:
        print(f"{RED}ВНИМАНИЕ: корзина неполная ({len(skipped)} "
              f"пропущено: {', '.join(skipped)}). Портфельные числа "
              f"НЕСРАВНИМЫ с прогонами на полной корзине!{RESET}")
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, strategy_name: str) -> None:
    """Печатает портфельную сводку по таблице результатов.

    Портфель = среднее по инструментам; max DD портфеля = worst-case
    среди инструментов (не средневзвешенная) — как в BENCHMARK_RESULTS.

    Args:
        df: Таблица результатов run_strategy_on_basket.
        strategy_name: Имя стратегии для заголовка.
    """
    if df.empty:
        print(f"{RED}Нет результатов (данные недоступны?){RESET}")
        return
    port_ret = df["return"].mean()
    worst_dd = df["max_dd"].min()
    profitable = int((df["return"] > 0).sum())
    total = len(df)
    med_ret = df["return"].median()

    print(f"\n{BOLD}=== {strategy_name} — портфель ==={RESET}")
    print(f"  Средняя доходность:  {port_ret:+.1%}")
    print(f"  Медианная доходность: {med_ret:+.1%}")
    print(f"  Worst-case DD:       {worst_dd:.1%}")
    print(f"  Прибыльных:          {profitable}/{total}")
    print(f"  Проходят DD<40%:     {int(df['passes_dd'].sum())}/{total}")


def main() -> None:
    """CLI-точка входа раннера."""
    parser = argparse.ArgumentParser(description="Прогон стратегии по корзине")
    parser.add_argument("--strategy", default="champion",
                        choices=list(STRATEGIES.keys()))
    parser.add_argument("--source", default="yf", choices=["yf", "databento"])
    parser.add_argument("--basket", default="commodity",
                        choices=["commodity", "equity"])
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument(
        "--vt", action="store_true",
        help="Обернуть стратегию vol-таргетингом "
             "(position * sizer(bars), см. --sizer)",
    )
    parser.add_argument(
        "--sizer", default="realized",
        choices=["realized", "garch"],
        help="Оценка волы для --vt: rolling realized (статус-кво) "
             "или GARCH(1,1) прогноз (core/garch.py)",
    )
    parser.add_argument(
        "--target-vol", type=float, default=0.15,
        help="Целевая годовая волатильность для --vt",
    )
    parser.add_argument(
        "--yearly", action="store_true",
        help="Годовая разбивка return/DD по каждому инструменту",
    )
    parser.add_argument(
        "--panel-dir", default=None,
        help="Каталог parquet-панелей для --source databento. По "
             "умолчанию выбирается по корзине: data/panels/futures "
             "(commodity) или data/panels/equities (equity).",
    )
    parser.add_argument(
        "--exclude", default=None,
        help="Инструменты через запятую, исключить из корзины "
             "(напр. тонкие H4-рынки: PA,PL)",
    )
    args = parser.parse_args()

    panel_dir = args.panel_dir
    if panel_dir is None:
        panel_dir = ("data/panels/equities" if args.basket == "equity"
                     else "data/panels/futures")
    source = (YFinanceSource() if args.source == "yf"
              else DatabentoSource(panel_dir=panel_dir))
    # Нотация тикеров зависит от источника: yfinance ждёт 'CL=F',
    # Databento-панели хранят корневые 'CL'. Для commodity+databento
    # берём COMMODITY_DATABENTO (чистые символы), иначе yfinance-словари.
    if args.basket == "equity":
        basket = EQUITY_BASKET
    elif args.source == "databento":
        basket = {s: s for s in COMMODITY_DATABENTO}
    else:
        basket = COMMODITY_YF

    if args.exclude:
        drop = {s.strip() for s in args.exclude.split(",")}
        removed = [k for k in basket if k in drop]
        basket = {k: v for k, v in basket.items() if k not in drop}
        if removed:
            print(f"Исключены из корзины: {', '.join(removed)}")

    strategy_fn = STRATEGIES[args.strategy]
    if args.vt:
        from core.sizing import make_sizer
        base_fn = strategy_fn
        sizer = make_sizer(args.sizer, target_vol=args.target_vol)

        def strategy_fn(bars):  # noqa: F811
            return base_fn(bars) * sizer(bars)

    vt_note = (f" | vt:{args.sizer}@{args.target_vol:.0%}"
               if args.vt else "")
    print(f"{BOLD}Стратегия {args.strategy} | {args.basket} | "
          f"{args.source} | {args.interval}{vt_note} | "
          f"{args.start}..{args.end}{RESET}")
    df = run_strategy_on_basket(
        strategy_fn, basket, source, args.start, args.end,
        args.interval, args.cost, yearly=args.yearly,
    )
    print_summary(df, args.strategy)


if __name__ == "__main__":
    main()
