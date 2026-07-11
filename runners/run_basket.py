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

import numpy as np
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
from strategies.advanced import ADVANCED
from strategies.ensemble import ENSEMBLES, mr_ensemble
from strategies.overlays import with_vol_gate
from strategies.trend_lab import TREND_LAB, TSMOM_VARIANTS
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
# Per-asset TSMOM (дедуп tr3_tsmom, 10.07.26): tsmom_eq/cr1d/cr4h/comm.
STRATEGIES.update(TSMOM_VARIANTS)
# Продвинутый мат-аппарат (2026-07e): Carver FDM, Hurst-аллокатор,
# volume-подтверждение (strategies/advanced.py).
STRATEGIES.update(ADVANCED)
# Hurst-аллокатор v2 (2026-07f): variance-ratio H, откалиброван по
# карте актив×стратегия (strategies/hurst_alloc.py).
from strategies.hurst_alloc import HURST_ALLOC  # noqa: E402
STRATEGIES.update(HURST_ALLOC)
# OU-лаборатория (2026-07f): 10 модификаций OU-реверсии под
# исследование крипто-ниши (strategies/ou_lab.py).
from strategies.ou_lab import OU_LAB  # noqa: E402
STRATEGIES.update(OU_LAB)
# Kalman trend (2026-07g): state-space уровень+наклон, непрерывный
# тренд с матмоделью — ответ на «пробой слишком просто» для крипты
# (strategies/kalman_trend.py).
from strategies.kalman_trend import KALMAN_TREND  # noqa: E402
STRATEGIES.update(KALMAN_TREND)
# Monday range (2026-07g): недельный пробой опорного диапазона —
# бенчмарк Александра для крипты (strategies/monday_range.py).
from strategies.monday_range import MONDAY_RANGE  # noqa: E402
STRATEGIES.update(MONDAY_RANGE)
# Импульсная лаборатория (2026-07g): 10 тренд-моделей для агрессивных
# импульсных рынков + вариации Carver (strategies/impulse_lab.py).
from strategies.impulse_lab import IMPULSE_LAB  # noqa: E402
STRATEGIES.update(IMPULSE_LAB)
# Carver-MR (2026-07g): реверсия аппаратом Карвера + мягкий vol-гейт
# (идеи Кирилла, доработанные; strategies/carver_mr.py).
from strategies.carver_mr import CARVER_MR  # noqa: E402
STRATEGIES.update(CARVER_MR)
# OU×Trend лаборатория (2026-07g): реверсия ВМЕСТЕ с трендом —
# ответ на «раздел OU не может быть бесполезен» (ou_trend_lab.py).
from strategies.ou_trend_lab import OU_TREND_LAB  # noqa: E402
STRATEGIES.update(OU_TREND_LAB)
# Trend Lab 2 (2026-07g): 10 трендовых на разном матаппарате —
# регрессия/Хольт/SuperTrend/VHF/фракталы/Ишимоку/ER/ZLEMA/Hull.
from strategies.trend_lab2 import TREND_LAB2  # noqa: E402
STRATEGIES.update(TREND_LAB2)
# Гейтованные версии: vol-percentile gate против структурных
# коллапсов (ответ на CL апрель-2020, см. strategies/overlays.py).
STRATEGIES["mr_ens_gate"] = with_vol_gate(mr_ensemble)
STRATEGIES["mr_atr_gate"] = with_vol_gate(
    MEANREV_LAB["mr_atr_stop"])
# MR-лаборатория 2 (2026-07j): 15 моделей реверсии разного
# матаппарата — state-space, непараметрика, робастная статистика,
# энтропия, VR/half-life гейты (strategies/meanrev_lab2.py).
from strategies.meanrev_lab2 import MEANREV_LAB2  # noqa: E402
STRATEGIES.update(MEANREV_LAB2)
# Trend Lab 3 (2026-07j): 15 трендовых — TSMOM, ленты, ADX,
# SuperTrend, KAMA, свинг-структура, fracdiff (LdP), ZLEMA,
# биномиальная персистентность, VR-гейт (strategies/trend_lab3.py).
from strategies.trend_lab3 import TREND_LAB3  # noqa: E402
STRATEGIES.update(TREND_LAB3)
# Крипто-AGGR (2026-07j): 10 агрессивных H4-моделей — пирамиды,
# squeeze, каскады, thrust, шорт-нога (strategies/crypto_aggr_lab.py).
from strategies.crypto_aggr_lab import CRYPTO_AGGR_LAB  # noqa: E402
STRATEGIES.update(CRYPTO_AGGR_LAB)
# Trend Lab 4 (2026-07k): 10 трендовых на незанятом аппарате — PSAR,
# Renko, Mann-Kendall, AR(1)-гейт, триггер Шмитта, декиклер Элерса,
# overnight-дрейф, CUSUM Пейджа, Келли, перцентиль momentum.
from strategies.trend_lab4 import TREND_LAB4  # noqa: E402
STRATEGIES.update(TREND_LAB4)
# Крипто-AGGR-2 (2026-07k): агрессивные ядра × кризисный матаппарат —
# CPPI, Grossman-Zhou, BNS-jump, CVaR, полудисперсия, vol-of-vol,
# Хоукс, skew-гейт, Келли-t, circuit breaker.
from strategies.crypto_aggr_lab2 import CRYPTO_AGGR_LAB2  # noqa: E402
STRATEGIES.update(CRYPTO_AGGR_LAB2)
# MR Lab 3 (2026-07k): реверсия с упором на извлечение — пороги
# Бертрама (OU-как-аппарат), GARCH-шок, Келли-MR, грид, овершут,
# DFA-Хёрст, ранговый шок, хвост-квантиль, AR(1)-прогноз, дивергенция.
from strategies.meanrev_lab3 import MEANREV_LAB3  # noqa: E402
STRATEGIES.update(MEANREV_LAB3)
# Schwartz-Smith (Roadmap D.1, 2026-07k): Kalman-разложение лог-цены
# на chi (OU) + xi (GBM); MR-нога строго на изолированном chi.
from strategies.schwartz_smith import SCHWARTZ_SMITH  # noqa: E402
STRATEGIES.update(SCHWARTZ_SMITH)
# Доработка mr_lowvol (2026-07k): 5 вариантов чемпиона, по одному
# узлу на вариант (размер/набор/гейт/выход/двойной гейт).
from strategies.mr_lowvol2 import MR_LOWVOL2  # noqa: E402
STRATEGIES.update(MR_LOWVOL2)


def _build_families() -> dict[str, str]:
    """Карта имя стратегии -> семейство ('trend' / 'mean-reversion' /
    'mixed').

    Источник — членство в лабораторных словарях, не порядок CLI-
    аргументов (фикс 2026-07j: run_strategy_chart подписывал роли
    по позиции --a/--b и врал, называя donchian_vt реверсией).
    """
    fam: dict[str, str] = {}
    trend_dicts = (TREND_LAB, TSMOM_VARIANTS, TREND_LAB2, TREND_LAB3,
                   TREND_LAB4, KALMAN_TREND, IMPULSE_LAB, MONDAY_RANGE,
                   CRYPTO_AGGR_LAB, CRYPTO_AGGR_LAB2)
    mr_dicts = (MEANREV_LAB, MEANREV_LAB2, MEANREV_LAB3, CARVER_MR,
                OU_LAB, SCHWARTZ_SMITH, MR_LOWVOL2)
    for d in trend_dicts:
        fam.update({k: "trend" for k in d})
    for d in mr_dicts:
        fam.update({k: "mean-reversion" for k in d})
    fam.update({k: "mixed" for k in OU_TREND_LAB})
    fam.update({
        "ema_cross": "trend", "ema_ensemble": "trend",
        "ema_vt": "trend", "ema_barbell": "trend",
        "donchian": "trend", "donchian_vt": "trend",
        "champion": "trend", "4step_pyr": "trend",
        "donchian_est_pyr": "trend",
        "bb_rsi": "mean-reversion", "bb_rsi_vt": "mean-reversion",
        "seasonal": "mixed", "seasonal_vt": "mixed",
        "donch_seasonal": "mixed", "donch_seasonal_vt": "mixed",
        "ou": "mean-reversion",
        "mr_ens": "mean-reversion", "mr_ens_gate": "mean-reversion",
        "mr_atr_gate": "mean-reversion",
        "ca_short_break": "trend (short)",
        "combo_tmr": "mixed", "trend_ens": "trend",
        "mr_kelt_confirm": "mean-reversion",
        "carver_fdm": "trend", "hurst_combo": "mixed",
        "donch_vol_confirm": "trend", "hurst_alloc": "mixed",
    })
    return fam


STRATEGY_FAMILY = _build_families()


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
) -> tuple[pd.DataFrame, dict]:
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
        (DataFrame сводки, dict {инструмент: кривая капитала}). Кривые
        нужны print_summary для настоящего портфельного Sharpe (equal-
        weight по дневным P&L), а не усреднения итоговых компаундов.
    """
    rows = []
    skipped = []
    equity_by_name = {}
    for name, ticker in basket.items():
        try:
            bars = source.load(ticker, start, end, interval)
        except Exception as exc:  # noqa: BLE001
            skipped.append(name)
            print(f"  {YELLOW}пропуск {name} ({ticker}): {exc}{RESET}")
            continue
        pos = strategy_fn(bars)
        # rf НЕ вычитается на уровне отдельной ноги: одиночный инструмент
        # несёт лишь долю риск-бюджета (реализованная вола ~2-4%, не 15%),
        # а обеспечение зарабатывает rf ОДИН раз на весь счёт, не по разу
        # на каждый инструмент. Вычет rf из каждой ноги давал абсурд
        # (NG Sharpe -2.34 при +5% компаунде). Excess-Sharpe осмыслен
        # только на портфельном пути (run_sleeves / полное комбо).
        res = run_engine(bars, pos, cost=cost)
        equity_by_name[name] = res.equity
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
    return pd.DataFrame(rows), equity_by_name


def print_summary(df: pd.DataFrame, strategy_name: str,
                  equity_by_name: dict | None = None,
                  rf: float = 0.0,
                  bars_per_year: float = 252.0) -> None:
    """Печатает портфельную сводку по таблице результатов.

    Портфель = equal-weight по дневным P&L инструментов (не усреднение
    итоговых компаундов). max DD портфеля = worst-case среди
    инструментов (как в BENCHMARK_RESULTS).

    Args:
        df: Таблица результатов run_strategy_on_basket.
        strategy_name: Имя стратегии для заголовка.
        equity_by_name: Кривые капитала по инструментам. Если заданы —
            строится настоящий equal-weight портфель и его полный
            (не среднегодовой) excess-Sharpe с вычетом rf ОДИН раз.
        rf: Годовая безрисковая ставка (вычитается один раз на уровне
            портфеля — обеспечение зарабатывает rf на весь счёт).
        bars_per_year: Для аннуализации.
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
    print(f"  Средняя доходность (комп.): {port_ret:+.1%}")
    print(f"  Медианная доходность (комп.): {med_ret:+.1%}")
    print(f"  Worst-case DD:       {worst_dd:.1%}")
    print(f"  Прибыльных:          {profitable}/{total}")
    print(f"  Проходят DD<40%:     {int(df['passes_dd'].sum())}/{total}")

    if equity_by_name:
        # Настоящий equal-weight портфель из дневных доходностей ног.
        rets = pd.DataFrame({
            n: eq.pct_change() for n, eq in equity_by_name.items()
        }).fillna(0.0)
        port = rets.mean(axis=1)  # equal-weight дневной P&L
        std = port.std(ddof=1)
        if std > 0:
            sharpe_gross = float(port.mean() / std
                                 * np.sqrt(bars_per_year))
            excess = port.mean() - rf / bars_per_year
            sharpe_ex = float(excess / std * np.sqrt(bars_per_year))
            ann_ret = float((1.0 + port).prod()
                            ** (bars_per_year / len(port)) - 1.0)
            port_eq = (1.0 + port).cumprod()
            port_dd = float((port_eq / port_eq.cummax() - 1.0).min())
            print(f"  {BOLD}Портфель EW — годовая доходность: "
                  f"{ann_ret:+.1%}{RESET}")
            if rf:
                # Фьючерсная рамка: обеспечение в T-bills зарабатывает
                # rf само -> excess счёта = gross Sharpe стратегии.
                print(f"  Портфель EW — Sharpe (cash-счёт, excess "
                      f"rf={rf:.1%}): {sharpe_ex:+.2f}")
                print(f"  {BOLD}Портфель EW — Sharpe (фьючерсный счёт, "
                      f"обеспечение в T-bills): {sharpe_gross:+.2f}"
                      f"{RESET}")
                print(f"  {BOLD}Доходность счёта с обеспечением: "
                      f"{ann_ret + rf:+.1%} годовых{RESET}")
            else:
                print(f"  {BOLD}Портфель EW — Sharpe (rf=0): "
                      f"{sharpe_gross:+.2f}{RESET}")
            print(f"  Портфель EW — реальная DD:  {port_dd:.1%}")


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
        "--rf", type=float, default=0.0,
        help="Годовая безрисковая ставка для excess-Sharpe "
             "(0.045 = 4.5%%). По умолчанию 0 — совместимо со старыми "
             "прогонами. Согласовано с run_bootstrap.",
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
    df, equity_by_name = run_strategy_on_basket(
        strategy_fn, basket, source, args.start, args.end,
        args.interval, args.cost, yearly=args.yearly,
    )
    print_summary(df, args.strategy, equity_by_name=equity_by_name,
                  rf=args.rf)


if __name__ == "__main__":
    main()
