"""Валидатор Hurst-аллокатора: H-вердикт против карты актив×стратегия.

Проверяет, воспроизводит ли variance-ratio H эмпирическую карту
(docs/REGIME_MAP_2026-07f.md). Для каждого актива считает статичный H
на всей истории + VR z-score, выводит H-вердикт (тренд/MR/нейтрал) и
сверяет с ожиданием из карты.

Логика валидации: если H(GC)>0.5, H(NG)<0.5, H(CL)>0.5 — модель
превращает «карту из результатов» в «сайзер из свойств инструмента».
Если H не различает — оценщик плох, аллокатор бесполезен.

Запуск:
    python -m runners.run_hurst_validation \\
        --basket commodity --source databento \\
        --panel-dir data/panels/futures --start 2019-01-01
"""

from __future__ import annotations

import argparse

import numpy as np

from core.config import (
    COMMODITY_DATABENTO, COMMODITY_YF, CRYPTO_YF, EQUITY_BASKET)
from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from strategies.variance_ratio import hurst_from_vr

BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"

# Ожидание из карты 2026-07f (робастные, совпавшие на 2 источниках).
# trend / mr / None(нейтрал). Ключи — символы Databento.
MAP_EXPECT = {
    "GC": "trend", "CL": "trend", "ZL": "trend",
    "NG": "mr", "SI": "mr", "ZC": "mr", "ZW": "mr", "ZS": "mr",
    "PA": None, "PL": None, "HG": None, "ZM": None,
}


def _verdict(h: float) -> str:
    if np.isnan(h):
        return "н/д"
    if h > 0.52:
        return "trend"
    if h < 0.48:
        return "mr"
    return "нейтрал"


def main() -> None:
    """CLI: H по каждому активу + сверка с картой."""
    p = argparse.ArgumentParser(
        description="Валидация VR-Hurst против карты режимов")
    p.add_argument("--source", default="yf",
                   choices=["yf", "databento"])
    p.add_argument("--basket", default="commodity",
                   choices=["commodity", "equity", "crypto"])
    p.add_argument("--panel-dir", default=None)
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--interval", default="1d")
    args = p.parse_args()

    panel_dir = args.panel_dir
    if panel_dir is None:
        panel_dir = ("data/panels/equities" if args.basket == "equity"
                     else "data/panels/futures")
    source = (YFinanceSource() if args.source == "yf"
              else DatabentoSource(panel_dir=panel_dir))
    if args.basket == "equity":
        basket = EQUITY_BASKET
    elif args.basket == "crypto":
        basket = CRYPTO_YF
    elif args.source == "databento":
        basket = {s: s for s in COMMODITY_DATABENTO}
    else:
        basket = COMMODITY_YF

    print(f"{BOLD}VR-Hurst валидация | {args.basket} | {args.source} "
          f"| {args.start}..{args.end}{RESET}")
    print(f"  {'актив':12s} {'H':>7s} {'z(VR)':>8s} "
          f"{'H-вердикт':>10s}  {'карта':>8s}  сверка")
    print("  " + "-" * 60)

    match, total = 0, 0
    for name, ticker in basket.items():
        try:
            bars = source.load(ticker, args.start, args.end,
                               args.interval)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:12s} пропуск: {exc}")
            continue
        logret = np.log(
            bars.close / bars.close.shift(1)).dropna().to_numpy()
        h, z = hurst_from_vr(logret)
        hv = _verdict(h)
        expect = MAP_EXPECT.get(name if args.source == "databento"
                                else _yf_to_sym(name))
        exp_str = expect if expect else "нейтрал"
        # Сверка: trend/mr должны совпасть; нейтрал карты — свободен.
        mark = ""
        if expect in ("trend", "mr"):
            total += 1
            if hv == expect:
                match += 1
                mark = f"{GREEN}✓{RESET}"
            elif hv == "нейтрал":
                mark = f"{YELLOW}~ (H слаб){RESET}"
            else:
                mark = f"{RED}✗ противоречит{RESET}"
        hs = f"{h:+.3f}" if not np.isnan(h) else "н/д"
        zs = f"{z:+.2f}" if not np.isnan(z) else "н/д"
        print(f"  {name:12s} {hs:>7s} {zs:>8s} {hv:>10s}  "
              f"{exp_str:>8s}  {mark}")

    if total:
        pct = 100.0 * match / total
        col = GREEN if pct >= 70 else (YELLOW if pct >= 50 else RED)
        print(f"\n  {BOLD}Совпадение с картой: {col}{match}/{total} "
              f"({pct:.0f}%){RESET}")
        print(f"  {YELLOW}>=70%: модель воспроизводит карту, аллокатор "
              f"обоснован. <50%: оценщик H не различает — пересмотреть."
              f"{RESET}")


def _yf_to_sym(name: str) -> str:
    """Грубый маппинг yfinance-названий на символы карты."""
    m = {
        "Gold": "GC", "Crude Oil": "CL", "Brent Oil": "CL",
        "Natural Gas": "NG", "Silver": "SI", "Copper": "HG",
        "Corn": "ZC", "Wheat": "ZW", "Soybeans": "ZS",
        "Soybean Oil": "ZL", "Soybean Meal": "ZM",
        "Palladium": "PA", "Platinum": "PL",
    }
    return m.get(name, name)


if __name__ == "__main__":
    main()
