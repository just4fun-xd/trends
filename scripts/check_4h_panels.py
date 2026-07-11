"""Проверка 4h-панелей Databento (10.07.26).

Кирилл выгрузил data/panels_4h/futures. Перед прогоном алгоритмов на
4h убеждаемся, что панели пригодны: все поля на месте, индексы
выровнены, покрытие символов достаточное, нет дыр/дубликатов времени,
bars_per_year выводится корректно (~1512 для 4h против 252 для 1d).

Запуск:
    python -m scripts.check_4h_panels
    python -m scripts.check_4h_panels --panel-dir data/panels_4h/futures
    python -m scripts.check_4h_panels --panel-dir data/panels_4h/equities

НИЧЕГО не пишет и не чинит — только диагностика. Если что-то красное,
перевыгрузить: python -m scripts.fetch_databento --interval 4h ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.bars import infer_bars_per_year

FIELDS = ("open", "high", "low", "close", "volume", "rollyield", "native")
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _load(panel_dir: Path, field: str) -> pd.DataFrame | None:
    path = panel_dir / f"panel_{field}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def main() -> None:
    """CLI-диагностика 4h-панелей."""
    ap = argparse.ArgumentParser(description="Проверка 4h-панелей")
    ap.add_argument("--panel-dir", default="data/panels_4h/futures")
    ap.add_argument("--min-bars", type=int, default=2000,
                    help="минимум баров на символ для 'достаточно'")
    args = ap.parse_args()
    pdir = Path(args.panel_dir)

    print(f"{BOLD}Проверка 4h-панелей: {pdir}{RESET}")
    if not pdir.exists():
        print(f"{RED}каталог не найден. Выгрузи: python -m "
              f"scripts.fetch_databento --interval 4h --out "
              f"{pdir}{RESET}")
        return

    close = _load(pdir, "close")
    if close is None:
        print(f"{RED}нет panel_close.parquet — панель непригодна{RESET}")
        return

    idx = close.index
    # 1) выравнивание всех обязательных полей по индексу close
    print(f"\n{BOLD}1. Поля и выравнивание{RESET}")
    ref_shape = close.shape
    for field in FIELDS:
        df = _load(pdir, field)
        if df is None:
            tag = (f"{YELLOW}отсутствует (опц.){RESET}"
                   if field in ("volume", "rollyield", "native")
                   else f"{RED}ОТСУТСТВУЕТ (обяз.){RESET}")
            print(f"  panel_{field:10s}: {tag}")
            continue
        aligned = df.index.equals(idx)
        same_cols = list(df.columns) == list(close.columns)
        mark = (f"{GREEN}ок{RESET}" if aligned and same_cols
                else f"{RED}РАССИНХРОН{RESET}")
        print(f"  panel_{field:10s}: {df.shape} idx={aligned} "
              f"cols={same_cols} {mark}")

    # 2) временная сетка: 4h-шаг, без дубликатов, монотонность
    print(f"\n{BOLD}2. Временная сетка{RESET}")
    # Интервал выводим ИЗ ДАННЫХ (медианный шаг), а не хардкодим 4h —
    # скрипт зовут и для 1h-панелей (фикс 10.07.26: раньше писал
    # bars_per_year(4h)=1512 даже для часовых, где верно ~6048).
    dt = pd.Series(idx)
    if len(idx) > 2:
        step_h = int(np.median(
            np.diff(idx.values).astype("timedelta64[h]").astype(int)))
    else:
        step_h = 4
    interval_guess = {1: "1h", 4: "4h", 24: "1d"}.get(step_h, f"{step_h}h")
    bpy = infer_bars_per_year(interval_guess)
    dups = int(dt.duplicated().sum())
    mono = bool(dt.is_monotonic_increasing)
    span_days = (idx[-1] - idx[0]).days if len(idx) > 1 else 0
    print(f"  баров всего: {len(idx)}")
    print(f"  диапазон: {idx[0]} .. {idx[-1]} ({span_days} дн)")
    print(f"  дубликаты времени: "
          f"{GREEN if dups == 0 else RED}{dups}{RESET}")
    print(f"  монотонность: "
          f"{GREEN if mono else RED}{mono}{RESET}")
    print(f"  интервал (из данных): {BOLD}{interval_guess}{RESET}, "
          f"bars_per_year: {bpy:.0f} "
          f"({GREEN}1d=252, 4h~1512, 1h~6048{RESET})")
    if len(idx) > 2:
        deltas = np.diff(idx.values).astype("timedelta64[h]").astype(int)
        from collections import Counter
        top = Counter(deltas).most_common(3)
        print(f"  шаги (часы, топ-3): {top} "
              f"{GREEN + 'преобладает 4ч' + RESET if top and top[0][0] == 4 else YELLOW + 'проверь: не 4ч?' + RESET}")

    # 3) покрытие по символам
    print(f"\n{BOLD}3. Покрытие символов (не-NaN close){RESET}")
    cov = close.notna().sum().sort_values()
    thin = cov[cov < args.min_bars]
    for sym, n in cov.items():
        bad = n < args.min_bars
        first = close[sym].first_valid_index()
        last = close[sym].last_valid_index()
        col = RED if bad else GREEN
        print(f"  {sym:6s}: {col}{n:6d}{RESET} баров "
              f"[{str(first)[:10]}..{str(last)[:10]}]")
    if len(thin):
        print(f"\n{YELLOW}  тонкое покрытие (<{args.min_bars}): "
              f"{list(thin.index)} — на 4h могут ломать стратегии, "
              f"рассмотри --exclude{RESET}")

    # 4) аномалии значений
    print(f"\n{BOLD}4. Аномалии{RESET}")
    nonpos = int((close <= 0).sum().sum())
    infs = int(np.isinf(close.to_numpy(dtype=float)).sum())
    print(f"  close <= 0: {YELLOW if nonpos else GREEN}{nonpos}{RESET} "
          f"(WTI-2020 может давать легитимные)")
    print(f"  inf в close: {RED if infs else GREEN}{infs}{RESET}")
    hi = _load(pdir, "high")
    lo = _load(pdir, "low")
    if hi is not None and lo is not None:
        bad_hl = int((hi < lo).sum().sum())
        print(f"  high < low: {RED if bad_hl else GREEN}{bad_hl}{RESET} "
              f"(нарушение OHLC-инварианта)")

    print(f"\n{BOLD}Вывод:{RESET} если пункты 1-2 зелёные и покрытие "
          f"ядра (CL/GC/BTC-эквив.) выше {args.min_bars} — панель "
          f"пригодна для 4h-прогона. Тонкие символы исключай флагом "
          f"--exclude в раннере.")


if __name__ == "__main__":
    main()
