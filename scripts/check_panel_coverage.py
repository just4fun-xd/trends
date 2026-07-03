"""Проверка покрытия (native) H4-панели по всем инструментам.

Сырые OHLCV-бары не пишутся Databento для часов без сделок — тонкие
рынки (палладий, платина) дадут много NaN/native=False в union-
календаре после ресемпла в H4. Перед боевым прогоном стратегий важно
знать, у кого из инструментов реально мало данных, а не просто
верить в общий размер панели (7994 строки — это union по ВСЕМ
инструментам, не гарантия покрытия для каждого).

Запуск:
    python scripts/check_panel_coverage.py data/panels_4h/futures
"""

from __future__ import annotations

import sys

import pandas as pd


def main() -> None:
    """Печатает % покрытия native-маски по каждому инструменту."""
    panel_dir = sys.argv[1] if len(sys.argv) > 1 else "data/panels_4h/futures"

    native = pd.read_parquet(f"{panel_dir}/panel_native.parquet")
    close = pd.read_parquet(f"{panel_dir}/panel_close.parquet")

    print(f"Union-календарь: {len(native)} H4-баров, "
          f"{native.shape[1]} инструментов\n")
    print(f"{'Инструмент':12s} {'native%':>8s} {'первый бар':>20s} "
          f"{'последний бар':>20s} {'NaN close%':>10s}")
    print("-" * 76)

    rows = []
    for sym in native.columns:
        cov = native[sym].mean()
        s = close[sym].dropna()
        first = s.index[0] if len(s) else None
        last = s.index[-1] if len(s) else None
        nan_pct = close[sym].isna().mean()
        rows.append((sym, cov, first, last, nan_pct))
        print(f"{sym:12s} {cov:7.1%}  {str(first):>20s} "
              f"{str(last):>20s} {nan_pct:9.1%}")

    print("\nИнструменты с покрытием < 50% — под вопросом для H4-"
          "бэктеста (много дыр, VT/Sharpe могут исказиться):")
    thin = [r for r in rows if r[1] < 0.5]
    if thin:
        for sym, cov, *_ in thin:
            print(f"  {sym}: {cov:.1%} покрытия")
    else:
        print("  таких нет")


if __name__ == "__main__":
    main()
