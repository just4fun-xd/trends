"""Диагностика H4-панели: фактический bars_per_year и качество границы.

Проверяет, совпадает ли реальное число H4-баров с зашитым в раннере
1512, и ровно ли легла граница ресемпла (медиана 6 баров/день = чисто).

Запуск:
    python check_h4_bars.py /tmp/fut_h4_one CL
"""

from __future__ import annotations

import sys

import pandas as pd


def main() -> None:
    """Печатает bars_per_year и распределение баров H4-панели."""
    panel_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fut_h4_one"
    sym = sys.argv[2] if len(sys.argv) > 2 else "CL"

    close = pd.read_parquet(f"{panel_dir}/panel_close.parquet")
    s = close[sym].dropna()
    print(f"всего H4-баров: {len(s)}")
    print(f"диапазон: {s.index[0]} .. {s.index[-1]}")

    years = (s.index[-1] - s.index[0]).days / 365.25
    bpy = len(s) / years
    print(f"лет: {years:.2f}")
    print(f"ФАКТИЧЕСКИЙ bars_per_year: {bpy:.0f}")
    print("раннер использует: 1512")
    print(f"поправка Sharpe: sqrt({bpy:.0f}/1512) = "
          f"{(bpy / 1512) ** 0.5:.3f}")

    bph = s.groupby(s.index.hour).size()
    print("\nбары по часам UTC (чистый H4 => ~6 равных групп):")
    print(bph.to_string())

    per_day = s.groupby(s.index.date).size()
    print(f"\nбаров в день: медиана {per_day.median():.0f}, "
          f"мин {per_day.min()}, макс {per_day.max()}")
    print("медиана != 6 или большой разброс => граница ресемпла кривая")


if __name__ == "__main__":
    main()
