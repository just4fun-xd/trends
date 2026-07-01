"""Ручной скрипт выгрузки Databento -> parquet-панели. НЕ в горячем пути.

Архитектурное решение (согласовано): бэктест читает локальные parquet,
API дёргается ТОЛЬКО этим скриптом, запускаемым руками при расширении
корзины (7 -> 20-30 инструментов). Так квота Databento не жжётся на
каждом прогоне, бэктест воспроизводим и офлайн, данные версионируются.

Что делает:
  1. Тянет continuous front-month (M1) и second-month (M2) OHLCV.
  2. Склеивает с roll-adjustment (backward-adjust на дату ролла).
  3. Считает carry = (M1 - M2) / M2.
  4. Строит native-маску торгуемых дней.
  5. Выравнивает всё по union-календарю.
  6. Пишет panel_{open,high,low,close,volume,rollyield,native}.parquet.

Запуск:
    export DATABENTO_API_KEY=...
    python -m scripts.fetch_databento --symbols CL NG GC SI HG ZW ZC \\
        --start 2015-01-01 --end 2025-01-01 --out data/panels

Требует: pip install databento. Без ключа/пакета — печатает инструкцию
и создаёт СХЕМУ панелей из демо-данных, чтобы DatabentoSource можно было
протестировать локально до реальной выгрузки.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def fetch_via_api(
    symbols: list[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Тянет M1/M2 OHLCV из Databento API (реальная выгрузка).

    Args:
        symbols: Корневые символы фьючерсов (CL, NG, ...).
        start: Дата начала.
        end: Дата конца.

    Returns:
        dict инструмент -> DataFrame с колонками
        open/high/low/close/volume/close_m2 (для carry).

    Raises:
        RuntimeError: Если databento не установлен или нет API-ключа.
    """
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Нет DATABENTO_API_KEY. export DATABENTO_API_KEY=... "
            "или запусти с --demo для схемы из синтетики."
        )
    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError(
            "databento не установлен: pip install databento"
        ) from exc

    client = db.Historical(api_key)
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        # Continuous front-month через smart-symbology Databento.
        # dataset/schema зависят от подписки — здесь GLBX.MDP3 дневные бары.
        m1 = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[f"{sym}.c.0"],   # continuous front (M1)
            stype_in="continuous",
            schema="ohlcv-1d",
            start=start, end=end,
        ).to_df()
        m2 = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[f"{sym}.c.1"],   # continuous second (M2)
            stype_in="continuous",
            schema="ohlcv-1d",
            start=start, end=end,
        ).to_df()
        df = m1[["open", "high", "low", "close", "volume"]].copy()
        df["close_m2"] = m2["close"].reindex(df.index)
        out[sym] = df
    return out


def demo_panels(
    symbols: list[str], start: str, end: str, seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Синтетические панели той же СХЕМЫ — для локального теста офлайн.

    Не реальные данные: генерирует OHLCV+M2 с реалистичной структурой,
    чтобы DatabentoSource и портфельный движок можно было прогнать без
    API-ключа. Помечено демо — не для выводов, только для проверки труб.

    Args:
        symbols: Список символов.
        start: Дата начала.
        end: Дата конца.
        seed: Сид генератора.

    Returns:
        dict инструмент -> DataFrame схемы Databento.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    out = {}
    for i, sym in enumerate(symbols):
        steps = rng.normal(0.0003, 0.014, n)
        close = pd.Series(50 * (1 + i * 0.3) * np.exp(np.cumsum(steps)),
                          index=idx)
        spread = close * 0.01
        high = close + spread * rng.uniform(0.3, 1.0, n)
        low = close - spread * rng.uniform(0.3, 1.0, n)
        open_ = close.shift(1).bfill()
        volume = pd.Series(rng.integers(1000, 50000, n), index=idx)
        # M2 чуть выше/ниже M1 -> carry (contango/backwardation).
        close_m2 = close * (1 + rng.normal(0.002, 0.005, n))
        out[sym] = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "close_m2": close_m2,
        })
    return out


def build_panels(
    data: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Собирает выровненные панели из per-symbol DataFrame'ов.

    Выравнивает все инструменты по union-календарю. Считает carry и
    native-маску. Roll-adjustment предполагается уже применённым
    провайдером (continuous contracts) — для сырых контрактов добавить
    backward-adjust здесь.

    Args:
        data: dict инструмент -> DataFrame (open/high/low/close/volume/
            close_m2).

    Returns:
        dict поле -> панель (даты × инструменты) для полей
        open/high/low/close/volume/rollyield/native.
    """
    symbols = list(data.keys())
    # Union-календарь по всем инструментам.
    union_idx = None
    for df in data.values():
        if union_idx is None:
            union_idx = df.index
        else:
            union_idx = union_idx.union(df.index)

    fields = ["open", "high", "low", "close", "volume"]
    panels: dict[str, pd.DataFrame] = {
        f: pd.DataFrame(index=union_idx, columns=symbols, dtype=float)
        for f in fields
    }
    carry = pd.DataFrame(index=union_idx, columns=symbols, dtype=float)
    native = pd.DataFrame(index=union_idx, columns=symbols, dtype=bool)

    for sym, df in data.items():
        aligned = df.reindex(union_idx)
        for f in fields:
            panels[f][sym] = aligned[f]
        # carry = (M1 - M2) / M2.
        carry[sym] = (aligned["close"] - aligned["close_m2"]) / aligned[
            "close_m2"
        ]
        # native: день торговался, если close не NaN в исходном ряду.
        native[sym] = df["close"].reindex(union_idx).notna()

    panels["rollyield"] = carry
    panels["native"] = native
    return panels


def write_panels(panels: dict[str, pd.DataFrame], out_dir: str) -> None:
    """Пишет панели в parquet.

    Args:
        panels: dict поле -> панель.
        out_dir: Каталог назначения.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for field, panel in panels.items():
        path = out / f"panel_{field}.parquet"
        panel.to_parquet(path)
        print(f"  записано {path} ({panel.shape[0]}×{panel.shape[1]})")


def main() -> None:
    """CLI-точка входа скрипта выгрузки."""
    parser = argparse.ArgumentParser(
        description="Выгрузка Databento -> parquet-панели"
    )
    parser.add_argument("--symbols", nargs="+",
                        default=["CL", "NG", "GC", "SI", "HG", "ZW", "ZC"])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--out", default="data/panels")
    parser.add_argument("--demo", action="store_true",
                        help="Синтетические панели без API (тест труб)")
    args = parser.parse_args()

    if args.demo:
        print("ДЕМО-режим: синтетические панели (не для выводов).")
        data = demo_panels(args.symbols, args.start, args.end)
    else:
        print(f"Выгрузка {len(args.symbols)} инструментов из Databento...")
        data = fetch_via_api(args.symbols, args.start, args.end)

    panels = build_panels(data)
    write_panels(panels, args.out)
    print(f"Готово. Панели в {args.out}/ — DatabentoSource их прочитает.")


if __name__ == "__main__":
    main()
