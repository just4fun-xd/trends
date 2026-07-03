"""Выгрузка commodity futures из Databento -> parquet-панели.

Рефактор исходного fetch_databento.py: тракт сборки/ресемпла/записи
переехал в panels_common, здесь остался только специфичный для
фьючерсов источник (M1/M2 continuous OHLCV + carry).

Пишет в СВОЙ каталог (по умолчанию data/panels/futures) — отдельный
sleeve от equities. Панели содержат rollyield (carry из M1-M2) и
native-маску торговых дней.

Что делает:
  1. Тянет continuous front-month (M1) и second-month (M2) OHLCV.
  2. Roll-adjustment — от continuous symbology провайдера.
  3. close_m2 -> build_panels считает carry = (M1 - M2) / M2.
  4. Выравнивает по union-календарю, пишет panel_*.parquet.

Запуск:
    export DATABENTO_API_KEY=...
    python -m scripts.fetch_databento_futures \\
        --symbols CL NG GC SI HG ZW ZC \\
        --start 2015-01-01 --end 2025-01-01 \\
        --out data/panels/futures

    # H4:
    python -m scripts.fetch_databento_futures --interval 4h ...

    # офлайн-проверка труб:
    python -m scripts.fetch_databento_futures --demo --out /tmp/fut_demo

Требует: pip install databento. Без ключа/пакета работает --demo.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from scripts import panels_common as pc
from scripts.fetch_resilient import fetch_symbols_cached


DATASET = "GLBX.MDP3"

DEFAULT_SYMBOLS = ["CL", "NG", "GC", "SI", "HG", "ZW", "ZC"]


def fetch_via_api(
    symbols: list[str], start: str, end: str, schema: str = "ohlcv-1d",
    cache_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Тянет M1/M2 OHLCV из Databento устойчиво: per-symbol кэш + retry.

    Args:
        symbols: Корневые символы фьючерсов (CL, NG, ...).
        start: Дата начала.
        end: Дата конца.
        schema: 'ohlcv-1d' — дневные; 'ohlcv-1h' — часовые (для H4).
        cache_dir: Каталог сырого per-symbol кэша (raw/{sym}.parquet).
            None -> без кэша (в память, как раньше).

    Returns:
        dict инструмент -> DataFrame (open/high/low/close/volume/
        close_m2). close_m2 порождает carry в build_panels.

    Raises:
        RuntimeError: Нет пакета databento или API-ключа.
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
    return fetch_symbols_cached(
        client, symbols, start, end, schema,
        dataset=DATASET,
        cache_dir=Path(cache_dir) if cache_dir else None,
        with_m2=True,
    )


def main() -> None:
    """CLI-точка входа futures-выгрузки."""
    parser = argparse.ArgumentParser(
        description="Выгрузка commodity futures (GLBX.MDP3) -> панели"
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    pc.add_cli_args(parser)
    parser.set_defaults(out="data/panels/futures")
    args = parser.parse_args()

    schema = "ohlcv-1h" if args.interval == "4h" else "ohlcv-1d"

    if args.demo:
        print(f"ДЕМО-режим ({args.interval}): синтетика (не для выводов).")
        data = pc.demo_panels(args.symbols, args.start, args.end,
                              interval=args.interval, carry=True)
    else:
        print(f"Выгрузка {len(args.symbols)} инструментов из {DATASET} "
              f"({args.interval}, schema={schema})...")
        cache_dir = str(Path(args.out) / "raw")
        data = fetch_via_api(args.symbols, args.start, args.end, schema,
                             cache_dir=cache_dir)

    panels = pc.build_panels(data, interval=args.interval)
    pc.write_panels(panels, args.out)
    print(f"Готово. Futures-панели ({args.interval}) в {args.out}/ — "
          f"укажи как sleeve в DatabentoSource(panel_dir=...).")


if __name__ == "__main__":
    main()
