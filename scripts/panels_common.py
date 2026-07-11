"""Общий pipeline сборки parquet-панелей (futures + equities).

Вынесено из fetch_databento.py, чтобы futures- и equities-выгрузки
делили один тракт ресемпла/выравнивания/записи. Различается только
источник данных (fetch_via_api в каждом враппере) и набор полей: у
фьючерсов есть close_m2 -> carry(rollyield), у акций его нет.

Архитектура (согласовано): бэктест читает локальные parquet, API
дёргается ТОЛЬКО враппером, запускаемым руками. Квота не жжётся на
прогонах, бэктест воспроизводим и офлайн, панели версионируются.

Публичный контракт:
  build_panels(data, interval)   -> dict поле -> панель
  write_panels(panels, out_dir)  -> пишет parquet
  demo_panels(...)               -> синтетика той же схемы (тест труб)
  add_cli_args(parser)           -> общие CLI-флаги
  base_fields                    -> ['open','high','low','close','volume']

Поле close_m2, если присутствует во входных DataFrame, порождает
панель rollyield (carry). Если его нет (equities) — rollyield не
пишется. native-маска пишется всегда (бар торговался, если close
не NaN в исходном ряду).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

base_fields = ["open", "high", "low", "close", "volume"]


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Добавляет CLI-флаги, общие для обоих врапперов.

    Символы намеренно НЕ добавляются здесь — дефолтная корзина у
    futures и equities разная, каждый враппер задаёт свою.

    Args:
        parser: Парсер враппера.
    """
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--out", default="data/panels")
    parser.add_argument("--interval", default="1d",
                        choices=["1d", "4h", "1h"],
                        help="Таймфрейм панелей: 1d, 4h (ресемпл 1h) "
                             "или 1h (часовые как есть, для MR)")
    parser.add_argument("--demo", action="store_true",
                        help="Синтетические панели без API (тест труб)")


def _resample_symbol(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Ресемплит один инструмент OHLCV(+M2) на крупный таймфрейм.

    Часовые бары -> H4: open=first, high=max, low=min, close=last,
    volume=sum. close_m2 (если есть) берётся последним в баре — для
    carry важен уровень на конец периода, не сумма.

    Args:
        df: DataFrame одного инструмента, индекс — часовые бары.
        rule: pandas-правило ресемпла ('4h').

    Returns:
        Ресемпленный DataFrame той же схемы.
    """
    agg = {
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "close_m2": "last",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}
    return df.resample(rule).agg(agg).dropna(subset=["close"])


def build_panels(
    data: dict[str, pd.DataFrame], interval: str = "1d"
) -> dict[str, pd.DataFrame]:
    """Собирает выровненные панели из per-symbol DataFrame'ов.

    Выравнивает все инструменты по union-календарю. Считает native-
    маску всегда; rollyield — только если во входных данных есть
    колонка close_m2 (фьючерсы). Roll-adjustment для фьючерсов
    предполагается уже применённым провайдером (continuous contracts).

    Если interval == '4h', часовые бары ресемплятся в 4H ДО
    выравнивания (union-календарь тогда — сетка H4-таймстемпов).

    Args:
        data: dict инструмент -> DataFrame (open/high/low/close/volume
            [+close_m2 для фьючерсов]).
        interval: '1d' — как есть; '4h' — ресемпл часовых баров в 4H.

    Returns:
        dict поле -> панель (время × инструменты). Всегда: open/high/
        low/close/volume/native. Плюс rollyield, если был close_m2.
    """
    if interval == "4h":
        data = {s: _resample_symbol(df, "4h") for s, df in data.items()}

    symbols = list(data.keys())
    has_carry = any("close_m2" in df.columns for df in data.values())

    union_idx: pd.Index | None = None
    for df in data.values():
        union_idx = df.index if union_idx is None else union_idx.union(
            df.index
        )

    panels: dict[str, pd.DataFrame] = {
        f: pd.DataFrame(index=union_idx, columns=symbols, dtype=float)
        for f in base_fields
    }
    native = pd.DataFrame(index=union_idx, columns=symbols, dtype=bool)
    carry = (
        pd.DataFrame(index=union_idx, columns=symbols, dtype=float)
        if has_carry else None
    )

    for sym, df in data.items():
        aligned = df.reindex(union_idx)
        for f in base_fields:
            panels[f][sym] = aligned[f]
        native[sym] = df["close"].reindex(union_idx).notna()
        if carry is not None and "close_m2" in df.columns:
            carry[sym] = (
                (aligned["close"] - aligned["close_m2"])
                / aligned["close_m2"]
            )

    panels["native"] = native
    if carry is not None:
        panels["rollyield"] = carry
    return panels


def write_panels(panels: dict[str, pd.DataFrame], out_dir: str) -> None:
    """Пишет панели в parquet.

    Args:
        panels: dict поле -> панель.
        out_dir: Каталог назначения (создаётся при отсутствии).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for field, panel in panels.items():
        path = out / f"panel_{field}.parquet"
        panel.to_parquet(path)
        print(f"  записано {path} ({panel.shape[0]}×{panel.shape[1]})")


def demo_panels(
    symbols: list[str], start: str, end: str, seed: int = 0,
    interval: str = "1d", carry: bool = True,
) -> dict[str, pd.DataFrame]:
    """Синтетические панели той же СХЕМЫ — для локального теста офлайн.

    Не реальные данные: генерирует OHLCV(+M2) с реалистичной
    структурой, чтобы источник и портфельный движок можно было
    прогнать без API-ключа. Помечено демо — не для выводов.

    Args:
        symbols: Список символов.
        start: Дата начала.
        end: Дата конца.
        seed: Сид генератора.
        interval: '1d' — дневная сетка; '4h' — часовая (её свернёт
            build_panels), чтобы проверить H4-тракт офлайн.
        carry: True — добавить close_m2 (фьючерсы); False — без него
            (equities-схема).

    Returns:
        dict инструмент -> DataFrame схемы источника.
    """
    rng = np.random.default_rng(seed)
    if interval == "4h":
        days = pd.bdate_range(start, end)
        idx = pd.DatetimeIndex(
            [d + pd.Timedelta(hours=h) for d in days for h in range(24)]
        )
        drift, vol = 0.0003 / 24, 0.014 / np.sqrt(24)
    else:
        idx = pd.bdate_range(start, end)
        drift, vol = 0.0003, 0.014
    n = len(idx)
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        steps = rng.normal(drift, vol, n)
        close = pd.Series(
            50 * (1 + i * 0.3) * np.exp(np.cumsum(steps)), index=idx
        )
        spread = close * 0.01
        high = close + spread * rng.uniform(0.3, 1.0, n)
        low = close - spread * rng.uniform(0.3, 1.0, n)
        open_ = close.shift(1).bfill()
        volume = pd.Series(rng.integers(1000, 50000, n), index=idx)
        cols = {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume,
        }
        if carry:
            cols["close_m2"] = close * (1 + rng.normal(0.002, 0.005, n))
        out[sym] = pd.DataFrame(cols)
    return out
