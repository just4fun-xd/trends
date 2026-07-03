"""Устойчивая per-symbol выгрузка Databento с retry и кэшем.

Проблема исходного fetch_via_api: вся корзина качается в одном
процессе и держится в памяти до записи. Обрыв ЛЮБОГО потока
(BentoError: Response ended prematurely — транзиентный обрыв при
стриминге больших H4-объёмов) убивает ВСЮ выгрузку, включая уже
скачанные (оплаченные!) инструменты. На 13-16 инструментах × 2
контракта × H4 это дорого и хрупко.

Решение:
  1. Каждый инструмент выгружается и СРАЗУ пишется в per-symbol
     parquet-кэш (raw/{sym}.parquet). Повторный запуск дособирает
     только недостающие — оплаченное не перекачивается.
  2. Транзиентные обрывы (Response ended prematurely и сетевые) —
     retry с экспоненциальным backoff.
  3. Нерезолвнутые символы (softs на ICE: KC/CC/SB не на GLBX.MDP3)
     не роняют процесс — логируются и пропускаются.

Использование — замена цикла в fetch_databento_futures.fetch_via_api:
    from scripts.fetch_resilient import fetch_symbols_cached
    out = fetch_symbols_cached(
        client, symbols, start, end, schema,
        dataset="GLBX.MDP3", cache_dir=Path(out_dir) / "raw",
        with_m2=True,
    )
Дальше out идёт в build_panels как раньше.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

# Подстроки транзиентных ошибок — retry по ним, а не по всем подряд
# (did not resolve / no data — НЕ транзиентные, их не ретраим).
_TRANSIENT = (
    "response ended prematurely",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "502", "503", "504",
)

# Подстроки «символ отсутствует в датасете» — пропуск без ошибки.
_UNRESOLVED = ("did not resolve", "no data found")


def _is_transient(exc: Exception) -> bool:
    """Похоже ли исключение на транзиентный сетевой обрыв."""
    msg = str(exc).lower()
    return any(t in msg for t in _TRANSIENT)


def _fetch_one_stream(
    client, dataset: str, symbol: str, schema: str,
    start: str, end: str, max_retries: int = 4,
) -> pd.DataFrame | None:
    """Тянет один continuous-поток с retry на транзиентных обрывах.

    Args:
        client: databento.Historical.
        dataset: Датасет (GLBX.MDP3).
        symbol: Полный continuous-символ (CL.c.0).
        schema: ohlcv-1h / ohlcv-1d.
        start: Дата начала.
        end: Дата конца.
        max_retries: Сколько раз повторить транзиентный обрыв.

    Returns:
        DataFrame потока, либо None если символ не резолвится в
        датасете (softs на ICE) или данных нет.

    Raises:
        Exception: Нетранзиентная ошибка после исчерпания ретраев.
    """
    attempt = 0
    while True:
        try:
            df = client.timeseries.get_range(
                dataset=dataset, symbols=[symbol],
                stype_in="continuous", schema=schema,
                start=start, end=end,
            ).to_df()
            if df is None or len(df) == 0:
                print(f"    {symbol}: пусто (нет данных в датасете) — "
                      f"пропуск")
                return None
            return df
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if any(u in msg for u in _UNRESOLVED):
                print(f"    {symbol}: не резолвится в {dataset} "
                      f"(другая биржа?) — пропуск")
                return None
            if _is_transient(exc) and attempt < max_retries:
                wait = 2.0 ** attempt
                attempt += 1
                print(f"    {symbol}: транзиентный обрыв, "
                      f"retry {attempt}/{max_retries} через {wait:.0f}s")
                time.sleep(wait)
                continue
            raise


def fetch_symbols_cached(
    client, symbols: list[str], start: str, end: str, schema: str,
    dataset: str = "GLBX.MDP3", cache_dir: Path | None = None,
    with_m2: bool = True,
) -> dict[str, pd.DataFrame]:
    """Выгружает корзину per-symbol с инкрементальным кэшем.

    Каждый инструмент, успешно скачанный, сразу пишется в
    cache_dir/{sym}.parquet. Повторный запуск читает готовое из кэша
    и докачивает только недостающее — оплаченные данные не теряются
    при обрыве на середине корзины.

    Args:
        client: databento.Historical.
        symbols: Корневые символы (CL, NG, ...).
        start: Дата начала.
        end: Дата конца.
        schema: ohlcv-1h (для H4) / ohlcv-1d.
        dataset: Датасет Databento.
        cache_dir: Каталог per-symbol кэша. None — без кэша (в памяти).
        with_m2: Тянуть ли второй контракт (.c.1) для carry.

    Returns:
        dict корневой_символ -> DataFrame (open/high/low/close/volume
        [+close_m2]). Нерезолвнутые/пустые символы отсутствуют в dict.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    base = ["open", "high", "low", "close", "volume"]

    for sym in symbols:
        cache_path = (cache_dir / f"{sym}.parquet"
                      if cache_dir is not None else None)
        if cache_path is not None and cache_path.exists():
            print(f"  {sym}: из кэша {cache_path}")
            out[sym] = pd.read_parquet(cache_path)
            continue

        print(f"  {sym}: выгрузка...")
        m1 = _fetch_one_stream(client, dataset, f"{sym}.c.0", schema,
                               start, end)
        if m1 is None:
            continue  # символ не в датасете — в корзину не попадёт
        df = m1[base].copy()

        if with_m2:
            m2 = _fetch_one_stream(client, dataset, f"{sym}.c.1", schema,
                                   start, end)
            if m2 is not None:
                df["close_m2"] = m2["close"].reindex(df.index)
            else:
                print(f"    {sym}: нет M2 — carry не будет посчитан")

        df = df[~df.index.duplicated(keep="last")]
        out[sym] = df
        if cache_path is not None:
            df.to_parquet(cache_path)
            print(f"    {sym}: закэшировано ({len(df)} баров)")

    resolved = list(out.keys())
    dropped = [s for s in symbols if s not in resolved]
    print(f"\n  Итог: {len(resolved)} инструментов "
          f"({', '.join(resolved)})")
    if dropped:
        print(f"  Выпали (не в датасете): {', '.join(dropped)}")
    return out
