"""Проверка доступа Databento перед тратой квоты (10.07.26).

Кирилл: ~$100 подаренного баланса, интрадей-акции требуют платного
датасета (EQUS.SUMMARY даёт только ohlcv-1d). Прежде чем выгружать —
проверяем ДЕШЁВО/БЕСПЛАТНО:

  1. list_datasets      — какие датасеты вообще доступны ключу
  2. list_schemas       — есть ли ohlcv-1h/1m на нужном датасете
  3. get_dataset_range  — глубина истории (чтобы не просить пустоту)
  4. get_cost           — СКОЛЬКО СТОИТ выгрузка В ДОЛЛАРАХ до оплаты
  5. list_publishers/get_dataset_condition — опционально

Все вызовы — metadata (дёшево) или get_cost (бесплатно, только оценка).
РЕАЛЬНОЙ выгрузки (get_range) скрипт НЕ делает — баланс не тратится.

Запуск:
    export DATABENTO_API_KEY=db-...
    python -m scripts.check_databento_access
    # оценить конкретную intraday-выгрузку акций:
    python -m scripts.check_databento_access --price-equity-intraday
    # оценить сырьё-1h:
    python -m scripts.check_databento_access --price-futures-1h

Документация стоимости: цена get_range считается по объёму данных
(число записей × байты), поэтому 1m/1h по многим тикерам за годы
может стоить ДОРОГО. get_cost показывает это ДО списания.
"""

from __future__ import annotations

import argparse
import os

# Кандидаты intraday-датасетов акций (у разных ключей разный доступ).
EQUITY_INTRADAY_DATASETS = ("XNAS.ITCH", "DBEQ.BASIC", "EQUS.MINI",
                            "IFEU.IMPACT", "EQUS.SUMMARY")
FUTURES_DATASET = "GLBX.MDP3"
EQUITY_TEST_SYMBOLS = ["AAPL", "MSFT", "NVDA"]
FUTURES_TEST_SYMBOLS = ["CL.n.0", "NG.n.0", "GC.n.0"]

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _client():
    """Databento Historical client (та же обёртка, что в fetch-скриптах)."""
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise SystemExit(
            f"{RED}Нет DATABENTO_API_KEY. "
            f"export DATABENTO_API_KEY=db-...{RESET}")
    try:
        import databento as db
    except ImportError:
        raise SystemExit(f"{RED}pip install databento{RESET}")
    return db.Historical(api_key)


def list_access(client) -> list[str]:
    """1. Датасеты, доступные ключу."""
    print(f"\n{BOLD}1. Доступные датасеты{RESET}")
    try:
        datasets = client.metadata.list_datasets()
    except Exception as exc:                              # noqa: BLE001
        print(f"  {RED}ошибка: {exc}{RESET}")
        return []
    for d in datasets:
        intr = " ← intraday-акции" if d in EQUITY_INTRADAY_DATASETS[:4] else ""
        fut = " ← сырьё" if d == FUTURES_DATASET else ""
        print(f"  {GREEN}{d}{RESET}{intr}{fut}")
    return list(datasets)


def check_schemas(client, dataset: str) -> None:
    """2. Схемы датасета (есть ли intraday ohlcv)."""
    print(f"\n{BOLD}2. Схемы {dataset}{RESET}")
    try:
        schemas = client.metadata.list_schemas(dataset=dataset)
    except Exception as exc:                              # noqa: BLE001
        print(f"  {RED}недоступен: {exc}{RESET}")
        return
    wanted = {"ohlcv-1h", "ohlcv-1m", "ohlcv-1d"}
    for s in schemas:
        mark = f" {GREEN}← есть{RESET}" if s in wanted else ""
        print(f"  {s}{mark}")
    got = wanted & set(schemas)
    if "ohlcv-1h" in got or "ohlcv-1m" in got:
        print(f"  {GREEN}intraday OHLCV доступен ✓{RESET}")
    else:
        print(f"  {YELLOW}intraday OHLCV НЕТ — только дневки{RESET}")


def check_range(client, dataset: str) -> None:
    """3. Диапазон истории датасета."""
    print(f"\n{BOLD}3. Диапазон {dataset}{RESET}")
    try:
        rng = client.metadata.get_dataset_range(dataset=dataset)
        print(f"  {GREEN}{rng}{RESET}")
    except Exception as exc:                              # noqa: BLE001
        print(f"  {RED}{exc}{RESET}")


def price(client, dataset: str, schema: str, symbols: list[str],
          start: str, end: str, stype: str = "raw_symbol") -> None:
    """4. СТОИМОСТЬ выгрузки в долларах (get_cost, бесплатно)."""
    print(f"\n{BOLD}4. Оценка стоимости{RESET}")
    print(f"  {dataset} · {schema} · {symbols} · {start}..{end}")
    try:
        cost = client.metadata.get_cost(
            dataset=dataset, schema=schema, symbols=symbols,
            start=start, end=end, stype_in=stype)
        size = client.metadata.get_billable_size(
            dataset=dataset, schema=schema, symbols=symbols,
            start=start, end=end, stype_in=stype)
        col = GREEN if cost < 5 else (YELLOW if cost < 25 else RED)
        print(f"  объём: {size/1e6:.1f} MB")
        print(f"  {BOLD}СТОИМОСТЬ: {col}${cost:.2f}{RESET} "
              f"(из ~$100 баланса)")
        if cost > 25:
            print(f"  {YELLOW}дорого — сузь символы/окно или бери "
                  f"1h вместо 1m{RESET}")
    except Exception as exc:                              # noqa: BLE001
        print(f"  {RED}get_cost упал: {exc}{RESET}")
        print(f"  {YELLOW}вероятно схема/датасет недоступны ключу{RESET}")


def main() -> None:
    """CLI-диагностика доступа."""
    ap = argparse.ArgumentParser(description="Проверка доступа Databento")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2021-02-01",
                    help="узкое окно для дешёвой оценки цены (месяц)")
    ap.add_argument("--price-equity-intraday", action="store_true",
                    help="оценить стоимость intraday-акций по датасетам")
    ap.add_argument("--price-futures-1h", action="store_true",
                    help="оценить стоимость сырья 1h (GLBX.MDP3)")
    ap.add_argument("--check-equity-symbology", action="store_true",
                    help="проверить, какие тикеры EQUITY_BASKET "
                         "резолвятся на XNAS.ITCH (Nasdaq vs NYSE)")
    ap.add_argument("--full-year", action="store_true",
                    help="считать цену за ГОД (умножь на историю сам)")
    args = ap.parse_args()

    client = _client()
    print(f"{BOLD}=== Проверка доступа Databento (баланс не тратится) "
          f"==={RESET}")

    available = list_access(client)

    # Схемы и диапазоны ключевых датасетов
    for ds in ("EQUS.SUMMARY", FUTURES_DATASET, *EQUITY_INTRADAY_DATASETS[:3]):
        if not available or ds in available:
            check_schemas(client, ds)
            check_range(client, ds)

    end = "2022-01-01" if args.full_year else args.end

    if args.price_equity_intraday:
        print(f"\n{BOLD}═══ ЦЕНА: intraday-акции ═══{RESET}")
        for ds in EQUITY_INTRADAY_DATASETS[:4]:
            for schema in ("ohlcv-1h", "ohlcv-1m"):
                price(client, ds, schema, EQUITY_TEST_SYMBOLS,
                      args.start, end)

    if args.price_futures_1h:
        print(f"\n{BOLD}═══ ЦЕНА: сырьё 1h ═══{RESET}")
        price(client, FUTURES_DATASET, "ohlcv-1h", FUTURES_TEST_SYMBOLS,
              args.start, end, stype="continuous")

    if args.check_equity_symbology:
        print(f"\n{BOLD}═══ Symbology EQUITY_BASKET на XNAS.ITCH "
              f"(Nasdaq-only!) ═══{RESET}")
        try:
            from core.config import EQUITY_BASKET
            tickers = list(EQUITY_BASKET.values())
        except Exception:                                # noqa: BLE001
            tickers = ["AAPL", "MSFT", "NVDA", "JPM", "V", "JNJ", "KO"]
        ok, bad = [], []
        # XNAS.ITCH данные с 2018-05; берём узкое валидное окно.
        for t in tickers:
            try:
                client.metadata.get_cost(
                    dataset="XNAS.ITCH", schema="ohlcv-1h", symbols=[t],
                    start="2023-01-01", end="2023-01-08",
                    stype_in="raw_symbol")
                ok.append(t)
            except Exception:                            # noqa: BLE001
                bad.append(t)
        print(f"  {GREEN}резолвятся ({len(ok)}): {ok}{RESET}")
        if bad:
            print(f"  {YELLOW}НЕ резолвятся ({len(bad)}): {bad}{RESET}")
            print(f"  {YELLOW}вероятно NYSE-листинг — для них нужен "
                  f"XNYS.PILLAR или EQUS-консолидированный intraday. "
                  f"Либо выгрузи Nasdaq-часть на XNAS.ITCH, NYSE-часть "
                  f"отдельно.{RESET}")

    print(f"\n{BOLD}Итог:{RESET} где 'intraday OHLCV доступен ✓' И "
          f"стоимость зелёная — можно выгружать. get_cost показал цену "
          f"ДО списания; умножь на полный список символов/лет для "
          f"реального бюджета.")


if __name__ == "__main__":
    main()
