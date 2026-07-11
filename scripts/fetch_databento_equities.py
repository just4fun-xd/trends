"""Выгрузка US equities из Databento -> parquet-панели (per-sleeve).

Отдельный sleeve от фьючерсов. Пишет в СВОЙ каталог (по умолчанию
data/panels/equities), НИКАКОГО rollyield: у акций нет M1/M2/carry.
Единый тракт сборки/ресемпла/записи переиспользуется из
panels_common.

Датасет: EQUS.SUMMARY — консолидированные EOD-цены (OHLCV) по всем
NMS-биржам и ATS, максимизирует покрытие CTA/UTP SIP. Это полный SIP-
объём, а не частичный IEX-фид (DBEQ.BASIC ~5% ADV) — для честного
бэктеста цены/ликвидность корректны. Schema: ohlcv-1d (дневки) или
ohlcv-1h (для H4, ресемпл в build_panels).

ВАЖНО, что отличает equities-путь от futures:
  1. Symbology. Databento по умолчанию отдаёт instrument_id, а не
     тикер. Для маппинга нужен request_symbology + вставка обратно.
     Здесь это делается через stype_in='raw_symbol' с явным
     watchlist (см. fetch_via_api).
  2. Нет carry. Колонки close_m2 нет -> build_panels не пишет
     rollyield.
  3. Splits/dividends. EQUS.SUMMARY отдаёт СЫРЫЕ цены (не adjusted).
     yfinance даёт adjusted close из коробки. Для честного A/B против
     yfinance-эталона на дневках включи --adjust: тогда цены/объёмы
     корректируются на сплиты через Databento corporate actions API.
     Без --adjust панели содержат raw-цены (сравнивай с raw yfinance).

Запуск:
    export DATABENTO_API_KEY=...
    python -m scripts.fetch_databento_equities \\
        --symbols AAPL MSFT NVDA AMD TSLA ... \\
        --start 2015-01-01 --end 2025-01-01 \\
        --out data/panels/equities --adjust

    # офлайн-проверка труб без ключа:
    python -m scripts.fetch_databento_equities --demo \\
        --out /tmp/eq_demo

Перед первой реальной выгрузкой ПРОВЕРЬ глубину истории EQUS.SUMMARY
на своём ключе (--check-range): для 10-летнего walk-forward нужно,
чтобы датасет покрывал весь период. Если история короче — держи
yfinance-дневки как длинный эталон, а Databento используй под intraday.

Требует: pip install databento. Без ключа/пакета работает --demo.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from scripts import panels_common as pc

DATASET = "EQUS.SUMMARY"          # дневки (EOD), история с 2024-07
# Intraday-акции: XNAS.ITCH — единственный пригодный (история с
# 2018-05, все US-тикеры, ohlcv-1h/1m ~$0). DBEQ.BASIC/EQUS.MINI
# отсеяны: история только с 2023-03 (проверено get_cost 10.07.26).
DATASET_INTRADAY = "XNAS.ITCH"

# Дефолтная корзина ~ твой валидированный equity-юниверс (крупный кап).
# Правь под фактический список в core/universe.
# Дефолтная корзина = боевой EQUITY_BASKET из config (единый источник
# правды). Хардкод расходился (UNH/XOM/CVX vs WMT/MRK), панель не
# совпадала с корзиной раннера -> «нет в panel_open». Фикс 10.07.26.
try:
    from core.config import EQUITY_BASKET as _EQB
    DEFAULT_SYMBOLS = list(_EQB.values())
except Exception:                                        # noqa: BLE001
    DEFAULT_SYMBOLS = [
        "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL", "META",
        "JPM", "V", "WMT", "JNJ", "PG", "HD", "MA", "MRK", "KO",
        "PEP", "COST",
    ]


def check_range(symbols: list[str]) -> None:
    """Печатает доступный диапазон дат EQUS.SUMMARY. Gate перед выгрузкой.

    Дёргает metadata.get_dataset_range — дешёвый вызов, квоту почти
    не тратит. Нужен, чтобы убедиться, что глубины хватает на walk-
    forward до реальной (платной) выгрузки баров.

    Args:
        symbols: Список символов (для справки в выводе; сам диапазон
            датасета от символов не зависит).

    Raises:
        RuntimeError: Нет ключа или пакета databento.
    """
    client = _client()
    rng = client.metadata.get_dataset_range(dataset=DATASET)
    print(f"EQUS.SUMMARY доступный диапазон: {rng}")
    print(f"(проверял под {len(symbols)} символов; "
          f"диапазон датасета от списка не зависит)")


def _client():
    """Создаёт Databento Historical client или падает с инструкцией.

    Returns:
        databento.Historical.

    Raises:
        RuntimeError: Нет DATABENTO_API_KEY или пакета databento.
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
    return db.Historical(api_key)


def fetch_via_api(
    symbols: list[str], start: str, end: str, schema: str = "ohlcv-1d",
    adjust: bool = False, dataset: str = DATASET,
) -> dict[str, pd.DataFrame]:
    """Тянет OHLCV акций из EQUS.SUMMARY с маппингом тикеров.

    Ключевое отличие от futures-пути — symbology. Запрашиваем по
    raw_symbol (тикер), но Databento индексирует ответ instrument_id;
    request_symbology + insert_symbology_json возвращает колонку
    'symbol' с тикером, по которой и раскладываем на per-symbol.

    Args:
        symbols: Тикеры (AAPL, MSFT, ...).
        start: Дата начала.
        end: Дата конца.
        schema: 'ohlcv-1d' (дневки) или 'ohlcv-1h' (для H4).
        adjust: True — скорректировать цены/объёмы на сплиты через
            corporate actions (для A/B против adjusted yfinance).

    Returns:
        dict тикер -> DataFrame (open/high/low/close/volume). Без
        close_m2 — у акций нет carry.

    Raises:
        RuntimeError: Нет ключа/пакета databento.
    """
    client = _client()
    data = client.timeseries.get_range(
        dataset=dataset,
        schema=schema,
        symbols=symbols,          # список тикеров
        stype_in="raw_symbol",    # входная symbology — тикер
        start=start, end=end,
    )
    # Маппинг instrument_id -> тикер обязателен: без него df придёт
    # с instrument_id вместо 'symbol'.
    symbology_json = data.request_symbology(client)
    data.insert_symbology_json(symbology_json, clear_existing=True)
    df_all = data.to_df()

    out: dict[str, pd.DataFrame] = {}
    cols = ["open", "high", "low", "close", "volume"]
    for sym, grp in df_all.groupby("symbol"):
        g = grp[cols].copy()
        g = g[~g.index.duplicated(keep="last")].sort_index()
        out[sym] = g

    missing = [s for s in symbols if s not in out]
    if missing:
        print(f"  ВНИМАНИЕ: нет данных по {missing} — пропущены.")

    if adjust:
        out = _apply_split_adjust(client, out, start, end)
    return out


def _apply_split_adjust(
    client, data: dict[str, pd.DataFrame], start: str, end: str,
) -> dict[str, pd.DataFrame]:
    """Backward-adjust цен/объёма на сплиты (для A/B с adjusted yfinance).

    EQUS.SUMMARY отдаёт сырые цены. yfinance adjusted close уже учёл
    сплиты. Чтобы A/B был честным, применяем тот же backward-adjust:
    цены до даты сплита делятся на кумулятивный коэффициент, объём —
    умножается. Дивиденды здесь НЕ учитываются (yfinance 'Close' без
    auto_adjust тоже сырой по дивидендам — сопоставляй режимы явно).

    ЗАГЛУШКА: точная схема запроса corporate actions зависит от
    подписки/датасета (adjustment/corporate actions API). Оставлена
    honest-заглушка в духе trust-layer: не делает молча неверное.

    Args:
        client: databento.Historical.
        data: dict тикер -> сырой DataFrame.
        start: Дата начала (диапазон corporate actions).
        end: Дата конца.

    Returns:
        dict тикер -> скорректированный DataFrame.

    Raises:
        NotImplementedError: Пока не подключён corporate actions API.
    """
    raise NotImplementedError(
        "Split-adjust не реализован: подключи Databento corporate "
        "actions/adjustment API под свою подписку и примени backward-"
        "adjust (цены /= кум.коэф, объём *= кум.коэф) до даты сплита. "
        "До этого запускай без --adjust и сравнивай с RAW yfinance "
        "('Close', auto_adjust=False), а не с adjusted."
    )


def main() -> None:
    """CLI-точка входа equities-выгрузки."""
    parser = argparse.ArgumentParser(
        description="Выгрузка US equities (EQUS.SUMMARY) -> панели"
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--adjust", action="store_true",
                        help="Split-adjust цен (для A/B с adjusted yf)")
    parser.add_argument("--check-range", action="store_true",
                        help="Только напечатать диапазон истории и выйти")
    pc.add_cli_args(parser)
    # equities-дефолт каталога — свой sleeve.
    parser.set_defaults(out="data/panels/equities")
    args = parser.parse_args()

    if args.check_range:
        check_range(args.symbols)
        return

    # Интервал -> (dataset, schema). 1d берёт EOD-датасет (дёшево,
    # но история с 2024-07!); 4h/1h -> XNAS.ITCH (intraday, с 2018).
    if args.interval == "1d":
        dataset, schema = DATASET, "ohlcv-1d"
    else:
        dataset = DATASET_INTRADAY
        schema = "ohlcv-1h"      # 4h ресемплится из 1h в build_panels
    if args.interval == "1h":
        # 1h — часовые как есть (build_panels не ресемплит на '1h').
        pass

    # Авто-клэмп start к доступному началу датасета. XNAS.ITCH с
    # 2018-05, EQUS.SUMMARY с 2024-07 — дефолтный --start 2015-01-01
    # даёт 422 data_start_before_available_start. Поджимаем и
    # предупреждаем, вместо падения (10.07.26).
    start = args.start
    if not args.demo:
        try:
            rng = _client().metadata.get_dataset_range(dataset=dataset)
            avail = str(rng.get("start", ""))[:10]
            if avail and start < avail:
                print(f"  ВНИМАНИЕ: --start {start} раньше начала "
                      f"{dataset} ({avail}). Поджимаю до {avail}.")
                start = avail
        except Exception:                                # noqa: BLE001
            pass  # клэмп best-effort; если range недоступен — идём как есть

    if args.demo:
        print(f"ДЕМО-режим ({args.interval}): синтетика (не для выводов).")
        data = pc.demo_panels(args.symbols, start, args.end,
                              interval=args.interval, carry=False)
    else:
        print(f"Выгрузка {len(args.symbols)} акций из {dataset} "
              f"({args.interval}, schema={schema}, "
              f"adjust={args.adjust}, start={start})...")
        data = fetch_via_api(args.symbols, start, args.end,
                             schema, adjust=args.adjust, dataset=dataset)

    panels = pc.build_panels(data, interval=args.interval)
    pc.write_panels(panels, args.out)
    print(f"Готово. Equities-панели ({args.interval}) в {args.out}/ — "
          f"укажи как отдельный sleeve в DatabentoSource(panel_dir=...).")


if __name__ == "__main__":
    main()
