"""Выгрузка крипто-свечей через ccxt в локальные parquet.

Ручной скрипт (как fetch_databento): запускается при расширении корзины
или таймфреймов, пишет data/crypto/{SYMBOL}_{TF}.parquet. Бэктест потом
читает их офлайн через data/ccxt_source.CCXTSource.

Требует: pip install ccxt

Запуск (Binance спот, вся крипто-корзина, H4 и H1):
    python -m scripts.fetch_ccxt --timeframe 4h --start 2019-01-01
    python -m scripts.fetch_ccxt --timeframe 1h --start 2021-01-01

Один символ:
    python -m scripts.fetch_ccxt --symbols BTC-USDT --timeframe 4h \
        --start 2019-01-01
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

# Корзина по умолчанию — зеркало CRYPTO_YF из core.config, но в
# нотации биржевых пар (USDT-споты Binance).
DEFAULT_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT", "SOL-USDT",
    "ADA-USDT", "TRX-USDT", "LTC-USDT", "DOT-USDT", "LINK-USDT",
    "BCH-USDT", "XLM-USDT", "ATOM-USDT", "NEAR-USDT", "AVAX-USDT",
    "UNI-USDT",
]


def fetch_symbol(exchange, symbol: str, timeframe: str,
                 since_ms: int, limit: int = 1000) -> pd.DataFrame:
    """Скачивает всю историю символа постраничо (ccxt fetch_ohlcv).

    Args:
        exchange: Инициализированный ccxt-экземпляр.
        symbol: Пара в нотации биржи ('BTC/USDT').
        timeframe: '1d' / '4h' / '1h'.
        since_ms: Начало в мс epoch.
        limit: Свечей на страницу (максимум биржи, обычно 1000).

    Returns:
        DataFrame OHLCV, индекс UTC DatetimeIndex, без дублей.
    """
    rows = []
    since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe,
                                     since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        # Следующая страница — после последней полученной свечи.
        since = batch[-1][0] + 1
        if len(batch) < limit:
            break
        # Уважение rate-limit биржи.
        time.sleep(exchange.rateLimit / 1000.0)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    # Последняя свеча может быть неполной (текущий бар) — выкинуть.
    return df.iloc[:-1]


def main() -> None:
    """CLI: выгрузка корзины в data/crypto/*.parquet."""
    p = argparse.ArgumentParser(description="ccxt -> parquet выгрузка")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS,
                   help="пары через дефис: BTC-USDT ETH-USDT ...")
    p.add_argument("--timeframe", default="4h",
                   choices=["1d", "4h", "1h", "30m", "15m"])
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--out-dir", default="data/crypto")
    args = p.parse_args()

    try:
        import ccxt
    except ImportError:
        raise SystemExit("нужен ccxt: pip install ccxt")

    ex_cls = getattr(ccxt, args.exchange)
    exchange = ex_cls({"enableRateLimit": True})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    since_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)

    for sym in args.symbols:
        pair = sym.replace("-", "/")
        print(f"{sym} {args.timeframe} ...", end=" ", flush=True)
        try:
            df = fetch_symbol(exchange, pair, args.timeframe, since_ms)
        except Exception as exc:  # noqa: BLE001
            print(f"ОШИБКА: {exc}")
            continue
        if df.empty:
            print("пусто")
            continue
        path = out / f"{sym.upper()}_{args.timeframe}.parquet"
        df.to_parquet(path)
        print(f"{len(df)} свечей -> {path}")


if __name__ == "__main__":
    main()
