"""CCXT-бэкенд DataSource — читает ЛОКАЛЬНЫЕ parquet-свечи крипты.

То же архитектурное решение, что у Databento: горячий путь бэктеста НЕ
ходит в сеть. Живой pull биржи вынесен в scripts/fetch_ccxt.py, который
пишет по файлу на (символ, таймфрейм):

  data/crypto/{SYMBOL}_{TF}.parquet   — колонки open/high/low/close/volume,
                                        индекс DatetimeIndex UTC.
  Пример: data/crypto/BTC-USDT_4h.parquet

Так бэктест воспроизводим и офлайн, rate-limit биржи не жжётся на
каждом прогоне, данные версионируются.

Интервалы: '1d', '4h', '1h' (любые, какие выгрузил fetch-скрипт).
bars_per_year выводится базовым классом из interval — единый механизм
с остальными источниками (крипта торгуется 24/7: 365 дней, не 252;
infer_bars_per_year это учитывает по флагу crypto24_7).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.source import DataSource


class CCXTSource(DataSource):
    """Источник крипто-свечей из локального parquet (выгрузка ccxt).

    Символы в нотации файла: 'BTC-USDT' (слэш пары заменён дефисом).
    """

    #: Крипта торгуется 24/7 — годовая база 365 дней и 24 часа,
    #: а не фьючерсные 252 дня / 23 часа из core.bars.BARS_PER_YEAR.
    _CRYPTO_BPY = {
        "1d": 365.0,
        "4h": 365.0 * 6,
        "1h": 365.0 * 24,
        "30m": 365.0 * 48,
        "15m": 365.0 * 96,
    }

    def __init__(self, data_dir: str = "data/crypto") -> None:
        """Источник по каталогу parquet-свечей.

        Args:
            data_dir: Каталог файлов {SYMBOL}_{TF}.parquet.
        """
        self._dir = Path(data_dir)

    def load(self, symbol: str, start: str, end: str,
             interval: str = "1d"):
        """Загружает бары и правит bars_per_year под режим 24/7.

        Базовый _normalize аннуализирует фьючерсной таблицей (252д/23ч);
        для крипты это занижает годовую базу и искажает VT/Sharpe.
        """
        import dataclasses
        bars = super().load(symbol, start, end, interval)
        bpy = self._CRYPTO_BPY.get(interval)
        if bpy is not None and bars.bars_per_year != bpy:
            bars = dataclasses.replace(bars, bars_per_year=bpy)
        return bars

    def _fetch_raw(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        """Читает свечи из parquet и режет по [start, end].

        Args:
            symbol: 'BTC-USDT' (или 'BTC/USDT' — нормализуется).
            start: Дата начала (включительно).
            end: Дата конца (включительно).
            interval: '1d' / '4h' / '1h' — суффикс имени файла.

        Returns:
            DataFrame OHLCV с DatetimeIndex.

        Raises:
            FileNotFoundError: Нет файла — сначала прогнать
                scripts/fetch_ccxt.py по этому символу/таймфрейму.
        """
        sym = symbol.replace("/", "-").upper()
        path = self._dir / f"{sym}_{interval}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} не найден. Сначала выгрузи свечи: "
                f"python -m scripts.fetch_ccxt --symbols {sym} "
                f"--timeframe {interval}")
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        return df.loc[str(start):str(end)]
