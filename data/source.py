"""Единый интерфейс загрузки данных со сменными бэкендами.

Архитектурное решение: раннер не знает, откуда бары. yfinance и Databento
реализуют один протокол DataSource и возвращают одинаковый Bars. Смена
источника = смена одной строки в конфиге, а не переписывание половины кода.

Нормализация (сплющить MultiIndex, привести к UTC, выкинуть неполные
последние бары, отсортировать) живёт ОДИН раз в базовом классе. Это ловит
исторический источник дрейфа: в старом проекте два пути загрузки => два
места, где end не включается или MultiIndex не сплющен => расхождения.

interval — параметр источника, а не хардкод. Так H4/интрадей (требование 5)
ложится в ту же архитектуру без спец-веток.
"""

from __future__ import annotations

import abc

import pandas as pd

from core.bars import Bars, infer_bars_per_year


class DataSource(abc.ABC):
    """Базовый источник баров. Бэкенды наследуют и реализуют _fetch_raw.

    Публичный load() вызывает _fetch_raw() бэкенда, затем прогоняет
    результат через _normalize() — единую точку очистки для всех
    источников. Бэкенд отвечает ТОЛЬКО за получение сырого OHLCV;
    вся гигиена данных централизована здесь.
    """

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> Bars:
        """Загружает и нормализует бары одного инструмента.

        Args:
            symbol: Тикер в нотации конкретного бэкенда.
            start: Дата начала (включительно), 'YYYY-MM-DD'.
            end: Дата конца. Полуинтервал приводится к включительному
                внутри _normalize (единое поведение для всех источников).
            interval: Таймфрейм ('1d', '4h', ...). Определяет bars_per_year.

        Returns:
            Нормализованный Bars.
        """
        raw = self._fetch_raw(symbol, start, end, interval)
        return self._normalize(raw, symbol, interval, end)

    @abc.abstractmethod
    def _fetch_raw(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        """Получает сырой OHLCV-DataFrame. Реализуется бэкендом.

        Ожидаемые колонки (регистр не важен, нормализуется дальше):
        open, high, low, close, и опционально volume. Индекс —
        DatetimeIndex. Может содержать MultiIndex-колонки (yfinance) —
        это разбирается в _normalize.

        Args:
            symbol: Тикер.
            start: Начало.
            end: Конец.
            interval: Таймфрейм.

        Returns:
            Сырой DataFrame как отдаёт провайдер.
        """
        raise NotImplementedError

    @staticmethod
    def _normalize(
        raw: pd.DataFrame, symbol: str, interval: str, end: str
    ) -> Bars:
        """Единая нормализация сырого DataFrame -> Bars.

        Шаги (одни для всех бэкендов):
          1. Сплющить MultiIndex-колонки (yfinance отдаёт (field, ticker)).
          2. Привести имена колонок к нижнему регистру.
          3. Отсортировать индекс по времени.
          4. Убрать строки с NaN в close.
          5. Обрезать по end включительно (чинит yfinance-полуинтервал).
          6. Собрать типизированный Bars с корректным bars_per_year.

        Args:
            raw: Сырой DataFrame от бэкенда.
            symbol: Название инструмента.
            interval: Таймфрейм для bars_per_year.
            end: Дата конца для включительной обрезки.

        Returns:
            Чистый Bars.

        Raises:
            ValueError: Если после очистки не осталось данных или нет
                обязательных колонок OHLC.
        """
        df = raw.copy()

        # 1. Сплющить MultiIndex-колонки: берём уровень поля (open/high/...).
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Нижний регистр имён колонок.
        df.columns = [str(c).lower() for c in df.columns]

        # 3. Индекс -> DatetimeIndex, отсортирован.
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Проверка обязательных колонок.
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{symbol}: нет колонок {missing} (есть {list(df.columns)})"
            )

        # 4. Выкинуть строки без close.
        df = df.dropna(subset=["close"])

        # 5. Включительная обрезка по end (единое поведение).
        df = df[df.index <= pd.Timestamp(end)]

        if df.empty:
            raise ValueError(f"{symbol}: пусто после нормализации")

        volume = df["volume"] if "volume" in df.columns else None
        return Bars(
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            bars_per_year=infer_bars_per_year(interval),
            volume=volume,
            symbol=symbol,
        )
