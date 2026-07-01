"""Databento-бэкенд DataSource — читает ЛОКАЛЬНЫЕ parquet-панели.

Архитектурное решение (согласовано): горячий путь бэктеста НЕ ходит в
сеть. Живой API-pull вынесен в отдельный ручной скрипт
scripts/fetch_databento.py, который запускается при расширении корзины
(7 -> 20-30 инструментов) и пишет parquet. Так:
  - бэктест воспроизводим и офлайн;
  - квота Databento не жжётся на каждом прогоне;
  - данные версионируются как parquet.

Схема панели (восстановлена из DUALMOM_RESULTS.md, расширена H/L):
  data/panels/panel_open.parquet    — цены открытия
  data/panels/panel_high.parquet    — максимумы (НУЖНЫ Дончиану, новое)
  data/panels/panel_low.parquet     — минимумы (новое)
  data/panels/panel_close.parquet   — цены закрытия
  data/panels/panel_volume.parquet  — объём
  data/panels/panel_rollyield.parquet — carry (M1-M2)/M2, отдельным треком
  data/panels/panel_native.parquet  — маска торгуемых дней (bool)

Каждый parquet: индекс — union-календарь (DatetimeIndex), колонки —
символы инструментов. Панели выровнены по одному индексу.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.bars import infer_bars_per_year
from data.source import DataSource

# Поля панели -> имя файла. high/low добавлены к исходной схеме, потому
# что старый close-only контракт их не нёс — та самая вычищаемая неточность.
PANEL_FIELDS = ("open", "high", "low", "close", "volume")


class DatabentoSource(DataSource):
    """Источник roll-adjusted continuous futures из локального parquet.

    Читает выровненные панели, собранные scripts/fetch_databento.py.
    Один инструмент = одна колонка во всех панелях. Carry и native-маска
    доступны отдельными методами (нужны кросс-секционным стратегиям).
    """

    def __init__(self, panel_dir: str = "data/panels") -> None:
        """Инициализирует источник, лениво загружает панели при первом load.

        Args:
            panel_dir: Каталог с panel_*.parquet файлами.
        """
        self.panel_dir = Path(panel_dir)
        self._panels: dict[str, pd.DataFrame] = {}

    def _panel(self, field: str) -> pd.DataFrame:
        """Лениво загружает и кэширует панель одного поля.

        Args:
            field: Одно из PANEL_FIELDS.

        Returns:
            DataFrame: индекс — даты, колонки — символы.

        Raises:
            FileNotFoundError: Если parquet поля отсутствует.
        """
        if field not in self._panels:
            path = self.panel_dir / f"panel_{field}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Панель не найдена: {path}. Собери её через "
                    f"python -m scripts.fetch_databento"
                )
            self._panels[field] = pd.read_parquet(path)
        return self._panels[field]

    def _fetch_raw(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        """Собирает OHLCV одного символа из панелей в единый DataFrame.

        Args:
            symbol: Колонка в панелях (напр. 'CL', 'NG', 'GC').
            start: Начало.
            end: Конец.
            interval: Таймфрейм (панели обычно '1d'; H4 — своя панель).

        Returns:
            DataFrame с колонками open/high/low/close/volume для символа.

        Raises:
            KeyError: Если символ отсутствует в панели close.
        """
        cols = {}
        for field in PANEL_FIELDS:
            try:
                panel = self._panel(field)
            except FileNotFoundError:
                if field == "volume":
                    continue  # volume опционален
                raise
            if symbol not in panel.columns:
                if field == "volume":
                    continue
                raise KeyError(
                    f"{symbol} нет в panel_{field} "
                    f"(есть: {list(panel.columns)[:8]}...)"
                )
            cols[field] = panel[symbol]

        df = pd.DataFrame(cols)
        df = df[(df.index >= pd.Timestamp(start))]
        return df

    def load_carry(self, symbols: list[str]) -> pd.DataFrame:
        """Возвращает carry-панель (M1-M2)/M2 для списка символов.

        Отдельный трек: carry не часть OHLC-контракта, но нужен
        double-sort и carry-ранжированию. Держится отдельной панелью.

        Args:
            symbols: Инструменты для выборки.

        Returns:
            DataFrame carry: индекс — даты, колонки — символы.
        """
        panel = self._panel_optional("rollyield")
        return panel[symbols] if panel is not None else pd.DataFrame()

    def load_native_mask(self, symbols: list[str]) -> pd.DataFrame:
        """Возвращает маску торгуемых дней (native-маска).

        Не торгуемые дни исключаются из кросс-секционного ранжирования —
        маска запекается в веса, а не фильтрует пост-фактум.

        Args:
            symbols: Инструменты.

        Returns:
            Bool-DataFrame: True = день торговался для инструмента.
        """
        panel = self._panel_optional("native")
        return panel[symbols] if panel is not None else pd.DataFrame()

    def _panel_optional(self, field: str) -> pd.DataFrame | None:
        """Загружает панель, возвращая None если файла нет.

        Args:
            field: Имя поля панели.

        Returns:
            DataFrame или None.
        """
        try:
            return self._panel(field)
        except FileNotFoundError:
            return None

    def load_panel_close(self, symbols: list[str]) -> pd.DataFrame:
        """Прямой доступ к close-панели (для портфельного движка).

        Кросс-секционные стратегии работают с матрицей close всех
        инструментов сразу, а не с Bars по одному. Отдаём срез панели.

        Args:
            symbols: Инструменты.

        Returns:
            DataFrame close: индекс — даты, колонки — символы.
        """
        return self._panel("close")[symbols]

    @staticmethod
    def bars_per_year(interval: str = "1d") -> float:
        """bars_per_year для интервала панели.

        Args:
            interval: Таймфрейм панели.

        Returns:
            Число баров в году.
        """
        return infer_bars_per_year(interval)
