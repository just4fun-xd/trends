"""Контракт данных между источником и стратегией.

Центральное архитектурное решение проекта. Стратегия получает не «широкий
DataFrame» (в который можно случайно залезть в будущие строки, объём или
посторонние колонки — источник look-ahead мимо единственного shift(1)),
а узкий типизированный объект Bars с полями open/high/low/close.

Старый close-only контракт становится частным случаем: bars.close — та же
pd.Series, что раньше приходила стратегии напрямую. Портирование
EMA-стратегий сводится к механической замене close -> bars.close.

Донечиан и ATR-стоп получают high/low НАТИВНО, без синтетики. Это ровно
та неточность, которую вычищаем: пробой по фейковому хаю — не тот пробой.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Bars:
    """Один инструмент, выровненные OHLCV-ряды на общем DatetimeIndex.

    Frozen — стратегия не может подменить данные под собой. Все ряды
    делят один индекс (проверяется в __post_init__). volume опционален:
    на фьючерсах/CFD из yfinance он ненадёжен и помечен None.

    Attributes:
        open: Цена открытия бара.
        high: Максимум бара. Нужен Дончиану/ATR — реальный, не синтетика.
        low: Минимум бара.
        close: Цена закрытия. Основной ряд для большинства сигналов.
        volume: Объём или None, если источник ненадёжен.
        bars_per_year: Число баров в году для этого таймфрейма. 252 для
            дневных, 252*6 для H4 фьючерсов и т.д. Используется движком
            для аннуализации волатильности — НЕ хардкодится больше нигде.
        symbol: Тикер/название инструмента (для логов и отчётов).
    """

    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    bars_per_year: float
    volume: pd.Series | None = None
    symbol: str = ""

    def __post_init__(self) -> None:
        """Проверяет выравнивание рядов и монотонность индекса."""
        ref = self.close.index
        for name in ("open", "high", "low"):
            series = getattr(self, name)
            if not series.index.equals(ref):
                raise ValueError(
                    f"Bars: индекс {name!r} не совпадает с close "
                    f"({self.symbol})"
                )
        if self.volume is not None and not self.volume.index.equals(ref):
            raise ValueError(
                f"Bars: индекс volume не совпадает ({self.symbol})"
            )
        if not ref.is_monotonic_increasing:
            raise ValueError(f"Bars: индекс не отсортирован ({self.symbol})")

    def __len__(self) -> int:
        """Число баров."""
        return len(self.close)

    @property
    def index(self) -> pd.DatetimeIndex:
        """Общий временной индекс всех рядов."""
        return self.close.index

    def returns(self) -> pd.Series:
        """Простые дневные (побарные) доходности по close.

        Returns:
            close / close.shift(1) - 1. Первый бар NaN.
        """
        return self.close / self.close.shift(1) - 1

    def true_range(self) -> pd.Series:
        """True Range по Wilder — основа ATR-стопов Дончиана.

        TR = max(high-low, |high-prev_close|, |low-prev_close|).
        Считается на РЕАЛЬНЫХ high/low, а не на синтетике из close.

        Returns:
            Ряд True Range, первый бар = high-low (нет prev_close).
        """
        prev_close = self.close.shift(1)
        hl = self.high - self.low
        hc = (self.high - prev_close).abs()
        lc = (self.low - prev_close).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr

    def atr(self, period: int = 20) -> pd.Series:
        """Average True Range (Wilder EMA сглаживание).

        Args:
            period: Окно сглаживания ATR.

        Returns:
            Ряд ATR той же длины, что и close.
        """
        return self.true_range().ewm(alpha=1.0 / period, adjust=False).mean()

    @classmethod
    def from_close(
        cls,
        close: pd.Series,
        bars_per_year: float = 252.0,
        symbol: str = "",
    ) -> "Bars":
        """Строит Bars из одного close-ряда (обратная совместимость).

        Для стратегий и тестов, которым реально нужен только close.
        high/low/open заполняются close — но эти Bars НЕЛЬЗЯ отдавать
        Дончиану/ATR (пробой по close-как-high бессмысленен). Помощник
        для миграции EMA-трека и синтетических тестов, не для продакшна
        breakout-стратегий.

        Args:
            close: Ряд цен закрытия.
            bars_per_year: Баров в году (252 дневные по умолчанию).
            symbol: Название инструмента.

        Returns:
            Bars, где open=high=low=close.
        """
        return cls(
            open=close.copy(),
            high=close.copy(),
            low=close.copy(),
            close=close.copy(),
            bars_per_year=bars_per_year,
            volume=None,
            symbol=symbol,
        )

    def slice(self, start=None, end=None) -> "Bars":
        """Возвращает Bars, обрезанные по датам [start, end].

        Args:
            start: Левая граница (включительно) или None.
            end: Правая граница (включительно) или None.

        Returns:
            Новый Bars с обрезанными рядами.
        """
        mask = pd.Series(True, index=self.index)
        if start is not None:
            mask &= self.index >= pd.Timestamp(start)
        if end is not None:
            mask &= self.index <= pd.Timestamp(end)
        vol = self.volume[mask] if self.volume is not None else None
        return Bars(
            open=self.open[mask],
            high=self.high[mask],
            low=self.low[mask],
            close=self.close[mask],
            bars_per_year=self.bars_per_year,
            volume=vol,
            symbol=self.symbol,
        )


# Стандартные значения bars_per_year по таймфреймам.
# Фьючерсы торгуются ~23ч/день; H4 => ~6 баров/день. Акции ~6.5ч => H1~7.
BARS_PER_YEAR = {
    "1d": 252.0,
    "4h": 252.0 * 6,
    "1h": 252.0 * 23,
    "30m": 252.0 * 46,
    "15m": 252.0 * 92,
}


def infer_bars_per_year(interval: str) -> float:
    """Возвращает bars_per_year для строки интервала.

    Args:
        interval: Строка таймфрейма ('1d', '4h', '1h', ...).

    Returns:
        Число баров в году. Дефолт 252 для неизвестных интервалов.
    """
    return BARS_PER_YEAR.get(interval, 252.0)
