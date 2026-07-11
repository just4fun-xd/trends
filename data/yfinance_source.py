"""yfinance-бэкенд DataSource.

Отвечает ТОЛЬКО за получение сырого OHLCV. Вся нормализация (MultiIndex,
UTC, обрезка end) — в базовом классе DataSource. Так бэкенд не может
разъехаться с Databento по деталям очистки.

Ограничения yfinance, зашитые в проектные знания:
  - volume на фьючерсах/CFD ненадёжен (не используется в стратегиях).
  - Интрадей (interval != '1d') доступен только за ~последние 730 дней —
    это лимит провайдера, релевантно для H4-требования на equity.
  - Колонки приходят MultiIndex (field, ticker) — разбирается в базе.
"""

from __future__ import annotations

import time

import pandas as pd

from data.source import DataSource


class YFinanceSource(DataSource):
    """Живой источник дневных и интрадей-баров через yfinance.

    Подходит для equity-корзины (чистые бары, без roll-проблемы) и для
    быстрых commodity-ETF прогонов. Для roll-adjusted continuous futures
    использовать DatabentoSource.
    """

    def _fetch_raw(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        """Скачивает сырой OHLCV через yfinance.download.

        Args:
            symbol: Тикер yfinance ('SPY', 'GC=F', ...).
            start: Дата начала.
            end: Дата конца (yfinance трактует как невключительную —
                чинится обрезкой в _normalize).
            interval: '1d', '4h'→маппится, '1h', '30m', '15m'.

        Returns:
            Сырой DataFrame (возможно с MultiIndex-колонками).

        Raises:
            RuntimeError: Если yfinance не установлен или вернул пусто.
        """
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "yfinance не установлен: pip install yfinance"
            ) from exc

        # yfinance не знает '4h' — качаем '1h' и ресемплим в базе? Нет:
        # ресемпл — тоже нормализация, но специфична для yfinance-интрадея.
        # Держим маппинг здесь, ресемпл делаем ниже (до _normalize).
        yf_interval, resample_to = _map_interval(interval)

        # yfinance-интрадей (1h/4h) доступен ТОЛЬКО за ~730 дней. Если
        # запрошено окно длиннее — данные молча обрежутся до 2 лет, что
        # ломает сравнимость с 5-летними databento/ccxt. Предупреждаем
        # (фикс 10.07.26). Для intraday-истории >2 лет: ccxt (крипта)
        # или databento (сырьё/акции).
        if yf_interval != "1d":
            import datetime as _dt
            try:
                s = pd.Timestamp(start)
                if (pd.Timestamp.now(tz=s.tz) - s).days > 725:
                    print(f"  [yfinance] ВНИМАНИЕ: интрадей {interval} "
                          f" limited to ~730 days; окно с {start} "
                          f"обрежется. Для длинной intraday-истории "
                          f"используй ccxt/databento.")
            except Exception:  # noqa: BLE001
                pass

        # RETRY (аудит 2026-07b): yfinance троттлит быстрые повторные
        # запросы и возвращает ПУСТО для живых тикеров («possibly
        # delisted» на GC=F). Молчаливый пропуск ломал сравнимость:
        # каждый прогон считался по РАЗНОЙ корзине. До 4 попыток с
        # экспоненциальной паузой; пусто после всех попыток — ошибка.
        df = None
        last_exc: Exception | None = None
        for attempt in range(4):
            if attempt:
                time.sleep(2.0 * (2 ** (attempt - 1)))  # 2, 4, 8 сек
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=yf_interval,
                    progress=False,
                    auto_adjust=True,
                )
            except Exception as exc:  # noqa: BLE001 — сетевые сбои
                last_exc = exc
                df = None
                continue
            if df is not None and not df.empty:
                break
        if df is None or df.empty:
            raise RuntimeError(
                f"yfinance вернул пусто для {symbol} после 4 попыток "
                f"(троттлинг/сеть){f': {last_exc}' if last_exc else ''}"
            )

        if resample_to is not None:
            df = _resample_ohlcv(df, resample_to)

        return df


def _map_interval(interval: str) -> tuple[str, str | None]:
    """Маппит запрошенный интервал на (yf_interval, resample_rule).

    yfinance принимает '1h'/'4h'/'1d', но НЕ 'h1'/'h4' (нотация крипто-
    раннера). Нормализуем обе формы. '4h' качается как '1h' + ресемпл
    (yfinance не отдаёт 4h нативно, хоть и указывает в списке).

    Args:
        interval: Запрошенный таймфрейм ('h1','h4','1h','4h','1d',...).

    Returns:
        (интервал для yfinance, правило ресемпла или None).
    """
    # h1/h4 (крипто-нотация) -> 1h/4h (нотация yfinance). Фикс 10.07.26:
    # раньше 'h1' уходил в yfinance как есть -> 422 "interval=h1 not
    # supported".
    norm = {"h1": "1h", "h4": "4h", "d1": "1d"}.get(interval, interval)
    if norm == "4h":
        return "1h", "4h"
    return norm, None


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Агрегирует OHLCV на более крупный таймфрейм.

    Args:
        df: Сырой DataFrame (может быть MultiIndex-колонки).
        rule: pandas-правило ресемпла ('4h').

    Returns:
        Ресемпленный DataFrame. Индекс остаётся, MultiIndex сохраняется
        для разбора в _normalize.
    """
    # Разбираем MultiIndex локально, чтобы применить agg по полям.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}
    return df.resample(rule).agg(agg).dropna(how="all")
