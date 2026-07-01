"""Единый конфиг: корзины инструментов и реестр стратегий.

Один источник истины для тикеров и стратегий. Раннеры импортируют отсюда,
а не хардкодят списки — это убирает исторический дрейф (разные скрипты с
разными корзинами).
"""

from __future__ import annotations

# --- Корзины инструментов ---

# Commodity ETF/фьючерсы через yfinance (дневные бары).
COMMODITY_YF = {
    "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F",
    "Crude Oil": "CL=F", "Natural Gas": "NG=F", "Brent Oil": "BZ=F",
    "Heating Oil": "HO=F", "Gasoline": "RB=F",
    "Wheat": "ZW=F", "Corn": "ZC=F", "Soybeans": "ZS=F",
    "Soybean Oil": "ZL=F", "Soybean Meal": "ZM=F",
    "Coffee": "KC=F", "Cocoa": "CC=F", "Sugar": "SB=F",
    "Cotton": "CT=F", "Palladium": "PA=F", "Zinc": "ZN=F",
}

# Equity-корзина (19 US large-cap) — чистые бары, без roll-проблемы.
EQUITY_BASKET = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Alphabet": "GOOGL",
    "Amazon": "AMZN", "Meta": "META", "Nvidia": "NVDA", "Tesla": "TSLA",
    "JPMorgan": "JPM", "Visa": "V", "Walmart": "WMT",
    "JnJ": "JNJ", "Procter": "PG", "Mastercard": "MA",
    "Home Depot": "HD", "Coca-Cola": "KO", "Merck": "MRK",
    "Pepsi": "PEP", "Costco": "COST", "AMD": "AMD",
}

# Мегакапы, искажающие статистику корзины (исключаются из медиан).
EQUITY_OUTLIERS = ("TSLA", "NVDA")

# Databento continuous futures (roll-adjusted, из parquet-панелей).
COMMODITY_DATABENTO = [
    "CL", "NG", "GC", "SI", "HG", "ZW", "ZC",  # база 7-инстр.
    # расширение до 20-30 добавляется через scripts/fetch_databento.py
]

# Индексы для GARCH vol-targeting (трек 3.1).
INDICES = {"S&P 500": "SPY", "Nasdaq": "QQQ"}


# --- Реестр стратегий: имя -> (модуль.функция, класс_актива) ---
# Класс актива подсказывает раннеру дефолтную корзину.

STRATEGY_CLASSES = {
    "equity": "Работает на структурном восходящем дрейфе акций",
    "commodity": "Работает на коротких волатильных трендах сырья",
    "range": "Mean-reversion, нужен боковой режим или роутер",
    "closed": "Закрыта отрицательно — для документации провала",
    "stub": "Заглушка под будущий трек (не реализована)",
}
