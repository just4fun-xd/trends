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
    "Cotton": "CT=F", "Palladium": "PA=F", "Platinum": "PL=F",
    # Аудит 2026-07: раньше стоял "Zinc": "ZN=F" — но ZN=F на Yahoo это
    # 10-Year T-Note futures, НЕ цинк (у цинка LME-листинг, нормального
    # yf-тикера нет). Казначейские ноты сидели в commodity-корзине под
    # видом металла. Заменён на платину. Сверить состав корзины со
    # старым проектом перед сравнением портфельных чисел!
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

# Крипто-корзина (yfinance -USD пары). Имена совпадают с локальными
# прогонами 2026-07f. Для intraday (H4/H1) — ccxt-тракт
# (data/ccxt_source.py + scripts/fetch_ccxt.py), yfinance intraday
# ограничен 730 днями и ненадёжен.
CRYPTO_YF = {
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "BNB": "BNB-USD",
    "XRP": "XRP-USD", "Solana": "SOL-USD", "Cardano": "ADA-USD",
    "Tron": "TRX-USD", "Litecoin": "LTC-USD", "Polkadot": "DOT-USD",
    "Chainlink": "LINK-USD", "Bitcoin Cash": "BCH-USD",
    "Stellar": "XLM-USD", "Cosmos": "ATOM-USD", "Near": "NEAR-USD",
    "Avalanche": "AVAX-USD",
    # Uniswap (UNI-USD) исключён: yfinance-история содержит артефакт
    # листинга/сплита, дающий +4 000 000% доходности и засоряющий
    # LOO-отчёты. Вернуть только после чистки данных или через ccxt.
}

# Та же корзина в нотации биржевых пар (USDT-споты) для --source ccxt
# (H4/H1 через data/ccxt_source.CCXTSource). Файлы:
# data/crypto/{SYMBOL}_{TF}.parquet, выгрузка scripts/fetch_ccxt.py.
CRYPTO_CCXT = {
    "Bitcoin": "BTC-USDT", "Ethereum": "ETH-USDT", "BNB": "BNB-USDT",
    "XRP": "XRP-USDT", "Solana": "SOL-USDT", "Cardano": "ADA-USDT",
    "Tron": "TRX-USDT", "Litecoin": "LTC-USDT", "Polkadot": "DOT-USDT",
    "Chainlink": "LINK-USDT", "Bitcoin Cash": "BCH-USDT",
    "Stellar": "XLM-USDT", "Cosmos": "ATOM-USDT", "Near": "NEAR-USDT",
    "Avalanche": "AVAX-USDT",
}

# Databento continuous futures (roll-adjusted, из parquet-панелей).
# 2026-07: расширено до фактически выгруженной H4-корзины (12
# инструментов; scripts/fetch_databento_futures.py --interval 4h).
# Softs (KC/CC/SB — ICE, не GLBX.MDP3) и CT (нет данных на ключе) не
# резолвятся этим трактом и сюда не входят. PA/PL присутствуют в
# списке (валидны на дневках), но на H4 дают патологически низкое
# покрытие (native < 15%, см. docs/REGIME_MAP.md) — вырезай их через
# --exclude PA,PL при H4-прогонах.
COMMODITY_DATABENTO = [
    "CL", "NG", "GC", "SI", "HG", "ZW", "ZC", "ZS", "ZL", "ZM",
    "PA", "PL",
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
