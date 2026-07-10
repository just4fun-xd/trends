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


# --- Боевые корзины ног (2026-07i, приоритет 1 роадмапа) ---
# Источник состава: instrument_contribution на ДВУХ источниках, не
# «из головы». Пересмотр — раз в квартал, не чаще (иначе подгонка).

# Тренд-нога сырьевого v2 (donchian_vt, чистая корзина без балласта).
DONCH_CORE_COMM = ["CL", "ZS", "SI", "ZL", "HG", "GC"]

# MR-нога сырьевого v2 (mr_lowvol).
MRLV_CORE_COMM = ["CL", "NG", "SI", "ZW", "ZC", "GC"]

# Балласт крипты — общий для обеих стратегий ансамбля
# (donchian_vt + tr_ichimoku), вырезается из корзины.
CRYPTO_BALLAST = ["Litecoin", "Cosmos"]

# Боевая крипто-корзина = полная минус балласт (имена общие для
# CRYPTO_YF и CRYPTO_CCXT — ключи совпадают).
CRYPTO_CORE = [n for n in CRYPTO_CCXT if n not in CRYPTO_BALLAST]


# --- Типизированные корзины по режиму (REGIME_MAP_2026-07f) ---
# Вердикты по совпадению LOO-направления на ДВУХ источниках.
# ТРЕНДОВЫЕ: держатели в тренд-ноге, балласт/нейтрал в MR.
# MR: держатели в реверсии, балласт в тренде.
# ФЛЕТ (нейтральные): ни одна нога не выигрывает устойчиво — кандидаты
# на исключение или отдельный аппарат.

COMM_TREND_ASSETS = ["GC", "CL", "ZL", "HG"]
COMM_MR_ASSETS = ["NG", "SI", "ZW", "ZC", "ZS"]
COMM_FLAT_ASSETS = ["PA", "PL", "ZM"]

# yfinance-эквиваленты (полные имена из COMMODITY_YF).
COMM_TREND_ASSETS_YF = ["Gold", "Crude Oil", "Soybean Oil", "Copper"]
COMM_MR_ASSETS_YF = [
    "Natural Gas", "Silver", "Wheat", "Corn", "Soybeans",
]
COMM_FLAT_ASSETS_YF = ["Palladium", "Platinum", "Soybean Meal"]


# --- Рабочие корзины стратегий-кандидатов (2026-07j скрининг) ---
# Из instrument_contribution на ДВУХ источниках: пересечение «держать»
# минус балласт, подтверждённый на обоих. Это НЕ чемпионы — это
# рабочие некоррелированные кандидаты в ансамбль (цель: пул, не корона).
HHHL_CORE_COMM = ["CL", "SI", "HG", "ZS", "GC", "NG"]   # tr3_hh_hl
RIBBON_CORE_COMM = ["ZC", "SI", "ZL", "CL", "GC"]        # tr3_ribbon

# Именованные корзины для CLI: --include @ИМЯ / --exclude @ИМЯ.
# Ключ — имя константы, значение — список тикеров ИЛИ имён корзины.
NAMED_BASKETS = {
    "DONCH_CORE_COMM": DONCH_CORE_COMM,
    "MRLV_CORE_COMM": MRLV_CORE_COMM,
    "HHHL_CORE_COMM": HHHL_CORE_COMM,
    "RIBBON_CORE_COMM": RIBBON_CORE_COMM,
    "CRYPTO_CORE": CRYPTO_CORE,
    "CRYPTO_BALLAST": CRYPTO_BALLAST,
    "COMM_TREND": COMM_TREND_ASSETS,
    "COMM_MR": COMM_MR_ASSETS,
    "COMM_FLAT": COMM_FLAT_ASSETS,
    "COMM_TREND_YF": COMM_TREND_ASSETS_YF,
    "COMM_MR_YF": COMM_MR_ASSETS_YF,
    "COMM_FLAT_YF": COMM_FLAT_ASSETS_YF,
}


def resolve_symbols(spec: str) -> list[str]:
    """Разворачивает CLI-спецификацию активов в список токенов.

    Поддерживает: 'CL,GC,ZS' (прямой список), '@DONCH_CORE_COMM'
    (именованная корзина из NAMED_BASKETS), смешанное
    '@COMM_TREND,NG'. Регистр имён корзин не важен.

    Args:
        spec: Строка из CLI (--include / --exclude).

    Returns:
        Плоский список токенов (тикеры или имена инструментов).

    Raises:
        SystemExit: Если @ИМЯ не найдено в NAMED_BASKETS.
    """
    out: list[str] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token.startswith("@"):
            name = token[1:].upper()
            if name not in NAMED_BASKETS:
                known = ", ".join(sorted(NAMED_BASKETS))
                raise SystemExit(
                    f"нет именованной корзины '@{name}'. "
                    f"Известные: {known}")
            out.extend(NAMED_BASKETS[name])
        else:
            out.append(token)
    return out


def filter_basket(
    basket: dict[str, str],
    include: str | None = None,
    exclude: str | None = None,
) -> dict[str, str]:
    """Фильтрует корзину {имя: тикер} по CLI-спецификациям.

    Токены матчатся и по имени, и по тикеру (без учёта регистра),
    поэтому одна и та же команда работает с 'NG' (databento) и
    'Natural Gas'/'NG=F' (yfinance). Порядок исходной корзины
    сохраняется. Сначала применяется include, затем exclude.

    Args:
        basket: Корзина {имя инструмента: тикер источника}.
        include: Спецификация включения (None = вся корзина).
        exclude: Спецификация исключения (None = ничего).

    Returns:
        Отфильтрованная корзина (тот же формат).

    Raises:
        SystemExit: Если include-токен не нашёл ни одного инструмента
            (защита от опечатки, из-за которой прогон молча пустеет).
    """
    def _norm(s: str) -> str:
        return s.upper().replace("=F", "").replace("-USDT", "") \
            .replace("-USD", "")

    def _match(name: str, ticker: str, tokens: list[str]) -> bool:
        keys = {_norm(name), _norm(ticker)}
        return any(_norm(t) in keys for t in tokens)

    result = dict(basket)
    if include:
        tokens = resolve_symbols(include)
        result = {n: t for n, t in result.items()
                  if _match(n, t, tokens)}
        if not result:
            raise SystemExit(
                f"--include '{include}' не совпал ни с одним "
                f"инструментом корзины: {list(basket)}")
        missing = [tok for tok in tokens
                   if not any(_match(n, t, [tok])
                              for n, t in basket.items())]
        if missing:
            print(f"ВНИМАНИЕ: --include токены без совпадения "
                  f"(пропущены): {missing}")
    if exclude:
        tokens = resolve_symbols(exclude)
        result = {n: t for n, t in result.items()
                  if not _match(n, t, tokens)}
    return result


# --- Реестр стратегий: имя -> (модуль.функция, класс_актива) ---
# Класс актива подсказывает раннеру дефолтную корзину.

STRATEGY_CLASSES = {
    "equity": "Работает на структурном восходящем дрейфе акций",
    "commodity": "Работает на коротких волатильных трендах сырья",
    "range": "Mean-reversion, нужен боковой режим или роутер",
    "closed": "Закрыта отрицательно — для документации провала",
    "stub": "Заглушка под будущий трек (не реализована)",
}
