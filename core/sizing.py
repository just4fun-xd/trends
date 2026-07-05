"""Реестр сайзеров: единый выбор realized-VT / GARCH-VT для раннеров.

Проблема, которую решает модуль: core/garch.py был готов, но НЕ был
подключён ни к одному раннеру — флаг --vt везде жёстко брал
vol_target_size (rolling realized). Теперь у всех раннеров один
контракт:

    sizer = make_sizer("garch", target_vol=0.25)
    pos = raw_signal * sizer(bars)

Имена реестра:
    realized — rolling std за 30 баров (vol_target_size, статус-кво);
    garch    — GARCH(1,1) one-step-ahead прогноз (garch_vol_target_size).

Оба сайзера делят потолок плеча и буфер ребалансировки (аудит 2026-07),
поэтому A/B «realized vs garch» изолирует ровно одну переменную —
оценку волатильности в знаменателе.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from core.bars import Bars
from core.engine import vol_target_size
from core.garch import garch_vol_target_size

SizerFn = Callable[[Bars], pd.Series]

SIZERS: dict[str, Callable] = {
    "realized": vol_target_size,
    "garch": garch_vol_target_size,
}


def make_sizer(
    name: str = "realized",
    target_vol: float = 0.15,
    max_leverage: float = 2.0,
    buffer: float = 0.10,
) -> SizerFn:
    """Фабрика сайзера по имени реестра.

    Args:
        name: 'realized' или 'garch'.
        target_vol: Целевая годовая волатильность.
        max_leverage: Потолок множителя.
        buffer: Мёртвая зона ребалансировки.

    Returns:
        Функция Bars -> Series множителей позиции.

    Raises:
        KeyError: Неизвестное имя сайзера.
    """
    if name not in SIZERS:
        raise KeyError(
            f"нет сайзера {name!r}; доступны: {sorted(SIZERS)}"
        )
    base = SIZERS[name]

    def sizer(bars: Bars) -> pd.Series:
        """Множитель позиции для данного инструмента."""
        return base(
            bars, target_vol=target_vol,
            max_leverage=max_leverage, buffer=buffer,
        )

    return sizer


def portfolio_vol_target(
    returns: pd.Series,
    target_vol: float = 0.20,
    window: int = 63,
    max_leverage: float = 4.0,
    bars_per_year: float = 252.0,
    funding_rate: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Масштабирует ряд ПОРТФЕЛЬНЫХ доходностей к целевой воле.

    Рычаг прибыли, который диагностировался портфельным DD-критерием,
    но не был реализован: комбо sleeve'ов после vol-parity работает на
    единицах процентов годовой волы (DD −1.6% за 7 лет = риск-бюджет
    задействован на ~5%), и Sharpe 1.2 превращается в +1.3%/год.
    Плечо на УРОВНЕ ПОРТФЕЛЯ доводит волу до target, конвертируя
    Sharpe в доходность: E[ret] ≈ Sharpe × target_vol.

    Без look-ahead: плечо бара t считается по воле окна, сдвинутого на
    1 бар. Стоимость фондирования (2026-07e): leverage_sweep на
    реальных данных проекта показал плечо 12+ как «максимум в рамках
    DD<40%», но это плечо занимает 11x капитала — при funding_rate
    ~5-6%/год это ~55-65% капитала В ГОД только на проценты по займу,
    способные съесть всю показанную доходность. Стоимость вычитается
    ПРОПОРЦИОНАЛЬНО ЗАЁМНОЙ ЧАСТИ (leverage − 1), не всему плечу —
    первый доллар позиции твой собственный капитал, процент платится
    только за занятые (leverage−1) доллара. Издержки внутри sleeve'ов
    (комиссии, спреды на ребалансировке позиций) здесь НЕ дублируются
    — это отдельная статья, уже учтённая движками sleeve'ов.

    Args:
        returns: Побарные доходности портфеля (после издержек ног).
        target_vol: Целевая годовая волатильность портфеля.
        window: Окно оценки реализованной волы (баров).
        max_leverage: Потолок портфельного плеча.
        bars_per_year: Баров в году для аннуализации.
        funding_rate: Годовая ставка фондирования ЗАЁМНОЙ части
            плеча (напр. 0.05 = 5%/год). 0.0 = не учитывать (как
            раньше — верхняя граница без costs). Типичный диапазон
            для маржинальных счетов брокеров/фьючерсных клиринговых
            депозитов: 0.04-0.08 в зависимости от валюты и брокера.

    Returns:
        (scaled, leverage): масштабированные доходности (после
        вычета funding, если funding_rate > 0) и ряд плеча.
    """
    vol = returns.rolling(window).std() * (bars_per_year ** 0.5)
    lev = (target_vol / vol.where(vol > 1e-12)).clip(
        upper=max_leverage)
    lev = lev.shift(1).fillna(0.0)
    gross = returns * lev
    if funding_rate > 0:
        daily_funding = funding_rate / bars_per_year
        borrowed = (lev - 1.0).clip(lower=0.0)  # только заёмная часть
        gross = gross - borrowed * daily_funding
    return gross, lev


def breakeven_funding_rate(
    returns: pd.Series,
    target_vol: float = 0.20,
    window: int = 63,
    max_leverage: float = 4.0,
    bars_per_year: float = 252.0,
    lo: float = 0.0,
    hi: float = 0.50,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """Ставка фондирования, при которой чистая доходность = 0.

    Прямой ответ на «какая ставка ещё оправдывает это плечо» вместо
    подстановки наугад. Бинарный поиск по net_return(funding_rate) —
    монотонно убывающая функция ставки (больше costs -> меньше
    доход), поэтому корень единственный в разумном диапазоне.

    ВАЖНО — область применимости этой ставки (см. модуль-докстринг
    portfolio_vol_target): модель «funding_rate на (leverage-1)»
    корректна для АКЦИЙ на марже (реальный займ у брокера). Для
    ФЬЮЧЕРСОВ она НЕ соответствует механике рынка — плечо там не
    заём наличных, а требование к гарантийному обеспечению;
    стоимость переноса позиции уже встроена в цену контракта
    (cost-of-carry/roll yield) и учтена в P&L самой стратегии.
    Применение этой функции к фьючерсному sleeve'у даёт число,
    отвечающее на другой вопрос: «какова была бы экономика, если бы
    это было akции на марже» — полезно как ориентир, но не как
    буквальная ставка для сырьевой ноги.

    Args:
        returns: Побарные доходности портфеля (после издержек ног).
        target_vol: Целевая годовая волатильность портфеля.
        window: Окно оценки реализованной волы (баров).
        max_leverage: Потолок портфельного плеча.
        bars_per_year: Баров в году.
        lo: Нижняя граница поиска (обычно 0.0).
        hi: Верхняя граница поиска. Если чистая доходность остаётся
            положительной даже при этой ставке — возвращается hi как
            нижняя оценка (истинный breakeven выше).
        tol: Точность по ставке (в долях, напр. 1e-4 = 0.01%).
        max_iter: Максимум итераций бисекции.

    Returns:
        Ставка фондирования (годовая, доля), при которой суммарная
        доходность за период обнуляется. 0.0, если стратегия убыточна
        уже без funding; hi, если прибыльна даже при funding=hi.
    """
    def net_return(fr: float) -> float:
        scaled, _ = portfolio_vol_target(
            returns, target_vol=target_vol, window=window,
            max_leverage=max_leverage, bars_per_year=bars_per_year,
            funding_rate=fr,
        )
        eq = (1.0 + scaled).cumprod()
        return float(eq.iloc[-1] - 1.0)

    if net_return(lo) <= 0:
        return 0.0
    if net_return(hi) > 0:
        return hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if net_return(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0
