"""Кросс-секционные стратегии — портфельный уровень (run_portfolio).

СТАТУС Dual Momentum (из DUALMOM_RESULTS.md): ЗАКРЫТ отрицательно на
ОБОИХ классах. Акции: положительный итог оказался концентрацией в
TSLA/NVDA, а не альфой ранжирования (убери 2 имени -> убыток). Сырьё:
убыточен на train И test — относительный ранг разворачивается резче
абсолютной цены на mean-reverting активах (прошлый чемпион уже
развернулся). Обе причины структурны.

Портировано для честной документации провала + как БАЗА: провалился
momentum-РАНГ, но ранг по carry или OU-z-score может работать (открытый
вопрос). Архитектура ранжирования переиспользуема, сигнал — заменяем.

Возвращают МАТРИЦУ весов (DataFrame) для run_portfolio, НЕ позицию.
Кросс-секционные стратегии нельзя гонять посерийным циклом.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _momentum_score(
    prices: pd.DataFrame, lookback: int = 126, skip: int = 21
) -> pd.DataFrame:
    """Моментум-скор: доходность за lookback с пропуском последних skip.

    Пропуск последнего месяца (skip) — против краткосрочного реверса
    (стандарт momentum-литературы).

    Args:
        prices: Матрица цен close (даты × инструменты).
        lookback: Окно моментума (126д ≈ полгода).
        skip: Пропуск последних дней (21 ≈ месяц).

    Returns:
        DataFrame моментум-скоров той же формы.
    """
    return prices.shift(skip) / prices.shift(lookback) - 1


def _rank_weights(
    score: pd.DataFrame,
    top_frac: float = 0.2,
    market_neutral: bool = True,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """Веса из ранга: лонг топ-фракции, (опц.) шорт дно-фракции.

    Аудит 2026-07: ре-ранжирование теперь ПЕРИОДИЧНОЕ (раз в
    rebalance_every баров, между датами веса удерживаются). Ежедневный
    ре-ранг заставлял границу топ/дно-фракции дрожать (126-дневный
    скор соседних имён почти равен), и с drift-aware издержками это
    дрожание стало платным. Месячный ребаланс — стандарт momentum-
    литературы.

    Args:
        score: Матрица скоров ранжирования (даты × инструменты).
        top_frac: Доля корзины в каждую ногу (0.2 = топ/дно 20%).
        market_neutral: True -> лонг топ + шорт дно (нейтраль);
            False -> только лонг топ (long-only).
        rebalance_every: Период ребаланса в барах (21 ~ месяц);
            1 = старое ежедневное поведение.

    Returns:
        DataFrame весов. Каждая нога equal-weight; между ребалансами
        целевые веса удерживаются (ffill).
    """
    n = score.shape[1]
    k = max(1, int(n * top_frac))

    rebal_dates = score.index[::rebalance_every]
    weights = pd.DataFrame(
        0.0, index=rebal_dates, columns=score.columns
    )
    for date in rebal_dates:
        row = score.loc[date].dropna()
        if len(row) < 2 * k:
            continue
        ranked = row.sort_values(ascending=False)
        longs = ranked.index[:k]
        weights.loc[date, longs] = 1.0 / k
        if market_neutral:
            shorts = ranked.index[-k:]
            weights.loc[date, shorts] = -1.0 / k
    # Между ребалансами держим последние целевые веса.
    return weights.reindex(score.index).ffill().fillna(0.0)


def dual_momentum(
    prices: pd.DataFrame,
    lookback: int = 126,
    skip: int = 21,
    top_frac: float = 0.2,
    market_neutral: bool = True,
    abs_filter_sma: int | None = 200,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """Cross-sectional dual momentum — ЗАКРЫТ отрицательно (документация).

    Ранжируем по моментуму, лонг топ (∧ absolute trend filter), опц.
    шорт дно. Не грааль — карта ограничения: momentum-ранг структурно
    несовместим с mean-reverting активами.

    Args:
        prices: Матрица цен close (даты × инструменты).
        lookback: Окно моментума.
        skip: Пропуск последнего периода.
        top_frac: Доля в каждую ногу.
        market_neutral: Нейтраль (лонг+шорт) или long-only.
        abs_filter_sma: Окно SMA absolute-фильтра (не лонговать имя ниже
            своей SMA); None -> без фильтра.

    Returns:
        Матрица весов для run_portfolio.
    """
    score = _momentum_score(prices, lookback, skip)
    weights = _rank_weights(
        score, top_frac, market_neutral, rebalance_every
    )

    if abs_filter_sma is not None:
        # Не держим лонг, если цена ниже своей SMA (absolute trend gate).
        sma = prices.rolling(abs_filter_sma).mean()
        below = prices < sma
        # Обнуляем лонги (положительные веса) там, где цена ниже SMA.
        long_mask = weights > 0
        weights = weights.mask(long_mask & below, 0.0)
    return weights


def carry_rank(
    carry: pd.DataFrame,
    top_frac: float = 0.2,
    market_neutral: bool = True,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """Ранжирование по carry (M1-M2)/M2 — ОТКРЫТЫЙ трек, не momentum.

    Гипотеза из DUALMOM_RESULTS.md: провалился momentum-ранг, но ранг по
    carry может нести альфу (backwardation vs contango). Переиспользует
    архитектуру ранжирования с carry-сигналом вместо моментума.

    ТРЕБУЕТ carry-панель из Databento (M1/M2) — недоступна в yfinance.

    Args:
        carry: Матрица carry-значений (даты × инструменты).
        top_frac: Доля в каждую ногу.
        market_neutral: Лонг backwardated + шорт contangoed, или long-only.

    Returns:
        Матрица весов для run_portfolio.
    """
    # Высокий carry (backwardation) -> лонг; низкий (contango) -> шорт.
    return _rank_weights(
        carry, top_frac, market_neutral, rebalance_every
    )


# --- ЗАГЛУШКА: Markowitz + Momentum (трек 1.2) ---
def markowitz_momentum(
    prices: pd.DataFrame,
    lookback: int = 126,
    lam: float = 1.0,
    shrinkage: bool = True,
) -> pd.DataFrame:
    """Markowitz с momentum-скором как μ — НЕ РЕАЛИЗОВАН (трек 1.2).

    Планируемая математика:
        max_w  w'μ - λ·w'Σw   s.t. Σw=1, w>=0
    где μ = momentum-скор, Σ = ковариация (Ledoit-Wolf shrinkage).

    Понижен в приоритете после провала Dual Momentum: он тоже
    cross-sectional и опирается на momentum-ранг как μ. Если базовый
    сигнал не несёт альфы, оптимизация весов поверх не спасёт (вывод
    DUALMOM_RESULTS.md, п.2). Реализовывать ПОСЛЕ переосмысления μ
    (carry/OU-скор вместо momentum). Заглушка, не выдумка результата.

    Args:
        prices: Матрица цен.
        lookback: Окно моментума для μ.
        lam: Риск-аверсия λ (явный риск-дайл).
        shrinkage: Ledoit-Wolf усадка ковариации.

    Raises:
        NotImplementedError: Трек 1.2 не пройден (и переосмысляется).
    """
    raise NotImplementedError(
        "markowitz_momentum — заглушка (трек 1.2). Понижен после провала "
        "Dual Momentum: μ на momentum-ранге не несёт альфы (DUALMOM_"
        "RESULTS.md). Реализация ждёт переосмысления μ (carry/OU-скор). "
        "Не выдумываем результат до прохождения трека."
    )


# ============================================================
# Research-треки dual momentum (аудит 2026-07, портированы из EMA1221).
# Базовый dual_momentum закрыт отрицательно; эти три — попытка ВОСКРЕСИТЬ
# трек через МЕХАНИЗМ, а не переоткрыть закрытое. Честная рамка: это
# hedged long-only, НЕ market-neutral (шорт-нога условна). Отвечают на
# вывод «шорт против растущих акций = только бета, не альфа»: включают
# шорт там/тогда, где спред реально положителен. Помечены research —
# требуют walk-forward перед любым боевым статусом.
# ============================================================


def _mom_and_rebal(
    prices: pd.DataFrame, lookback: int, skip: int, rebalance: str,
):
    """Общий блок: 12-1 моментум + даты ребаланса (для research-вариантов).

    Args:
        prices: Матрица цен close.
        lookback: Окно моментума в барах.
        skip: Пропуск последних баров (анти-реверс).
        rebalance: Частота ребаланса pandas ('ME', 'W').

    Returns:
        (mom, rebal_dates): матрица моментума и индекс дат ребаланса.
    """
    past = prices.shift(lookback)
    recent = prices.shift(skip) if skip else prices
    mom = recent / past - 1
    rebal_dates = prices.resample(rebalance).last().index
    return mom, rebal_dates


def _above_sma(prices: pd.DataFrame, window: int | None) -> pd.DataFrame:
    """Маска «цена выше своей SMA» (absolute-trend фильтр шорта).

    Args:
        prices: Матрица цен close.
        window: Окно SMA; None -> маска сплошь False (фильтр выключен).

    Returns:
        Bool-матрица той же формы.
    """
    if window is None:
        return pd.DataFrame(False, index=prices.index,
                            columns=prices.columns)
    return prices > prices.rolling(window).mean()


def dual_momentum_tilt(
    prices: pd.DataFrame, benchmark: pd.Series | None = None,
    lookback: int = 126, top_frac: float = 0.2,
    sma_filter: int | None = 200, rebalance: str = "ME",
    skip: int = 21, short_frac: float = 0.5, market_sma: int = 200,
) -> pd.DataFrame:
    """RESEARCH: long-tilt с динамическим gross шорта.

    Две правки против симметричной базы: (1) асимметрия — лонги в сумме
    +1.0, шорты −short_frac (напр. 0.5): держим большую часть
    структурного дрейфа, но с хеджем; (2) динамический gross — в бычьем
    рынке (benchmark выше своей SMA) шорт-нога ужимается вдвое: платим
    меньше за хедж в бычьи годы. Net long by construction -> бета НЕ ~0
    (это осознанный размен нейтральности на меньший дрэг в бычьи годы).

    Args:
        prices: Матрица цен close (даты × инструменты).
        benchmark: Прокси рынка (напр. SPY). None -> без ужатия.
        lookback: Окно моментума в барах.
        top_frac: Доля корзины в ногу.
        sma_filter: Окно absolute-фильтра шорта (None -> выкл).
        rebalance: Частота ребаланса.
        skip: Пропуск последних баров.
        short_frac: Базовый gross шорт-ноги (лонг = 1.0).
        market_sma: Окно рыночного режим-фильтра.

    Returns:
        Матрица весов для run_portfolio (лонги +1, шорты −short_frac).
    """
    mom, rebal_dates = _mom_and_rebal(prices, lookback, skip, rebalance)
    above = _above_sma(prices, sma_filter)
    if benchmark is not None:
        bench = benchmark.reindex(prices.index).ffill()
        mkt_up = bench > bench.rolling(market_sma).mean()
    else:
        mkt_up = pd.Series(False, index=prices.index)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for dt in rebal_dates:
        sub = mom.loc[:dt]
        if sub.empty:
            continue
        row = sub.iloc[-1].dropna()
        if len(row) < 5:
            continue
        eff = sub.index[-1]
        n = max(1, int(len(row) * top_frac))
        ranked = row.sort_values(ascending=False)
        longs = list(ranked.head(n).index)
        shorts = [s for s in ranked.tail(n).index
                  if not bool(above.loc[eff, s])]
        sf = short_frac * 0.5 if bool(mkt_up.reindex([eff]).iloc[0]) \
            else short_frac
        w = pd.Series(0.0, index=prices.columns)
        if longs:
            w[longs] = 1.0 / len(longs)
        if shorts:
            w[shorts] = -sf / len(shorts)
        weights.loc[dt:] = w.values
    return weights


def dual_momentum_regime(
    prices: pd.DataFrame, benchmark: pd.Series,
    lookback: int = 126, top_frac: float = 0.2,
    sma_filter: int | None = 200, rebalance: str = "ME",
    skip: int = 21, market_sma: int = 200,
) -> pd.DataFrame:
    """RESEARCH: шорт-нога включается ТОЛЬКО в risk-off режиме.

    Лонг-нога всегда (+1). Шорт добавляется, только когда рынок ниже
    своей SMA — хедж появляется именно в медвежьем режиме, где спред
    2022 был положителен, и ВЫКЛЮЧЕН в бычьи годы, где шорт только жёг
    деньги. Самый обоснованный из трёх: концентрирует ценность шорта
    там, где данные говорят, что она есть. Честная рамка для Александра:
    это хеджированный long-only, не market-neutral.

    Args:
        prices: Матрица цен close.
        benchmark: Прокси рынка (SPY) — ОБЯЗАТЕЛЕН здесь.
        lookback: Окно моментума.
        top_frac: Доля корзины в ногу.
        sma_filter: Окно absolute-фильтра шорта.
        rebalance: Частота ребаланса.
        skip: Пропуск последних баров.
        market_sma: Окно рыночного режим-переключателя.

    Returns:
        Матрица весов: long-only в бычьем режиме, long+short в медвежьем.
    """
    mom, rebal_dates = _mom_and_rebal(prices, lookback, skip, rebalance)
    above = _above_sma(prices, sma_filter)
    bench = benchmark.reindex(prices.index).ffill()
    risk_off = bench < bench.rolling(market_sma).mean()

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for dt in rebal_dates:
        sub = mom.loc[:dt]
        if sub.empty:
            continue
        row = sub.iloc[-1].dropna()
        if len(row) < 5:
            continue
        eff = sub.index[-1]
        n = max(1, int(len(row) * top_frac))
        ranked = row.sort_values(ascending=False)
        longs = list(ranked.head(n).index)
        w = pd.Series(0.0, index=prices.columns)
        if longs:
            w[longs] = 1.0 / len(longs)
        if bool(risk_off.reindex([eff]).iloc[0]):
            shorts = [s for s in ranked.tail(n).index
                      if not bool(above.loc[eff, s])]
            if shorts:
                w[shorts] = -1.0 / len(shorts)
        weights.loc[dt:] = w.values
    return weights


def dual_momentum_volscaled(
    prices: pd.DataFrame, lookback: int = 126, top_frac: float = 0.2,
    sma_filter: int | None = 200, rebalance: str = "ME",
    skip: int = 21, vol_window: int = 60,
) -> pd.DataFrame:
    """RESEARCH: равный риск-вклад на имя (inverse-vol внутри ноги).

    Та же конструкция long-top / short-bottom, но вес внутри ноги
    пропорционален 1/волатильность, а не равные доллары. Прыгучее имя
    (Tesla, Nvidia) получает меньший вес, чем спокойное (KO, PG) — ни
    одно волатильное имя не доминирует риск ноги. Прямо адресует
    concentration-артефакт (TSLA/NVDA = 66% P&L), закрывший базовый DM.

    Args:
        prices: Матрица цен close.
        lookback: Окно моментума.
        top_frac: Доля корзины в ногу.
        sma_filter: Окно absolute-фильтра шорта.
        rebalance: Частота ребаланса.
        skip: Пропуск последних баров.
        vol_window: Окно inverse-vol взвешивания.

    Returns:
        Матрица весов: inverse-vol внутри ноги, каждая нога до unit gross.
    """
    mom, rebal_dates = _mom_and_rebal(prices, lookback, skip, rebalance)
    above = _above_sma(prices, sma_filter)
    inv_vol = 1.0 / (prices.pct_change().rolling(vol_window).std()
                     * np.sqrt(252.0))

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for dt in rebal_dates:
        sub = mom.loc[:dt]
        if sub.empty:
            continue
        row = sub.iloc[-1].dropna()
        if len(row) < 5:
            continue
        eff = sub.index[-1]
        n = max(1, int(len(row) * top_frac))
        ranked = row.sort_values(ascending=False)
        longs = list(ranked.head(n).index)
        shorts = [s for s in ranked.tail(n).index
                  if not bool(above.loc[eff, s])]
        iv = inv_vol.loc[eff].replace([np.inf], 0.0).fillna(0.0)
        w = pd.Series(0.0, index=prices.columns)
        if longs:
            lw = iv[longs]
            lw = lw / lw.sum() if lw.sum() > 0 else lw
            w[longs] = lw.values
        if shorts:
            sw = iv[shorts]
            sw = sw / sw.sum() if sw.sum() > 0 else sw
            w[shorts] = -sw.values
        weights.loc[dt:] = w.values
    return weights
