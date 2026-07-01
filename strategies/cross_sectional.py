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
) -> pd.DataFrame:
    """Веса из ранга: лонг топ-фракции, (опц.) шорт дно-фракции.

    Args:
        score: Матрица скоров ранжирования (даты × инструменты).
        top_frac: Доля корзины в каждую ногу (0.2 = топ/дно 20%).
        market_neutral: True -> лонг топ + шорт дно (нейтраль);
            False -> только лонг топ (long-only).

    Returns:
        DataFrame весов. Каждая нога equal-weight, суммы нормированы.
    """
    weights = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    n = score.shape[1]
    k = max(1, int(n * top_frac))

    for date in score.index:
        row = score.loc[date].dropna()
        if len(row) < 2 * k:
            continue
        ranked = row.sort_values(ascending=False)
        longs = ranked.index[:k]
        weights.loc[date, longs] = 1.0 / k
        if market_neutral:
            shorts = ranked.index[-k:]
            weights.loc[date, shorts] = -1.0 / k
    return weights


def dual_momentum(
    prices: pd.DataFrame,
    lookback: int = 126,
    skip: int = 21,
    top_frac: float = 0.2,
    market_neutral: bool = True,
    abs_filter_sma: int | None = 200,
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
    weights = _rank_weights(score, top_frac, market_neutral)

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
    return _rank_weights(carry, top_frac, market_neutral)


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
