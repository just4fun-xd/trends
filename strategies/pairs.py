"""Парный трейдинг с Kalman-оценкой хедж-беты — RESEARCH, не боевой.

ДВОЙНАЯ ПОМЕТКА (важно для честности проекта):
  1. Математика ВАЛИДНА: Kalman-бета оценивается online, без look-ahead,
     и корректно адресует риск №1 из Karpathy-списка (beta look-ahead —
     статический OLS на всём окне «подглядывает»). Это настоящее
     улучшение НАД статической бетой.
  2. НО край НЕ ПОДТВЕРЖДЁН: OU_RESULTS.md закрыл парный трейдинг —
     сигнал не имеет края даже при НУЛЕВЫХ издержках на дневных
     commodity-данных. Kalman чинит look-ahead беты, но не создаёт
     край там, где его нет. Это ремонт двигателя у машины без колёс.

Поэтому трек — исследовательский: инструмент проверки гипотезы «даёт ли
адаптивная бета край там, где статическая не дала». Прогонять на парах
с ПОДТВЕРЖДЁННОЙ коинтеграцией, не вслепую. Не боевая стратегия до
walk-forward с положительным краем на train И test.

dX = θ(μ−X)dt + σdW для спреда; бета [β,α] — случайное блуждание,
оценка Калмана: predict (inflate P), update по сегодняшней цене.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine


def kalman_beta(
    a: pd.Series, b: pd.Series, delta: float = 1e-4, R: float = 1e-3
) -> pd.Series:
    """Online хедж-бета β[t] (и α[t]) через фильтр Калмана.

    Оценивает меняющуюся во времени связь A[t] = β[t]·B[t] + α[t], где
    состояние [β, α] — случайное блуждание. Оценка в момент t использует
    ТОЛЬКО данные до t включительно — в этом весь смысл против
    статического OLS на всём окне (тот подглядывает).

    Зачем: коинтеграция не фиксирована. При структурном сдвиге (кризис
    2022, дислокация crack-спредов) статическая бета сидит против
    убежавшего спреда, который не вернётся, — систематический убыток.
    Kalman-бета адаптируется к новой связи.

    Два параметра — размен стабильности и адаптивности (источник
    известной чувствительности Kalman pairs):
        delta: масштаб process-noise. Больше -> бета быстрее, шумнее.
        R: дисперсия observation-noise. Больше -> меньше доверия новым
           данным, бета глаже.
    Дефолты (1e-4, 1e-3) — общие значения QuantStart.

    Args:
        a: Цены актива A (лонг-нога), выровнены с b.
        b: Цены актива B (хедж-нога), выровнены с a.
        delta: Масштаб process-noise.
        R: Дисперсия observation-noise.

    Returns:
        beta: Ряд дневных хедж-бет β[t], индекс как у a.
    """
    n = len(a)
    beta_out = np.zeros(n)
    P = np.zeros((2, 2))                     # ковариация состояния
    x = np.zeros(2)                          # состояние [β, α]
    Vw = delta / (1 - delta) * np.eye(2)     # ковариация process-noise
    a_v = a.values
    b_v = b.values

    for t in range(n):
        if t > 0:
            P = P + Vw                       # predict: рост неопределённости
        F = np.array([b_v[t], 1.0])          # строка наблюдения [B, 1]
        yhat = F @ x                         # прогноз A из ПРОШЛОГО состояния
        e = a_v[t] - yhat                    # инновация (ошибка прогноза)
        S = F @ P @ F + R                    # дисперсия инновации
        K = (P @ F) / S                      # усиление Калмана
        x = x + K * e                        # апдейт сегодняшней ценой
        P = P - np.outer(K, F) @ P
        beta_out[t] = x[0]
    return pd.Series(beta_out, index=a.index)


def zscore_spread_kalman(
    close_a: pd.Series, close_b: pd.Series, window: int = 20,
    entry: float = 2.0, exit_z: float = 0.5,
    delta: float = 1e-4, R: float = 1e-3,
):
    """Z-score mean-reversion на спреде пары с Kalman-бетой (research).

    Бета оценивается online (без look-ahead) вместо статического OLS.
    Синтетическая equity — доллар-нейтральный портфель спреда, чья
    дневная доходность хорошо масштабирована, так что (1+r).cumprod()
    не уходит в ноль. Возвращает (synth_bars, position) для run_engine.

    Args:
        close_a: Цены A (лонг-нога).
        close_b: Цены B (хедж-нога).
        window: Окно z-score (среднее/std спреда), в барах.
        entry: |z| для открытия позиции.
        exit_z: |z| для закрытия во флэт.
        delta: Kalman process-noise.
        R: Kalman observation-noise.

    Returns:
        (synth_bars, position): Bars синтетической equity спреда (для
        подачи в run_engine) и позиция +1/0/−1.
    """
    df = pd.concat([close_a, close_b], axis=1,
                   keys=["A", "B"], sort=True).dropna()
    a, b = df["A"], df["B"]

    beta = kalman_beta(a, b, delta=delta, R=R)

    # Доллар-нейтральная equity спреда (engine-safe).
    ret_a = a.pct_change()
    ret_b = b.pct_change()
    abs_beta = beta.abs()
    w_a = 1.0 / (1.0 + abs_beta)
    w_b = abs_beta / (1.0 + abs_beta)
    leg_return = w_a * ret_a - w_b * ret_b
    synth = (1 + leg_return).cumprod()
    synth.iloc[0] = 1.0

    # Z-score меняющегося во времени спреда (сигнал).
    spread = a - beta * b
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    std = std.where(std > 1e-12)             # аудит: без деления на ~0
    z = (spread - mean) / std

    position = pd.Series(0.0, index=df.index)
    state = 0
    z_v = z.values
    for i in range(len(df)):
        zi = z_v[i]
        if np.isnan(zi):
            position.iloc[i] = float(state)  # держим состояние на NaN
            continue
        if state == 0:
            if zi > entry:
                state = -1
            elif zi < -entry:
                state = 1
        else:
            if abs(zi) < exit_z:
                state = 0
        position.iloc[i] = float(state)

    synth_bars = Bars.from_close(synth, symbol="PAIR_SPREAD")
    return synth_bars, position


def run_pair_kalman(
    close_a: pd.Series, close_b: pd.Series, window: int = 20,
    entry: float = 2.0, exit_z: float = 0.5, cost: float = 0.0002,
):
    """Прогоняет Kalman-пару через движок (research-удобство).

    Собирает synth-equity + позицию и подаёт в run_engine. Возвращает
    BacktestResult — можно сразу смотреть годовую разбивку.

    Args:
        close_a: Цены A.
        close_b: Цены B.
        window: Окно z-score.
        entry: Порог входа |z|.
        exit_z: Порог выхода |z|.
        cost: Издержки на смену позиции.

    Returns:
        BacktestResult прогона пары.
    """
    synth_bars, position = zscore_spread_kalman(
        close_a, close_b, window, entry, exit_z
    )
    return run_engine(synth_bars, position, cost=cost)
