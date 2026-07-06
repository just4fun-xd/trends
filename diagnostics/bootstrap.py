"""Stationary bootstrap (Политис-Романо, 1994): CI на Sharpe.

Зачем: сравнения вида «kama +4.4% против donchian_vt +3.4%» на 7
годовых окнах — сравнение точечных чисел, разница может быть шумом.
Bootstrap строит доверительный интервал на Sharpe И на РАЗНОСТЬ Sharpe
двух стратегий: если CI разности накрывает ноль — стратегии
статистически неразличимы, и «победа» не основание менять чемпиона.

Почему stationary bootstrap, а не обычный (iid) ресемплинг:
дневные доходности автокоррелированы (vol-кластеризация), iid-бутстрап
разрушает эту структуру и занижает дисперсию оценок. Stationary
bootstrap ресемплит БЛОКАМИ случайной длины (геометрическое
распределение со средним avg_block), склеивая их циклически —
сохраняет краткосрочную зависимость, оставаясь стационарным.

Выбор avg_block: эвристика ~ n^(1/3) (Politis-White); для 1760 дневных
баров это ~12 баров. Параметр открыт для sweep'а.

Разность Sharpe считается НА ОДНИХ И ТЕХ ЖЕ ресемплированных индексах
(paired bootstrap): обе стратегии переживают одинаковые «истории», что
резко сужает CI разности при коррелированных стратегиях — именно то,
что нужно для сравнения двух тренд-моделей на одной корзине.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _stationary_bootstrap_indices(
    n: int, avg_block: float, rng: np.random.Generator,
) -> np.ndarray:
    """Индексы одного ресемпла stationary bootstrap.

    Args:
        n: Длина ряда.
        avg_block: Средняя длина блока (геом. распределение).
        rng: Генератор случайности.

    Returns:
        Массив индексов длины n (циклическая склейка блоков).
    """
    p = 1.0 / avg_block
    idx = np.empty(n, dtype=np.int64)
    pos = 0
    while pos < n:
        start = rng.integers(0, n)
        # Длина блока ~ Geometric(p), минимум 1.
        length = min(int(rng.geometric(p)), n - pos)
        block = (start + np.arange(length)) % n  # циклически
        idx[pos:pos + length] = block
        pos += length
    return idx


def _sharpe(rets: np.ndarray, bars_per_year: float,
            rf: float = 0.0) -> float:
    """Годовой Sharpe с вычетом risk-free.

    Args:
        rets: Побарные доходности.
        bars_per_year: Баров в году.
        rf: Годовая безрисковая ставка (вычитается: excess Sharpe).

    Returns:
        (mean − rf_daily)/std × √bars_per_year; 0 при нулевой std.
    """
    std = rets.std()
    if std <= 0:
        return 0.0
    excess = rets.mean() - rf / bars_per_year
    return float(excess / std * np.sqrt(bars_per_year))


def sharpe_ci(
    returns: pd.Series,
    n_boot: int = 2000,
    avg_block: float | None = None,
    ci: float = 0.90,
    bars_per_year: float = 252.0,
    rf: float = 0.0,
    seed: int = 42,
) -> dict:
    """Доверительный интервал Sharpe одной стратегии.

    Args:
        returns: Побарные доходности (после издержек).
        n_boot: Число ресемплов.
        avg_block: Средняя длина блока; None -> эвристика n^(1/3).
        ci: Уровень доверия (0.90 -> перцентили 5/95).
        bars_per_year: Баров в году.
        rf: Годовая безрисковая ставка (excess Sharpe).
        seed: Сид генератора.

    Returns:
        dict: sharpe (точечный), lo, hi (границы CI), avg_block.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if avg_block is None:
        avg_block = max(2.0, n ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, avg_block, rng)
        boots[b] = _sharpe(r[idx], bars_per_year, rf)
    alpha = (1.0 - ci) / 2.0
    return {
        "sharpe": _sharpe(r, bars_per_year, rf),
        "lo": float(np.quantile(boots, alpha)),
        "hi": float(np.quantile(boots, 1.0 - alpha)),
        "avg_block": float(avg_block),
    }


def sharpe_diff_ci(
    returns_a: pd.Series,
    returns_b: pd.Series,
    n_boot: int = 2000,
    avg_block: float | None = None,
    ci: float = 0.90,
    bars_per_year: float = 252.0,
    rf: float = 0.0,
    seed: int = 42,
) -> dict:
    """CI на разность Sharpe(A) − Sharpe(B), paired bootstrap.

    Обе стратегии ресемплируются ОДНИМИ индексами (переживают те же
    «истории») — корректное сравнение на общей корзине/периоде.

    Args:
        returns_a: Доходности стратегии A.
        returns_b: Доходности стратегии B (общий индекс с A).
        n_boot: Число ресемплов.
        avg_block: Средняя длина блока; None -> n^(1/3).
        ci: Уровень доверия.
        bars_per_year: Баров в году.
        rf: Годовая безрисковая ставка.
        seed: Сид.

    Returns:
        dict: sharpe_a, sharpe_b, diff, lo, hi, significant (bool:
        CI разности не накрывает 0 -> различие значимо на уровне ci).
    """
    df = pd.concat([returns_a, returns_b], axis=1, join="inner")
    df = df.dropna()
    a = df.iloc[:, 0].to_numpy()
    b = df.iloc[:, 1].to_numpy()
    n = len(df)
    if avg_block is None:
        avg_block = max(2.0, n ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = _stationary_bootstrap_indices(n, avg_block, rng)
        diffs[k] = (_sharpe(a[idx], bars_per_year, rf)
                    - _sharpe(b[idx], bars_per_year, rf))
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(diffs, alpha))
    hi = float(np.quantile(diffs, 1.0 - alpha))
    sa = _sharpe(a, bars_per_year, rf)
    sb = _sharpe(b, bars_per_year, rf)
    return {
        "sharpe_a": sa,
        "sharpe_b": sb,
        "diff": sa - sb,
        "lo": lo,
        "hi": hi,
        "significant": bool(lo > 0 or hi < 0),
        "avg_block": float(avg_block),
    }
