"""HRP — Hierarchical Risk Parity (Лопес де Прадо) для весов ног.

Зачем: vol-parity не видит КЛАСТЕРЫ корреляций. Если в портфеле два
скоррелированных крипто-сигнала и одна сырьевая нога, inverse-vol
перегрузит крипто-кластер (два голоса против одного). HRP сначала
группирует ноги в дерево по корреляционной близости, потом делит риск
МЕЖДУ ветвями (кластер получает одну долю, внутри — делится дальше).

Когда применять: ноги >= 3-4 с неоднородными корреляциями
(мульти-портфель: сырьё-тренд + сырьё-MR + крипто-импульс + ...).
На 2 ногах HRP вырождается в обычный inverse-vol — кластеризовать
нечего. Реализация — классические три шага де Прадо:
  1) дерево: scipy linkage на distance = sqrt((1-corr)/2);
  2) квазидиагонализация: порядок листьев дерева;
  3) рекурсивная бисекция: риск делится обратно кластерной дисперсии.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    """Дисперсия минимально-дисперсного портфеля внутри кластера."""
    sub = cov.loc[items, items]
    ivp = 1.0 / np.diag(sub.values)
    ivp /= ivp.sum()
    return float(ivp @ sub.values @ ivp)


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Веса HRP по матрице доходностей ног (колонки — ноги).

    Args:
        returns: Дневные P&L ног (после выравнивания).

    Returns:
        Веса (сумма = 1), индекс — имена ног.
    """
    rets = returns.dropna(how="all").fillna(0.0)
    cols = list(rets.columns)
    if len(cols) == 1:
        return pd.Series([1.0], index=cols)
    corr = rets.corr().clip(-1.0, 1.0)
    cov = rets.cov()
    if len(cols) == 2:
        # Вырожденный случай: HRP == inverse-variance.
        iv = 1.0 / np.diag(cov.values)
        w = iv / iv.sum()
        return pd.Series(w, index=cols)
    # 1) Дерево по корреляционной дистанции де Прадо.
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    # 2) Квазидиагонализация: порядок листьев.
    order = [cols[i] for i in leaves_list(link)]
    # 3) Рекурсивная бисекция.
    weights = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) < 2:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            var_l = _cluster_var(cov, left)
            var_r = _cluster_var(cov, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            weights[left] *= alpha
            weights[right] *= (1.0 - alpha)
            nxt.extend([left, right])
        clusters = nxt
    return weights.reindex(cols) / weights.sum()
