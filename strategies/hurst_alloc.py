"""Hurst-аллокатор v2 — вес тренд/MR ноги по variance-ratio H.

Наследник hurst_combo (агрегированная дисперсия), но:
  - оценка H через variance-ratio Ло-МакКинлея (strategies.variance_ratio)
    — со статистикой значимости, устойчивее на коротких окнах;
  - откалиброван против эмпирической карты актив×стратегия
    (docs/REGIME_MAP_2026-07f.md): GC/CL трендовые (H>0.5), NG/металлы/
    соя реверсионные (H<0.5).

Механизм (тот же, что заложен в roadmap):
  w_trend = clip((H − h_lo)/(h_hi − h_lo), 0, 1)
  position = w_trend · trend_leg + (1 − w_trend) · mr_leg
H — свойство ИНСТРУМЕНТА на trailing-окне (не «режим времени»),
веса непрерывны (нет переключений с лагом). VT — снаружи (--vt).

Валидация: H-вердикт должен воспроизводить LOO-вердикт карты. Прогон
проверки — diagnostics/hurst_validation.py.
"""

from __future__ import annotations

import pandas as pd

from core.bars import Bars
from strategies.donchian import donchian_champion_raw
from strategies.ensemble import mr_keltner_confirm
from strategies.variance_ratio import rolling_hurst_vr


def hurst_alloc(
    bars: Bars,
    h_lo: float = 0.45,
    h_hi: float = 0.55,
    window: int = 504,
) -> pd.Series:
    """Комбо тренд+MR с весами ног по variance-ratio H инструмента.

    w_trend = clip((H − h_lo)/(h_hi − h_lo), 0, 1);
    position = w_trend · donchian_raw + (1 − w_trend) · mr_kelt_confirm.
    На прогреве H (NaN) — 50/50 (нейтрально, не ставим на класс).

    Порог [h_lo, h_hi] уже центра [0.4, 0.6] прежнего hurst_combo:
    VR-оценка H менее смещена, реальные H сырья группируются ближе к
    0.5, поэтому узкое окно вокруг 0.5 даёт более контрастное
    разделение. Откалибровано по карте, НЕ по результату комбо
    (one-shot дисциплина).

    Args:
        bars: Данные инструмента.
        h_lo: H чистого MR (ниже — весь бюджет реверсии).
        h_hi: H чистого тренда (выше — весь бюджет тренду).
        window: Окно оценки H (~2 года).

    Returns:
        position [0, 1] (обе ноги long-only сырые, VT снаружи).
    """
    h = rolling_hurst_vr(bars.close, window=window)
    span = max(h_hi - h_lo, 1e-9)
    w_trend = ((h - h_lo) / span).clip(0.0, 1.0).fillna(0.5)
    trend = donchian_champion_raw(bars).fillna(0.0)
    mr = mr_keltner_confirm(bars).fillna(0.0)
    return w_trend * trend + (1.0 - w_trend) * mr


HURST_ALLOC = {"hurst_alloc": hurst_alloc}
