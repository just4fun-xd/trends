"""Интерфейс детекции режима — задел под HMM-роутер (трек 2.2-2.3).

СТАТУС: интерфейсная заглушка, НЕ реализация. Согласовано: непройденные
мат-модели стаблятся интерфейсом, а не выдумываются. HMM — двухнедельный
трек, ещё не отработан. Здесь только форма контракта, чтобы будущий HMM
встал без ломки, плюс тривиальный always-trend детектор для тестов
роутера до появления HMM.

Почему это нужный компонент, а не преждевременный (из OU_RESULTS.md):
OU и Donchian НЕ МОГУТ сосуществовать без детектора режима — включённые
вместе гасят друг друга, вслепую каждый тонет в чужом режиме. HMM —
недостающий переключатель:
    Режим TREND -> Donchian champion (сырьё +168%, акции EMA x2)
    Режим RANGE -> OU z-score (работает где тренд буксует)
    Режим CRISIS -> кэш / vol-only
"""

from __future__ import annotations

import abc
from enum import Enum

import numpy as np
import pandas as pd

from core.bars import Bars


class Regime(Enum):
    """Дискретные режимы рынка для роутинга стратегий."""

    TREND = "trend"
    RANGE = "range"
    CRISIS = "crisis"


class RegimeDetector(abc.ABC):
    """Базовый детектор режима. HMM встанет сюда как подкласс.

    Контракт: detect() возвращает ВЕРОЯТНОСТИ режимов по времени БЕЗ
    look-ahead (forward-алгоритм: P(режим_t | данные до t включительно)).
    Роутер берёт эти вероятности и переключает стратегии.
    """

    @abc.abstractmethod
    def detect(self, bars: Bars) -> pd.DataFrame:
        """Оценивает вероятности режимов по каждому бару.

        КРИТИЧНО: без look-ahead. Вероятность в момент t использует
        только данные до t включительно (forward-фильтрация, не
        сглаживание Витерби по всему ряду).

        Args:
            bars: Данные инструмента.

        Returns:
            DataFrame: индекс = bars.index, колонки = значения Regime,
            строки суммируются в 1.0 (распределение вероятностей).
        """
        raise NotImplementedError


class AlwaysTrendDetector(RegimeDetector):
    """Тривиальный детектор: всегда режим TREND (P=1).

    Заглушка для тестирования архитектуры роутера ДО реализации HMM.
    С ним роутер = чистый Donchian champion, что проверяемо против
    прямого прогона. Заменяется на HMMDetector, когда трек 2.2 пройден.
    """

    def detect(self, bars: Bars) -> pd.DataFrame:
        """Возвращает P(TREND)=1 на всём периоде.

        Args:
            bars: Данные инструмента.

        Returns:
            DataFrame вероятностей, TREND=1, остальные 0.
        """
        df = pd.DataFrame(0.0, index=bars.index,
                          columns=[r.value for r in Regime])
        df[Regime.TREND.value] = 1.0
        return df


class VolatilityRegimeDetector(RegimeDetector):
    """Простой эвристический детектор по реализованной волатильности.

    НЕ HMM — грубая эвристика-плейсхолдер, дающая осмысленные режимы для
    отладки роутера: низкая вола+тренд -> TREND, низкая вола+боковик ->
    RANGE, высокая вола -> CRISIS. Честно помечена как эвристика, не
    выдаёт себя за откалиброванную модель. HMM заменит её математикой
    Baum-Welch, когда трек будет пройден.
    """

    def __init__(
        self,
        vol_lookback: int = 20,
        trend_lookback: int = 60,
        crisis_vol_pct: float = 0.75,
    ) -> None:
        """Настраивает окна эвристики.

        Args:
            vol_lookback: Окно реализованной волатильности.
            trend_lookback: Окно оценки силы тренда.
            crisis_vol_pct: Перцентиль волы для режима CRISIS.
        """
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback
        self.crisis_vol_pct = crisis_vol_pct

    def detect(self, bars: Bars) -> pd.DataFrame:
        """Грубая классификация режима (эвристика, без look-ahead).

        Args:
            bars: Данные инструмента.

        Returns:
            DataFrame вероятностей режимов (мягкая, через сигмоиды).
        """
        rets = bars.returns()
        vol = rets.rolling(self.vol_lookback).std()
        # Порог кризиса — расширяющийся перцентиль (без look-ahead).
        vol_thresh = vol.expanding(min_periods=self.vol_lookback).quantile(
            self.crisis_vol_pct
        )
        # Сила тренда: |наклон| нормированной цены за окно.
        sma = bars.close.rolling(self.trend_lookback).mean()
        trend_strength = (bars.close - sma).abs() / (bars.close * 0.1)

        df = pd.DataFrame(index=bars.index,
                          columns=[r.value for r in Regime], dtype=float)
        is_crisis = (vol > vol_thresh).astype(float)
        is_trend = ((trend_strength.clip(0, 1)) * (1 - is_crisis))
        is_range = ((1 - trend_strength.clip(0, 1)) * (1 - is_crisis))

        df[Regime.CRISIS.value] = is_crisis
        df[Regime.TREND.value] = is_trend
        df[Regime.RANGE.value] = is_range
        # Нормировка строк в 1.
        row_sum = df.sum(axis=1).replace(0, np.nan)
        df = df.div(row_sum, axis=0).fillna(0.0)
        df[Regime.TREND.value] = df[Regime.TREND.value].where(
            row_sum.notna(), 1.0
        )
        return df


# --- ЗАГЛУШКА: сюда встанет HMM, когда трек 2.2 будет пройден ---
class HMMDetector(RegimeDetector):
    """Скрытая марковская модель режимов — НЕ РЕАЛИЗОВАНА (трек 2.2).

    Планируемая математика: скрытая цепь Маркова, K состояний,
    наблюдаемые = дневные доходности + реализованная вола, калибровка
    Baum-Welch (EM), фильтрация forward-алгоритмом (без look-ahead),
    K выбирается на train / оценивается на hold-out.

    Намеренно оставлена заглушкой: реализовывать HMM сейчас = выдумывать
    результаты для непройденного трека. Форма контракта задана выше
    (RegimeDetector.detect), HMM встанет без ломки роутера.
    """

    def __init__(self, n_states: int = 3) -> None:
        """Args:
            n_states: Планируемое число скрытых состояний K.
        """
        self.n_states = n_states

    def detect(self, bars: Bars) -> pd.DataFrame:
        """Не реализовано — трек 2.2 не пройден.

        Args:
            bars: Данные инструмента.

        Raises:
            NotImplementedError: Всегда. Заглушка под будущую реализацию.
        """
        raise NotImplementedError(
            "HMMDetector is a stub for track 2.2 (HMM Regime Detection). "
            "Implementation needs the track: Baum-Welch calibration, "
            "forward filtering, choosing K on train. We do not fabricate a "
            "result before the track. Use VolatilityRegimeDetector to "
            "debug the router architecture."
        )
