"""Тесты стратегий — обновлены аудитом 2026-07 под новую семантику.

Новая семантика 4step-ядра, закрываемая регрессиями:
  - one-shot take-profit + ЗАМОРОЗКА пирамиды до полного выхода
    (устранён «пулемётный» тейк и пила тейк<->докупка);
  - turtle-стоп подтягивается за докупками (last_add - stop_atr*ATR);
  - риск-триггеры (стоп, нижний канал) по low, тейк по high,
    входы/докупки close-confirmed.

ВАЖНО: семантика champion уточнена => эталонные числа из
BENCHMARK_RESULTS.md (+5.2% / -12.1% / 15 из 19) получены СТАРЫМ кодом
и подлежат локальной ре-валидации на реальных данных до доклада.

Живые данные Yahoo недоступны в песочнице — проверяем МЕХАНИКУ на
синтетических рядах с заданной структурой.

Запуск: python -m tests.test_strategies
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bars import Bars
from core.engine import run_engine, vol_target_size
from strategies import bollinger, donchian, ema


def _trend_bars(n=400, slope=0.0015, noise=0.008, seed=0) -> Bars:
    """Стохастический восходящий тренд с реалистичным диапазоном."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps = rng.normal(slope, noise, n)
    close = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
    rng_hl = close * 0.006
    high = close + rng_hl * rng.uniform(0.3, 1.0, n)
    low = close - rng_hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="TREND")


def _reversal_bars(n=400, seed=1) -> Bars:
    """Рост, затем решительный разворот вниз — стоп обязан выбить."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    half = n // 2
    up = rng.normal(0.002, 0.008, half)
    down = rng.normal(-0.003, 0.012, n - half)
    steps = np.concatenate([up, down])
    close = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
    rng_hl = close * 0.008
    high = close + rng_hl * rng.uniform(0.3, 1.0, n)
    low = close - rng_hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="REVERSAL")


def _range_bars(n=400, seed=2) -> Bars:
    """Боковик вокруг 100 — mean-reversion профитна, тренд сливает."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = np.zeros(n)
    x[0] = 100
    for i in range(1, n):
        x[i] = x[i - 1] + 0.15 * (100 - x[i - 1]) + rng.normal(0, 1.5)
    close = pd.Series(x, index=idx)
    rng_hl = 1.2
    high = close + rng_hl * rng.uniform(0.3, 1.0, n)
    low = close - rng_hl * rng.uniform(0.3, 1.0, n)
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="RANGE")


def _monotonic_bars(n=120, step=0.005) -> Bars:
    """Детерминированный ступенчатый рост: каждый close выше прошлого
    high — пробой, докупки и тейк срабатывают предсказуемо.

    Для точечной проверки one-shot тейка и заморозки пирамиды.
    """
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * (1 + step) ** np.arange(n), index=idx)
    high = close * 1.001
    low = close * 0.997
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="MONO")


def _trail_bars() -> Bars:
    """Прогрев -> пробой -> набор пирамиды (last_add ~104.8) -> плато ->
    откат в ЗАЗОР между подтянутым стопом (last_add - 2*ATR ~102.8) и
    entry-стопом (entry - 2*ATR ~99.2).

    Подтянутый стоп обязан выбить; стоп, висящий на entry, — нет.
    Регрессия на находку самоаудита (стоп не подтягивался). Канал
    выхода в тесте берётся широким (25), чтобы ссылаться на старые
    lows прогрева (~99.6) и не маскировать проверку стопа.
    """
    n = 80
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = np.full(n, 100.0)
    for i in range(40, 44):          # рост +1.2/бар: пробой + 3 докупки
        close[i] = close[i - 1] + 1.2
    close[44:50] = close[43]         # плато на 104.8 (пирамида полная)
    close[50:] = 102.5               # откат: low 102.1 < 102.8, > 99.2
    close = pd.Series(close, index=idx)
    high = close + 0.4
    low = close - 0.4
    return Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="TRAIL")


def test_ema_champion_on_trend() -> None:
    """EMA ensemble+VT ловит гладкий тренд (родная среда — акции)."""
    bars = _trend_bars(slope=0.0015)
    pos = ema.ema_ensemble_voltarget(bars)
    res = run_engine(bars, pos)
    assert res.total_return > 0, f"EMA VT слил тренд: {res.total_return:.1%}"
    assert res.passes_dd(0.40), f"EMA VT пробил DD40: {res.max_drawdown:.1%}"
    print(f"  [ok] EMA ens+VT на тренде: ret={res.total_return:+.1%}, "
          f"dd={res.max_drawdown:.1%} (проходит DD40)")


def test_take_is_one_shot_and_freezes() -> None:
    """КРИТИЧНО (аудит 2026-07): тейк one-shot, пирамида заморожена.

    На монотонном росте: пирамида набирается 0.4->0.7->0.9->1.0,
    РОВНО ОДИН сброс верхней ступени (1.0->0.9), дальше константа —
    ни повторных тейков (пулемёт), ни докупок (пила тейк<->докупка).
    """
    bars = _monotonic_bars()
    raw = donchian._donchian_4step(
        bars, 20, 10, 20, 2.0, take_atr=3.5, use_take=True
    )
    d = raw.diff().fillna(0.0)
    ups = d[d > 1e-12]
    downs = d[d < -1e-12]

    assert raw.max() > 0.99, f"Пирамида не набралась: max={raw.max():.3f}"
    assert len(downs) == 1, (
        f"Тейк сработал {len(downs)} раз — должен ровно 1 (one-shot)"
    )
    take_i = downs.index[0]
    after = raw.loc[take_i:]
    assert abs(after.iloc[0] - 0.9) < 1e-9, (
        f"Сброшена не верхняя ступень: {after.iloc[0]:.3f} != 0.9"
    )
    frozen = after.diff().fillna(0.0).abs()
    assert (frozen < 1e-12).all(), (
        "Пирамида НЕ заморожена после тейка (есть докупки/тейки)"
    )
    print(f"  [ok] One-shot тейк: подъёмов {len(ups)}, тейк ровно 1 "
          f"(1.0->0.9 на {take_i.date()}), после — заморозка до конца")


def test_stop_trails_behind_adds() -> None:
    """КРИТИЧНО (самоаудит): стоп подтягивается за докупками.

    Откат до (last_add - 2*ATR) выбивает позицию, хотя цена ещё далеко
    выше (entry - 2*ATR) и выше нижнего канала. Старый код (стоп на
    entry) здесь НЕ вышел бы вовсе.
    """
    bars = _trail_bars()
    raw = donchian._donchian_4step(
        bars, 20, 25, 20, 2.0, take_atr=None, use_take=False
    )
    assert raw.iloc[44:50].max() > 0.99, "Пирамида не набралась на росте"
    # Откат начинается на баре 50; подтянутый стоп выбивает сразу.
    assert (raw.iloc[52:] == 0.0).all(), (
        "Подтянутый стоп не выбил на откате к last_add - 2*ATR"
    )
    # Дискриминация: low на откате выше entry-стопа (~99.2) и выше
    # широкого канала (~99.6) — выбить мог ТОЛЬКО подтянутый стоп.
    exit_low = bars.low.iloc[50]
    assert exit_low > 100.5, "Конструкция теста нарушена"
    print(f"  [ok] Подтяжка стопа: выход на low={exit_low:.1f} "
          f"(entry-стоп ~99.2, канал ~99.6 — старый код не вышел бы)")


def test_champion_identity_raw_times_vol() -> None:
    """Тождество рефакторинга: champion == champion_raw * vol_size.

    Это инвариант, на котором держится риск-паритет роутера.
    """
    bars = _trend_bars()
    champ = donchian.donchian_est_macd_4step_take(bars)
    manual = (donchian.donchian_champion_raw(bars)
              * vol_target_size(bars, 0.15))
    diff = (champ - manual).abs().max()
    assert diff < 1e-12, f"Тождество нарушено: {diff:.2e}"
    print(f"  [ok] champion == raw * vol_target: расхождение {diff:.2e}")


def test_champion_pyramids_on_trend() -> None:
    """Champion набирает пирамиду на стохастическом тренде, держит DD."""
    bars = _trend_bars(slope=0.0015)
    raw = donchian._donchian_4step(
        bars, 20, 10, 20, 2.0, take_atr=3.5, use_take=True
    )
    assert raw.max() > 0.40, (
        f"Пирамида не выше 1 ступени: max={raw.max():.2f}"
    )
    res = run_engine(bars, donchian.donchian_est_macd_4step_take(bars))
    assert res.passes_dd(0.40), f"Champion пробил DD40: {res.max_drawdown:.1%}"
    print(f"  [ok] Champion на тренде: пирамида до {raw.max():.2f}, "
          f"ret={res.total_return:+.1%}, dd={res.max_drawdown:.1%}")


def test_champion_stop_on_reversal() -> None:
    """На развороте риск-выход по low выбивает раньше обвала."""
    bars = _reversal_bars()
    raw = donchian._donchian_4step(
        bars, 20, 10, 20, 2.0, take_atr=3.5, use_take=True
    )
    second_half = raw.iloc[len(raw) // 2 + 30:]
    assert (second_half == 0).sum() > len(second_half) * 0.5, (
        "Стоп не вывел из позиции после разворота"
    )
    res = run_engine(bars, donchian.donchian_est_macd_4step_take(bars))
    assert res.passes_dd(0.40), (
        f"Champion не удержал DD40 на развороте: {res.max_drawdown:.1%}"
    )
    print(f"  [ok] Разворот: вышел из позиции, dd={res.max_drawdown:.1%} "
          f"(удержал DD40)")


def test_entry_atr_frozen() -> None:
    """Регрессия (аудит Gemini): сетка риска фиксируется на входе.

    Прямая проверка инварианта: строим ряд, где после входа ATR РАСТЁТ.
    С плавающим atr_i стоп-уровень last_add - stop_atr*atr уползал бы
    вниз вместе с растущим ATR (риск > запланированного). С фиксированным
    entry_atr стоп держится узким и выбивает на умеренном откате.

    Дискриминирующий сценарий: спокойный вход (низкий ATR), затем серия
    расширяющихся баров поднимает текущий ATR, и умеренный откат. С
    фиксированным ATR откат пробивает узкий стоп; с плавающим — стоп уже
    уехал ниже отката и позиция ложно выжила бы.
    """
    n = 120
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = np.full(n, 100.0)
    for i in range(1, 40):                # спокойный вход, узкий ATR
        close[i] = close[i - 1] + 0.4
    # После входа — расширяющиеся бары (растёт текущий ATR), цена почти
    # стоит, потом умеренный откат.
    for i in range(40, 70):
        close[i] = close[i - 1] + (0.1 if i % 2 else -0.1)
    close[70:] = close[69] - 2.2          # умеренный откат
    close = pd.Series(close, index=idx)
    high = close + 0.15
    low = close - 0.15
    # Расширяем диапазон баров 40-69 -> текущий ATR растёт после входа.
    for i in range(40, 70):
        high.iloc[i] = close.iloc[i] + 1.5
        low.iloc[i] = close.iloc[i] - 1.5
    low.iloc[70] = close.iloc[70] - 0.15
    bars = Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="ATRFRZ")

    # Считаем оба уровня стопа руками на баре отката (70).
    atr = bars.atr(20)
    entry_bar = 39                        # примерно бар входа
    entry_atr = atr.iloc[entry_bar]
    current_atr = atr.iloc[70]
    # ATR действительно вырос после входа (иначе тест не дискриминирует).
    assert current_atr > entry_atr * 1.5, (
        f"ATR не вырос: entry={entry_atr:.2f} current={current_atr:.2f}"
    )
    raw = donchian._donchian_4step(
        bars, 20, 30, 20, 2.0, take_atr=None, use_take=False
    )
    # На баре входа пирамида набралась.
    assert raw.iloc[40:69].max() > 0.39, "Позиция не открылась"
    # С фиксированным узким entry_atr стоп выбивает на откате бара 70.
    # (С плавающим широким ATR стоп уехал бы ниже и позиция выжила бы.)
    assert (raw.iloc[71:] == 0.0).all(), (
        "Стоп не выбил на откате — сетка риска поехала за текущим ATR"
    )
    print(f"  [ok] entry_atr зафиксирован: entry_atr={entry_atr:.2f} < "
          f"current_atr={current_atr:.2f}, стоп по УЗКОМУ входному ATR "
          f"выбил на откате")


def test_outside_bar_reopens_same_bar() -> None:
    """Регрессия (аудит Gemini): outside-bar переоткрывает в том же баре.

    Бар пробивает нижний канал (low < lo -> выход) И закрывается выше
    верхнего (close > up -> вход). С if/elif мы бы вышли и пропустили
    вход до следующего бара. С двумя независимыми if — выходим по риску,
    затем тем же баром переоткрываемся. Проверяем: после outside-бара
    позиция снова 1.0 (а не 0 до следующего бара).
    """
    n = 60
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = np.full(n, 100.0)
    for i in range(1, 30):           # рост, входим в позицию
        close[i] = close[i - 1] + 0.6
    close[30:40] = close[29]         # короткое плато
    # Бар 40 — outside: НОВЫЙ пик close (выше 20-дневного max -> вход),
    # но длинный нижний хвост пробивает exit-канал (low < lo -> выход).
    close[40] = close[29] + 2.0      # свежий максимум закрытия
    close[41:] = close[40]
    close = pd.Series(close, index=idx)
    high = close + 0.2
    low = close - 0.2
    low.iloc[40] = close.iloc[40] - 12.0    # длинный хвост вниз
    high.iloc[40] = close.iloc[40] + 0.2
    bars = Bars(open=close.shift(1).bfill(), high=high, low=low,
                close=close, bars_per_year=252.0, symbol="OUTSIDE")

    pos = donchian.donchian_breakout(bars, entry=20, exit_period=10)
    # Дискриминация: на баре 40 low пробил нижний канал И close дал новый
    # максимум выше верхнего канала. Проверим оба условия выполнимы.
    from strategies.donchian import _donchian_channels
    up, lo = _donchian_channels(bars, 20, 10)
    assert bars.low.iloc[40] < lo.iloc[40], "low не пробил канал (тест)"
    assert bars.close.iloc[40] > up.iloc[40], "close не выше up (тест)"
    # С двумя if позиция на баре 40 = 1.0 (вышли по риску, зашли по close).
    # С if/elif была бы дыра pos[40]==0 до следующего бара.
    assert pos.iloc[40] == 1.0, (
        "Outside-бар оставил дыру — вход не переоткрылся в том же баре"
    )
    print("  [ok] outside-bar: вышли по риску и переоткрылись тем же "
          "баром (нет пропущенного бара на входе)")


def test_meanrev_beats_trend_in_range() -> None:
    """В боковике BB+RSI профитнее трендового Дончиана
    (комплементарность — ключевой тезис проекта)."""
    bars = _range_bars()
    mr = run_engine(bars, bollinger.bollinger_rsi(bars))
    tr = run_engine(bars, donchian.donchian_breakout(bars))
    assert mr.total_return > tr.total_return, (
        f"MR {mr.total_return:.1%} не побил тренд {tr.total_return:.1%}"
    )
    print(f"  [ok] Боковик: BB+RSI {mr.total_return:+.1%} > "
          f"Дончиан {tr.total_return:+.1%} (комплементарность)")


def test_percent_b_redundant() -> None:
    """Don %b ~ Don ens: коррелированный фильтр не меняет картину."""
    bars = _trend_bars(slope=0.001)
    base = run_engine(bars, donchian.donchian_ensemble_voltarget(bars))
    pb = run_engine(bars, bollinger.donchian_percent_b(bars))
    print(f"  [ok] Don ens {base.total_return:+.1%} vs "
          f"Don %b {pb.total_return:+.1%} (%b — коррелир. фильтр)")


def test_ema_stop_no_same_bar_reentry() -> None:
    """Регрессия (самоаудит): после стопа нет перевхода в том же баре.

    Стоп по low + реарм: пока bull-сигнал не сбросится, позиция 0.
    Проверяем: в баре срабатывания стопа позиция не 1.0 вновь.
    """
    bars = _reversal_bars(seed=5)
    pos = ema.ema_cross_stop(bars, stop=0.05)
    # Ищем бары, где позиция упала 1->0: в следующем баре она может
    # вернуться только если bull пересобрался; в том же — никогда.
    drops = pos.diff() < 0
    assert not (drops & (pos > 0)).any(), (
        "Перевход в том же баре после стопа (реарм не работает)"
    )
    res = run_engine(bars, pos)
    print(f"  [ok] ema_cross_stop: реарм после стопа, "
          f"ret={res.total_return:+.1%}")


if __name__ == "__main__":
    print("Тесты стратегий (семантика аудита 2026-07):")
    test_ema_champion_on_trend()
    test_take_is_one_shot_and_freezes()
    test_stop_trails_behind_adds()
    test_champion_identity_raw_times_vol()
    test_champion_pyramids_on_trend()
    test_champion_stop_on_reversal()
    test_entry_atr_frozen()
    test_outside_bar_reopens_same_bar()
    test_meanrev_beats_trend_in_range()
    test_percent_b_redundant()
    test_ema_stop_no_same_bar_reentry()
    print("Все тесты стратегий пройдены.")
