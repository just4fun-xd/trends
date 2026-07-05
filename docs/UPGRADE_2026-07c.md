# Пакет 2026-07c: фиксы данных, vol-gate, лаборатория тренда

## Фиксы

1. **yfinance retry (критично для сравнимости).** Троттлинг ронял живые
   тикеры («possibly delisted» на GC=F), каждый прогон считался по
   РАЗНОЙ корзине — портфельные числа были несравнимы. Теперь: до 4
   попыток с паузами 2/4/8 сек; пусто после всех — ошибка, не молчание.
   Плюс run_basket/run_vol_sweep печатают красное ВНИМАНИЕ, если
   корзина неполная. Правило: числа с этим предупреждением в отчёты
   не идут.
2. **KAMA NaN-отравление** (найден при разработке): одиночный NaN на
   границе прогрева отравлял рекурсию KAMA навсегда — позиция вечный 0.
   Исправлено + регрессионный тест.
3. **champion разжалован в реестре** (комментарий): проигрывает
   donchian_vt на walk-forward (60% окон против 80%), механизм —
   take-profit режет правый хвост. Имя сохранено для сравнимости.

## Новое

- `strategies/overlays.py` — **vol_percentile_gate**: блок позиции,
  когда реализованная вола выше q-перцентиля своей trailing-истории.
  Точечный ответ на CL апрель-2020 (−64%/−86%): long-only MR не
  отличает откат от структурного коллапса; диагностический признак
  коллапса — вола за пределами исторического распределения. Это НЕ
  закрытые HMM/EMA200 (те угадывали режим тренда с лагом) — гейт меряет
  режим ВОЛАТИЛЬНОСТИ, прямую причину провала. Зарегистрированы
  `mr_ens_gate`, `mr_atr_gate`.
- `strategies/trend_lab.py` — **8 тренд-моделей пакетом** (см. каталог
  ниже), все long-only, сырые (VT снаружи), с дисклеймером multiple
  testing. Реестр: 38 стратегий, 61 тест, flake8 чист.

## Batch-прогон (один конвейер для всех)

```bash
# ШАГ 0 — Databento-панели (go/no-go для MR-ветки: артефакт роллов)
python -m scripts.fetch_databento_futures --interval 1d \
    --start 2019-01-01 --end 2026-01-01

# ШАГ 1 — пакет тренда: пере-выбор чемпиона, ПОЛНЫЙ цикл с ковидом
python -m runners.run_walkforward --strategy donchian_vt tsmom \
    tsmom_multi ewmac donch_multi channel_pos kama chandelier adx_donch \
    --basket commodity --start 2019-01-01 --end 2026-01-01 --vt --matrix
# ВАЖНО: donchian_vt здесь единственный со встроенным VT — гони его
# отдельно БЕЗ --vt, остальные С --vt (гигиена двойного VT).

# ШАГ 2 — гейт против ковида: A/B на полном цикле
python -m runners.run_basket --strategy mr_ens --basket commodity \
    --vt --target-vol 0.35 --start 2019-01-01 --yearly
python -m runners.run_basket --strategy mr_ens_gate --basket commodity \
    --vt --target-vol 0.35 --start 2019-01-01 --yearly
# Критерий: гейт обязан радикально срезать CL-2020, потеряв <20%
# доходности спокойных лет.

# ШАГ 3 — sweep выживших на Databento
python -m runners.run_vol_sweep --strategy <winner> --basket commodity \
    --source databento --vols 0.25 0.35 0.50 --start 2019-01-01

# ШАГ 4 — комбо лучшего тренда + гейтованного MR
python -m runners.run_sleeves --sleeve <trend>:commodity \
    --sleeve mr_ens_gate:commodity:vt --parity
```

## Каталог мат-моделей и методов (15)

Реализовано в этом пакете (8):

| # | Имя | Математика | Механизм-гипотеза |
|---|-----|-----------|-------------------|
| 1 | tsmom | sign(ret 12м), MOP-2012 | Контроль: если Donchian не бьёт голый TSMOM — каналы не оправданы |
| 2 | tsmom_multi | среднее знаков 1/3/12м | Диверсификация горизонта momentum |
| 3 | ewmac | Carver: (EMAf−EMAs)/vol, клип, 3 пары | Непрерывная сила тренда вместо бинарного входа |
| 4 | donch_multi | среднее пробоев 10/20/40/80 | Убирает магию lookback=20 |
| 5 | channel_pos | clip((C−mid)/(up−mid)), EMA | Плавный вход/выход — меньше пилы на границе |
| 6 | kama | адаптивная MA Кауфмана (ER) | MA замирает в шуме — меньше ложных кроссов |
| 7 | chandelier | пробой + трейл high−3·ATR | Правый хвост открыт полностью (анти-take-profit) |
| 8 | adx_donch | Дончиан ∧ ADX>20 | Фильтр СИЛЫ тренда (не направления — урок EMA200) |

Плюс overlay: **vol_percentile_gate** (9) — режим волатильности как
условие допуска MR.

Очередь (в порядке ценности, после результатов пакета):

| # | Метод | Что даёт |
|----|------|----------|
| 10 | Forecast diversification multiplier (Carver) | Корректный масштаб комбинированного прогноза ewmac+donchian с учётом корреляции ног |
| 11 | Ledoit–Wolf shrinkage ковариации sleeve'ов | Устойчивые min-variance веса вместо parity при 3+ sleeve |
| 12 | Triple-barrier + meta-labeling (Lopez de Prado) | Логрегрессия на numpy: пропускать/нет сделку победителя — ML без нарушения minimal stack |
| 13 | Schwartz–Smith 2-фактор (Kalman) | Правильная модель сырья: OU-остаток → MR-нога, GBM-тренд → тренд-нога |
| 14 | EVT/POT-оценка хвоста | Порог vol-гейта из теории экстремумов вместо эмпирического перцентиля |
| 15 | Block bootstrap кривых P&L | Доверительные интервалы Sharpe/DD — «значимо ли А лучше Б» вместо сравнения точечных чисел |

Закрыто и не возвращается: GARCH-VT (проиграл честный A/B 2019-2025,
углубил CL-2020: квартальный рефит держал плечо на входе в коллапс),
HMM, EMA200-фильтр, L/S Donchian, carry, cross-sectional momentum.
