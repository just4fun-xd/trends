# Апгрейд 2026-07b: ревью, унификация, ансамбли, GARCH в раннерах

## 1. Найдено и исправлено при ревью

| # | Место | Проблема | Фикс |
|---|-------|----------|------|
| 1 | `diagnostics/vol_sweep.py` | Оборот считался старой `diff()`-формулой, которую движок сам объявил багом (занижает оборот на дробных весах). Издержки начислялись drift-aware, оборот печатался diff — колонки «издержк» и «оборот» были из РАЗНЫХ моделей. | Оборот теперь `drift_turnover()` — та же функция, что списывает издержки. Тест `test_sweep_turnover_matches_engine_formula` фиксирует совпадение. |
| 2 | Все раннеры | `core/garch.py` готов и покрыт тестами, но НЕ подключён ни к одному раннеру — трек 3 роадмапа (A/B garch vs realized) физически нельзя было запустить. | Реестр `core/sizing.py` (`realized`/`garch`) + флаг `--sizer` в `run_basket`, `run_vol_sweep`, `run_walkforward`; в `run_sleeves` — спецификация `strategy:basket:garch`. |
| 3 | `run_basket`, `run_walkforward` | Дублированные inline-обёртки `--vt` с локальным импортом `vol_target_size` внутри замыкания — два места с копипастой, target_vol не настраивался. | Единый `make_sizer(name, target_vol)`; `--target-vol` в run_basket. |
| 4 | `core/config.py` | Нет перевода строки в конце файла (W292). | Исправлено. |
| 5 | `scripts/check_h4_bars.py` | f-string без плейсхолдеров (F541). | Исправлено. |
| 6 | `strategies/donchian.py:479` | Строка 84 символа (E501). | Разбита. |

flake8 чист, 54 теста зелёные (48 старых + 6 новых).

## 2. Напрашивавшаяся диагностика: портфельный DD в vol-sweep

Твой прогон `mr_atr_stop` упирался в DD<40% на target_vol 50% по
критерию **worst-case per-instrument**. Но лимит Александра — про
стратегию (портфель), а худший из 19 инструментов — заведомо более
жёсткий критерий: equal-weight портфель диверсифицирует просадки по
инструментам и его DD всегда мельче worst-case.

`vol_sweep_basket` теперь считает оба: `worstDD` (старый, консервативный)
и `портDD` + `портРет` + `Sharpe` (equal-weight дневного P&L). Раннер
печатает две рекомендации. Ожидание на твоих данных: портфельный
критерий позволит target_vol 50–70% там, где worst-case запрещал
50% — это самый дешёвый источник доходности (тот же сигнал, больше
риска ровно до лимита).

## 3. Ансамбли (`strategies/ensemble.py`)

- **`mr_ens`** — среднее позиций 4 MR-вариантов с разными выходами
  (atr_stop / time_stop / keltner / confirm). Убирает сам шаг «выбор
  лучшего из 10» (multiple testing): сигнал общий, шум выходов
  декоррелирован, позиция [0..1] = conviction sizing по согласию ног.
- **`combo_tmr`** — Donchian champion (raw) + mr_ens на одном
  инструменте, веса на уровне ПОЗИЦИИ. В отличие от run_sleeves
  (комбинация P&L), противоположные изменения ног неттингуются ДО
  издержек — дешевле по обороту.

Оба зарегистрированы в `STRATEGIES`, сырые (VT снаружи: `--vt --sizer garch`).

## 4. Как гонять GARCH A/B (трек готов к запуску)

```bash
# A/B на sweep'е: одна переменная — знаменатель VT
python -m runners.run_vol_sweep --strategy mr_atr_stop --basket commodity \
    --vols 0.15 0.25 0.35 0.50 0.70 --sizer realized
python -m runners.run_vol_sweep --strategy mr_atr_stop --basket commodity \
    --vols 0.15 0.25 0.35 0.50 0.70 --sizer garch

# Walk-forward стабильность обоих сайзеров
python -m runners.run_walkforward --strategy mr_atr_stop mr_ens \
    --basket commodity --vt --sizer garch

# Sleeve-комбо с GARCH-сайзером на MR-ноге
python -m runners.run_sleeves --sleeve champion:commodity \
    --sleeve mr_ens:commodity:garch --parity
```

Вердикт по A/B: GARCH принят, только если (а) DD мельче при том же
доходе ИЛИ доход выше при том же DD, (б) эффект стабилен по годам
walk-forward, (в) оборот не взорвался (alpha-шоки дёргают размер).
