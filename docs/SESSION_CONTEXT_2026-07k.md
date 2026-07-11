# SESSION CONTEXT — 2026-07k (10.07.26): лаборатории 37+D.1, три класса

Дайджест сессии для восстановления контекста. Хронология сжата до
решений, находок, механизмов и команд подтверждения. Реестр вырос
132 → 169 (37 новых стратегий).

---

## ЧТО СДЕЛАНО

### 5 новых модулей стратегий (37 функций)
- `strategies/trend_lab4.py` (10) — трендовые на НЕЗАНЯТОМ аппарате:
  PSAR, Renko (событийное время), Mann-Kendall (ранговый тест),
  AR(1)-гейт, триггер Шмитта (гистерезис), декиклер Элерса (DSP),
  overnight-дрейф (единств. модель на bars.open), CUSUM Пейджа,
  дробный Келли, перцентиль momentum.
- `strategies/crypto_aggr_lab2.py` (10) — агрессивные крипто × КРИЗИСНЫЙ
  матаппарат защиты: CPPI (Black-Perold), Grossman-Zhou DD-контроль,
  BNS jump-тест (bipower variation), CVaR-сайзинг (EVT), полудисперсия
  (Sortino), vol-of-vol гейт, интенсивность Хоукса, skew-гейт, Келли+
  куртозис-штраф, circuit-breaker.
- `strategies/meanrev_lab3.py` (10) — реверсия с упором на ИЗВЛЕЧЕНИЕ:
  пороги Бертрама (OU-как-измеритель), GARCH-шок, Келли-MR, грид,
  овершут-выход, DFA-Хёрст, ранговый шок (Уилкоксон), хвост-квантиль,
  AR(1)-прогноз, дивергенция.
- `strategies/mr_lowvol2.py` (5) — доработка чемпиона mr_lowvol, по
  ОДНОМУ узлу на вариант: размер (cont), набор (scale), гейт (garch),
  выход (zexit), двойной гейт (vr).
- `strategies/schwartz_smith.py` (2, Roadmap D.1) — Kalman-разложение
  ln(S)=chi+xi, MR-нога на изолированном chi. ss_chi_mr / ss_chi_soft.

### Инфраструктура
- `runners/run_contribution_sweep.py` — оркестратор instrument_
  contribution по РАБОЧИМ парам (карта WORKING_PAIRS: стратегия →
  актив-класс где прошла ≥TIE). Свод в один CSV с колонкой asset_class.
- Guard в `run_category_ranking`: `--source ccxt` требует
  `--basket crypto` (иначе брал COMMODITY_YF молча → пустой рейтинг).
- `tests/test_labs_2026_07k.py` — контракт, префикс-устойчивость
  (look-ahead) всех 37, регрессия двойного shift, режимное поведение,
  восстановление OU-свойств chi в Schwartz-Smith. 8/8 зелёные.

---

## РЕЗУЛЬТАТЫ ПО ТРЁМ КЛАССАМ (bootstrap, 2 источника где возможно)

### Сырьё (databento + yf) — champion-стек НЕПОКОЛЕБИМ
- Все 37 → TIE или WORSE против donchian_vt / mr_lowvol.
- Рейтинги наврали дважды: mr_lv2_zexit (рейтинг 1.51) и tr4_mk
  (рейтинг 1.27) лидировали в grubом ранжировании, но bootstrap дал
  TIE с разными знаками diff на двух источниках → МИРАЖ. Правило
  «рейтинг ≠ арбитраж» подтверждено эмпирически.
- Лучшие «рабочие» ([2]): mr_lv2_zexit/scale/vr/garch (corr 0.7-0.9 с
  чемпионом — двойники), mr3_overshoot/bertram/tail_q/dfa, tr4_mk/
  schmitt. В ансамбль не идут (corr высок).

### Крипта (ccxt 4h + yf 1d) — защиты работают, но не бьют дончиана
- donchian_vt соло недостижим (4h 1.82). Все ca2_* ниже.
- ca2_breaker/semi/vvol: WORSE на 4h, corr 0.80-0.87 — ДВОЙНИКИ
  Дончиана с урезанной доходностью (защита стоит Sharpe).
- **ca2_cppi — единственная ценная**: TIE на обоих, corr 0.51/0.45,
  ensemble_ok=1 дважды, combo 1.93 > дончиан-соло 1.82 при DD -0.035.
  Декоррелированный оборонительный поток — кандидат в крипто-оборону.
- Кризисный матаппарат РАБОТАЕТ по назначению: DD защит -0.03..-0.07
  против -0.07 у незащищённых ca_burst/ca_squeeze (отриц. Sharpe).
- ПОДТВЕРЖДЕНО (из fullcycle.csv прошлых прогонов): ca_vol_ride на
  крипте Sharpe 1.945, combo 2.085, corr 0.49, ensemble_ok — действ.
  3-й leg крипто-ансамбля.

### Акции (yf) — ДЫРА ЗАКРЫТА (не гонялись в fullcycle до этой сессии)
- Тренд: 4 стратегии ЗНАЧИМО BEATS ema_vt (0.828), но ci_lo впритык:
  tr3_tsmom (1.19, ci_lo +0.004), tsmom_multi (1.14, ci_lo +0.0001),
  trend_ens, ema_ensemble. ВСЕ corr 0.90+, ensemble_ok=0.
  → tr3_tsmom = watch-кандидат в чемпионы equity-тренда, НО:
  пограничная значимость + одна история (2019-26, один бычий режим) +
  риск beta-не-alpha (урок equity short leg). ЧЕМПИОНА НЕ МЕНЯЕМ до
  второго независимого куска истории / медвежьего режима.
- trend_lab2/impulse на equity → почти все WORSE (те же механизмы,
  что провалились на сырье; качество-фильтры тренда универсально слабы).
- MR: ноль значимых, mr_ens (0.696) не обойдён. НО декоррелированные
  кандидаты в ансамбль: mr_connors (corr 0.21!), mr_trend (0.38),
  mr3_overshoot (0.43), mr_lowvol (0.59) — ensemble_ok=1.
- ПОДТВЕРЖДЕНО run_sleeves: mr_ens + mr_connors, vol-parity, corr 0.31,
  combo Sharpe 1.33 (соло 0.95/0.99), DD -1.7%, ни одного убыточного
  года. НО годовая +2.2% при воле 1.7% — профиль «Sharpe хорош, деньги
  ниже bonds» (прямо критика Александра). Ансамбль = диверсификация
  касательного портфеля, НЕ доходность.

### D.1 Schwartz-Smith — ЗАКРЫТ С МЕХАНИЗМОМ
- ss_chi_soft 0.09, ss_chi_mr -0.03 vs mr_lowvol 1.35. Глубокий провал.
- Причина НЕ баг (тесты: фильтр восстанавливает OU-свойства chi).
  Причина структурная: (1) недоидентификация chi на ОДНОМ roll-adj
  ряду — нет term structure, факторы разделяются только по скорости
  затухания автокорр., MLE уходит в вырождение; (2) rho=0 (нельзя
  оценить на одном ряду) убивает канал сигнала; (3) даже идеальный
  chi = детрендир. остаток, а mr_lowvol ту же реверсию берёт проще.
- Roadmap предсказал дословно: «неясно, что chi добавит сверху».
  Ответ эмпирический: НИЧЕГО. OU-как-двигатель закрыт по ВСЕМ формам.
- OU-как-измеритель ЖИВ: kappa→пороги (mr3_bertram), half-life→окна.
- Последний шанс (если давать): панель РЕАЛЬНЫХ фьючерсных сроков
  (ближний+дальний контракт) вместо roll-adj — term structure
  идентифицирует факторы честно. Другой data pipeline, низкий приор.
  РЕКОМЕНДАЦИЯ: закрыть, не воскрешать.

---

## БАЛЛАСТ ПО КОРЗИНАМ (run_contribution_sweep + LOO)

Балласт — свойство ПАРЫ (актив × стратегия), не актива. Инструменты-
балласт у нескольких стратегий класса (кандидаты на пересмотр корзины):
- **Сырьё**: PL (балласт у 5 трендовых, solo Sh -0.42), NG (у 2 тренд,
  но лучший держатель в MR — классика). CL балласт у MR/hurst, но
  ДЕРЖАТЕЛЬ у donchian_vt (loo -0.168). ZL/ZM/ZW/Sugar размывают MR.
- **Крипта**: Litecoin (балласт у 4), Cosmos/BCH (у 2). Ядро = Bitcoin.
- **Акции**: JnJ/Pepsi (балласт у тренда), Coca-Cola (у MR). Стейплы
  без тренда. Ядро тренда = Costco/Nvidia/Tesla.

**ВЫРЕЗ PL,NG ПОДТВЕРЖДЁН** (donchian_vt, databento):
полная корзина (12) Sharpe +1.27 → без PL,NG (10) Sharpe **+1.41**
(+0.14 бесплатно). После выреза балласта не осталось. Механизм ясен
(PL/NG не торгуются трендом). Осталось подтвердить на yf (гигиена).

---

## КОМАНДЫ ПОДТВЕРЖДЕНИЯ (рабочие, сверены с кодом)

```bash
# Рейтинг категорий (грубый скрининг, НЕ арбитраж)
python -m runners.run_category_ranking --families cat_trend_lab4 \
    --source databento --vt --target-vol 0.4 --csv cat_trend_lab4_vt.csv
python -m runners.run_category_ranking --families cat_mr_lab3 cat_mr_lowvol2 \
    --source databento --csv cat_mr_lab3.csv
# КРИПТА: обязательно --basket crypto И --interval 4h (данные H4/H1!)
python -m runners.run_category_ranking --families cat_crypto_aggr2 \
    --source ccxt --basket crypto --interval 4h --vt --target-vol 0.4 \
    --csv cat_crypto_aggr2.csv

# Bootstrap-арбитраж (--candidates + --champion; кандидат=чемпион
# ОТФИЛЬТРУЕТСЯ, само-сравнение невозможно)
python -m runners.run_fullcycle --candidates mr_lv2_zexit mr3_overshoot \
    --champion mr_lowvol --source databento --basket commodity \
    --vt --target-vol 0.4 --csv fc_mr_db.csv
python -m runners.run_fullcycle --candidates tr4_mk tr4_page \
    --champion donchian_vt --source databento --basket commodity \
    --vt --target-vol 0.4 --csv fc_tr_db.csv
# КРИПТА-защиты
python -m runners.run_fullcycle --candidates ca2_breaker ca2_semi ca2_vvol ca2_cppi \
    --champion donchian_vt --source ccxt --basket crypto --interval 4h \
    --vt --target-vol 0.4 --csv fc_ca2_4h.csv
# EQUITY (ДЫРА): --family против equity-чемпионов
python -m runners.run_fullcycle --family trend --champion ema_vt \
    --source yf --basket equity --vt --target-vol 0.20 \
    --skip-prefix ca ca2 --csv fc_eq_trend.csv
python -m runners.run_fullcycle --family mean-reversion --champion mr_ens \
    --source yf --basket equity --vt --target-vol 0.20 \
    --skip-prefix ou ca2 --csv fc_eq_mr.csv

# Свод вклада инструментов по рабочим парам (три класса разом)
python -m runners.run_contribution_sweep --vt --csv contrib_sweep.csv

# Балласт-вырез: ДВА прогона contribution, сравнить «Полный портфель Sharpe»
python -m runners.run_instrument_contribution --strategy donchian_vt \
    --basket commodity --source databento --vt --target-vol 0.20
python -m runners.run_instrument_contribution --strategy donchian_vt \
    --basket commodity --source databento --exclude PL,NG --vt --target-vol 0.20

# Equity-MR ансамбль (VT как :vt в спеке ноги, НЕ флаг; equity без @)
python -m runners.run_sleeves --sleeve mr_ens:equity:vt \
    --sleeve mr_connors:equity:vt --source yf --target-vol 0.20 --parity
```

---

## УРОКИ СИНТАКСИСА (грабли сессии — свериться перед командой)
- `@ИМЯ` работает ТОЛЬКО для 12 ключей NAMED_BASKETS (COMM_*, CRYPTO_*,
  *_CORE_COMM). Equity там НЕТ. Прочее — список тикеров через запятую.
- `run_fullcycle`: кандидат==чемпион → отфильтрован (0 кандидатов).
  Само-сравнение стратегии на 2 корзинах невозможно этим раннером.
- `run_sleeves`: НЕТ флага `--vt`. VT = третье поле спека (`strat:eq:vt`).
- `run_category_ranking --source ccxt`: обязателен `--basket crypto` +
  `--interval 4h` (крипта выгружена в H4/H1, дефолт 1d не находит).
- Балласт-вырез проверяется через `run_instrument_contribution
  --exclude`, НЕ через fullcycle (тот сравнивает стратегии, не корзины).

---

## ОТКРЫТЫЕ ХВОСТЫ
1. Вырез PL,NG подтвердить на yf (второй источник) — гигиена, не блокер.
2. tr3_tsmom vs ema_vt — прямой поединок дал BEATS на грани; НЕ менять
   чемпиона (одна история, corr 0.9). Ждать 2-й режим.
3. Финальный .docx-отчёт Александру (структура согласована).

## ИТОГ
37 новых + D.1 протестированы на 3 классах. Champion-стек сырьё/крипта
непоколебим. Equity догнан (пограничный watch tr3_tsmom, декоррелир.
mr_connors в ансамбль). D.1 закрыт с механизмом. Балласт локализован,
вырез PL,NG подтверждён (+0.14 Sharpe). Реестр 132→169. Тесты зелёные.
Приобретения: ca2_cppi (крипто-оборона), mr_connors (equity-MR divers).

---

## ДОПОЛНЕНО В КОНЦЕ СЕССИИ (дедуп, баг-фиксы, таймфреймы)

### Дедупликация каталога (реестр 169 -> 170)
- tr3_tsmom = tsmom_multi -> унифицирован в параметризованный движок +
  per-asset варианты tsmom_eq/cr1d/cr4h/comm. Удалены tr3_supertrend
  (=tr_supertrend), tr3_kama_slope (=kama_trend). tr3_zlema помечен.

### БАГ-ФИКСЫ (влияют на прошлые числа!)
1. tsmom_multi: NaN прогрева/дыр давал ЛОЖНЫЙ 0 (шорт на склейке
   фьючерсов). Теперь NaN->движок пропускает. Перемерить sweep.
2. kama_trend (живой!): NaN-вирус prepend=close[0] -> вечный 0.
   Исправлено.
3. yfinance h1/h4 -> нормализация в 1h/4h.
Двойной лаг (channel/chandelier/adx_donchian): проверено — НЕ баг,
.shift на канале это сигнал, лаг ровно 1.

### ТАЙМФРЕЙМЫ (гипотеза Кирилла проверена)
- 1h деградирует ВСЁ (MR-сырьё -0.07, крипто-тренд 0.79): микрошум+
  издержки. Гипотеза «1h для MR» ОПРОВЕРГНУТА.
- 4h оптимален для крипто/сырья-ТРЕНДА (сырьё 4h 1.25 vs 1d 0.60),
  нейтрален для MR. Таймфрейм > горизонтов.
- 1d правильный для акций (4h деградирует 1.19->0.79) и heavy-крипты.
- Акциям НЕ нужны длинные горизонты (единое окно: 21,63,252 не хуже
  63,126,252).

### LOO-МЕТОДИКА (критика Кирилла)
- |LOO|<0.03 = зона ШУМА, не ранжировать. Добавлена колонка corr с
  корзиной: держатель обоснован декорреляцией, не solo-доходностью.
- Полные имена активов (Natural Gas вместо NG). Авто-путь панели по
  --interval. ROI: нейтральные ОСТАВЛЯТЬ (диверсификация), вырезать
  только балласт вне зоны шума, потом перевзвесить.

### intraday DATA готова
XNAS.ITCH (акции, с 2018, ~$0), GLBX.MDP3 1h (сырьё). Панели 4h
(сырьё-2021, акции-2018) и 1h (сырьё) выгружены и проверены.

### ФИНАЛ: реестр 170, тесты 127 passed. Отчёт tsmom_multi_report_EN.md.
