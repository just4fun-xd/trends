#!/usr/bin/env bash
# =====================================================================
# СКРИНИНГ 40 НОВЫХ СТРАТЕГИЙ (2026-07j) — проверенная последовательность
#
# Философия (см. критику «гонять только на рабочих активах»):
#   ЭТАП A — СКРИНИНГ идёт по ПОЛНОЙ вселенной. «Рабочая корзина» новой
#            стратегии ещё НЕ известна: балласт — свойство пары
#            (актив×стратегия), а @DONCH_CORE_COMM скроена под donchian.
#            Отбор на чужой корзине = фора чемпиону структурой теста.
#            --exclude только для битых данных (тонкие H4 PA/PL).
#   ЭТАП B — instrument_contribution СТРОИТ корзину выжившего из данных.
#   ЭТАП C — АРБИТРАЖ (bootstrap) идёт на корзине выжившего против
#            чемпиона. Разность Sharpe с CI — единственный судья.
#   ЭТАП D — робастность (walk-forward) + второй источник = вердикт.
#
# Правило: одна гипотеза за раз, без ре-тюнинга после теста.
# 40 моделей => ~6-8 ложных победителей; спасают ТОЛЬКО два источника.
#
# Подставь свои пути к данным:
PANEL=data/panels/futures        # databento-панели сырья
CRYPTO=data/crypto               # ccxt parquet-свечи
S=2020-01-01                     # сырьё/акции
SC=2021-01-01                    # крипта H4
# =====================================================================


# ─────────────────────────────────────────────────────────────────────
# ЭТАП A — СКРИНИНГ «кто вообще в игре» (полная вселенная)
# ─────────────────────────────────────────────────────────────────────

# A1. Трендовая лаборатория 3 vs donchian_vt (сырьё, databento).
#     donchian_vt включён в семейство якорем — видно, кто дотягивается.
python -m runners.run_category_ranking --families trend3 \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.20

# A2. То же на втором источнике (yfinance) — совпадение верхушки
#     рейтинга между источниками отсекает случайных лидеров.
python -m runners.run_category_ranking --families trend3 \
    --source yf --start "$S" --vt --target-vol 0.20

# A3. MR-лаборатория 2 vs mr_lowvol (сырьё, оба источника).
python -m runners.run_category_ranking --families mr2 \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.20
python -m runners.run_category_ranking --families mr2 \
    --source yf --start "$S" --vt --target-vol 0.20

# A4. Крипто-агрессивные vs donchian_vt + tr_ichimoku (H4, ccxt).
#     VT 40% — как весь крипто-трек. PA/PL тут не при чём (сырьё).
python -m runners.run_category_ranking --families crypto_aggr \
    --basket crypto --source ccxt --crypto-dir "$CRYPTO" \
    --interval 4h --start "$SC" --vt --target-vol 0.40

# A5. Режимная карта: какие крипто-активы трендовые/реверсионные для
#     новых стратегий (перед тем как строить корзины ног).
python -m runners.run_regime_map \
    --trend donchian_vt tr3_supertrend tr3_adx_di ca_turbo_don \
    --mr mr_lowvol mr2_kalman_z mr2_vr \
    --basket crypto --source ccxt --crypto-dir "$CRYPTO" \
    --interval 4h --start "$SC" --vt --target-vol 0.40


# ─────────────────────────────────────────────────────────────────────
# ЭТАП B — КОРЗИНА ВЫЖИВШЕГО (для КАЖДОГО, кто прошёл A на 2 источниках)
# ─────────────────────────────────────────────────────────────────────
# Пример для tr3_supertrend — замени именем своего выжившего.
# Смотришь строки «держать» -> это и есть корзина ноги для ЭТАПА C.
# ПОЛНАЯ вселенная (без --include!) — иначе балласт не выявить.
python -m runners.run_instrument_contribution \
    --strategy tr3_supertrend \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.20
python -m runners.run_instrument_contribution \
    --strategy tr3_supertrend \
    --source yf --start "$S" --vt --target-vol 0.20
# Балласт исключать, только если он балласт на ОБОИХ источниках И есть
# механизм, почему актив не торгуется этой стратегией.


# ─────────────────────────────────────────────────────────────────────
# ЭТАП C — АРБИТРАЖ выжившего против чемпиона (на корзине выжившего)
# ─────────────────────────────────────────────────────────────────────
# Здесь --include УМЕСТЕН: корзина построена из данных на ЭТАПЕ B, а не
# заимствована у чемпиона. Пример на @DONCH_CORE_COMM — подставь свою.
# Оба источника; bootstrap на gross (без --rf — rf на портфельном слое).
python -m runners.run_bootstrap --a tr3_supertrend --b donchian_vt \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.20 --include @DONCH_CORE_COMM
python -m runners.run_bootstrap --a tr3_supertrend --b donchian_vt \
    --source yf --start "$S" \
    --vt --target-vol 0.20 --include @DONCH_CORE_COMM

# MR-выживший против mr_lowvol на реверсионной корзине.
python -m runners.run_bootstrap --a mr2_kalman_z --b mr_lowvol \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.20 --include @MRLV_CORE_COMM
python -m runners.run_bootstrap --a mr2_kalman_z --b mr_lowvol \
    --source yf --start "$S" \
    --vt --target-vol 0.20 --include @MRLV_CORE_COMM

# Крипто-выживший против donchian_vt (H4, честный bpy внутри раннера).
python -m runners.run_bootstrap --a ca_turbo_don --b donchian_vt \
    --basket crypto --source ccxt --crypto-dir "$CRYPTO" \
    --interval 4h --start "$SC" --vt --target-vol 0.40 \
    --include @CRYPTO_CORE


# ─────────────────────────────────────────────────────────────────────
# ЭТАП D — РОБАСТНОСТЬ выжившего (walk-forward, по годам, 2 источника)
# ─────────────────────────────────────────────────────────────────────
python -m runners.run_walkforward \
    --strategy tr3_supertrend donchian_vt \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.20 --by year --include @DONCH_CORE_COMM
python -m runners.run_walkforward \
    --strategy tr3_supertrend donchian_vt \
    --source yf --start "$S" \
    --vt --target-vol 0.20 --by year --include @DONCH_CORE_COMM

# Крипта — walk-forward выжившего против чемпиона на H4.
python -m runners.run_walkforward \
    --strategy ca_turbo_don donchian_vt \
    --basket crypto --source ccxt --crypto-dir "$CRYPTO" \
    --interval 4h --start "$SC" --vt --target-vol 0.40 \
    --by year --include @CRYPTO_CORE


# ─────────────────────────────────────────────────────────────────────
# ЭТАП E — КОРРЕЛЯЦИЯ (кандидат в ансамбль, только прошедшие C+D)
# ─────────────────────────────────────────────────────────────────────
# Некоррелированность (<0.6) с чемпионом — единственная причина брать
# выжившего в ансамбль. НО: урок OU — некоррелированный СЛАБЫЙ член
# всё равно портит комбо. Сначала не хуже чемпиона (C), потом corr.
python -m runners.run_sleeves \
    --sleeve donchian_vt:commodity:vt@DONCH_CORE_COMM \
    --sleeve tr3_supertrend:commodity:vt@DONCH_CORE_COMM \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --target-vol 0.60 --parity
# Смотришь корреляцию ног в выводе run_sleeves; <0.6 -> кандидат.
