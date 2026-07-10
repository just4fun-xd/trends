#!/usr/bin/env bash
# =====================================================================
# ЕДИНЫЙ КАТАЛОГ 132 СТРАТЕГИЙ (2026-07j) — прогон для сводного отчёта
#
# Собирает Sharpe-рейтинг ВСЕХ 132 стратегий реестра по всем классам
# активов в один CSV (catalog.csv), из которого строится единый
# отчёт-каталог (как STRATEGY_CATALOG, но полный и на фактах).
#
# 11 наборов cat_* покрывают реестр без пропусков; в каждом наборе
# чемпион ниши как якорь. --csv дописывает результаты в общий файл.
#
# Пути под себя:
PANEL=data/panels/futures
CRYPTO=data/crypto
OUT=catalog.csv
S=2020-01-01
SC=2021-01-01
rm -f "$OUT"

# ── СЫРЬЁ (databento + yf) — трендовые и MR наборы ──────────────────
# 'all' = все cat_*; но крипто-наборы на сырье бессмысленны, поэтому
# перечисляем применимые к классу наборы явно.
for SRC_ARGS in \
  "databento --panel-dir $PANEL" \
  "yf"; do
  python -m runners.run_category_ranking \
      --families cat_trend_core cat_trend_lab2 cat_trend_lab3 \
                 cat_impulse cat_mr_core cat_mr_lab2 cat_ou \
                 cat_ou_trend cat_mixed \
      --source $SRC_ARGS --start "$S" \
      --vt --target-vol 0.20 --csv "$OUT"
done

# ── АКЦИИ (yf) — трендовые, MR, импульсные, смешанные ──────────────
python -m runners.run_category_ranking \
    --families cat_trend_core cat_trend_lab3 cat_impulse \
               cat_mr_core cat_mr_lab2 cat_mixed \
    --source yf --basket equity --start "$S" \
    --vt --target-vol 0.20 --csv "$OUT"

# ── КРИПТА H4 (ccxt) — тренд/крипто-агрессивные/MR/monday ──────────
python -m runners.run_category_ranking \
    --families cat_trend_core cat_trend_lab2 cat_trend_lab3 \
               cat_crypto_aggr cat_monday cat_mr_core cat_mr_lab2 \
    --basket crypto --source ccxt --crypto-dir "$CRYPTO" \
    --interval 4h --start "$SC" --vt --target-vol 0.40 --csv "$OUT"

echo ""
echo "Готово. Единый CSV: $OUT"
echo "Пришли его — соберу единый отчёт-каталог (xlsx) по всем 132."
