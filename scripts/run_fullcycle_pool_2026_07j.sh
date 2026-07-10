#!/usr/bin/env bash
# =====================================================================
# ПОЛНЫЙ ЦИКЛ ДЛЯ ВСЕГО ПУЛА (2026-07j) — 94 стратегии без данных
#
# Прогоняет run_fullcycle (bootstrap разности Sharpe + корреляция с
# чемпионом) для КАЖDОЙ непроверенной стратегии, пишет всё в один CSV.
# Уже прошедшие цикл (tr3_hh_hl, ca_vol_ride, hurst_alloc и т.д.) и
# закрытые с механизмом (OU-раздел, mr_short, ca_squeeze_pop...) НЕ
# включены — их вердикты уже зафиксированы.
#
# Чемпионы ниш:  сырьё/крипта-тренд donchian_vt · сырьё-MR mr_lowvol
#                акции-тренд ema_vt · акции-MR mr_ens
#
# Порядок вывода в CSV одинаков -> ты кидаешь fullcycle.csv, я свожу
# в единый каталог с финальными вердиктами (BEATS/TIE/WORSE + ★ENSEMBLE).
#
# Пути под себя:
PANEL=data/panels/futures
CRYPTO=data/crypto
OUT=fullcycle.csv
S=2020-01-01
SC=2021-01-01
rm -f "$OUT"

TREND_VS_DONCH="--champion donchian_vt --champion-basket @DONCH_CORE_COMM"
MR_VS_LOWVOL="--champion mr_lowvol --champion-basket @MRLV_CORE_COMM"

# ─────────────────────────────────────────────────────────────────────
# СЫРЬЁ — трендовые кандидаты vs donchian_vt (оба источника)
# --family trend авто-набирает ВСЕ trend реестра; чемпион и
# уже-закрытые фильтруются самим раннером (champion исключается,
# остальные просто получат свой вердикт — дубли не страшны, CSV
# дедуплицируется на моей стороне по (strategy,source)).
# ─────────────────────────────────────────────────────────────────────
for SRC in "databento --panel-dir $PANEL" "yf"; do
  python -m runners.run_fullcycle --family trend $TREND_VS_DONCH \
      --source $SRC --start "$S" --vt --target-vol 0.20 --csv "$OUT"
done

# СЫРЬЁ — MR-кандидаты vs mr_lowvol (оба источника)
for SRC in "databento --panel-dir $PANEL" "yf"; do
  python -m runners.run_fullcycle --family mean-reversion $MR_VS_LOWVOL \
      --source $SRC --start "$S" --vt --target-vol 0.20 --csv "$OUT"
done

# СЫРЬЁ — mixed/роутеры vs donchian_vt (оба источника)
# OU-раздел закрыт целиком (--skip-prefix ou); ou_gap_fade и прочие
# провалили на обоих источниках во всех формах — не переоткрываем.
for SRC in "databento --panel-dir $PANEL" "yf"; do
  python -m runners.run_fullcycle --family mixed --champion donchian_vt \
      --skip-prefix ou \
      --source $SRC --start "$S" --vt --target-vol 0.20 --csv "$OUT"
done

# ─────────────────────────────────────────────────────────────────────
# КРИПТА H4 — трендовые + mixed кандидаты vs donchian_vt (VT 40%)
# MR на крипте не гоняем: ниша закрыта (max Sharpe +0.19 по скринингу).
# ─────────────────────────────────────────────────────────────────────
python -m runners.run_fullcycle --family trend --champion donchian_vt \
    --basket crypto --source ccxt --crypto-dir "$CRYPTO" \
    --candidate-basket @CRYPTO_CORE --champion-basket @CRYPTO_CORE \
    --interval 4h --start "$SC" --vt --target-vol 0.40 --csv "$OUT"

python -m runners.run_fullcycle --family mixed --champion donchian_vt \
    --skip-prefix ou \
    --basket crypto --source ccxt --crypto-dir "$CRYPTO" \
    --candidate-basket @CRYPTO_CORE --champion-basket @CRYPTO_CORE \
    --interval 4h --start "$SC" --vt --target-vol 0.40 --csv "$OUT"

echo ""
echo "Готово. Единый CSV полного цикла: $OUT"
echo "Пришли его — сведу в финальный каталог с вердиктами и ансамблями."
