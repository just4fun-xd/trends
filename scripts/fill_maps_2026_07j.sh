#!/usr/bin/env bash
# =====================================================================
# ЗАПОЛНЕНИЕ КАРТ АКТИВ×СТРАТЕГИЯ (2026-07j) — для листов 5/7/8 отчёта
#
# Гоняет run_instrument_contribution по ВСЕМ не отклонённым стратегиям
# (рейтинг [1]/[2] из каталога; seasonal исключены) на трёх классах
# активов и собирает единый maps.csv (strategy,basket,source,asset,
# solo_ret,solo_dd,solo_sharpe,loo_delta,verdict).
#
# Из maps.csv я строю полные карты:
#   лист 5 — сырьё (databento + yf),
#   лист 7 — акции (yf),
#   лист 8 — крипта (ccxt H4).
#
# Пути под себя:
PANEL=data/panels/futures
CRYPTO=data/crypto
OUT=maps.csv
S=2020-01-01
SC=2021-01-01
rm -f "$OUT"

# Списки не отклонённых по нишам (рейтинг [1]/[2]).
TREND="4step_pyr adx_donch ca_accel ca_chand ca_double_don \
ca_pyramid_max ca_thrust ca_turbo_don ca_vol_ride carver_fast \
carver_fdm carver_hicap champion chandelier channel_pos donch_multi \
donchian donchian_est_pyr donchian_vt ema_barbell ema_cross \
ema_ensemble ema_vt ewmac imp_52h imp_accel imp_skip_mom kama \
tr3_adx_di tr3_atr_mom tr3_extreme_t tr3_fracdiff tr3_hh_hl \
tr3_mid_ride tr3_persist tr3_range_exp tr3_ribbon tr3_supertrend \
tr3_tsmom tr3_vote3 tr3_zlema tr_holt tr_ichimoku trend_ens tsmom \
tsmom_multi"

MR="bb_rsi bb_rsi_vt mr2_ddband mr2_halflife mr2_kalman_z mr2_mad \
mr2_percb_bw mr2_quantile mr2_runs mr2_vr mr_atr_gate mr_atr_stop \
mr_ens mr_ens_gate mr_kelt_confirm mr_keltner mr_ladder mr_lowvol \
mr_time_stop mr_trend"

MIXED="combo_tmr hurst_alloc hurst_combo"

# ── СЫРЬЁ (databento + yf) — тренд+MR+mixed ────────────────────────
for SRC in "databento --panel-dir $PANEL" "yf"; do
  python -m runners.run_instrument_contribution --strategy $TREND $MR \
      $MIXED --source $SRC --start "$S" --vt --target-vol 0.20 \
      --csv "$OUT"
done

# ── АКЦИИ (yf) — те же (крипто-специфичные ca_* дадут пусто, ок) ──
python -m runners.run_instrument_contribution --strategy $TREND $MR \
    $MIXED --basket equity --source yf --start "$S" \
    --vt --target-vol 0.20 --csv "$OUT"

# ── КРИПТА H4 (ccxt) — тренд+mixed (MR-ниша закрыта) ──────────────
python -m runners.run_instrument_contribution --strategy $TREND \
    $MIXED --basket crypto --source ccxt --crypto-dir "$CRYPTO" \
    --interval 4h --start "$SC" --vt --target-vol 0.40 --csv "$OUT"

echo ""
echo "Готово. Единый CSV карт: $OUT"
echo "Пришли — заполню карты актив×стратегия (листы 5/7/8) по всем."
