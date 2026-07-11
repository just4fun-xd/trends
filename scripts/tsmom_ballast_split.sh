#!/usr/bin/env bash
# tsmom по корзинам БЕЗ БАЛЛАСТА + отдельно ТОЛЬКО ДЕРЖАТЕЛИ.
# Источник состава — sweep 10.07.26 (единое окно 2021-2025, v24 с corr).
# Тикеры по вердиктам: БАЛЛАСТ (LOO>+0.05, вне зоны шума) исключён,
# «держать» (LOO<-0.05) — второй прогон отдельно. Нейтраль/шум НЕ
# входит ни в тот, ни в другой список ниже (сознательно — см. таблицу
# под каждым блоком), при желании добавь их в --assets вручную.
#
# Замечание про честность: это ГИПОТЕЗА по LOO на ОДНОМ окне/классе
# горизонтов, не bootstrap-подтверждение. Перед тем как менять боевую
# корзину — прогнать fullcycle против чемпиона на этой урезанной
# корзине и втором источнике (см. финал файла).
set -euo pipefail

# ── СЫРЬЁ 4h (tsmom_cr4h 42,126,504 — лучший вариант, Sharpe 1.25) ────
# БАЛЛАСТ: Soybeans(ZS) LOO+0.13, Corn(ZC) LOO+0.08
# ДЕРЖАТЬ:  Crude Oil(CL) -0.05, Natural Gas(NG) -0.10,
#           Soybean Oil(ZL) -0.11, Silver(SI) -0.15, Gold(GC) -0.19
# шум/нейтраль исключены из обоих списков: Wheat(ZW), Soybean Meal(ZM), Copper(HG)
echo "=== СЫРЬЁ 4h — БЕЗ БАЛЛАСТА (без ZS,ZC) ==="
python -m runners.run_strategy_test --strategy tsmom_cr4h \
    --basket commodity --source databento --panel-dir data/panels_4h/futures \
    --interval 4h --assets CL,NG,GC,SI,HG,ZW,ZL,ZM --start 2021-01-01 \
    --vt --target-vol 0.20

echo "=== СЫРЬЁ 4h — ТОЛЬКО ДЕРЖАТЕЛИ (CL,NG,ZL,SI,GC) ==="
python -m runners.run_strategy_test --strategy tsmom_cr4h \
    --basket commodity --source databento --panel-dir data/panels_4h/futures \
    --interval 4h --assets CL,NG,ZL,SI,GC --start 2021-01-01 \
    --vt --target-vol 0.20


# ── СЫРЬЁ 1d (tsmom_comm 21,63,252 — Sharpe 0.70) ─────────────────────
# БАЛЛАСТ: Wheat(ZW) LOO+0.08 (единственный вне шума)
# ДЕРЖАТЬ:  Silver(SI) -0.07, Gold(GC) -0.19
echo "=== СЫРЬЁ 1d — БЕЗ БАЛЛАСТА (без ZW) ==="
python -m runners.run_strategy_test --strategy tsmom_comm \
    --basket commodity --source databento --panel-dir data/panels/futures \
    --assets CL,NG,GC,SI,HG,ZC,ZS,ZL,ZM --start 2021-01-01 \
    --vt --target-vol 0.20

echo "=== СЫРЬЁ 1d — ТОЛЬКО ДЕРЖАТЕЛИ (SI,GC) ==="
python -m runners.run_strategy_test --strategy tsmom_comm \
    --basket commodity --source databento --panel-dir data/panels/futures \
    --assets SI,GC --start 2021-01-01 --vt --target-vol 0.20


# ── АКЦИИ (tsmom_comm 21,63,252 — Sharpe 1.18, лучше tsmom_eq в этом окне) ─
# БАЛЛАСТ: Mastercard(MA) LOO+0.08, Pepsi(PEP) LOO+0.07
# ДЕРЖАТЬ:  JPMorgan(JPM) -0.05, Meta(META) -0.05, Costco(COST) -0.07,
#           Nvidia(NVDA) -0.12
echo "=== АКЦИИ — БЕЗ БАЛЛАСТА (без MA,PEP) ==="
python -m runners.run_strategy_test --strategy tsmom_comm \
    --basket equity --source yf \
    --assets AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,V,WMT,JNJ,PG,HD,KO,MRK,COST,AMD \
    --start 2021-01-01 --vt --target-vol 0.20

echo "=== АКЦИИ — ТОЛЬКО ДЕРЖАТЕЛИ (JPM,META,COST,NVDA) ==="
python -m runners.run_strategy_test --strategy tsmom_comm \
    --basket equity --source yf --assets JPM,META,COST,NVDA \
    --start 2021-01-01 --vt --target-vol 0.20


# ── КРИПТА 1d (tsmom_cr1d 21,63,252 — Sharpe 0.87) ────────────────────
# БАЛЛАСТ: Litecoin(LTC) LOO+0.07
# ДЕРЖАТЬ:  BNB -0.06, Solana(SOL) -0.06, Tron(TRX) -0.11
echo "=== КРИПТА 1d — БЕЗ БАЛЛАСТА (без LTC) ==="
python -m runners.run_strategy_test --strategy tsmom_cr1d \
    --basket crypto --source yf \
    --assets BTC-USD,ETH-USD,BNB-USD,SOL-USD,XRP-USD,ADA-USD,AVAX-USD,DOT-USD,TRX-USD,LINK-USD,BCH-USD,XLM-USD,ATOM-USD,NEAR-USD \
    --start 2021-01-01 --vt --target-vol 0.40

echo "=== КРИПТА 1d — ТОЛЬКО ДЕРЖАТЕЛИ (BNB,SOL,TRX) ==="
python -m runners.run_strategy_test --strategy tsmom_cr1d \
    --basket crypto --source yf --assets BNB-USD,SOL-USD,TRX-USD \
    --start 2021-01-01 --vt --target-vol 0.40


# ── КРИПТА 4h (tsmom_cr4h 42,126,504, ccxt — Sharpe 0.90) ─────────────
# БАЛЛАСТ: нет вне зоны шума в этом прогоне (все нейтраль/шум кроме
# держателей) — блок «без балласта» СОВПАДАЕТ с полной вселенной,
# пропущен намеренно.
# ДЕРЖАТЬ: Solana(SOL) -0.07, Tron(TRX) -0.09
echo "=== КРИПТА 4h — ТОЛЬКО ДЕРЖАТЕЛИ (SOL,TRX) ==="
python -m runners.run_strategy_test --strategy tsmom_cr4h \
    --basket crypto --source ccxt --interval 4h \
    --assets SOL-USDT,TRX-USDT --start 2021-01-01 --vt --target-vol 0.40


echo
echo "=== ВСЁ РАЗОМ (контроль, полная вселенная, все классы) ==="
python -m runners.run_strategy_test --strategy tsmom_comm --basket all \
    --panel-dir data/panels/futures --crypto-dir data/crypto \
    --start 2021-01-01 --vt --target-vol 0.20

echo
echo "ВАЖНО: числа выше — гипотеза (урезка по LOO на одном окне), НЕ"
echo "арбитраж. Перед фиксацией боевой корзины подтверди bootstrap:"
echo "  python -m runners.run_fullcycle --candidates tsmom_cr4h \\"
echo "      --champion donchian_vt --source databento --basket commodity \\"
echo "      --candidate-basket 'CL,NG,ZL,SI,GC' --interval 4h \\"
echo "      --vt --target-vol 0.20 --csv fc_tsmom_holders_only.csv"
