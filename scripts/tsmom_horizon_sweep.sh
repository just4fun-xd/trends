#!/usr/bin/env bash
# TSMOM horizon sweep (10.07.26, ред. окна) — вклад инструментов для
# РАЗНЫХ наборов горизонтов: найти лучший per-asset и на КАКИХ активах.
#
# tsmom_multi параметризован; гоняем ЗАРЕГИСТРИРОВАННЫЕ варианты
# tsmom_eq (63,126,252) / tsmom_comm=cr1d (21,63,252) / tsmom_cr4h
# (42,126,504). Каждый прогон печатает solo_ret/DD/Sharpe + LOO.
#
# ЕДИНОЕ ОКНО: всё с --start 2021-01-01. Причина — 4h-панель сырья
# существует только с 2021, и честное сравнение горизонтов/таймфреймов
# требует ОДНОГО периода. Раньше часть прогонов уходила в 2019 и числа
# были несравнимы (tsmom_comm давал 0.59 в окне-2021 и 0.98 в окне-2019
# — это разница ОКОН, не горизонтов). Теперь окно фиксировано везде.
#
# Использование:
#   bash scripts/tsmom_horizon_sweep.sh            # все классы
#   bash scripts/tsmom_horizon_sweep.sh commodity  # сырьё 1d
#   bash scripts/tsmom_horizon_sweep.sh commodity4h
#   bash scripts/tsmom_horizon_sweep.sh equity | crypto1d | crypto4h
set -euo pipefail

WHICH="${1:-all}"
OUT="reports/tsmom_sweep"
START="2021-01-01"          # единое окно для ВСЕХ прогонов
mkdir -p "$OUT"

run() {  # $1=variant $2=basket $3=source $4=extra-flags $5=tag
  echo "=============================================================="
  echo ">>> TSMOM $1 | $2 | $3  (start=$START) $4"
  echo "=============================================================="
  python -m runners.run_instrument_contribution \
      --strategy "$1" --basket "$2" --source "$3" \
      --start "$START" \
      --vt --target-vol "${TVOL:-0.20}" $4 \
      | tee "$OUT/tsmom_${5}.txt"
  echo
}

# ── Сырьё 1d: три набора горизонтов, ОДНО окно ───────────────────────
# Сравниваем ГОРИЗОНТЫ при фиксированных ТФ(1d) и окне(2021).
if [[ "$WHICH" == "all" || "$WHICH" == "commodity" ]]; then
  TVOL=0.20
  run tsmom_comm commodity databento "--exclude PA,PL" "comm_1d_21_63_252"
  run tsmom_eq   commodity databento "--exclude PA,PL" "comm_1d_63_126_252"
  run tsmom_cr4h commodity databento "--exclude PA,PL" "comm_1d_42_126_504"
fi

# ── Сырьё 4h: та же сетка на 4h-панели (авто-путь по --interval) ──────
# Сравниваем ТАЙМФРЕЙМ (4h vs 1d выше) при том же окне и горизонтах.
# PA,PL исключены (тонкое 4h-покрытие). --panel-dir больше не нужен:
# раннер сам берёт data/panels_4h/futures по --interval 4h.
if [[ "$WHICH" == "all" || "$WHICH" == "commodity4h" ]]; then
  TVOL=0.20
  run tsmom_cr4h commodity databento \
      "--interval 4h --exclude PA,PL" "comm_4h_42_126_504"
  run tsmom_comm commodity databento \
      "--interval 4h --exclude PA,PL" "comm_4h_21_63_252"
fi

# ── Акции 1d ─────────────────────────────────────────────────────────
if [[ "$WHICH" == "all" || "$WHICH" == "equity" ]]; then
  TVOL=0.20
  run tsmom_eq equity yf "" "equity_1d_63_126_252"
  run tsmom_comm equity yf "" "equity_1d_21_63_252"
fi

# ── Крипта 1d ────────────────────────────────────────────────────────
if [[ "$WHICH" == "all" || "$WHICH" == "crypto1d" ]]; then
  TVOL=0.40
  run tsmom_cr1d crypto yf "" "crypto_1d_21_63_252"
fi

# ── Крипта 4h (ccxt) ─────────────────────────────────────────────────
if [[ "$WHICH" == "all" || "$WHICH" == "crypto4h" ]]; then
  TVOL=0.40
  run tsmom_cr4h crypto ccxt "--interval 4h" "crypto_4h_42_126_504"
fi

echo "Готово. Тексты в $OUT/. Всё в окне $START — числа СРАВНИМЫ."
echo "Сравни 'Полный портфель Sharpe' между наборами: выбери per-asset"
echo "горизонт по лучшему full_sharpe. Сырьё любит короткий старт (21),"
echo "акции — длинный (63); проверь, устойчиво ли это в едином окне."
