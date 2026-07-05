#!/usr/bin/env bash
# run_core_reports.sh — компактный прогон: 4 РАБОЧИХ sleeve + 5
# сильнейших альтернатив + LOO-разбор. ~12 запусков вместо 85 полного
# run_all_reports.sh — реалистично для одного присеста без троттлинга
# yfinance/Databento. Каждый вывод сохранён в свой .txt.
#
# Запуск: bash run_core_reports.sh

set -uo pipefail

OUT="reports/core_$(date +%Y-%m-%d)"
mkdir -p "$OUT"

echo "=== 0. Юнит-тесты + flake8 ===" | tee "$OUT/00_pytest.txt"
python -m pytest tests/ -q >> "$OUT/00_pytest.txt" 2>&1
flake8 --max-line-length 79 . >> "$OUT/00_pytest.txt" 2>&1
echo "  -> $OUT/00_pytest.txt"

echo "=== 1. РАБОЧИЕ: сырьё, обе ноги, walk-forward (Databento) ==="
python -m runners.run_walkforward --strategy donchian_vt mr_kelt_confirm \
    --basket commodity --source databento --panel-dir data/panels/futures \
    --start 2019-01-01 --vt --matrix \
    > "$OUT/01_commodity_legs_walkforward.txt" 2>&1
echo "  -> $OUT/01_commodity_legs_walkforward.txt"

echo "=== 2. РАБОЧИЕ: акции, обе ноги, walk-forward (yfinance) ==="
python -m runners.run_walkforward --strategy ema_vt mr_ens \
    --basket equity --start 2019-01-01 --vt --matrix \
    > "$OUT/02_equity_legs_walkforward.txt" 2>&1
echo "  -> $OUT/02_equity_legs_walkforward.txt"

echo "=== 3. РАБОЧИЕ: сырьё, финальное комбо + плечо ==="
python -m runners.run_sleeves --sleeve donchian_vt:commodity \
    --sleeve mr_kelt_confirm:commodity:vt --parity --start 2019-01-01 \
    --source databento --panel-dir data/panels/futures --lev-sweep \
    > "$OUT/03_commodity_combo_leverage.txt" 2>&1
echo "  -> $OUT/03_commodity_combo_leverage.txt"

echo "=== 4. РАБОЧИЕ: акции, финальное комбо + плечо ==="
python -m runners.run_sleeves --sleeve ema_vt:equity \
    --sleeve mr_ens:equity:vt --parity --start 2019-01-01 --lev-sweep \
    > "$OUT/04_equity_combo_leverage.txt" 2>&1
echo "  -> $OUT/04_equity_combo_leverage.txt"

echo "=== 5. АЛЬТЕРНАТИВЫ (сильнейшие): keltner-соло, mr_ens, mr_ens_gate ==="
python -m runners.run_walkforward \
    --strategy mr_keltner mr_ens mr_ens_gate \
    --basket commodity --source databento --panel-dir data/panels/futures \
    --start 2019-01-01 --vt --matrix \
    > "$OUT/05_commodity_alt_mr_variants.txt" 2>&1
echo "  -> $OUT/05_commodity_alt_mr_variants.txt"

echo "=== 6. АЛЬТЕРНАТИВЫ: пограничные тренд-модели (правило 2 источников) ==="
python -m runners.run_walkforward --strategy tsmom_multi ewmac trend_ens \
    --basket commodity --start 2019-01-01 --vt --matrix \
    > "$OUT/06_commodity_alt_trend_yf.txt" 2>&1
python -m runners.run_walkforward --strategy tsmom_multi ewmac trend_ens \
    --basket commodity --source databento --panel-dir data/panels/futures \
    --start 2019-01-01 --vt --matrix \
    > "$OUT/06_commodity_alt_trend_databento.txt" 2>&1
echo "  -> $OUT/06_commodity_alt_trend_yf.txt"
echo "  -> $OUT/06_commodity_alt_trend_databento.txt"

echo "=== 7. LOO-вклад: состав MR-ансамбля (сырьё и акции) ==="
python -m runners.run_member_contribution --ensemble mr \
    --basket commodity --source databento --panel-dir data/panels/futures \
    --start 2019-01-01 --vt --target-vol 0.35 \
    > "$OUT/07_loo_commodity.txt" 2>&1
python -m runners.run_member_contribution --ensemble mr \
    --basket equity --start 2019-01-01 --vt --target-vol 0.35 \
    > "$OUT/07_loo_equity.txt" 2>&1
echo "  -> $OUT/07_loo_commodity.txt"
echo "  -> $OUT/07_loo_equity.txt"

echo "=== 8. Плечо на сильнейшей альтернативе (keltner-соло) ==="
python -m runners.run_sleeves --sleeve mr_keltner:commodity:vt --parity \
    --start 2019-01-01 --source databento --panel-dir data/panels/futures \
    --lev-sweep \
    > "$OUT/08_commodity_alt_keltner_leverage.txt" 2>&1
echo "  -> $OUT/08_commodity_alt_keltner_leverage.txt"

echo ""
echo "ГОТОВО. Результаты: $OUT/"
