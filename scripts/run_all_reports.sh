#!/usr/bin/env bash
# run_all_reports.sh — прогон юнит-тестов + walk-forward ВСЕХ 40
# стратегий реестра (commodity: yfinance И Databento; equity: yfinance)
# с сохранением каждого вывода в свой .txt через перенаправление.
#
# Запуск: bash run_all_reports.sh
# Результат: папка reports/ с одним .txt на каждый прогон + summary.txt
#
# Список стратегий получен из живого реестра STRATEGIES (не вручную),
# поэтому скрипт не разъедется с кодом при добавлении новых стратегий —
# перегенерировать список: python -c "from runners.run_basket import
# STRATEGIES; print(' '.join(sorted(STRATEGIES)))"

set -uo pipefail  # без -e: одна упавшая стратегия не должна рвать batch

OUT="reports/$(date +%Y-%m-%d)"
mkdir -p "$OUT"

# Equity-семейство идёт на equity-корзину, всё остальное — commodity.
EQUITY_STRATS="ema_barbell ema_cross ema_ensemble ema_vt"

# Полный список стратегий (commodity-семейство: всё, что не equity).
ALL_STRATS="4step_pyr adx_donch bb_rsi bb_rsi_vt champion chandelier \
channel_pos combo_tmr donch_multi donch_seasonal donch_seasonal_vt \
donchian donchian_est_pyr donchian_vt ewmac kama mr_atr_gate \
mr_atr_stop mr_confirm mr_connors mr_ens mr_ens_gate mr_kelt_confirm \
mr_keltner mr_ladder mr_lowvol mr_scaled mr_short mr_time_stop \
mr_trend ou seasonal seasonal_vt trend_ens tsmom tsmom_multi"

echo "=== 1/4: юнит-тесты проекта ===" | tee "$OUT/00_pytest.txt"
python -m pytest tests/ -q >> "$OUT/00_pytest.txt" 2>&1
echo "flake8:" >> "$OUT/00_pytest.txt"
flake8 --max-line-length 79 . >> "$OUT/00_pytest.txt" 2>&1
echo "  -> $OUT/00_pytest.txt"

echo "=== 2/4: commodity, walk-forward, yfinance ===" 
for s in $ALL_STRATS; do
    f="$OUT/commodity_yf_${s}.txt"
    echo "  $s -> $f"
    python -m runners.run_walkforward --strategy "$s" \
        --basket commodity --start 2019-01-01 --vt --matrix \
        > "$f" 2>&1
done

echo "=== 3/4: commodity, walk-forward, Databento (roll-adjusted) ==="
for s in $ALL_STRATS; do
    f="$OUT/commodity_databento_${s}.txt"
    echo "  $s -> $f"
    python -m runners.run_walkforward --strategy "$s" \
        --basket commodity --source databento \
        --panel-dir data/panels/futures \
        --start 2019-01-01 --vt --matrix \
        > "$f" 2>&1
done

echo "=== 4/4: equity, walk-forward, yfinance ==="
for s in $EQUITY_STRATS mr_ens mr_ens_gate mr_kelt_confirm mr_atr_stop \
         mr_time_stop mr_keltner mr_confirm; do
    f="$OUT/equity_yf_${s}.txt"
    echo "  $s -> $f"
    python -m runners.run_walkforward --strategy "$s" \
        --basket equity --start 2019-01-01 --vt --matrix \
        > "$f" 2>&1
done

# Сводка: какие РОБАСТНА / ПОД ВОПРОСОМ по каждому файлу — быстрый
# просмотр без открытия всех txt по отдельности.
SUMMARY="$OUT/summary.txt"
echo "Сводка вердиктов walk-forward ($(date))" > "$SUMMARY"
echo "==========================================" >> "$SUMMARY"
for f in "$OUT"/commodity_*.txt "$OUT"/equity_*.txt; do
    verdict=$(grep -oE "РОБАСТНА|ПОД ВОПРОСОМ" "$f" | head -1)
    name=$(basename "$f" .txt)
    printf "%-45s %s\n" "$name" "${verdict:-ОШИБКА/НЕТ ДАННЫХ}" >> "$SUMMARY"
done
echo "" >> "$SUMMARY"
echo "Готово. Все файлы в $OUT/, сводка выше." >> "$SUMMARY"
cat "$SUMMARY"

echo ""
echo "ГОТОВО. Результаты: $OUT/"
echo "Сводка: $SUMMARY"
