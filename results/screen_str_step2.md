python -m runners.run_walkforward \
    --strategy tr3_hh_hl tr3_ribbon tr3_mid_ride tr3_atr_mom \
    tr3_fracdiff tr3_vote3 tr3_zlema donchian_vt \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.20 --by year
Walk-forward (anchored, by=year) | commodity | databento | 2020-01-01..2026-01-01

tr3_hh_hl +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +7.0% / +9.2%
  Лучшее / худшее:   +15.8% / -3.0%
  Разброс окон:      18.8% (мираж на 1 годе, если велик)

tr3_ribbon +VT(realized) — ПОД ВОПРОСОМ
  Окон прибыльных:  67% (6 окон)
  Средний / медиана: +6.6% / +7.4%
  Лучшее / худшее:   +21.3% / -4.5%
  Разброс окон:      25.8% (мираж на 1 годе, если велик)

tr3_mid_ride +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +6.0% / +5.0%
  Лучшее / худшее:   +20.6% / -4.4%
  Разброс окон:      25.0% (мираж на 1 годе, если велик)

tr3_atr_mom +VT(realized) — ПОД ВОПРОСОМ
  Окон прибыльных:  67% (6 окон)
  Средний / медиана: +3.1% / +2.5%
  Лучшее / худшее:   +11.1% / -2.4%
  Разброс окон:      13.5% (мираж на 1 годе, если велик)

tr3_fracdiff +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +6.1% / +6.5%
  Лучшее / худшее:   +17.7% / -6.7%
  Разброс окон:      24.4% (мираж на 1 годе, если велик)

tr3_vote3 +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +6.8% / +6.4%
  Лучшее / худшее:   +20.8% / -5.3%
  Разброс окон:      26.1% (мираж на 1 годе, если велик)

tr3_zlema +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +4.5% / +3.5%
  Лучшее / худшее:   +10.7% / -1.7%
  Разброс окон:      12.4% (мираж на 1 годе, если велик)

donchian_vt +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +4.6% / +3.7%
  Лучшее / худшее:   +14.1% / -1.2%
  Разброс окон:      15.3% (мираж на 1 годе, если велик)

python -m runners.run_walkforward \
    --strategy tr3_hh_hl tr3_ribbon tr3_mid_ride tr3_atr_mom \
    tr3_fracdiff tr3_vote3 tr3_zlema donchian_vt \
    --source yf --start "$S" --vt --target-vol 0.20 --by year
Walk-forward (anchored, by=year) | commodity | yf | 2020-01-01..2026-01-01

tr3_hh_hl +VT(realized) — ПОД ВОПРОСОМ
  Окон прибыльных:  67% (6 окон)
  Средний / медиана: +5.7% / +6.7%
  Лучшее / худшее:   +14.1% / -0.8%
  Разброс окон:      14.9% (мираж на 1 годе, если велик)

tr3_ribbon +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +6.0% / +6.0%
  Лучшее / худшее:   +12.5% / -2.0%
  Разброс окон:      14.5% (мираж на 1 годе, если велик)

tr3_mid_ride +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +4.6% / +3.4%
  Лучшее / худшее:   +16.8% / -3.6%
  Разброс окон:      20.4% (мираж на 1 годе, если велик)

tr3_atr_mom +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +3.0% / +2.6%
  Лучшее / худшее:   +7.7% / -1.7%
  Разброс окон:      9.3% (мираж на 1 годе, если велик)

tr3_fracdiff +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +5.9% / +6.0%
  Лучшее / худшее:   +16.1% / -6.6%
  Разброс окон:      22.7% (мираж на 1 годе, если велик)

tr3_vote3 +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +5.3% / +4.1%
  Лучшее / худшее:   +14.9% / -3.2%
  Разброс окон:      18.1% (мираж на 1 годе, если велик)

tr3_zlema +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +3.7% / +4.1%
  Лучшее / худшее:   +8.6% / -3.0%
  Разброс окон:      11.6% (мираж на 1 годе, если велик)

donchian_vt +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +4.0% / +3.9%
  Лучшее / худшее:   +9.7% / -1.0%
  Разброс окон:      10.7% (мираж на 1 годе, если велик)

python -m runners.run_walkforward \
    --strategy mr2_percb_bw mr2_runs mr2_quantile mr2_vr mr_lowvol \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.20 --by year

Walk-forward (anchored, by=year) | commodity | databento | 2020-01-01..2026-01-01

mr2_percb_bw +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +2.7% / +3.2%
  Лучшее / худшее:   +5.8% / -2.1%
  Разброс окон:      7.9% (мираж на 1 годе, если велик)

mr2_runs +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +1.7% / +1.2%
  Лучшее / худшее:   +4.4% / -1.0%
  Разброс окон:      5.4% (мираж на 1 годе, если велик)

mr2_quantile +VT(realized) — РОБАСТНА
  Окон прибыльных:  100% (6 окон)
  Средний / медиана: +3.6% / +3.2%
  Лучшее / худшее:   +7.7% / +1.1%
  Разброс окон:      6.6% (мираж на 1 годе, если велик)
/Users/shalygin/dev/Python_work/trends/.venv/lib/python3.12/site-packages/pandas/core/arraylike.py:402: RuntimeWarning: invalid value encountered in log
  result = getattr(ufunc, method)(*inputs, **kwargs)

mr2_vr +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +1.9% / +2.3%
  Лучшее / худшее:   +5.3% / -2.9%
  Разброс окон:      8.2% (мираж на 1 годе, если велик)

mr_lowvol +VT(realized) — РОБАСТНА
  Окон прибыльных:  100% (6 окон)
  Средний / медиана: +2.9% / +2.7%
  Лучшее / худшее:   +4.7% / +1.8%
  Разброс окон:      2.9% (мираж на 1 годе, если велик)

python -m runners.run_walkforward \
    --strategy mr2_percb_bw mr2_runs mr2_quantile mr2_vr mr_lowvol \
    --source yf --start "$S" --vt --target-vol 0.20 --by year
Walk-forward (anchored, by=year) | commodity | yf | 2020-01-01..2026-01-01

mr2_percb_bw +VT(realized) — РОБАСТНА
  Окон прибыльных:  100% (6 окон)
  Средний / медиана: +3.8% / +4.2%
  Лучшее / худшее:   +6.6% / +0.7%
  Разброс окон:      5.9% (мираж на 1 годе, если велик)

mr2_runs +VT(realized) — РОБАСТНА
  Окон прибыльных:  100% (6 окон)
  Средний / медиана: +1.9% / +1.9%
  Лучшее / худшее:   +3.2% / +0.0%
  Разброс окон:      3.2% (мираж на 1 годе, если велик)

mr2_quantile +VT(realized) — РОБАСТНА
  Окон прибыльных:  100% (6 окон)
  Средний / медиана: +4.1% / +4.5%
  Лучшее / худшее:   +6.7% / +0.9%
  Разброс окон:      5.8% (мираж на 1 годе, если велик)
/Users/shalygin/dev/Python_work/trends/.venv/lib/python3.12/site-packages/pandas/core/arraylike.py:402: RuntimeWarning: invalid value encountered in log
  result = getattr(ufunc, method)(*inputs, **kwargs)

mr2_vr +VT(realized) — ПОД ВОПРОСОМ
  Окон прибыльных:  67% (6 окон)
  Средний / медиана: +1.8% / +1.4%
  Лучшее / худшее:   +6.2% / -1.3%
  Разброс окон:      7.5% (мираж на 1 годе, если велик)

mr_lowvol +VT(realized) — РОБАСТНА
  Окон прибыльных:  100% (6 окон)
  Средний / медиана: +2.8% / +2.5%
  Лучшее / худшее:   +4.3% / +1.7%
  Разброс окон:      2.6% (мираж на 1 годе, если велик)


((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_bootstrap --a tr3_hh_hl --b donchian_vt \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.20 --include CL,SI,HG,ZS,GC,NG
python -m runners.run_bootstrap --a tr3_hh_hl --b donchian_vt \
    --source yf --start "$S" --vt --target-vol 0.20 \
    --include Crude_Oil,Silver,Copper,Soybeans,Gold,Natural_Gas
Bootstrap tr3_hh_hl vs donchian_vt | commodity | databento | rf=0.0% | CI 90%
  tr3_hh_hl          Sharpe +1.49  CI [+0.86, +2.11]
  donchian_vt        Sharpe +1.34  CI [+0.71, +1.99]

  Разность Sharpe (tr3_hh_hl − donchian_vt): +0.15  CI [-0.37, +0.67] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 12 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.
ВНИМАНИЕ: --include токены без совпадения (пропущены): ['Crude_Oil', 'Natural_Gas']
Bootstrap tr3_hh_hl vs donchian_vt | commodity | yf | rf=0.0% | CI 90%
  tr3_hh_hl          Sharpe +1.25  CI [+0.54, +1.91]
  donchian_vt        Sharpe +1.09  CI [+0.37, +1.79]

  Разность Sharpe (tr3_hh_hl − donchian_vt): +0.16  CI [-0.45, +0.76] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 11 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.
((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_bootstrap --a tr3_hh_hl --b donchian_vt \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.40 --include CL,SI,HG,ZS,GC,NG
python -m runners.run_bootstrap --a tr3_hh_hl --b donchian_vt \
    --source yf --start "$S" --vt --target-vol 0.40 \
    --include Crude_Oil,Silver,Copper,Soybeans,Gold,Natural_Gas
Bootstrap tr3_hh_hl vs donchian_vt | commodity | databento | rf=0.0% | CI 90%
  tr3_hh_hl          Sharpe +1.38  CI [+0.76, +2.00]
  donchian_vt        Sharpe +1.27  CI [+0.66, +1.90]

  Разность Sharpe (tr3_hh_hl − donchian_vt): +0.11  CI [-0.39, +0.62] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 12 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.
ВНИМАНИЕ: --include токены без совпадения (пропущены): ['Crude_Oil', 'Natural_Gas']
Bootstrap tr3_hh_hl vs donchian_vt | commodity | yf | rf=0.0% | CI 90%
  tr3_hh_hl          Sharpe +1.18  CI [+0.49, +1.83]
  donchian_vt        Sharpe +0.98  CI [+0.27, +1.66]

  Разность Sharpe (tr3_hh_hl − donchian_vt): +0.20  CI [-0.39, +0.78] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 11 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.

  ((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_bootstrap --a tr3_ribbon --b donchian_vt \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.20 --include ZC,SI,ZL,CL,GC
python -m runners.run_bootstrap --a tr3_ribbon --b donchian_vt \
    --source yf --start "$S" --vt --target-vol 0.20 \
    --include Corn,Silver,Soybean_Oil,Crude_Oil,Gold
Bootstrap tr3_ribbon vs donchian_vt | commodity | databento | rf=0.0% | CI 90%
  tr3_ribbon         Sharpe +1.38  CI [+0.71, +2.04]
  donchian_vt        Sharpe +1.28  CI [+0.57, +1.97]

  Разность Sharpe (tr3_ribbon − donchian_vt): +0.10  CI [-0.34, +0.54] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 12 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.
ВНИМАНИЕ: --include токены без совпадения (пропущены): ['Soybean_Oil', 'Crude_Oil']
Bootstrap tr3_ribbon vs donchian_vt | commodity | yf | rf=0.0% | CI 90%
  tr3_ribbon         Sharpe +0.98  CI [+0.29, +1.65]
  donchian_vt        Sharpe +0.72  CI [+0.01, +1.43]

  Разность Sharpe (tr3_ribbon − donchian_vt): +0.26  CI [-0.21, +0.69] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 11 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.
((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_bootstrap --a tr3_ribbon --b donchian_vt \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.40 --include ZC,SI,ZL,CL,GC
python -m runners.run_bootstrap --a tr3_ribbon --b donchian_vt \
    --source yf --start "$S" --vt --target-vol 0.40 \
    --include Corn,Silver,Soybean_Oil,Crude_Oil,Gold
Bootstrap tr3_ribbon vs donchian_vt | commodity | databento | rf=0.0% | CI 90%
  tr3_ribbon         Sharpe +1.31  CI [+0.64, +1.97]
  donchian_vt        Sharpe +1.25  CI [+0.55, +1.92]

  Разность Sharpe (tr3_ribbon − donchian_vt): +0.05  CI [-0.37, +0.50] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 12 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.
ВНИМАНИЕ: --include токены без совпадения (пропущены): ['Soybean_Oil', 'Crude_Oil']
Bootstrap tr3_ribbon vs donchian_vt | commodity | yf | rf=0.0% | CI 90%
  tr3_ribbon         Sharpe +0.89  CI [+0.22, +1.55]
  donchian_vt        Sharpe +0.63  CI [-0.07, +1.33]

  Разность Sharpe (tr3_ribbon − donchian_vt): +0.26  CI [-0.19, +0.67] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 11 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.

  ((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_bootstrap --a mr2_percb_bw --b mr_lowvol \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --vt --target-vol 0.20 --include @MRLV_CORE_COMM
python -m runners.run_bootstrap --a mr2_percb_bw --b mr_lowvol \
    --source yf --start "$S" --vt --target-vol 0.20 --include @MRLV_CORE_COMM
Bootstrap mr2_percb_bw vs mr_lowvol | commodity | databento | rf=0.0% | CI 90%
  mr2_percb_bw       Sharpe +1.35  CI [+0.77, +1.95]
  mr_lowvol          Sharpe +1.70  CI [+1.16, +2.21]

  Разность Sharpe (mr2_percb_bw − mr_lowvol): -0.35  CI [-0.88, +0.23] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 12 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.
Bootstrap mr2_percb_bw vs mr_lowvol | commodity | yf | rf=0.0% | CI 90%
  mr2_percb_bw       Sharpe +1.30  CI [+0.66, +2.01]
  mr_lowvol          Sharpe +1.56  CI [+0.97, +2.12]

  Разность Sharpe (mr2_percb_bw − mr_lowvol): -0.25  CI [-0.78, +0.33] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 11 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.

  ((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_sleeves \
    --sleeve donchian_vt:commodity:vt@DONCH_CORE_COMM \
    --sleeve tr3_hh_hl:commodity:vt@HHHL_CORE_COMM \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --target-vol 0.60 --parity
Sleeve donchian_vt_commodity_vt [6 инстр] (databento, 1d) ...
Sleeve tr3_hh_hl_commodity_vt [6 инстр] (databento, 1d) ...

=== Sleeve'ы по отдельности (gross Sharpe) ===
  donchian_vt_commodity_vt     ret +144.5%  DD -13.1%  Sharpe +1.32
  tr3_hh_hl_commodity_vt       ret +283.0%  DD -15.1%  Sharpe +1.26

Корреляция дневных P&L (главное число прогона):
                          donchian_vt_commodity_vt  tr3_hh_hl_commodity_vt
donchian_vt_commodity_vt                      1.00                    0.59
tr3_hh_hl_commodity_vt                        0.59                    1.00

=== КОМБО (vol-parity (trailing 63, shift 1)) ===
  Доходность: +194.1%  (годовая: +15.7%)
  Max DD:     -11.0%
  Sharpe:     +1.44
  Годовая вола комбо: 10.5% (доля лимита DD<40%: ~26%)
  Проходит DD<40%: да

Комбо по годам (gross Sharpe)
   Год     Return     MaxDD   Sharpe   Бары
  ------------------------------------------
  2020    +26.7%    -5.5%    +2.20    311
  2021    +34.5%    -5.7%    +2.24    310
  2022    +24.9%   -10.5%    +1.56    310
  2023     -0.3%    -7.5%    +0.01    309
  2024    +10.0%    -8.9%    +0.86    313
  2025    +25.9%    -8.6%    +1.50    312
  ------------------------------------------
  ИТОГ   +194.1%   -10.5%    +1.39         (компаунд / худший год / ср.Sharpe)

  ((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_sleeves \
    --sleeve donchian_vt:commodity:vt@DONCH_CORE_COMM \
    --sleeve tr3_ribbon:commodity:vt@RIBBON_CORE_COMM \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --target-vol 0.60 --parity
Sleeve donchian_vt_commodity_vt [6 инстр] (databento, 1d) ...
Sleeve tr3_ribbon_commodity_vt [5 инстр] (databento, 1d) ...

=== Sleeve'ы по отдельности (gross Sharpe) ===
  donchian_vt_commodity_vt     ret +144.5%  DD -13.1%  Sharpe +1.32
  tr3_ribbon_commodity_vt      ret +361.3%  DD -24.4%  Sharpe +1.22

Корреляция дневных P&L (главное число прогона):
                          donchian_vt_commodity_vt  tr3_ribbon_commodity_vt
donchian_vt_commodity_vt                      1.00                     0.74
tr3_ribbon_commodity_vt                       0.74                     1.00

=== КОМБО (vol-parity (trailing 63, shift 1)) ===
  Доходность: +215.2%  (годовая: +16.8%)
  Max DD:     -15.8%
  Sharpe:     +1.38
  Годовая вола комбо: 11.7% (доля лимита DD<40%: ~29%)
  Проходит DD<40%: да

Комбо по годам (gross Sharpe)
   Год     Return     MaxDD   Sharpe   Бары
  ------------------------------------------
  2020    +41.2%    -5.7%    +2.52    311
  2021    +33.6%   -10.2%    +1.94    310
  2022    +26.0%   -15.8%    +1.45    310
  2023     -0.2%    -6.2%    +0.02    309
  2024     +5.6%   -10.8%    +0.51    313
  2025    +25.7%   -10.8%    +1.41    312
  ------------------------------------------
  ИТОГ   +215.2%   -15.8%    +1.31         (компаунд / худший год / ср.Sharpe)

  ((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_sleeves \
    --sleeve mr_lowvol:commodity:vt@MRLV_CORE_COMM \
    --sleeve mr2_percb_bw:commodity:vt@MRLV_CORE_COMM \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --target-vol 0.60 --parity
Sleeve mr_lowvol_commodity_vt [6 инстр] (databento, 1d) ...
Sleeve mr2_percb_bw_commodity_vt [6 инстр] (databento, 1d) ...

=== Sleeve'ы по отдельности (gross Sharpe) ===
  mr_lowvol_commodity_vt       ret +120.6%  DD  -6.3%  Sharpe +1.73
  mr2_percb_bw_commodity_vt    ret +139.1%  DD -14.8%  Sharpe +1.30

Корреляция дневных P&L (главное число прогона):
                           mr_lowvol_commodity_vt  mr2_percb_bw_commodity_vt
mr_lowvol_commodity_vt                       1.00                       0.59
mr2_percb_bw_commodity_vt                    0.59                       1.00

=== КОМБО (vol-parity (trailing 63, shift 1)) ===
  Доходность: +121.6%  (годовая: +11.4%)
  Max DD:     -10.6%
  Sharpe:     +1.57
  Годовая вола комбо: 7.0% (доля лимита DD<40%: ~18%)
  Проходит DD<40%: да

Комбо по годам (gross Sharpe)
   Год     Return     MaxDD   Sharpe   Бары
  ------------------------------------------
  2020    +17.9%    -1.7%    +2.33    311
  2021    +20.1%    -2.2%    +2.56    310
  2022     +6.8%   -10.6%    +0.66    310
  2023    +24.5%    -5.1%    +2.00    309
  2024     +8.2%    -5.0%    +1.10    313
  2025     +8.9%    -3.1%    +1.18    312
  ------------------------------------------
  ИТОГ   +121.6%   -10.6%    +1.64         (компаунд / худший год / ср.Sharpe)



  ((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_bootstrap --a ca_vol_ride --b donchian_vt \
    --basket crypto --source ccxt --crypto-dir "$CRYPTO" \
    --interval 4h --start "$SC" --vt --target-vol 0.40 --include @CRYPTO_CORE

python -m runners.run_bootstrap --a ca_thrust --b donchian_vt \
    --basket crypto --source ccxt --crypto-dir "$CRYPTO" \
    --interval 4h --start "$SC" --vt --target-vol 0.40 --include @CRYPTO_CORE
Bootstrap ca_vol_ride vs donchian_vt | crypto | ccxt | rf=0.0% | CI 90%
  ca_vol_ride        Sharpe +1.95  CI [+1.24, +2.65]
  donchian_vt        Sharpe +1.66  CI [+0.85, +2.43]

  Разность Sharpe (ca_vol_ride − donchian_vt): +0.29  CI [-0.39, +1.00] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 22 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.
Bootstrap ca_thrust vs donchian_vt | crypto | ccxt | rf=0.0% | CI 90%
  ca_thrust          Sharpe +1.49  CI [+0.69, +2.28]
  donchian_vt        Sharpe +1.66  CI [+0.85, +2.43]

  Разность Sharpe (ca_thrust − donchian_vt): -0.17  CI [-0.68, +0.36] -> НЕРАЗЛИЧИМЫ
  (paired stationary bootstrap, средний блок 22 баров, 2000 ресемплов)
  CI разности накрывает 0: смена чемпиона по этим данным НЕ обоснована.
((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_sleeves \
    --sleeve donchian_vt:crypto:vt \
    --sleeve ca_vol_ride:crypto:vt \
    --sleeve kalman_trend_long:crypto:vt \
    --source ccxt --crypto-dir "$CRYPTO" --interval 4h --start "$SC" \
    --target-vol 0.40 --parity
Sleeve donchian_vt_crypto_vt (ccxt, 4h) ...
Sleeve ca_vol_ride_crypto_vt (ccxt, 4h) ...
Sleeve kalman_trend_long_crypto_vt (ccxt, 4h) ...

=== Sleeve'ы по отдельности (gross Sharpe) ===
  donchian_vt_crypto_vt        ret  +31.6%  DD  -3.7%  Sharpe +1.49
  ca_vol_ride_crypto_vt        ret  +84.9%  DD  -6.1%  Sharpe +1.79
  kalman_trend_long_crypto_vt  ret  +92.6%  DD -17.0%  Sharpe +1.16

Корреляция дневных P&L (главное число прогона):
                             donchian_vt_crypto_vt  ca_vol_ride_crypto_vt  kalman_trend_long_crypto_vt
donchian_vt_crypto_vt                         1.00                   0.50                         0.83
ca_vol_ride_crypto_vt                         0.50                   1.00                         0.61
kalman_trend_long_crypto_vt                   0.83                   0.61                         1.00

=== КОМБО (vol-parity (trailing 63, shift 1)) ===
  Доходность: +57.6%  (годовая: +9.5%)
  Max DD:     -3.5%
  Sharpe:     +1.87
  Годовая вола комбо: 4.9% (доля лимита DD<40%: ~12%)
  Проходит DD<40%: да

Комбо по годам (gross Sharpe)
   Год     Return     MaxDD   Sharpe   Бары
  ------------------------------------------
  2021     +6.9%    -2.3%    +1.88   2190
  2022     -0.3%    -2.6%    -0.07   2190
  2023    +21.5%    -2.7%    +2.76   2190
  2024    +15.1%    -3.4%    +2.89   2196
  2025     +5.7%    -3.5%    +1.19   2190
  2026     +0.0%    +0.0%    +0.00      1
  ------------------------------------------
  ИТОГ    +57.6%    -3.5%    +1.44         (компаунд / худший год / ср.Sharpe)

  ((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_walkforward \
    --strategy ca_vol_ride ca_thrust kalman_trend_long donchian_vt \
    --basket crypto --source ccxt --crypto-dir "$CRYPTO" \
    --interval 4h --start "$SC" --vt --target-vol 0.40 --by year
Walk-forward (anchored, by=year) | crypto | ccxt | 2021-01-01..2026-01-01

ca_vol_ride +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +11.6% / +5.2%
  Лучшее / худшее:   +35.8% / +0.0%
  Разброс окон:      35.8% (мираж на 1 годе, если велик)

ca_thrust +VT(realized) — ПОД ВОПРОСОМ
  Окон прибыльных:  67% (6 окон)
  Средний / медиана: +17.0% / +16.5%
  Лучшее / худшее:   +46.9% / -9.4%
  Разброс окон:      56.3% (мираж на 1 годе, если велик)

kalman_trend_long +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +12.9% / +15.6%
  Лучшее / худшее:   +29.3% / -11.9%
  Разброс окон:      41.2% (мираж на 1 годе, если велик)

donchian_vt +VT(realized) — РОБАСТНА
  Окон прибыльных:  83% (6 окон)
  Средний / медиана: +5.0% / +4.4%
  Лучшее / худшее:   +12.5% / -1.7%
  Разброс окон:      14.3% (мираж на 1 годе, если велик)


  ((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_sleeves \
    --sleeve donchian_vt:crypto:vt \
    --sleeve ca_vol_ride:crypto:vt \
    --source ccxt --crypto-dir "$CRYPTO" --interval 4h --start "$SC" \
    --target-vol 0.40 --parity

python -m runners.run_sleeves \
    --sleeve donchian_vt:crypto:vt \
    --sleeve tr_ichimoku:crypto:vt \
    --sleeve ca_vol_ride:crypto:vt \
    --source ccxt --crypto-dir "$CRYPTO" --interval 4h --start "$SC" \
    --target-vol 0.40 --parity
Sleeve donchian_vt_crypto_vt (ccxt, 4h) ...
Sleeve ca_vol_ride_crypto_vt (ccxt, 4h) ...

=== Sleeve'ы по отдельности (gross Sharpe) ===
  donchian_vt_crypto_vt        ret  +31.6%  DD  -3.7%  Sharpe +1.49
  ca_vol_ride_crypto_vt        ret  +84.9%  DD  -6.1%  Sharpe +1.79

Корреляция дневных P&L (главное число прогона):
                       donchian_vt_crypto_vt  ca_vol_ride_crypto_vt
donchian_vt_crypto_vt                    1.0                    0.5
ca_vol_ride_crypto_vt                    0.5                    1.0

=== КОМБО (vol-parity (trailing 63, shift 1)) ===
  Доходность: +49.7%  (годовая: +8.4%)
  Max DD:     -3.1%
  Sharpe:     +1.99
  Годовая вола комбо: 4.1% (доля лимита DD<40%: ~10%)
  Проходит DD<40%: да

Комбо по годам (gross Sharpe)
   Год     Return     MaxDD   Sharpe   Бары
  ------------------------------------------
  2021     +4.6%    -1.9%    +1.67   2190
  2022     +1.1%    -1.9%    +0.41   2190
  2023    +20.5%    -2.1%    +3.10   2190
  2024    +12.4%    -3.1%    +2.94   2196
  2025     +4.6%    -2.8%    +1.15   2190
  2026     +0.0%    +0.0%    +0.00      1
  ------------------------------------------
  ИТОГ    +49.7%    -3.1%    +1.54         (компаунд / худший год / ср.Sharpe)
Sleeve donchian_vt_crypto_vt (ccxt, 4h) ...
Sleeve tr_ichimoku_crypto_vt (ccxt, 4h) ...
Sleeve ca_vol_ride_crypto_vt (ccxt, 4h) ...

=== Sleeve'ы по отдельности (gross Sharpe) ===
  donchian_vt_crypto_vt        ret  +31.6%  DD  -3.7%  Sharpe +1.49
  tr_ichimoku_crypto_vt        ret +275.2%  DD -22.5%  Sharpe +1.11
  ca_vol_ride_crypto_vt        ret  +84.9%  DD  -6.1%  Sharpe +1.79

Корреляция дневных P&L (главное число прогона):
                       donchian_vt_crypto_vt  tr_ichimoku_crypto_vt  ca_vol_ride_crypto_vt
donchian_vt_crypto_vt                   1.00                   0.47                   0.50
tr_ichimoku_crypto_vt                   0.47                   1.00                   0.29
ca_vol_ride_crypto_vt                   0.50                   0.29                   1.00

=== КОМБО (vol-parity (trailing 63, shift 1)) ===
  Доходность: +64.5%  (годовая: +10.5%)
  Max DD:     -3.6%
  Sharpe:     +2.05
  Годовая вола комбо: 4.9% (доля лимита DD<40%: ~12%)
  Проходит DD<40%: да

Комбо по годам (gross Sharpe)
   Год     Return     MaxDD   Sharpe   Бары
  ------------------------------------------
  2021     +6.9%    -2.2%    +1.87   2190
  2022     +2.5%    -2.3%    +0.78   2190
  2023    +21.8%    -2.7%    +2.78   2190
  2024    +15.4%    -3.4%    +2.90   2196
  2025     +6.9%    -3.6%    +1.46   2190
  2026     +0.0%    +0.0%    +0.00      1
  ------------------------------------------
  ИТОГ    +64.5%    -3.6%    +1.63         (компаунд / худший год / ср.Sharpe)
((.venv) ) shalygin@MacBook-Pro-Kirill-2 trends % python -m runners.run_sleeves \
    --sleeve donchian_vt:commodity:vt@DONCH_CORE_COMM \
    --sleeve mr_lowvol:commodity:vt@MRLV_CORE_COMM \
    --sleeve tr3_hh_hl:commodity:vt@HHHL_CORE_COMM \
    --source databento --panel-dir "$PANEL" --start "$S" \
    --target-vol 0.60 --hrp
Sleeve donchian_vt_commodity_vt [6 инстр] (databento, 1d) ...
Sleeve mr_lowvol_commodity_vt [6 инстр] (databento, 1d) ...
Sleeve tr3_hh_hl_commodity_vt [6 инстр] (databento, 1d) ...

=== Sleeve'ы по отдельности (gross Sharpe) ===
  donchian_vt_commodity_vt     ret +144.5%  DD -13.1%  Sharpe +1.32
  mr_lowvol_commodity_vt       ret +120.6%  DD  -6.3%  Sharpe +1.73
  tr3_hh_hl_commodity_vt       ret +283.0%  DD -15.1%  Sharpe +1.26

Корреляция дневных P&L (главное число прогона):
                          donchian_vt_commodity_vt  mr_lowvol_commodity_vt  tr3_hh_hl_commodity_vt
donchian_vt_commodity_vt                      1.00                    0.04                    0.59
mr_lowvol_commodity_vt                        0.04                    1.00                    0.08
tr3_hh_hl_commodity_vt                        0.59                    0.08                    1.00

=== КОМБО (HRP (де Прадо): donchian_vt_commodity_vt=21%, mr_lowvol_commodity_vt=71%, tr3_hh_hl_commodity_vt=8%) ===
  Доходность: +138.6%  (годовая: +12.5%)
  Max DD:     -5.0%
  Sharpe:     +2.18
  Годовая вола комбо: 5.5% (доля лимита DD<40%: ~14%)
  Проходит DD<40%: да

Комбо по годам (gross Sharpe)
   Год     Return     MaxDD   Sharpe   Бары
  ------------------------------------------
  2020    +18.8%    -2.3%    +3.03    311
  2021    +18.2%    -1.6%    +2.79    310
  2022    +20.4%    -4.7%    +2.68    310
  2023    +16.5%    -5.0%    +1.81    309
  2024     +8.6%    -2.3%    +1.47    313
  2025    +11.5%    -4.2%    +1.60    312
  ------------------------------------------
  ИТОГ   +138.6%    -5.0%    +2.23         (компаунд / худший год / ср.Sharpe)
