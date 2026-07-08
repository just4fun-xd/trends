"""Визуализация работы стратегий на одном инструменте (HTML-график).

Строит самодостаточный интерактивный HTML: цена актива + позиции двух
стратегий (когда каждая в рынке) + их кривые капитала. Видно, какая
стратегия отрабатывает тренд, какая — флэт, где они дополняют друг
друга. Опционально — комбо-sleeve (equal-weight двух).

График на английском (по требованию). Chart.js из CDN, один файл,
открывается в браузере.

Запуск:
    python -m runners.run_strategy_chart \\
        --ticker TSLA --a ema_vt --b mr_ens \\
        --source yf --start 2022-01-01 --end 2024-01-01 \\
        --vt --target-vol 0.30 --out chart.html
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from core.engine import run_engine
from core.sizing import make_sizer
from data.ccxt_source import CCXTSource
from data.databento_source import DatabentoSource
from data.yfinance_source import YFinanceSource
from runners.run_basket import STRATEGIES


def _series(bars, strat_fn, sizer, cost):
    """Позиция и кривая капитала стратегии на инструменте."""
    pos = strat_fn(bars)
    if sizer is not None:
        pos = pos * sizer(bars)
    res = run_engine(bars, pos, cost=cost)
    return pos.fillna(0.0), res.equity


def main() -> None:
    """CLI: HTML-график цена + 2 стратегии (+ опц. combo)."""
    p = argparse.ArgumentParser(
        description="HTML-график работы двух стратегий на инструменте")
    p.add_argument("--ticker", required=True,
                   help="тикер источника (TSLA, GC=F, BTC-USDT...)")
    p.add_argument("--a", required=True, help="стратегия A (тренд)")
    p.add_argument("--b", required=True, help="стратегия B (MR)")
    p.add_argument("--source", default="yf",
                   choices=["yf", "databento", "ccxt"])
    p.add_argument("--panel-dir", default="data/panels/futures")
    p.add_argument("--crypto-dir", default="data/crypto")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--interval", default="1d")
    p.add_argument("--cost", type=float, default=0.0002)
    p.add_argument("--vt", action="store_true")
    p.add_argument("--sizer", default="realized",
                   choices=["realized", "garch"])
    p.add_argument("--target-vol", type=float, default=0.30)
    p.add_argument("--combo", action="store_true",
                   help="добавить equal-weight combo двух стратегий")
    p.add_argument("--out", default="strategy_chart.html")
    args = p.parse_args()

    for name in (args.a, args.b):
        if name not in STRATEGIES:
            raise SystemExit(f"нет стратегии '{name}' в реестре")

    if args.source == "yf":
        source = YFinanceSource()
    elif args.source == "ccxt":
        source = CCXTSource(data_dir=args.crypto_dir)
    else:
        source = DatabentoSource(panel_dir=args.panel_dir)

    bars = source.load(args.ticker, args.start, args.end, args.interval)
    sizer = (make_sizer(args.sizer, target_vol=args.target_vol)
             if args.vt else None)

    pos_a, eq_a = _series(bars, STRATEGIES[args.a], sizer, args.cost)
    pos_b, eq_b = _series(bars, STRATEGIES[args.b], sizer, args.cost)

    dates = [d.strftime("%Y-%m-%d %H:%M") if args.interval != "1d"
             else d.strftime("%Y-%m-%d") for d in bars.close.index]
    price = bars.close.to_numpy()
    # Нормируем цену и equity к 100 на старте (общая шкала).
    price_n = (price / price[0] * 100.0).round(2).tolist()
    eq_a_n = (eq_a.to_numpy() / eq_a.iloc[0] * 100.0).round(2).tolist()
    eq_b_n = (eq_b.to_numpy() / eq_b.iloc[0] * 100.0).round(2).tolist()
    pa = pos_a.to_numpy().round(3).tolist()
    pb = pos_b.to_numpy().round(3).tolist()

    combo_eq = None
    if args.combo:
        ra = eq_a.pct_change().fillna(0.0)
        rb = eq_b.pct_change().fillna(0.0)
        combo = (1.0 + 0.5 * ra + 0.5 * rb).cumprod()
        combo_eq = (combo / combo.iloc[0] * 100.0).round(2).tolist()

    payload = {
        "dates": dates, "price": price_n,
        "eqA": eq_a_n, "eqB": eq_b_n,
        "posA": pa, "posB": pb, "combo": combo_eq,
        "nameA": args.a, "nameB": args.b,
        "ticker": args.ticker,
        "sharpeA": round(_sharpe(eq_a, bars.bars_per_year), 2),
        "sharpeB": round(_sharpe(eq_b, bars.bars_per_year), 2),
    }
    html = _HTML.replace("__DATA__", json.dumps(payload))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"График сохранён: {args.out}")
    print(f"  {args.a}: Sharpe {payload['sharpeA']:+.2f} | "
          f"{args.b}: Sharpe {payload['sharpeB']:+.2f}")
    print(f"  Открой в браузере: open {args.out}")


def _sharpe(eq, bpy):
    r = eq.pct_change().dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(bpy))


_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Strategy Behavior</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/\
chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/\
hammerjs/2.0.8/hammer.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/\
chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js"></script>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;
   background:#0f1117;color:#e6e6e6;margin:0;padding:20px;
   box-sizing:border-box}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 .sub{color:#8b93a7;font-size:13px;margin-bottom:14px}
 .card{background:#171a23;border:1px solid #252a37;border-radius:12px;
   padding:14px 18px;margin-bottom:14px}
 .lbl{font-size:12px;color:#8b93a7;margin:0 0 8px;font-weight:600;
   text-transform:uppercase;letter-spacing:.04em;
   display:flex;justify-content:space-between;align-items:center}
 .wrap{position:relative;width:100%}
 .wrap.top{height:56vh}.wrap.bot{height:26vh}
 canvas{width:100%!important;height:100%!important}
 .btn{background:#252a37;color:#c3c9d6;border:1px solid #333a4a;
   border-radius:6px;padding:3px 10px;font-size:11px;cursor:pointer;
   margin-left:6px}
 .btn:hover{background:#2f3646}
 .legend{display:flex;gap:18px;font-size:12px;color:#c3c9d6;
   margin-top:8px;flex-wrap:wrap}
 .dot{display:inline-block;width:10px;height:10px;border-radius:2px;
   margin-right:5px;vertical-align:middle}
 .hint{color:#6b7280;font-size:11px;font-weight:400;
   text-transform:none;letter-spacing:0}
</style></head><body>
<h1 id="ttl"></h1>
<div class="sub" id="sub"></div>
<div class="card"><p class="lbl">
 <span>Equity curves (left axis) &amp; price (right axis)</span>
 <span class="hint">wheel = zoom &middot; drag = pan &middot; synced
  <button class="btn" onclick="resetAll()">Reset</button>
  <button class="btn" id="logBtn" onclick="tog()">Log scale</button>
 </span></p>
 <div class="wrap top"><canvas id="main"></canvas></div></div>
<div class="card"><p class="lbl">
 <span>Position over time &mdash; when each strategy is in the market</span>
 <span class="hint">
  <button class="btn" onclick="resetAll()">Reset</button></span></p>
 <div class="wrap bot"><canvas id="pos"></canvas></div>
 <div class="legend" id="poslegend"></div></div>
<script>
// Плагин зума в UMD-сборке кладётся в window как ChartZoom / \
chartjs-plugin-zoom;
// без явной регистрации zoom/pan молча не работают, а resetZoom() — no-op.
if (window.ChartZoom) { Chart.register(window.ChartZoom); }
else if (window['chartjs-plugin-zoom']) {
  Chart.register(window['chartjs-plugin-zoom']); }
const D = __DATA__;
document.getElementById('ttl').textContent =
  D.ticker+' \u2014 '+D.nameA+' (trend) vs '+D.nameB+' (mean-reversion)';
document.getElementById('sub').textContent =
  'Sharpe: '+D.nameA+' '+D.sharpeA+'  |  '+D.nameB+' '+D.sharpeB+
  (D.combo ? '  |  50/50 combo shown' : '');
const gA='#4f9dff',gB='#ff8c42',gP='#5a6070',gC='#3ddc84';
let syncing=false;
// Синхронизация обоих графиков: когда один зумят/панят по X, второй
// повторяет тот же диапазон X.
function syncFrom(src,dst){
 if(syncing) return; syncing=true;
 const sx=src.scales.x;
 dst.options.scales.x.min=sx.min; dst.options.scales.x.max=sx.max;
 dst.update('none'); syncing=false;
}
function mkZoom(getPartner){return{
 zoom:{wheel:{enabled:true},pinch:{enabled:true},mode:'x',
   onZoom:({chart})=>syncFrom(chart,getPartner())},
 pan:{enabled:true,mode:'x',threshold:5,
   onPanComplete:({chart})=>syncFrom(chart,getPartner()),
   onPan:({chart})=>syncFrom(chart,getPartner())}};}
function resetAll(){
 // Синк оставляет min/max на партнёре — их надо снять, иначе resetZoom
 // сбрасывает только «родной» график, а второй держит старый диапазон.
 [mainC,posC].forEach(c=>{
   c.options.scales.x.min=undefined;
   c.options.scales.x.max=undefined;});
 mainC.resetZoom(); posC.resetZoom();
 mainC.update('none'); posC.update('none');}
const eqDs=[
 {label:D.nameA+' equity',data:D.eqA,borderColor:gA,borderWidth:1.8,
  pointRadius:0,tension:.1,yAxisID:'y'},
 {label:D.nameB+' equity',data:D.eqB,borderColor:gB,borderWidth:1.8,
  pointRadius:0,tension:.1,yAxisID:'y'},
 {label:'Price',data:D.price,borderColor:gP,borderWidth:1.2,
  pointRadius:0,tension:.1,yAxisID:'yP'}];
if(D.combo) eqDs.splice(2,0,{label:'Combo equity',data:D.combo,
  borderColor:gC,borderWidth:2,pointRadius:0,tension:.1,
  borderDash:[4,3],yAxisID:'y'});
const mainC=new Chart(document.getElementById('main'),{type:'line',
 data:{labels:D.dates,datasets:eqDs},
 options:{responsive:true,maintainAspectRatio:false,
  interaction:{mode:'index',intersect:false},
  plugins:{legend:{labels:{color:'#c3c9d6',boxWidth:14,
   font:{size:11}}},zoom:mkZoom(()=>posC)},
  scales:{
   x:{ticks:{color:'#6b7280',maxTicksLimit:12},
    grid:{color:'#1e2330'}},
   y:{position:'left',ticks:{color:'#4f9dff'},
    grid:{color:'#1e2330'},
    title:{display:true,text:'equity (start=100)',color:'#8b93a7'}},
   yP:{position:'right',ticks:{color:'#5a6070'},
    grid:{drawOnChartArea:false},
    title:{display:true,text:'price (start=100)',color:'#8b93a7'}}}}});
const posC=new Chart(document.getElementById('pos'),{type:'line',
 data:{labels:D.dates,datasets:[
  {label:D.nameA,data:D.posA,borderColor:gA,backgroundColor:gA+'22',
   borderWidth:1.4,pointRadius:0,fill:true,stepped:true},
  {label:D.nameB,data:D.posB,borderColor:gB,backgroundColor:gB+'22',
   borderWidth:1.4,pointRadius:0,fill:true,stepped:true}]},
 options:{responsive:true,maintainAspectRatio:false,
  interaction:{mode:'index',intersect:false},
  plugins:{legend:{display:false},zoom:mkZoom(()=>mainC)},
  scales:{x:{ticks:{color:'#6b7280',maxTicksLimit:12},
   grid:{color:'#1e2330'}},
   y:{ticks:{color:'#6b7280'},grid:{color:'#1e2330'},
    title:{display:true,text:'position size',color:'#8b93a7'}}}}});
let isLog=false;
function tog(){isLog=!isLog;
 mainC.options.scales.y.type=isLog?'logarithmic':'linear';
 mainC.options.scales.yP.type=isLog?'logarithmic':'linear';
 document.getElementById('logBtn').textContent=
  isLog?'Linear scale':'Log scale';mainC.update();}
document.getElementById('poslegend').innerHTML=
 '<span><span class="dot" style="background:'+gA+'"></span>'+D.nameA+
 ' \u2014 trend: in market during directional moves</span>'+
 '<span><span class="dot" style="background:'+gB+'"></span>'+D.nameB+
 ' \u2014 mean-reversion: in market during pullbacks/chop</span>';
</script></body></html>
"""


if __name__ == "__main__":
    main()
