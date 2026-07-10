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
from runners.run_basket import STRATEGIES, STRATEGY_FAMILY


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
        # Семейства из реестра, НЕ из порядка аргументов (фикс
        # 2026-07j: подпись врала, называя donchian_vt реверсией,
        # если он передан вторым).
        "famA": STRATEGY_FAMILY.get(args.a, "strategy"),
        "famB": STRATEGY_FAMILY.get(args.b, "strategy"),
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
 .wrap.top{height:52vh}.wrap.bot{height:24vh}
 canvas{width:100%!important;height:100%!important;cursor:grab}
 canvas.dragging{cursor:grabbing}
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
 details{margin-bottom:14px}
 summary{cursor:pointer;color:#8b93a7;font-size:13px;font-weight:600}
 .help{color:#aab2c4;font-size:13px;line-height:1.55;max-width:900px}
 .help b{color:#e6e6e6}
 .help .sw{display:inline-block;width:10px;height:10px;
   border-radius:2px;margin:0 4px 0 2px;vertical-align:baseline}
</style></head><body>
<h1 id="ttl"></h1>
<div class="sub" id="sub"></div>
<details><summary>How to read this chart</summary>
<div class="help"><p><b>Everything starts at 100 — that is
normalization, not a bug.</b> All three lines are rescaled so that
their first value equals 100. This puts dollars-per-barrel, index
points and strategy P&amp;L on one comparable scale: a value of 117
simply means +17% cumulative growth since the first date.</p>
<p><span class="sw" style="background:#5a6070"></span><b>Price
(grey, right axis)</b> — growth of $100 invested in the asset
itself (buy &amp; hold). This is the benchmark: if a strategy line
is below the grey line over a stretch, the strategy underperformed
simply holding the asset there (usually with far less risk —
compare how smooth the lines are, not only how high).</p>
<p><span class="sw" style="background:#4f9dff"></span><span
class="sw" style="background:#ff8c42"></span><b>Equity curves
(left axis)</b> — growth of $100 given to each strategy. Flat
shelves = the strategy is out of the market (P&amp;L frozen);
sloped segments = in a position. The two y-axes have different
ranges, so compare shapes across axes, not absolute heights.</p>
<p><b>Bottom panel — position size over time.</b> Height is the
applied position after vol-targeting, so it can exceed 1.0 (e.g.
1.08 = 108% notional because volatility was low) or stay below it.
Zero = out of the market. This panel explains the shelves above:
equity only moves while a position bar is non-zero.</p>
<p><b>Controls:</b> mouse wheel = zoom around cursor, drag = pan,
◀ ▶ = step left/right, Reset = full range. Both panels always show
the same time window.</p></div></details>
<div class="card"><p class="lbl">
 <span>Equity curves (left axis) &amp; price (right axis)</span>
 <span class="hint">wheel = zoom &middot; drag = pan &middot; synced
  <button class="btn" onclick="stepView(-0.2)">&#9664;</button>
  <button class="btn" onclick="stepView(0.2)">&#9654;</button>
  <button class="btn" onclick="resetView()">Reset</button>
  <button class="btn" id="logBtn" onclick="tog()">Log scale</button>
 </span></p>
 <div class="wrap top"><canvas id="main"></canvas></div></div>
<div class="card"><p class="lbl">
 <span>Position over time &mdash; when each strategy is in the market</span>
 <span class="hint">
  <button class="btn" onclick="stepView(-0.2)">&#9664;</button>
  <button class="btn" onclick="stepView(0.2)">&#9654;</button>
  <button class="btn" onclick="resetView()">Reset</button></span></p>
 <div class="wrap bot"><canvas id="pos"></canvas></div>
 <div class="legend" id="poslegend"></div></div>
<script>
const D = __DATA__;
document.getElementById('ttl').textContent =
  D.ticker+' \u2014 '+D.nameA+' ('+D.famA+') vs '+D.nameB+
  ' ('+D.famB+')';
document.getElementById('sub').textContent =
  'Sharpe: '+D.nameA+' '+D.sharpeA+'  |  '+D.nameB+' '+D.sharpeB+
  (D.combo ? '  |  50/50 combo shown' : '');
const gA='#4f9dff',gB='#ff8c42',gP='#5a6070',gC='#3ddc84';
const N=D.dates.length;
// ЕДИНОЕ состояние вьюпорта (индексы в category-оси). Плагин зума
// удалён: его состояние конфликтовало с ручной синхронизацией осей
// (Reset масштабировал панели по-разному), а drag-pan требовал
// Hammer.js и молча не работал. Здесь всё детерминированно.
let view={min:0,max:N-1};
function applyView(){
 for(const c of [mainC,posC]){
  c.options.scales.x.min=view.min;
  c.options.scales.x.max=view.max;
  c.update('none');
 }
}
function clampView(){
 const span=Math.max(10,view.max-view.min);
 if(view.min<0){view.min=0;view.max=Math.min(N-1,span);}
 if(view.max>N-1){view.max=N-1;view.min=Math.max(0,N-1-span);}
}
function resetView(){view={min:0,max:N-1};applyView();}
function stepView(frac){
 const span=view.max-view.min;
 const d=Math.round(span*frac)||Math.sign(frac);
 view.min+=d;view.max+=d;clampView();applyView();
}
function zoomAt(canvas,evt,factor){
 const rect=canvas.getBoundingClientRect();
 const fx=(evt.clientX-rect.left)/rect.width;   // 0..1 позиция курсора
 const span=view.max-view.min;
 const newSpan=Math.max(10,Math.min(N-1,Math.round(span*factor)));
 const anchor=view.min+span*fx;
 view.min=Math.round(anchor-newSpan*fx);
 view.max=view.min+newSpan;
 clampView();applyView();
}
function attachNav(canvas){
 canvas.addEventListener('wheel',e=>{
  e.preventDefault();
  zoomAt(canvas,e,e.deltaY>0?1.25:0.8);
 },{passive:false});
 let dragging=false,lastX=0;
 canvas.addEventListener('pointerdown',e=>{
  dragging=true;lastX=e.clientX;
  canvas.classList.add('dragging');
  canvas.setPointerCapture(e.pointerId);
 });
 canvas.addEventListener('pointermove',e=>{
  if(!dragging)return;
  const rect=canvas.getBoundingClientRect();
  const span=view.max-view.min;
  const dIdx=Math.round((e.clientX-lastX)/rect.width*span);
  if(dIdx!==0){
   view.min-=dIdx;view.max-=dIdx;lastX=e.clientX;
   clampView();applyView();
  }
 });
 const stop=e=>{dragging=false;canvas.classList.remove('dragging');};
 canvas.addEventListener('pointerup',stop);
 canvas.addEventListener('pointerleave',stop);
}
const eqDs=[
 {label:D.nameA+' equity',data:D.eqA,borderColor:gA,borderWidth:1.8,
  pointRadius:0,tension:.1,yAxisID:'y'},
 {label:D.nameB+' equity',data:D.eqB,borderColor:gB,borderWidth:1.8,
  pointRadius:0,tension:.1,yAxisID:'y'},
 {label:'Price (buy&hold)',data:D.price,borderColor:gP,
  borderWidth:1.2,pointRadius:0,tension:.1,yAxisID:'yP'}];
if(D.combo) eqDs.splice(2,0,{label:'Combo equity',data:D.combo,
  borderColor:gC,borderWidth:2,pointRadius:0,tension:.1,
  borderDash:[4,3],yAxisID:'y'});
const mainC=new Chart(document.getElementById('main'),{type:'line',
 data:{labels:D.dates,datasets:eqDs},
 options:{responsive:true,maintainAspectRatio:false,animation:false,
  interaction:{mode:'index',intersect:false},
  plugins:{legend:{labels:{color:'#c3c9d6',boxWidth:14,
   font:{size:11}}}},
  scales:{
   x:{ticks:{color:'#6b7280',maxTicksLimit:12},
    grid:{color:'#1e2330'}},
   y:{position:'left',ticks:{color:'#4f9dff'},
    grid:{color:'#1e2330'},
    title:{display:true,text:'equity (start=100)',color:'#8b93a7'}},
   yP:{position:'right',ticks:{color:'#5a6070'},
    grid:{drawOnChartArea:false},
    title:{display:true,text:'price (start=100)',
     color:'#8b93a7'}}}}});
const posC=new Chart(document.getElementById('pos'),{type:'line',
 data:{labels:D.dates,datasets:[
  {label:D.nameA,data:D.posA,borderColor:gA,backgroundColor:gA+'22',
   borderWidth:1.4,pointRadius:0,fill:true,stepped:true},
  {label:D.nameB,data:D.posB,borderColor:gB,backgroundColor:gB+'22',
   borderWidth:1.4,pointRadius:0,fill:true,stepped:true}]},
 options:{responsive:true,maintainAspectRatio:false,animation:false,
  interaction:{mode:'index',intersect:false},
  plugins:{legend:{display:false}},
  scales:{x:{ticks:{color:'#6b7280',maxTicksLimit:12},
   grid:{color:'#1e2330'}},
   y:{ticks:{color:'#6b7280'},grid:{color:'#1e2330'},
    title:{display:true,text:'position size',color:'#8b93a7'}}}}});
attachNav(document.getElementById('main'));
attachNav(document.getElementById('pos'));
let isLog=false;
function tog(){isLog=!isLog;
 mainC.options.scales.y.type=isLog?'logarithmic':'linear';
 mainC.options.scales.yP.type=isLog?'logarithmic':'linear';
 document.getElementById('logBtn').textContent=
  isLog?'Linear scale':'Log scale';mainC.update();}
// Роли в легенде — из реестра семейств (D.famA/famB), не из порядка
// аргументов: раньше первая стратегия всегда подписывалась trend,
// вторая mean-reversion, что врало при обратном порядке.
function famNote(f){
 if(f==='trend')
  return ' \u2014 trend: in market during directional moves';
 if(f.indexOf('trend')===0)
  return ' \u2014 '+f+': in market during directional moves';
 if(f==='mean-reversion')
  return ' \u2014 mean-reversion: in market during pullbacks/chop';
 return ' \u2014 '+f;
}
document.getElementById('poslegend').innerHTML=
 '<span><span class="dot" style="background:'+gA+'"></span>'+D.nameA+
 famNote(D.famA)+'</span>'+
 '<span><span class="dot" style="background:'+gB+'"></span>'+D.nameB+
 famNote(D.famB)+'</span>';
</script></body></html>
"""


if __name__ == "__main__":
    main()
