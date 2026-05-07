#!/usr/bin/env python3
"""
AI Supply Chain Stock Screener
*** pip install yfinance pandas numpy 설치되어야함
매 실행 시 yfinance로 실시간 데이터를 갱신하여 HTML 리포트 생성
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import webbrowser
import os
import warnings
warnings.filterwarnings('ignore')

# ── AI Supply Chain 티커 목록 ──────────────────────────────────────────────────
TICKERS = [
    ("AAOI", "Applied Optoelectronics"),
    ("AXTI", "AXT Inc"),
    ("OPTX", "Syntec Optics"),
    ("SNDK", "SanDisk"),
    ("LWLG", "Lightwave Logic"),
    ("AEHR", "Aehr Test Systems"),
    ("ICHR", "Ichor"),
    ("BE",   "Bloom Energy"),
    ("VIAV", "VIAVI Solutions"),
    """
    ("LITE", "Lumentum"),
    ("FORM", "FormFactor"),
    ("WDC",  "Western Digital"),
    ("GLW",  "Corning"),
    ("CIEN", "Ciena"),
    ("STX",  "Seagate Technology"),
    ("MOD",  "Modine Manufacturing"),
    ("TER",  "Teradyne"),
    ("VRT",  "Vertiv"),
    ("AEIS", "Advanced Energy Industries"),
    ("TSEM", "Tower Semi"),
    ("FIX",  "Comfort Systems"),
    ("LASR", "nLIGHT"),
    ("VICR", "Vicor Corporation"),
    ("INTC", "Intel"),
    ("COHR", "Coherent"),
    ("SOLS", "Solstive"),
    ("CAMT", "Camtek"),
    ("ONTO", "Onto Innovation"),
    ("ASX",  "ASE"),
    ("MU",   "Micron Technology"),
    ("MRVL", "Marvell"),
    ("LRCX", "Lam Research"),
    ("MTSI", "MACOM Technology"),
    ("GEV",  "GE Vernova"),
    ("AMKR", "Amkor"),
    ("AOSL", "Alpha & Omega Semi"),
    ("MPWR", "Monolithic Power Systems"),
    ("FN",   "Fabrinet"),
    ("PSIX", "Power Solutions"),
    ("ARM",  "ARM"),
    ("KLAC", "KLA Corporation"),
    ("FTAI", "FTAI Aviation"),
    ("ASML", "ASML"),
    ("ON",   "ON Semi"),
    ("LPTH", "LightPath Technologies"),
    ("RMBS", "Rambus"),
    ("ETN",  "Eaton"),
    ("ACMR", "ACM Research"),
    ("TSM",  "TSMC"),
    ("SMTC", "Semtech"),
    ("AMD",  "Advanced Micro Devices"),
    ("ANET", "Arista Networks"),
    ("APH",  "Amphenol"),
    ("AVGO", "Broadcom"),
    ("CRDO", "Credo Technology"),
    ("NVDA", "NVIDIA"),
    ("ALAB", "AsteraLabs"),"""
]

# ── 데이터 수집 ────────────────────────────────────────────────────────────────
def fetch_data(tickers):
    symbols = [t[0] for t in tickers]
    name_map = {t[0]: t[1] for t in tickers}
    
    print(f"📡 {len(symbols)}개 종목 데이터 다운로드 중...")
    
    now = datetime.now()
    start_1y = (now - timedelta(days=365)).strftime('%Y-%m-%d')
    start_ytd = datetime(now.year, 1, 1).strftime('%Y-%m-%d')

    # 1년치 일봉 다운로드
    raw = yf.download(symbols, start=start_1y, auto_adjust=True, progress=False)
    close = raw['Close'] if 'Close' in raw.columns else raw.xs('Close', axis=1, level=0)

    rows = []
    for sym in symbols:
        try:
            if sym not in close.columns:
                continue
            s = close[sym].dropna()
            if len(s) < 5:
                continue

            price = float(s.iloc[-1])
            price_1y_ago = float(s.iloc[0])
            pct_1y = (price / price_1y_ago - 1) * 100

            # YTD
            ytd_start_idx = s.index.searchsorted(pd.Timestamp(now.year, 1, 1))
            ytd_start_price = float(s.iloc[ytd_start_idx]) if ytd_start_idx < len(s) else price
            pct_ytd = (price / ytd_start_price - 1) * 100

            # SMA
            sma20  = float(s.rolling(20).mean().iloc[-1])
            sma50  = float(s.rolling(50).mean().iloc[-1])
            sma200 = float(s.rolling(200).mean().iloc[-1])

            # 52주 고점 대비
            high_52w = float(s.rolling(252).max().iloc[-1])
            vs_high  = (price / high_52w - 1) * 100

            # yfinance info (PER, P/S, Market Cap)
            info = yf.Ticker(sym).fast_info
            mkt_cap = getattr(info, 'market_cap', None)
            pe  = getattr(info, 'pe_ratio', None)

            ticker_obj = yf.Ticker(sym)
            info_full  = ticker_obj.info
            ps  = info_full.get('priceToSalesTrailing12Months', None)

            # 차트 스파크라인 (최근 50일)
            spark = list(s.tail(50).values)

            rows.append({
                'ticker':   sym,
                'company':  name_map[sym],
                'price':    price,
                'mkt_cap':  mkt_cap,
                'ps':       ps,
                'pe':       pe,
                'pct_ytd':  pct_ytd,
                'pct_1y':   pct_1y,
                'vs_high':  vs_high,
                'sma20_ok': price > sma20,
                'sma50_ok': price > sma50,
                'sma200_ok':price > sma200,
                'spark':    spark,
            })
            print(f"  ✓ {sym}")
        except Exception as e:
            print(f"  ✗ {sym}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('pct_1y', ascending=False).reset_index(drop=True)
    return df

# ── 포맷 헬퍼 ─────────────────────────────────────────────────────────────────
def fmt_price(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if v >= 1000:
        return f"${v:,.0f}"
    return f"${v:.2f}"

def fmt_cap(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if v >= 1e12: return f"${v/1e12:.1f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    if v >= 1e6:  return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"

def fmt_ratio(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span class="na">n/a</span>'
    return f"{v:.2f}"

def fmt_pct(v, cls=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span class="na">—</span>'
    color = "pos" if v >= 0 else "neg"
    sign  = "+" if v >= 0 else ""
    return f'<span class="{color} {cls}">{sign}{v:.2f}%</span>'

def sma_badge(ok):
    return '▲' if ok else '▽'

def sma_class(ok):
    return 'sma-up' if ok else 'sma-dn'

def spark_svg(vals):
    if not vals or len(vals) < 2:
        return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1
    W, H = 80, 28
    pts = []
    for i, v in enumerate(vals):
        x = i / (len(vals) - 1) * W
        y = H - (v - mn) / rng * H
        pts.append(f"{x:.1f},{y:.1f}")
    color = "#22c55e" if vals[-1] >= vals[0] else "#ef4444"
    return (f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="1.5" stroke-linejoin="round"/></svg>')

def vs_high_badge(v):
    if v is None or np.isnan(v): return "—"
    if v >= -5:   cls = "badge-green"
    elif v >= -15: cls = "badge-yellow"
    else:          cls = "badge-red"
    return f'<span class="badge {cls}">{v:+.2f}%</span>'

# ── HTML 생성 ─────────────────────────────────────────────────────────────────
def build_html(df):
    ts = datetime.now().strftime("%Y년 %m월 %d일  %H:%M")

    avg_ps = df['ps'].dropna().mean()
    avg_pe = df['pe'].dropna().mean()

    rows_html = ""
    for i, r in df.iterrows():
        rows_html += f"""
        <tr>
          <td class="rank">{i+1}</td>
          <td class="ticker">{r['ticker']}</td>
          <td class="company">{r['company']}</td>
          <td class="num">{fmt_price(r['price'])}</td>
          <td class="num">{fmt_cap(r['mkt_cap'])}</td>
          <td class="num">{fmt_ratio(r['ps'])}</td>
          <td class="num">{fmt_ratio(r['pe'])}</td>
          <td>{fmt_pct(r['pct_ytd'], 'bold')}</td>
          <td class="spark">{spark_svg(r['spark'])}</td>
          <td>{fmt_pct(r['pct_1y'], 'bold')}</td>
          <td>{vs_high_badge(r['vs_high'])}</td>
          <td class="{sma_class(r['sma20_ok'])}">{sma_badge(r['sma20_ok'])}</td>
          <td class="{sma_class(r['sma50_ok'])}">{sma_badge(r['sma50_ok'])}</td>
          <td class="{sma_class(r['sma200_ok'])}">{sma_badge(r['sma200_ok'])}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Supply Chain Screener</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+KR:wght@400;600;700&display=swap');

  :root {{
    --bg:      #0d0f14;
    --surface: #141720;
    --border:  #1e2230;
    --text:    #d4d8e8;
    --muted:   #5a607a;
    --accent:  #4f8ef7;
    --green:   #22c55e;
    --red:     #ef4444;
    --yellow:  #f59e0b;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans KR', sans-serif;
    font-size: 13px;
    min-height: 100vh;
  }}

  header {{
    padding: 32px 40px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
  }}

  header h1 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: #fff;
  }}

  header h1 span {{ color: var(--accent); }}

  .meta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    text-align: right;
    line-height: 1.7;
  }}

  .stats-bar {{
    display: flex;
    gap: 24px;
    padding: 16px 40px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }}

  .stat {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}

  .stat-label {{
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }}

  .stat-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px;
    font-weight: 600;
    color: #fff;
  }}

  .table-wrap {{
    overflow-x: auto;
    padding: 0 40px 40px;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
  }}

  thead th {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--muted);
    padding: 10px 8px;
    border-bottom: 1px solid var(--border);
    text-align: right;
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
    transition: color 0.15s;
  }}

  thead th:hover {{ color: var(--accent); }}
  thead th.th-left {{ text-align: left; }}

  tbody tr {{
    border-bottom: 1px solid var(--border);
    transition: background 0.1s;
  }}

  tbody tr:hover {{ background: rgba(79,142,247,0.05); }}

  td {{
    padding: 9px 8px;
    text-align: right;
    white-space: nowrap;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
  }}

  td.rank  {{ color: var(--muted); font-size: 11px; }}
  td.ticker {{ text-align: left; font-weight: 600; color: var(--accent); }}
  td.company {{ text-align: left; color: var(--text); font-family: 'IBM Plex Sans KR', sans-serif; font-size: 12px; min-width: 180px; }}
  td.num   {{ color: #e2e5f0; }}
  td.spark {{ padding: 4px 8px; }}

  .pos  {{ color: var(--green); }}
  .neg  {{ color: var(--red); }}
  .bold {{ font-weight: 600; }}
  .na   {{ color: var(--muted); }}

  .badge {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }}

  .badge-green  {{ background: rgba(34,197,94,0.15); color: var(--green); }}
  .badge-yellow {{ background: rgba(245,158,11,0.15); color: var(--yellow); }}
  .badge-red    {{ background: rgba(239,68,68,0.15);  color: var(--red); }}

  .sma-up {{ color: var(--green); font-size: 11px; }}
  .sma-dn {{ color: var(--red);   font-size: 11px; }}

  footer {{
    padding: 16px 40px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }}
</style>
</head>
<body>

<header>
  <h1>AI <span>Supply Chain</span> Screener</h1>
  <div class="meta">
    <div>업데이트: {ts}</div>
    <div>종목 수: {len(df)}개 · 데이터: Yahoo Finance</div>
  </div>
</header>

<div class="stats-bar">
  <div class="stat">
    <div class="stat-label">총 종목</div>
    <div class="stat-val">{len(df)}</div>
  </div>
  <div class="stat">
    <div class="stat-label">200SMA 위</div>
    <div class="stat-val" style="color:var(--green)">{df['sma200_ok'].sum()}</div>
  </div>
  <div class="stat">
    <div class="stat-label">200SMA 아래</div>
    <div class="stat-val" style="color:var(--red)">{(~df['sma200_ok']).sum()}</div>
  </div>
  <div class="stat">
    <div class="stat-label">평균 P/S</div>
    <div class="stat-val">{avg_ps:.1f}x</div>
  </div>
  <div class="stat">
    <div class="stat-label">평균 P/E</div>
    <div class="stat-val">{avg_pe:.1f}x</div>
  </div>
  <div class="stat">
    <div class="stat-label">YTD 상승 종목</div>
    <div class="stat-val" style="color:var(--green)">{(df['pct_ytd']>0).sum()}</div>
  </div>
  <div class="stat">
    <div class="stat-label">1Y 상승 종목</div>
    <div class="stat-val" style="color:var(--green)">{(df['pct_1y']>0).sum()}</div>
  </div>
</div>

<div class="table-wrap">
  <table id="tbl">
    <thead>
      <tr>
        <th class="th-left" onclick="sortTable(0)">#</th>
        <th class="th-left" onclick="sortTable(1)">Ticker</th>
        <th class="th-left" onclick="sortTable(2)">Company</th>
        <th onclick="sortTable(3)">Price</th>
        <th onclick="sortTable(4)">Mkt Cap</th>
        <th onclick="sortTable(5)">P/S</th>
        <th onclick="sortTable(6)">P/E</th>
        <th onclick="sortTable(7)">% YTD</th>
        <th>Chart 1Y</th>
        <th onclick="sortTable(9)">% 1Y</th>
        <th onclick="sortTable(10)">△ Highs</th>
        <th>20SMA</th>
        <th>50SMA</th>
        <th>200SMA</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>

<footer>
  <span>데이터 출처: Yahoo Finance (yfinance) · 정보 제공 목적으로만 사용, 투자 권유 아님</span>
  <span>Python으로 자동 생성 · {ts}</span>
</footer>

<script>
let sortDir = {{}};
function sortTable(col) {{
  const tbl = document.getElementById('tbl');
  const tb = tbl.tBodies[0];
  const rows = Array.from(tb.rows);
  const asc = !sortDir[col];
  sortDir = {{}};
  sortDir[col] = asc;

  rows.sort((a, b) => {{
    let av = a.cells[col].innerText.replace(/[^\\d.\\-%+]/g, '');
    let bv = b.cells[col].innerText.replace(/[^\\d.\\-%+]/g, '');
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});

  rows.forEach(r => tb.appendChild(r));
}}
</script>
</body>
</html>"""
    return html

# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  AI Supply Chain Stock Screener")
    print("=" * 50)

    df = fetch_data(TICKERS)

    if df.empty:
        print("❌ 데이터를 가져오지 못했습니다.")
        return

    html = build_html(df)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ai_supply_chain_report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ 리포트 저장 완료: {output_path}")
    print("🌐 브라우저에서 열기...")
    webbrowser.open(f"file://{output_path}")

if __name__ == "__main__":
    main()