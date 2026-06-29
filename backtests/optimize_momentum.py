"""
Momentum Optimization Grid
Tests: Cross-Sectional 12-1 (current) vs RS-vs-SPY
Lookback: 3,4,5,6,7,8,9,10,11,12,15,18,24 months
Skip:     0, 1 months
Outputs:  opt_results.json for dashboard embedding
"""

import json, math
import pandas as pd
import numpy as np

# ── Load previously downloaded price data ────────────────────────────
print("Loading price data from backtest20y.json...", flush=True)
with open('/sessions/tender-dreamy-mendel/mnt/outputs/backtest20y.json') as f:
    bt = json.load(f)

# We need raw monthly prices. Re-read them from the saved data.
# Actually we need to reconstruct from equity curves -- but we don't have raw prices.
# Instead, re-use the same fetch logic but load from the JSON.
# The JSON has hybrid_eq / spy_eq but not raw prices.
# We need to re-download. Let's check if we still have network access.

import requests, time, sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

START_DATA = "2004-01-01"
END_DATE   = "2026-06-30"
START_BT   = "2006-01-01"
SPY_FLOOR  = 0.60
ACTIVE_PCT = 0.40
TOP_N      = 3
MA_WINDOW  = 200
HEADERS    = {"User-Agent": "Mozilla/5.0"}

NQ100 = [
    "AAPL","ABNB","ACGL","ADBE","ADI","ADP","ADSK","ALGN","AMAT","AMD",
    "AMGN","AMZN","ASML","AVGO","AZN","BIIB","BKNG","BKR","CDNS","CEG",
    "CHTR","CINF","CMCSA","COST","CPRT","CRWD","CSCO","CSGP","CSX","CTAS",
    "DASH","DDOG","DLTR","DXCM","EA","EBAY","ENPH","EXC","FANG","FAST",
    "FTNT","GEHC","GFS","GILD","GOOG","GOOGL","HON","IDXX","ILMN","INTC",
    "INTU","ISRG","KDP","KLAC","LRCX","LULU","MAR","MCHP","MDLZ","MELI",
    "META","MNST","MRNA","MRVL","MSFT","MU","NFLX","NVDA","NXPI","ODFL",
    "ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PYPL","QCOM","REGN",
    "ROP","ROST","SBUX","SIRI","SNPS","TEAM","TMUS","TSLA","TTD","TTWO",
    "TXN","VRSK","VRTX","WBD","WDAY","XEL","ZM","ZS"
]

def to_unix(d):
    return int(datetime.strptime(d, "%Y-%m-%d").timestamp())

def fetch_monthly(ticker, retries=3):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={to_unix(START_DATA)}&period2={to_unix(END_DATE)}"
           f"&interval=1mo&events=adjsplit")
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200: return ticker, None
            j = r.json()
            res = j.get("chart", {}).get("result", [])
            if not res: return ticker, None
            data = res[0]
            ts  = data["timestamp"]
            adj = (data.get("indicators",{}).get("adjclose",[{}])[0].get("adjclose")
                   or data["indicators"]["quote"][0].get("close"))
            if not adj: return ticker, None
            dates  = [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d") for t in ts]
            series = pd.Series(adj, index=pd.to_datetime(dates), name=ticker)
            return ticker, series[series.notna()]
        except:
            if attempt < retries-1: time.sleep(1)
    return ticker, None

def fetch_daily_regime():
    """Stable proxy for 200d MA regime."""
    proxy = ["MSFT","AAPL","AMZN","INTC","NVDA","CSCO","QCOM","ADBE","COST",
             "GILD","CMCSA","AMGN","EBAY","FAST","KLAC","LRCX","MCHP","PAYX","PCAR","TXN"]
    daily = {}
    for tk in proxy:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{tk}"
               f"?period1={to_unix(START_DATA)}&period2={to_unix(END_DATE)}"
               f"&interval=1d&events=adjsplit")
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            j = r.json()
            res = j["chart"]["result"][0]
            ts  = res["timestamp"]
            adj = (res.get("indicators",{}).get("adjclose",[{}])[0].get("adjclose")
                   or res["indicators"]["quote"][0].get("close"))
            dates = [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d") for t in ts]
            s = pd.Series(adj, index=pd.to_datetime(dates), name=tk)
            s = s[s.notna()]
            if len(s) > 200: daily[tk] = s
        except: pass
        time.sleep(0.1)
    return daily

print("Downloading monthly prices...", flush=True)
monthly_prices = {}
all_tickers = NQ100 + ["SPY"]
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(fetch_monthly, t): t for t in all_tickers}
    done = 0
    for f in as_completed(futures):
        tk, series = f.result()
        done += 1
        if series is not None and len(series) >= 6:
            monthly_prices[tk] = series
        sys.stdout.write(f"\r  {done}/{len(all_tickers)} done  ")
        sys.stdout.flush()
print(f"\nGot {len(monthly_prices)} tickers")

print("Downloading daily regime data...", flush=True)
daily_series = fetch_daily_regime()
print(f"Got {len(daily_series)} daily tickers")

# ── Build DataFrames ────────────────────────────────────────────────
mp_df = pd.DataFrame(monthly_prices).sort_index()
mp_df.index = pd.to_datetime(mp_df.index)

nq_monthly  = mp_df[[c for c in mp_df.columns if c != "SPY"]]
spy_monthly  = mp_df["SPY"]

# Daily regime
daily_df  = pd.DataFrame(daily_series).sort_index()
daily_df.index = pd.to_datetime(daily_df.index)
daily_norm = daily_df.divide(daily_df.ffill().bfill().iloc[0])
ew_daily   = daily_norm.mean(axis=1)
ma200      = ew_daily.rolling(MA_WINDOW, min_periods=100).mean()
regime_d   = (ew_daily > ma200).astype(int).shift(1).fillna(0)
regime_monthly = regime_d.resample("ME").last()

nq_ret  = nq_monthly.pct_change().fillna(0)
spy_ret = spy_monthly.pct_change().fillna(0)

bt_start = pd.Timestamp(START_BT)

# ── Performance metrics helper ───────────────────────────────────────
def calc_metrics(eq, rets):
    bt_eq   = eq[eq.index >= bt_start]
    bt_rets = rets[rets.index >= bt_start]
    if len(bt_eq) < 12: return None
    n_years = len(bt_rets) / 12
    cagr    = (bt_eq.iloc[-1] / bt_eq.iloc[0]) ** (1/n_years) - 1
    ann_vol = bt_rets.std() * math.sqrt(12)
    sharpe  = (bt_rets.mean() * 12) / ann_vol if ann_vol > 0 else 0
    dd      = bt_eq / bt_eq.cummax() - 1
    max_dd  = dd.min()
    calmar  = cagr / abs(max_dd) if max_dd != 0 else 0
    down    = bt_rets[bt_rets < 0]
    sortino = (bt_rets.mean()*12) / (down.std()*math.sqrt(12)) if len(down) > 0 else 0
    total   = bt_eq.iloc[-1] / bt_eq.iloc[0] - 1
    return dict(cagr=round(cagr*100,2), sharpe=round(sharpe,3),
                sortino=round(sortino,3), max_dd=round(max_dd*100,2),
                calmar=round(calmar,3), total=round(total*100,1),
                ann_vol=round(ann_vol*100,2))

def run_backtest(lookback, skip, method):
    """
    method: 'cs' = cross-sectional within NQ100
            'rs' = relative strength vs SPY
    skip: number of most-recent months to exclude from lookback
    """
    # Momentum score
    if method == 'cs':
        # Cross-sectional: rank NQ100 stocks against each other
        if skip > 0:
            score = nq_monthly.pct_change(periods=lookback) - nq_monthly.pct_change(periods=skip)
        else:
            score = nq_monthly.pct_change(periods=lookback)
    else:
        # RS vs SPY: stock's excess return over SPY
        stock_ret  = nq_monthly.pct_change(periods=lookback)
        spy_lb_ret = spy_monthly.pct_change(periods=lookback)
        excess     = stock_ret.subtract(spy_lb_ret, axis=0)
        if skip > 0:
            stock_sk = nq_monthly.pct_change(periods=skip)
            spy_sk   = spy_monthly.pct_change(periods=skip)
            excess_sk = stock_sk.subtract(spy_sk, axis=0)
            score = excess - excess_sk
        else:
            score = excess

    # Rank and select top-N (shift 1 month: signal → next month)
    ranks      = score.rank(axis=1, ascending=False)
    top_mask   = (ranks <= TOP_N).shift(1)
    top_wts    = top_mask.div(TOP_N).fillna(0)

    # Regime
    reg = regime_monthly.reindex(nq_monthly.index, method="ffill").fillna(0)

    # Strategy returns
    mom_ret    = (nq_ret * top_wts).sum(axis=1)
    spy_ret_m  = spy_ret.reindex(nq_monthly.index).fillna(0)
    active_ret = reg * mom_ret
    total_ret  = SPY_FLOOR * spy_ret_m + ACTIVE_PCT * active_ret

    eq = (1 + total_ret).cumprod() * 100
    return eq, total_ret

# ── Optimization grid ─────────────────────────────────────────────────
LOOKBACKS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 24]
SKIPS     = [0, 1, 2]
METHODS   = ['cs', 'rs']

print("\nRunning optimization grid...", flush=True)
results = []
total_runs = len(LOOKBACKS) * len(SKIPS) * len(METHODS)
done = 0
for method in METHODS:
    for lb in LOOKBACKS:
        for sk in SKIPS:
            if sk >= lb: continue   # skip must be less than lookback
            eq, rets = run_backtest(lb, sk, method)
            m = calc_metrics(eq, rets)
            if m:
                results.append({**m, 'lookback': lb, 'skip': sk, 'method': method})
            done += 1
            sys.stdout.write(f"\r  {done}/{total_runs} runs done  ")
            sys.stdout.flush()
print(f"\nDone. {len(results)} valid results")

# ── Find best by Sharpe ──────────────────────────────────────────────
best_sharpe  = max(results, key=lambda x: x['sharpe'])
best_cagr    = max(results, key=lambda x: x['cagr'])
best_calmar  = max(results, key=lambda x: x['calmar'])
current_cs   = next((r for r in results if r['method']=='cs' and r['lookback']==12 and r['skip']==1), None)
best_rs      = max([r for r in results if r['method']=='rs'], key=lambda x: x['sharpe'])
best_cs      = max([r for r in results if r['method']=='cs'], key=lambda x: x['sharpe'])

print(f"\nCurrent (CS 12-1): {current_cs}")
print(f"Best CS: {best_cs}")
print(f"Best RS: {best_rs}")
print(f"Best overall (Sharpe): {best_sharpe}")
print(f"Best overall (CAGR): {best_cagr}")

# ── Run equity curves for key configs ────────────────────────────────
configs = {
    'current_cs12':  (12, 1, 'cs'),
    'best_cs':       (best_cs['lookback'],   best_cs['skip'],   'cs'),
    'best_rs':       (best_rs['lookback'],   best_rs['skip'],   'rs'),
    'best_sharpe':   (best_sharpe['lookback'], best_sharpe['skip'], best_sharpe['method']),
}
# Deduplicate
seen = set()
unique_configs = {}
for name, cfg in configs.items():
    key = cfg
    if key not in seen:
        seen.add(key)
        unique_configs[name] = cfg

curves = {}
for name, (lb, sk, meth) in unique_configs.items():
    eq, rets = run_backtest(lb, sk, meth)
    eq_bt = eq[eq.index >= bt_start]
    curves[name] = {
        'dates': [d.strftime('%Y-%m-%d') for d in eq_bt.index],
        'values': [round(v, 2) for v in eq_bt.values],
        'label': f"{'CS' if meth=='cs' else 'RS'} {lb}-{sk}",
        'metrics': calc_metrics(eq, rets)
    }

# SPY baseline
spy_bt = spy_ret.reindex(nq_monthly.index).fillna(0)
spy_eq = (1 + spy_bt).cumprod() * 100
spy_eq_bt = spy_eq[spy_eq.index >= bt_start]
curves['spy'] = {
    'dates': [d.strftime('%Y-%m-%d') for d in spy_eq_bt.index],
    'values': [round(v, 2) for v in spy_eq_bt.values],
    'label': 'SPY',
    'metrics': calc_metrics(spy_eq, spy_bt)
}

# ── Build grid tables (Sharpe heatmap by method) ─────────────────────
def build_grid(method):
    """Returns {lookback: {skip: sharpe}} dict."""
    grid = {}
    for r in results:
        if r['method'] != method: continue
        lb, sk = r['lookback'], r['skip']
        if lb not in grid: grid[lb] = {}
        grid[lb][sk] = r
    return grid

grid_cs = build_grid('cs')
grid_rs = build_grid('rs')

# Full results list for table
results_sorted = sorted(results, key=lambda x: -x['sharpe'])

# ── Save ──────────────────────────────────────────────────────────────
output = {
    'results': results_sorted,
    'best_sharpe': best_sharpe,
    'best_cagr': best_cagr,
    'best_cs': best_cs,
    'best_rs': best_rs,
    'current': current_cs,
    'curves': curves,
    'grid_cs': {str(k): v for k, v in grid_cs.items()},
    'grid_rs': {str(k): v for k, v in grid_rs.items()},
    'lookbacks': LOOKBACKS,
    'skips': SKIPS
}

out_path = '/sessions/tender-dreamy-mendel/mnt/outputs/opt_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f)

print(f"\nSaved to {out_path}")
