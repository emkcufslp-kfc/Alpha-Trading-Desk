"""
Risk Analysis Engine
1. Breadth filter: % of NQ100 stocks with positive momentum → add to regime
2. Rolling entry stress test: simulate starting every possible month, track max DD
3. Optimization grid: SPY floor × breadth threshold × lookback → Calmar (CAGR/|MaxDD|)
Outputs: risk_results.json
"""

import json, math, sys, time
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

START_DATA = "2004-01-01"
END_DATE   = "2026-06-30"
START_BT   = "2006-01-01"
HEADERS    = {"User-Agent": "Mozilla/5.0"}
TOP_N      = 3

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
            s = pd.Series(adj, index=pd.to_datetime(dates), name=ticker)
            return ticker, s[s.notna()]
        except:
            if attempt < retries-1: time.sleep(1)
    return ticker, None

def fetch_daily_regime():
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

# ── Download ─────────────────────────────────────────────────────────
print("Downloading prices...", flush=True)
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

# ── Build base DataFrames ─────────────────────────────────────────────
mp_df = pd.DataFrame(monthly_prices).sort_index()
mp_df.index = pd.to_datetime(mp_df.index)
nq_monthly  = mp_df[[c for c in mp_df.columns if c != "SPY"]]
spy_monthly  = mp_df["SPY"]

daily_df  = pd.DataFrame(daily_series).sort_index()
daily_df.index = pd.to_datetime(daily_df.index)
daily_norm = daily_df.divide(daily_df.ffill().bfill().iloc[0])
ew_daily   = daily_norm.mean(axis=1)

nq_ret  = nq_monthly.pct_change().fillna(0)
spy_ret = spy_monthly.pct_change().fillna(0)
bt_start = pd.Timestamp(START_BT)

# ── Build regime components ────────────────────────────────────────────
# 1. 200d MA regime (existing)
def build_ma_regime(ma_window=200):
    ma = ew_daily.rolling(ma_window, min_periods=100).mean()
    regime_d = (ew_daily > ma).astype(int).shift(1).fillna(0)
    return regime_d.resample("ME").last().reindex(nq_monthly.index, method="ffill").fillna(0)

# 2. Breadth: % of NQ100 stocks with positive 12-1 momentum score (shifted 1M)
def build_breadth(lookback=12, skip=1):
    """Monthly breadth = fraction of NQ100 stocks with positive score."""
    score = nq_monthly.pct_change(periods=lookback) - nq_monthly.pct_change(periods=skip)
    # Count how many have positive score at each month-end
    breadth = (score > 0).sum(axis=1) / score.notna().sum(axis=1)
    # Shift 1: signal at t → used in t+1 portfolio
    return breadth.shift(1).fillna(0)

# 3. Short-term breadth: % stocks with positive 1-month return (market pulse)
def build_short_breadth():
    pos = (nq_monthly.pct_change(1) > 0).sum(axis=1) / nq_monthly.pct_change(1).notna().sum(axis=1)
    return pos.shift(1).fillna(0)

# ── Core backtest function ────────────────────────────────────────────
def run_strategy(lookback=12, skip=1, spy_floor=0.60,
                 breadth_thresh=0.0, ma_window=200,
                 use_short_breadth=False):
    """
    Combined regime: MA regime AND breadth >= threshold → bull
    breadth_thresh=0 means no breadth filter (original behavior).
    """
    active_pct = 1.0 - spy_floor

    # Momentum signal
    score = nq_monthly.pct_change(lookback) - nq_monthly.pct_change(skip)
    ranks = score.rank(axis=1, ascending=False)
    top_mask = (ranks <= TOP_N).shift(1)
    top_wts  = top_mask.div(TOP_N).fillna(0)

    # Regime components
    ma_reg = build_ma_regime(ma_window)

    if breadth_thresh > 0:
        if use_short_breadth:
            breadth = build_short_breadth()
        else:
            breadth = build_breadth(lookback, skip)
        combined_regime = ((ma_reg >= 0.5) & (breadth >= breadth_thresh)).astype(float)
    else:
        combined_regime = ma_reg

    combined_regime = combined_regime.reindex(nq_monthly.index, method="ffill").fillna(0)

    # Returns
    spy_r  = spy_ret.reindex(nq_monthly.index).fillna(0)
    mom_r  = (nq_ret * top_wts).sum(axis=1)
    active = combined_regime * mom_r
    total  = spy_floor * spy_r + active_pct * active

    eq = (1 + total).cumprod() * 100
    return eq, total, combined_regime

def metrics(eq, rets, start=None):
    if start: eq = eq[eq.index >= start]; rets = rets[rets.index >= start]
    if len(eq) < 6: return None
    n = len(rets) / 12
    cagr   = (eq.iloc[-1]/eq.iloc[0])**(1/n) - 1
    vol    = rets.std() * math.sqrt(12)
    sharpe = (rets.mean()*12)/vol if vol > 0 else 0
    dd     = eq/eq.cummax()-1
    mdd    = dd.min()
    calmar = cagr/abs(mdd) if mdd != 0 else 0
    down   = rets[rets < 0]
    sortino= (rets.mean()*12)/(down.std()*math.sqrt(12)) if len(down)>0 else 0
    return dict(cagr=round(cagr*100,2), sharpe=round(sharpe,3),
                max_dd=round(mdd*100,2), calmar=round(calmar,3),
                sortino=round(sortino,3), vol=round(vol*100,2),
                total=round((eq.iloc[-1]/eq.iloc[0]-1)*100,1))

# ── Key strategies to compare ─────────────────────────────────────────
print("\nBuilding key strategy equity curves...", flush=True)

STRATEGIES = {
    "current":     dict(lookback=12, skip=1, spy_floor=0.60, breadth_thresh=0.0),
    "opt_cs11":    dict(lookback=11, skip=1, spy_floor=0.60, breadth_thresh=0.0),
    "breadth50":   dict(lookback=11, skip=1, spy_floor=0.60, breadth_thresh=0.50),
    "breadth60":   dict(lookback=11, skip=1, spy_floor=0.60, breadth_thresh=0.60),
    "breadth70":   dict(lookback=11, skip=1, spy_floor=0.60, breadth_thresh=0.70),
    "floor70":     dict(lookback=11, skip=1, spy_floor=0.70, breadth_thresh=0.0),
    "floor70_b60": dict(lookback=11, skip=1, spy_floor=0.70, breadth_thresh=0.60),
    "floor80":     dict(lookback=11, skip=1, spy_floor=0.80, breadth_thresh=0.0),
}

strat_results = {}
for name, params in STRATEGIES.items():
    eq, rets, regime = run_strategy(**params)
    eq_bt   = eq[eq.index >= bt_start]
    rets_bt = rets[rets.index >= bt_start]
    m = metrics(eq, rets, start=bt_start)
    strat_results[name] = {
        "params": params,
        "metrics": m,
        "dates":  [d.strftime("%Y-%m-%d") for d in eq_bt.index],
        "values": [round(v,2) for v in eq_bt.values],
    }
    print(f"  {name}: Calmar={m['calmar']:.3f} CAGR={m['cagr']:.1f}% MaxDD={m['max_dd']:.1f}%")

# SPY baseline
spy_eq_full = (1 + spy_ret.reindex(nq_monthly.index).fillna(0)).cumprod() * 100
spy_bt = spy_eq_full[spy_eq_full.index >= bt_start]
spy_rets_bt = spy_ret.reindex(nq_monthly.index).fillna(0)
spy_m = metrics(spy_eq_full, spy_rets_bt, start=bt_start)
strat_results["spy"] = {
    "params": {}, "metrics": spy_m,
    "dates": [d.strftime("%Y-%m-%d") for d in spy_bt.index],
    "values": [round(v,2) for v in spy_bt.values],
}

# ── Breadth time-series for chart ─────────────────────────────────────
breadth_series = build_breadth(11, 1)
breadth_bt = breadth_series[breadth_series.index >= bt_start]
short_breadth = build_short_breadth()
short_breadth_bt = short_breadth[short_breadth.index >= bt_start]

# ── Rolling entry stress test ─────────────────────────────────────────
print("\nRunning rolling entry stress test...", flush=True)

def rolling_entry_mdd(eq_series, window_months=36):
    """
    For each entry month t, simulate investing $100 at t,
    return the max drawdown experienced in the next `window_months`.
    """
    results = []
    idx = eq_series.index.tolist()
    vals = eq_series.values
    n = len(idx)
    for i in range(n - 3):
        end = min(i + window_months, n)
        sub = vals[i:end] / vals[i] * 100
        dd = (sub / np.maximum.accumulate(sub) - 1).min() * 100
        results.append({"date": idx[i].strftime("%Y-%m-%d"), "mdd": round(float(dd), 2)})
    return results

print("  Computing rolling entry DDmax for each strategy...", flush=True)
rolling_entry = {}
for name in ["current", "opt_cs11", "breadth60", "floor70_b60", "spy"]:
    eq_bt = pd.Series(
        strat_results[name]["values"],
        index=pd.to_datetime(strat_results[name]["dates"])
    )
    rolling_entry[name] = rolling_entry_mdd(eq_bt, window_months=36)

# ── Full grid optimization: Calmar maximization ───────────────────────
print("\nRunning full optimization grid (SPY floor × breadth × lookback)...", flush=True)

SPY_FLOORS      = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
BREADTH_THRESH  = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70]
LOOKBACKS_GRID  = [(11,1), (12,1), (9,1), (10,1)]

grid_results = []
total_runs = len(SPY_FLOORS) * len(BREADTH_THRESH) * len(LOOKBACKS_GRID)
done = 0
for lb, sk in LOOKBACKS_GRID:
    for spy_fl in SPY_FLOORS:
        for br_th in BREADTH_THRESH:
            eq, rets, _ = run_strategy(lookback=lb, skip=sk,
                                       spy_floor=spy_fl, breadth_thresh=br_th)
            m = metrics(eq, rets, start=bt_start)
            if m:
                grid_results.append({
                    **m,
                    "lookback": lb, "skip": sk,
                    "spy_floor": spy_fl, "breadth": br_th
                })
            done += 1
            sys.stdout.write(f"\r  {done}/{total_runs}  ")
            sys.stdout.flush()
print(f"\n  {len(grid_results)} valid configs")

# Sort by Calmar
grid_results.sort(key=lambda x: -x["calmar"])
best_calmar = grid_results[0]
print(f"  Best Calmar: {best_calmar}")

# Also find best for Sharpe and CAGR
best_sharpe_g = max(grid_results, key=lambda x: x["sharpe"])
best_cagr_g   = max(grid_results, key=lambda x: x["cagr"])
print(f"  Best Sharpe: {best_sharpe_g}")
print(f"  Best CAGR:   {best_cagr_g}")

# Run equity curve for best Calmar config
eq_bc, rets_bc, regime_bc = run_strategy(
    lookback=best_calmar["lookback"], skip=best_calmar["skip"],
    spy_floor=best_calmar["spy_floor"], breadth_thresh=best_calmar["breadth"]
)
eq_bc_bt = eq_bc[eq_bc.index >= bt_start]
strat_results["best_calmar"] = {
    "params": best_calmar,
    "metrics": metrics(eq_bc, rets_bc, start=bt_start),
    "dates": [d.strftime("%Y-%m-%d") for d in eq_bc_bt.index],
    "values": [round(v,2) for v in eq_bc_bt.values],
}
rolling_entry["best_calmar"] = rolling_entry_mdd(eq_bc_bt, window_months=36)

# ── Pareto frontier: CAGR vs MaxDD ───────────────────────────────────
# Find Pareto-optimal points (non-dominated: can't improve CAGR without worsening MaxDD)
def pareto_frontier(results):
    """Return configs on the Pareto frontier (maximize CAGR, minimize |MaxDD|)."""
    pts = sorted(results, key=lambda x: -x["cagr"])
    frontier = []
    best_dd = float("inf")
    for p in pts:
        if abs(p["max_dd"]) < best_dd:
            best_dd = abs(p["max_dd"])
            frontier.append(p)
    return frontier

pareto = pareto_frontier(grid_results)

# ── Breadth series statistics ─────────────────────────────────────────
breadth_stats = {
    "dates":  [d.strftime("%Y-%m-%d") for d in breadth_bt.index],
    "values": [round(float(v), 3) for v in breadth_bt.values],
    "short_breadth": [round(float(v), 3) for v in short_breadth_bt.values],
    "mean":   round(float(breadth_bt.mean()), 3),
    "median": round(float(breadth_bt.median()), 3),
}

# ── Save ──────────────────────────────────────────────────────────────
output = {
    "strat_results": strat_results,
    "rolling_entry": rolling_entry,
    "grid_results":  grid_results[:100],  # top-100 by calmar
    "all_grid":      grid_results,         # full grid for scatter
    "best_calmar":   best_calmar,
    "best_sharpe_g": best_sharpe_g,
    "best_cagr_g":   best_cagr_g,
    "pareto":        pareto,
    "breadth_stats": breadth_stats,
}

out_path = "/sessions/tender-dreamy-mendel/mnt/outputs/risk_results.json"
with open(out_path, "w") as f:
    json.dump(output, f)
print(f"\nSaved → {out_path}")
print(f"Grid configs: {len(grid_results)}, Pareto: {len(pareto)}")
