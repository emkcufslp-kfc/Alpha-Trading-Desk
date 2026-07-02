"""
MaxDD < 20% Optimization Engine

Root cause: with 60% fixed in SPY, SPY's -51% MaxDD alone costs 30% portfolio hit.
Key insight: must make SPY allocation regime-sensitive too.

Levers tested:
  A) Dynamic SPY floor (reduce SPY in bear, not just active sleeve)
  B) Full cash in bear (both SPY + active go to 0%)
  C) TLT rotation (bond ETF in bear instead of SPY)
  D) Volatility targeting (scale total position by rolling vol)
  E) Drawdown circuit breaker (cut exposure when DD > threshold)
  F) Combined best-of-all (A + breadth + best lookback)

Grid:
  bull_spy:  [40,50,60,70]%   ← SPY in bull regime
  bear_spy:  [0,10,20,30]%    ← SPY in bear regime (instead of fixed floor)
  breadth:   [0,30,50]%
  lookback:  [11,12]M, skip=1
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
            dates = [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d") for t in ts]
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
all_tickers = NQ100 + ["SPY","TLT"]
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
print(f"\nGot {len(monthly_prices)} tickers, TLT={'yes' if 'TLT' in monthly_prices else 'NO'}")

print("Downloading daily regime proxy...", flush=True)
daily_series = fetch_daily_regime()
print(f"Got {len(daily_series)} daily tickers")

# Also get SPY daily for its own MA filter
spy_daily_raw = None
try:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/SPY"
           f"?period1={to_unix(START_DATA)}&period2={to_unix(END_DATE)}"
           f"&interval=1d&events=adjsplit")
    r = requests.get(url, headers=HEADERS, timeout=20)
    j = r.json()
    res = j["chart"]["result"][0]
    ts  = res["timestamp"]
    adj = (res.get("indicators",{}).get("adjclose",[{}])[0].get("adjclose")
           or res["indicators"]["quote"][0].get("close"))
    dates = [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d") for t in ts]
    spy_daily_raw = pd.Series(adj, index=pd.to_datetime(dates), name="SPY")
    spy_daily_raw = spy_daily_raw[spy_daily_raw.notna()]
    print(f"SPY daily: {len(spy_daily_raw)} days")
except Exception as e:
    print(f"SPY daily failed: {e}")

# ── Build DataFrames ─────────────────────────────────────────────────
mp_df = pd.DataFrame(monthly_prices).sort_index()
mp_df.index = pd.to_datetime(mp_df.index)
nq_monthly  = mp_df[[c for c in mp_df.columns if c not in ("SPY","TLT")]]
spy_monthly  = mp_df["SPY"]
tlt_monthly  = mp_df.get("TLT", None)

nq_ret  = nq_monthly.pct_change().fillna(0)
spy_ret = spy_monthly.pct_change().fillna(0)
tlt_ret = tlt_monthly.pct_change().fillna(0) if tlt_monthly is not None else None

# Daily EW regime
daily_df  = pd.DataFrame(daily_series).sort_index()
daily_df.index = pd.to_datetime(daily_df.index)
daily_norm = daily_df.divide(daily_df.ffill().bfill().iloc[0])
ew_daily   = daily_norm.mean(axis=1)
ma200_ew   = ew_daily.rolling(200, min_periods=100).mean()
regime_d   = (ew_daily > ma200_ew).astype(int).shift(1).fillna(0)
ma_regime_monthly = regime_d.resample("ME").last().reindex(nq_monthly.index, method="ffill").fillna(0)

# SPY daily MA regime (separate filter for SPY sleeve)
if spy_daily_raw is not None:
    spy_ma200    = spy_daily_raw.rolling(200, min_periods=100).mean()
    spy_regime_d = (spy_daily_raw > spy_ma200).astype(int).shift(1).fillna(0)
    spy_ma_regime_monthly = spy_regime_d.resample("ME").last().reindex(nq_monthly.index, method="ffill").fillna(0)
else:
    spy_ma_regime_monthly = ma_regime_monthly.copy()

# Faster MA (100d) for early warning
ma100_ew   = ew_daily.rolling(100, min_periods=60).mean()
regime100_d = (ew_daily > ma100_ew).astype(int).shift(1).fillna(0)
ma100_regime_monthly = regime100_d.resample("ME").last().reindex(nq_monthly.index, method="ffill").fillna(0)

bt_start = pd.Timestamp(START_BT)

# ── Breadth helper ────────────────────────────────────────────────────
def get_breadth(lb=11, sk=1):
    score = nq_monthly.pct_change(lb) - nq_monthly.pct_change(sk)
    return ((score > 0).sum(axis=1) / score.notna().sum(axis=1)).shift(1).fillna(0)

# ── Core strategy: dynamic allocation ────────────────────────────────
def run_dynamic(lookback=11, skip=1,
                bull_spy=0.50, bull_active=0.50,
                bear_spy=0.00, bear_active=0.00,
                breadth_thresh=0.30,
                use_tlt=False,
                spy_own_ma=False,
                vol_target=None,
                dd_brake=None):
    """
    Dynamic allocation:
      Bull regime + breadth OK: bull_spy in SPY + bull_active in top-N
      Bear regime (or breadth fails): bear_spy in SPY (or TLT) + bear_active in cash

    vol_target: if set (e.g. 0.12), scale total position to target annual vol
    dd_brake:   if set (e.g. 0.10), cut to 50% when portfolio DD > dd_brake
    spy_own_ma: apply separate 200d MA filter to SPY sleeve
    use_tlt:    in bear, replace SPY with TLT
    """
    # Momentum signal (top-N selection)
    score    = nq_monthly.pct_change(lookback) - nq_monthly.pct_change(skip)
    ranks    = score.rank(axis=1, ascending=False)
    top_mask = (ranks <= TOP_N).shift(1)
    top_wts  = top_mask.div(TOP_N).fillna(0)

    # Regime
    ma_reg = ma_regime_monthly.copy()
    if breadth_thresh > 0:
        breadth = get_breadth(lookback, skip)
        combined_regime = ((ma_reg >= 0.5) & (breadth >= breadth_thresh)).astype(float)
    else:
        combined_regime = ma_reg

    # SPY sleeve regime
    spy_reg = spy_ma_regime_monthly if spy_own_ma else ma_regime_monthly
    spy_reg_bin = (spy_reg >= 0.5).astype(float)

    # Monthly returns
    spy_r   = spy_ret.reindex(nq_monthly.index).fillna(0)
    mom_r   = (nq_ret * top_wts).sum(axis=1)
    tlt_r   = tlt_ret.reindex(nq_monthly.index).fillna(0) if (use_tlt and tlt_ret is not None) else pd.Series(0.0, index=nq_monthly.index)

    # Dynamic weights
    is_bull = (combined_regime >= 0.5).values
    is_spy_bull = (spy_reg_bin >= 0.5).values

    w_spy_arr    = np.where(is_spy_bull, bull_spy, bear_spy)
    w_active_arr = np.where(is_bull,     bull_active, bear_active)

    # When bear and use_tlt: put bear_spy into TLT
    if use_tlt:
        w_tlt_arr = np.where(~is_spy_bull, bear_spy, 0.0)
        w_spy_arr = np.where(is_spy_bull, bull_spy, 0.0)
    else:
        w_tlt_arr = np.zeros(len(is_bull))

    # Total strategy return
    total = (w_spy_arr * spy_r.values +
             w_active_arr * mom_r.values +
             w_tlt_arr * tlt_r.values)
    total = pd.Series(total, index=nq_monthly.index)

    # Volatility targeting
    if vol_target is not None:
        roll_vol = total.rolling(12, min_periods=3).std() * math.sqrt(12)
        scale = (vol_target / roll_vol.shift(1)).clip(0.2, 2.0).fillna(1.0)
        total = total * scale

    # Drawdown circuit breaker
    if dd_brake is not None:
        eq_raw = (1 + total.fillna(0)).cumprod()
        dd     = eq_raw / eq_raw.cummax() - 1
        brake  = (dd.shift(1) < -dd_brake).astype(float) * 0.5  # cut to 50% when in brake
        total  = total * (1 - brake)

    eq = (1 + total.fillna(0)).cumprod() * 100
    return eq, total

def metrics(eq, rets, start=bt_start):
    eq_bt   = eq[eq.index >= start]
    rets_bt = rets[rets.index >= start]
    if len(eq_bt) < 12: return None
    n   = len(rets_bt) / 12
    cagr   = (eq_bt.iloc[-1]/eq_bt.iloc[0])**(1/n) - 1
    vol    = rets_bt.std() * math.sqrt(12)
    sharpe = (rets_bt.mean()*12)/vol if vol > 0 else 0
    dd     = eq_bt/eq_bt.cummax()-1
    mdd    = dd.min()
    calmar = cagr/abs(mdd) if mdd != 0 else 0
    down   = rets_bt[rets_bt < 0]
    sortino= (rets_bt.mean()*12)/(down.std()*math.sqrt(12)) if len(down)>0 else 0
    return dict(cagr=round(cagr*100,2), sharpe=round(sharpe,3),
                max_dd=round(mdd*100,2), calmar=round(calmar,3),
                sortino=round(sortino,3), vol=round(vol*100,2),
                total=round((eq_bt.iloc[-1]/eq_bt.iloc[0]-1)*100,1))

# ── Named strategy comparison ─────────────────────────────────────────
print("\nBuilding named strategies...", flush=True)

NAMED = {
    "A_current":     dict(bull_spy=0.60, bull_active=0.40, bear_spy=0.60, bear_active=0.00, lookback=12),
    "B_bestCalmar":  dict(bull_spy=0.50, bull_active=0.50, bear_spy=0.50, bear_active=0.00, breadth_thresh=0.30, lookback=11),
    "C_fullCash":    dict(bull_spy=0.60, bull_active=0.40, bear_spy=0.00, bear_active=0.00, lookback=11),
    "D_dynSPY":      dict(bull_spy=0.60, bull_active=0.40, bear_spy=0.20, bear_active=0.00, breadth_thresh=0.30, lookback=11),
    "E_dynAggressive":dict(bull_spy=0.40, bull_active=0.60, bear_spy=0.00, bear_active=0.00, breadth_thresh=0.40, lookback=11),
    "F_TLT":         dict(bull_spy=0.60, bull_active=0.40, bear_spy=0.00, bear_active=0.00, use_tlt=True, breadth_thresh=0.30, lookback=11),
    "G_spyMA":       dict(bull_spy=0.60, bull_active=0.40, bear_spy=0.00, bear_active=0.00, spy_own_ma=True, breadth_thresh=0.30, lookback=11),
    "H_volTarget15": dict(bull_spy=0.60, bull_active=0.40, bear_spy=0.00, bear_active=0.00, vol_target=0.15, breadth_thresh=0.30, lookback=11),
    "I_volTarget12": dict(bull_spy=0.50, bull_active=0.50, bear_spy=0.00, bear_active=0.00, vol_target=0.12, breadth_thresh=0.30, lookback=11),
    "J_ddBrake":     dict(bull_spy=0.50, bull_active=0.50, bear_spy=0.00, bear_active=0.00, dd_brake=0.08, breadth_thresh=0.30, lookback=11),
    "K_combined":    dict(bull_spy=0.50, bull_active=0.50, bear_spy=0.00, bear_active=0.00,
                          spy_own_ma=True, breadth_thresh=0.40, vol_target=0.15, lookback=11),
    "L_ultraDef":    dict(bull_spy=0.40, bull_active=0.60, bear_spy=0.00, bear_active=0.00,
                          spy_own_ma=True, breadth_thresh=0.50, vol_target=0.12, lookback=11),
}

named_results = {}
for name, params in NAMED.items():
    eq, rets = run_dynamic(**params)
    m = metrics(eq, rets)
    eq_bt = eq[eq.index >= bt_start]
    named_results[name] = {
        "params": params, "metrics": m,
        "dates":  [d.strftime("%Y-%m-%d") for d in eq_bt.index],
        "values": [round(v,2) for v in eq_bt.values],
        "label":  name
    }
    print(f"  {name}: CAGR={m['cagr']:.1f}% MaxDD={m['max_dd']:.1f}% Calmar={m['calmar']:.3f}")

# ── Full grid for scatter: maximize CAGR subject to MaxDD < 20% ───────
print("\nRunning full grid...", flush=True)

BULL_SPY_G    = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
BEAR_SPY_G    = [0.00, 0.10, 0.20]
BREADTH_G     = [0.00, 0.20, 0.30, 0.40, 0.50]
LOOKBACKS_G   = [(11,1), (12,1)]
SPY_OWN_MA_G  = [False, True]
VOL_TARGETS_G = [None, 0.12, 0.15]

grid_results = []
total_g = len(BULL_SPY_G)*len(BEAR_SPY_G)*len(BREADTH_G)*len(LOOKBACKS_G)*len(SPY_OWN_MA_G)*len(VOL_TARGETS_G)
done = 0
for lb, sk in LOOKBACKS_G:
    for b_spy in BULL_SPY_G:
        for br_spy in BEAR_SPY_G:
            for brd in BREADTH_G:
                for spy_ma in SPY_OWN_MA_G:
                    for vt in VOL_TARGETS_G:
                        b_act = 1.0 - b_spy   # active = remainder in bull
                        eq, rets = run_dynamic(
                            lookback=lb, skip=sk,
                            bull_spy=b_spy, bull_active=b_act,
                            bear_spy=br_spy, bear_active=0.0,
                            breadth_thresh=brd,
                            spy_own_ma=spy_ma,
                            vol_target=vt
                        )
                        m = metrics(eq, rets)
                        if m:
                            grid_results.append({
                                **m,
                                "lb": lb, "bull_spy": b_spy, "bear_spy": br_spy,
                                "breadth": brd, "spy_ma": spy_ma,
                                "vol_target": vt if vt else 0
                            })
                        done += 1
                        sys.stdout.write(f"\r  {done}/{total_g}  ")
                        sys.stdout.flush()
print(f"\n  {len(grid_results)} results")

# Filter for MaxDD < 20%
under20 = [r for r in grid_results if abs(r["max_dd"]) < 20.0]
under25 = [r for r in grid_results if abs(r["max_dd"]) < 25.0]
under20.sort(key=lambda x: -x["cagr"])
under25.sort(key=lambda x: -x["cagr"])

print(f"  Configs with MaxDD < 20%: {len(under20)}")
print(f"  Configs with MaxDD < 25%: {len(under25)}")

if under20:
    best20 = under20[0]
    print(f"  Best CAGR with MaxDD<20%: {best20}")
else:
    best20 = None
    print("  No config achieves MaxDD < 20% — best near-miss:")
    near = sorted(grid_results, key=lambda x: abs(abs(x["max_dd"])-20))[0]
    print(f"  Nearest: {near}")

# Best under 25% for fallback
best25 = under25[0] if under25 else max(grid_results, key=lambda x: x["calmar"])

# Run equity curve for best under-20% (or best under-25%)
best_target = best20 or best25
print(f"\nBest target config: {best_target}")

b_act = 1.0 - best_target["bull_spy"]
eq_bt_eq, rets_bt = run_dynamic(
    lookback=best_target["lb"], skip=1,
    bull_spy=best_target["bull_spy"], bull_active=b_act,
    bear_spy=best_target["bear_spy"], bear_active=0.0,
    breadth_thresh=best_target["breadth"],
    spy_own_ma=best_target["spy_ma"],
    vol_target=best_target["vol_target"] or None
)
eq_best_bt = eq_bt_eq[eq_bt_eq.index >= bt_start]
named_results["Z_target"] = {
    "params": best_target, "metrics": best_target,
    "dates":  [d.strftime("%Y-%m-%d") for d in eq_best_bt.index],
    "values": [round(v,2) for v in eq_best_bt.values],
    "label":  f"Target (MaxDD<{'20' if best20 else '25'}%)"
}

# ── Rolling entry for key strategies ─────────────────────────────────
def rolling_entry_mdd(eq_series, window=36):
    idx = list(eq_series.index)
    vals = eq_series.values
    out = []
    for i in range(len(vals)-3):
        end = min(i+window, len(vals))
        sub = vals[i:end]/vals[i]*100
        dd  = float((sub/np.maximum.accumulate(sub)-1).min()*100)
        out.append({"date": idx[i].strftime("%Y-%m-%d"), "mdd": round(dd,2)})
    return out

print("\nRolling entry for key strategies...", flush=True)
rolling = {}
for name in ["A_current","C_fullCash","G_spyMA","K_combined","Z_target"]:
    if name not in named_results: continue
    eq_s = pd.Series(named_results[name]["values"],
                     index=pd.to_datetime(named_results[name]["dates"]))
    rolling[name] = rolling_entry_mdd(eq_s)
    worst = min(rolling[name], key=lambda r: r["mdd"])
    print(f"  {name}: worst entry DD = {worst['mdd']:.1f}% on {worst['date']}")

# ── Drawdown decomposition for current vs target ──────────────────────
def dd_series(eq_list, dates_list):
    eq = pd.Series(eq_list, index=pd.to_datetime(dates_list))
    return ((eq/eq.cummax()-1)*100).round(2).tolist()

# ── SPY bear periods ──────────────────────────────────────────────────
# Annotate which months were bear (for reference)
bear_months = []
for dt, rv in zip(ma_regime_monthly.index, ma_regime_monthly.values):
    if rv < 0.5 and dt >= bt_start:
        bear_months.append(dt.strftime("%Y-%m-%d"))

# ── Save ──────────────────────────────────────────────────────────────
output = {
    "named_results": named_results,
    "grid_results":  grid_results,
    "under20":       under20[:30],
    "under25":       under25[:30],
    "best20":        best20,
    "best25":        best25,
    "best_target":   best_target,
    "rolling":       rolling,
    "bear_months":   bear_months,
    "has_tlt":       tlt_monthly is not None,
}
out_path = "/sessions/tender-dreamy-mendel/mnt/outputs/maxdd_results.json"
with open(out_path, "w") as f:
    json.dump(output, f)
print(f"\nSaved → {out_path}")
print(f"Under-20% configs: {len(under20)}, Under-25%: {len(under25)}")
