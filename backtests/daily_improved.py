"""
Daily Regime Trigger + System Improvement Analysis
"""
import json, time, math, sys, requests
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

START_DATA = "2004-01-01"
END_DATE   = "2026-06-30"
START_BT   = "2006-01-01"
HEADERS    = {"User-Agent": "Mozilla/5.0"}
TOP_N      = 3
LOOKBACK   = 11
SKIP       = 1
BULL_SPY   = 0.35
BULL_ACT   = 0.65
BEAR_SPY   = 0.10
BREADTH_TH = 0.40

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
REGIME_PROXY = ["MSFT","AAPL","AMZN","INTC","NVDA","CSCO","QCOM","ADBE","COST","GILD",
                "CMCSA","AMGN","EBAY","FAST","KLAC","LRCX","MCHP","PAYX","PCAR","TXN"]

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
            res = j.get("chart",{}).get("result",[])
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

def fetch_daily(ticker, retries=3):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={to_unix(START_DATA)}&period2={to_unix(END_DATE)}"
           f"&interval=1d&events=adjsplit")
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200: return ticker, None
            j = r.json()
            res = j["chart"]["result"][0]
            ts  = res["timestamp"]
            adj = (res.get("indicators",{}).get("adjclose",[{}])[0].get("adjclose")
                   or res["indicators"]["quote"][0].get("close"))
            dates = [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d") for t in ts]
            s = pd.Series(adj, index=pd.to_datetime(dates), name=ticker)
            return ticker, s[s.notna()]
        except:
            if attempt < retries-1: time.sleep(1)
    return ticker, None

# ── Download ────────────────────────────────────────────────────────
print("Downloading monthly prices...", flush=True)
monthly_prices = {}
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(fetch_monthly, t): t for t in NQ100 + ["SPY"]}
    done = 0
    for f in as_completed(futures):
        tk, s = f.result()
        done += 1
        if s is not None and len(s) >= 12:
            monthly_prices[tk] = s
        sys.stdout.write(f"\r  monthly {done}/{len(NQ100)+1} ")
        sys.stdout.flush()
print(f"\nGot {len(monthly_prices)} monthly tickers")

print("Downloading daily prices (regime proxy + SPY + key stocks)...", flush=True)
daily_fetch = list(set(REGIME_PROXY + ["SPY","MU","LRCX","AMD","AVGO","WBD","NVDA","NFLX","TSLA","INTC","CRWD","DASH","AAPL","MSFT"]))
daily_prices = {}
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(fetch_daily, t): t for t in daily_fetch}
    done = 0
    for f in as_completed(futures):
        tk, s = f.result()
        done += 1
        if s is not None and len(s) > 200:
            daily_prices[tk] = s
        sys.stdout.write(f"\r  daily {done}/{len(daily_fetch)} ")
        sys.stdout.flush()
print(f"\nGot {len(daily_prices)} daily tickers")

# ── Build frames ────────────────────────────────────────────────────
mp = pd.DataFrame(monthly_prices).sort_index()
mp.index = pd.to_datetime(mp.index)
nq_m  = mp[[c for c in mp.columns if c != "SPY"]]
spy_m = mp["SPY"]

dp = pd.DataFrame(daily_prices).sort_index()
dp.index = pd.to_datetime(dp.index)
spy_d = dp["SPY"] if "SPY" in dp.columns else None

proxy_cols = [c for c in dp.columns if c in REGIME_PROXY]
ew_d     = dp[proxy_cols].divide(dp[proxy_cols].ffill().bfill().iloc[0]).mean(axis=1)
ma200_d  = ew_d.rolling(200, min_periods=100).mean()
nq_bull_d = (ew_d > ma200_d)

spy_ma_d  = spy_d.rolling(200, min_periods=100).mean() if spy_d is not None else ew_d.rolling(200,min_periods=100).mean()
spy_bull_d = (spy_d > spy_ma_d) if spy_d is not None else nq_bull_d

# Dual regime: NQ100 EW AND SPY both above 200d MA — shifted T+1
dual_bull_d  = (nq_bull_d & spy_bull_d).shift(1).fillna(False)

# Monthly regime (last trading day of month, carry forward)
regime_m = dual_bull_d.resample("ME").last().astype(int)

# Monthly momentum
nq_ret_m  = nq_m.pct_change().fillna(0)
spy_ret_m = spy_m.pct_change().fillna(0)
score_m   = nq_m.pct_change(periods=LOOKBACK) - nq_m.pct_change(periods=SKIP)
ranks_m   = score_m.rank(axis=1, ascending=False)
top3_m    = (ranks_m <= TOP_N).shift(1)
top3_wts  = top3_m.div(TOP_N).fillna(0)
breadth_m = (score_m > 0).sum(axis=1) / score_m.notna().sum(axis=1)
breadth_m = breadth_m.shift(1).fillna(0)

bt_start = pd.Timestamp(START_BT)

def calc_metrics(eq, rets, label=""):
    eq   = eq[eq.index >= bt_start]
    rets = rets[rets.index >= bt_start]
    if len(eq) < 10: return {}
    # Detect daily vs monthly by index frequency
    is_daily = len(rets) > 500
    ann = 252 if is_daily else 12
    n   = len(rets) / ann
    cagr  = (eq.iloc[-1]/eq.iloc[0])**(1/n)-1
    vol   = rets.std() * math.sqrt(ann)
    shrp  = (rets.mean()*ann) / vol if vol > 0 else 0
    dd    = eq / eq.cummax() - 1
    mdd   = dd.min()
    calmar= cagr / abs(mdd) if mdd != 0 else 0
    down  = rets[rets < 0]
    sort  = (rets.mean()*ann) / (down.std()*math.sqrt(ann)) if len(down)>0 else 0
    total = eq.iloc[-1]/eq.iloc[0]-1
    return dict(label=label, cagr=round(cagr*100,2), sharpe=round(shrp,3),
                sortino=round(sort,3), max_dd=round(mdd*100,2),
                calmar=round(calmar,3), total=round(total*100,1),
                ann_vol=round(vol*100,2))

def monthly_curve(eq, label=""):
    eq = eq[eq.index >= bt_start]
    eq_r = eq.resample("ME").last().ffill()
    return {"label": label,
            "dates":  [d.strftime("%Y-%m-%d") for d in eq_r.index],
            "values": [round(float(v),2) for v in eq_r.values]}

# ══════════════════════════════════════════════════════════════════════
# SYSTEM A: Monthly baseline
# ══════════════════════════════════════════════════════════════════════
print("\nSystem A: monthly baseline...", flush=True)
reg_A  = regime_m.reindex(nq_m.index, method="ffill").fillna(0)
brd_A  = (breadth_m >= BREADTH_TH).astype(int).reindex(nq_m.index, method="ffill").fillna(0)
bull_A = (reg_A * brd_A).astype(int)
w_spy_A = bull_A * BULL_SPY + (1-bull_A) * BEAR_SPY
w_act_A = bull_A * BULL_ACT
mom_A   = (nq_ret_m * top3_wts).sum(axis=1)
ret_A   = w_spy_A * spy_ret_m + w_act_A * mom_A
eq_A    = (1+ret_A).cumprod()*100
m_A     = calc_metrics(eq_A, ret_A, "A: Monthly (optimised baseline)")
print(f"  {m_A}")

# ══════════════════════════════════════════════════════════════════════
# SYSTEM B: Daily regime, monthly momentum (daily equity curve)
# ══════════════════════════════════════════════════════════════════════
print("\nSystem B: daily regime trigger (T+1 exit), monthly momentum...", flush=True)

# Build daily active returns from available daily stock prices
# Carry forward monthly top-3 holdings to daily
all_m_dates = sorted(d for d in top3_wts.index if d >= bt_start)
top3_by_month = {}
for dt in all_m_dates:
    row = top3_wts.loc[dt]
    top3_by_month[dt] = row[row > 0].index.tolist()

all_days = spy_d.dropna().index if spy_d is not None else dp.dropna().index
all_days = all_days[all_days >= bt_start]

def get_top3(day):
    prev = [d for d in all_m_dates if d <= day]
    return top3_by_month[prev[-1]] if prev else []

spy_ret_d = spy_d.pct_change().fillna(0) if spy_d is not None else pd.Series(0.0, index=all_days)

# Daily active return (vectorised approach)
daily_stock_rets = {}
for tk in daily_prices:
    if tk not in ["SPY"]:
        s = daily_prices[tk]
        daily_stock_rets[tk] = s.pct_change().fillna(0)

active_d = pd.Series(0.0, index=all_days, dtype=float)
for day in all_days:
    holdings = get_top3(day)
    if not holdings: continue
    rets = []
    for tk in holdings:
        if tk in daily_stock_rets and day in daily_stock_rets[tk].index:
            rets.append(float(daily_stock_rets[tk][day]))
        elif day in spy_ret_d.index:
            rets.append(float(spy_ret_d[day]))  # SPY proxy fallback
    if rets: active_d[day] = float(np.mean(rets))

# Daily bull signal (breadth stays monthly)
brd_daily = breadth_m.reindex(all_days, method="ffill").fillna(0)
brd_ok_d  = (brd_daily >= BREADTH_TH)
bull_d_B  = (dual_bull_d.reindex(all_days, method="ffill").fillna(False) & brd_ok_d)

spy_d_aligned = spy_ret_d.reindex(all_days).fillna(0)
w_spy_B = bull_d_B * BULL_SPY + (~bull_d_B) * BEAR_SPY
w_act_B = bull_d_B * BULL_ACT
ret_B   = w_spy_B * spy_d_aligned + w_act_B * active_d
eq_B    = (1+ret_B).cumprod()*100
m_B     = calc_metrics(eq_B, ret_B, "B: Daily regime + monthly momentum")
print(f"  {m_B}")

# ══════════════════════════════════════════════════════════════════════
# SYSTEM C: Daily regime + weekly momentum refresh
# ══════════════════════════════════════════════════════════════════════
print("\nSystem C: daily regime + weekly momentum...", flush=True)
weekly_dates = pd.date_range(START_BT, END_DATE, freq="W-FRI")
score_w  = score_m.reindex(weekly_dates, method="ffill")
ranks_w  = score_w.rank(axis=1, ascending=False)
top3_w   = (ranks_w <= TOP_N).shift(1)
top3_wts_w = top3_w.div(TOP_N).fillna(0)
brd_w    = breadth_m.reindex(weekly_dates, method="ffill").fillna(0)
brd_ok_w = (brd_w >= BREADTH_TH)
regime_w = dual_bull_d.resample("W-FRI").last().reindex(weekly_dates, method="ffill").fillna(False)
bull_w_C = (regime_w & brd_ok_w)

# Weekly returns — use daily data resampled
nq_ret_w  = {}
for tk in nq_m.columns:
    if tk in daily_stock_rets:
        nq_ret_w[tk] = (1+daily_stock_rets[tk]).resample("W-FRI").prod()-1
    else:
        nq_ret_w[tk] = (1+nq_ret_m[tk]).resample("W-FRI").prod()-1
nq_ret_w = pd.DataFrame(nq_ret_w)
spy_ret_w = (1+spy_ret_d).resample("W-FRI").prod()-1 if spy_d is not None else spy_ret_m.resample("W-FRI").apply(lambda x:(1+x).prod()-1)

mom_C  = (nq_ret_w * top3_wts_w).sum(axis=1)
w_spy_C = bull_w_C * BULL_SPY + (~bull_w_C) * BEAR_SPY
w_act_C = bull_w_C * BULL_ACT
ret_C   = w_spy_C * spy_ret_w + w_act_C * mom_C
eq_C    = (1+ret_C).cumprod()*100
m_C     = calc_metrics(eq_C, ret_C, "C: Daily regime + weekly momentum")
print(f"  {m_C}")

# ══════════════════════════════════════════════════════════════════════
# SYSTEM D: Daily regime + monthly momentum + per-position stop -12%
# ══════════════════════════════════════════════════════════════════════
print("\nSystem D: daily regime + monthly momentum + per-position stop -12%...", flush=True)
STOP = -0.12
# Build daily equity but with stop-loss per position
# If any holding drops STOP% from its entry price (start of month), exit to cash T+1
# Entry price = closing price of last day of prior month

# Build entry prices per month for each top-3 holding
entry_prices = {}  # {month_start_dt: {ticker: entry_price}}
for i, dt in enumerate(all_m_dates):
    holdings = top3_by_month.get(dt, [])
    eps = {}
    for tk in holdings:
        if tk in daily_prices:
            # Entry at first trading day open of the month ≈ last day of prior month close
            prior = daily_prices[tk].index[daily_prices[tk].index < dt]
            if len(prior) > 0:
                eps[tk] = float(daily_prices[tk][prior[-1]])
    entry_prices[dt] = eps

# Simulate daily with stop
ret_D = pd.Series(0.0, index=all_days, dtype=float)
stopped_out = set()  # tickers stopped out this month, reset at month-end

current_month_start = None
for day in all_days:
    # Detect new month
    month_start = day.to_period("M").to_timestamp()
    if month_start != current_month_start:
        current_month_start = month_start
        stopped_out = set()
        # Find which monthly signal applies
        prev_m = [d for d in all_m_dates if d <= day]
        cur_holdings_month = top3_by_month.get(prev_m[-1], []) if prev_m else []

    bull = bool(bull_d_B.get(day, False))
    if not bull:
        ret_D[day] = float(BEAR_SPY * spy_d_aligned.get(day, 0))
        continue

    holdings = [t for t in cur_holdings_month if t not in stopped_out]
    spy_r = float(spy_d_aligned.get(day, 0))

    # Check stop-loss for each holding
    new_stops = set()
    for tk in holdings:
        # Find entry price for this month
        prev_m = [d for d in all_m_dates if d <= day]
        m_key  = prev_m[-1] if prev_m else None
        entry  = entry_prices.get(m_key, {}).get(tk)
        if entry and tk in daily_prices and day in daily_prices[tk].index:
            cur_p  = float(daily_prices[tk][day])
            dd_pos = (cur_p / entry) - 1
            if dd_pos < STOP:
                new_stops.add(tk)

    # Exit stopped positions T+1 (approximate: apply stop at today's return)
    active_r = 0.0
    n_active = len(holdings)
    for tk in holdings:
        if tk in new_stops:
            # Capped at stop level
            if tk in daily_stock_rets and day in daily_stock_rets[tk].index:
                r = max(float(daily_stock_rets[tk][day]), STOP)
            else:
                r = STOP
            active_r += r / TOP_N
            stopped_out.add(tk)
        else:
            if tk in daily_stock_rets and day in daily_stock_rets[tk].index:
                active_r += float(daily_stock_rets[tk][day]) / TOP_N
            else:
                active_r += spy_r / TOP_N

    n_cash_slots = TOP_N - len([t for t in cur_holdings_month if t not in stopped_out])
    total_act = active_r
    ret_D[day] = BULL_SPY * spy_r + BULL_ACT * total_act

eq_D  = (1+ret_D).cumprod()*100
m_D   = calc_metrics(eq_D, ret_D, "D: Daily regime + monthly mom + stop -12%")
print(f"  {m_D}")

# ══════════════════════════════════════════════════════════════════════
# SYSTEM E: Daily regime + weekly momentum + stop-loss (combined best)
# ══════════════════════════════════════════════════════════════════════
print("\nSystem E: combined (daily regime + weekly momentum + stop)...", flush=True)
# Apply stop-loss to weekly system
STOP_W = -0.10

ret_E = ret_C.copy()
# For each week, check if any holding's weekly return < stop
for dt in ret_C.index:
    if not bool(bull_w_C.get(dt, False)): continue
    wts_dt = top3_wts_w.loc[dt] if dt in top3_wts_w.index else pd.Series()
    holdings_e = wts_dt[wts_dt > 0].index.tolist() if len(wts_dt)>0 else []
    if not holdings_e: continue
    spy_r = float(spy_ret_w.get(dt, 0))
    act_r = 0.0
    for tk in holdings_e:
        if tk in nq_ret_w.columns and dt in nq_ret_w.index:
            r = float(nq_ret_w.loc[dt, tk])
            act_r += max(r, STOP_W) / TOP_N
        else:
            act_r += spy_r / TOP_N
    ret_E[dt] = BULL_SPY * spy_r + BULL_ACT * act_r

eq_E  = (1+ret_E).cumprod()*100
m_E   = calc_metrics(eq_E, ret_E, "E: Daily regime + weekly + stop -10%")
print(f"  {m_E}")

# ══════════════════════════════════════════════════════════════════════
# SPY baseline
# ══════════════════════════════════════════════════════════════════════
spy_eq_m = (1+spy_ret_m[spy_ret_m.index>=bt_start]).cumprod()*100
m_spy    = calc_metrics(spy_eq_m, spy_ret_m[spy_ret_m.index>=bt_start], "SPY (buy & hold)")

# ══════════════════════════════════════════════════════════════════════
# REGIME LAG ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\nRegime lag analysis...", flush=True)
bear_periods = [
    ("GFC 2008-09",  "2007-07-01", "2009-06-30"),
    ("COVID 2020",   "2020-01-01", "2020-06-30"),
    ("Bear 2022",    "2021-11-01", "2023-01-31"),
]
lag_analysis = []
for name, s, e in bear_periods:
    sd, ed = pd.Timestamp(s), pd.Timestamp(e)
    d_window = dual_bull_d.loc[sd:ed]
    bear_d   = d_window[~d_window]
    m_window = regime_m.loc[sd:ed]
    bear_m   = m_window[m_window == 0]
    fd = bear_d.index[0].strftime("%Y-%m-%d") if len(bear_d)>0 else "n/a"
    fm = bear_m.index[0].strftime("%Y-%m-%d") if len(bear_m)>0 else "n/a"
    lag = (bear_m.index[0] - bear_d.index[0]).days if len(bear_d)>0 and len(bear_m)>0 else None
    # SPY DD in window
    if spy_d is not None and len(spy_d.loc[sd:ed])>0:
        sw = spy_d.loc[sd:ed]
        pk = sw.cummax()
        dd_series = ((sw/pk)-1)*100
        mdd = round(float(dd_series.min()),1)
    else:
        mdd = None
    lag_analysis.append({"period":name,"daily_trigger":fd,"monthly_trigger":fm,"lag_days":lag,"spy_max_dd":mdd})
    print(f"  {name}: daily={fd}, monthly={fm}, lag={lag}d, SPY_DD={mdd}%")

# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════
output = {
    "metrics": {
        "A": m_A, "B": m_B, "C": m_C, "D": m_D, "E": m_E, "spy": m_spy
    },
    "curves": {
        "A":   monthly_curve(eq_A, "A: Monthly"),
        "B":   monthly_curve(eq_B, "B: Daily Regime"),
        "C":   monthly_curve(eq_C, "C: Weekly Momentum"),
        "D":   monthly_curve(eq_D, "D: +Stop Loss"),
        "E":   monthly_curve(eq_E, "E: Combined Best"),
        "spy": monthly_curve(spy_eq_m, "SPY"),
    },
    "lag_analysis": lag_analysis,
}

with open("/sessions/tender-dreamy-mendel/mnt/outputs/daily_improved.json","w") as f:
    json.dump(output, f)

print("\n=== SUMMARY ===")
for k, m in output["metrics"].items():
    if m:
        print(f"  {m.get('label',''):<48} CAGR={m.get('cagr'):>6}%  MaxDD={m.get('max_dd'):>7}%  Calmar={m.get('calmar'):>5}  Sharpe={m.get('sharpe')}")
