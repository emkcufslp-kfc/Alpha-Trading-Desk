"""
20-Year Hybrid Strategy Backtest (2006–2026)
60% SPY floor + 40% NQ100 Top-3 Monthly 12-1 Momentum + 200d MA Regime
Outputs JSON data file for embedding in dashboard.html
"""

import json, time, math, sys
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────
START_DATA  = "2004-01-01"   # data warmup start
START_BT    = "2006-01-01"   # equity curve start (after warmup)
END_DATE    = "2026-06-30"
SPY_FLOOR   = 0.60
ACTIVE_PCT  = 0.40
TOP_N       = 3
MA_WINDOW   = 200            # daily
HEADERS     = {"User-Agent": "Mozilla/5.0 (compatible; backtest/1.0)"}

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

# ── Helpers ─────────────────────────────────────────────────────────────
def to_unix(d):
    return int(datetime.strptime(d, "%Y-%m-%d").timestamp())

def fetch_monthly(ticker, retries=3):
    """Monthly adjusted-close from Yahoo Finance v8."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={to_unix(START_DATA)}&period2={to_unix(END_DATE)}"
        f"&interval=1mo&events=adjsplit"
    )
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                return ticker, None
            j = r.json()
            res = j.get("chart", {}).get("result", [])
            if not res:
                return ticker, None
            data = res[0]
            ts   = data["timestamp"]
            adj  = (data.get("indicators", {})
                       .get("adjclose", [{}])[0]
                       .get("adjclose", None))
            if adj is None:
                adj = data["indicators"]["quote"][0].get("close")
            if not adj:
                return ticker, None
            dates  = [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d") for t in ts]
            series = pd.Series(adj, index=pd.to_datetime(dates), name=ticker)
            series = series[series.notna()]
            return ticker, series
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
    return ticker, None

def fetch_daily(ticker):
    """Daily adjusted-close for regime calculation."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={to_unix(START_DATA)}&period2={to_unix(END_DATE)}"
        f"&interval=1d&events=adjsplit"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        j = r.json()
        res = j["chart"]["result"][0]
        ts  = res["timestamp"]
        adj = (res.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
               or res["indicators"]["quote"][0].get("close"))
        dates  = [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d") for t in ts]
        series = pd.Series(adj, index=pd.to_datetime(dates), name=ticker)
        return series[series.notna()]
    except:
        return pd.Series(dtype=float, name=ticker)

# ── Download data ────────────────────────────────────────────────────────
print("Downloading monthly prices for NQ100 + SPY...", flush=True)
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
        sys.stdout.write(f"\r  {done}/{len(all_tickers)} tickers done  ")
        sys.stdout.flush()
print(f"\nGot {len(monthly_prices)} tickers with data")

print("Downloading daily NQ100 prices for 200d MA regime...", flush=True)
# Use a stable proxy: MSFT, AAPL, AMZN, GOOGL, INTC, NVDA, CSCO, QCOM, ADBE, COST
regime_proxy = ["MSFT","AAPL","AMZN","INTC","NVDA","CSCO","QCOM","ADBE","COST","GILD",
                "CMCSA","AMGN","EBAY","FAST","KLAC","LRCX","MCHP","PAYX","PCAR","TXN"]
daily_series = {}
for tk in regime_proxy:
    s = fetch_daily(tk)
    if len(s) > 200:
        daily_series[tk] = s
    time.sleep(0.1)
print(f"Got {len(daily_series)} tickers for daily regime")

# ── Build price DataFrames ────────────────────────────────────────────────
# Monthly prices (end-of-month)
mp_df = pd.DataFrame(monthly_prices)
mp_df.index = pd.to_datetime(mp_df.index)
mp_df = mp_df.sort_index()

# NQ100 monthly (exclude SPY)
nq_monthly = mp_df[[c for c in mp_df.columns if c != "SPY"]]
spy_monthly = mp_df["SPY"] if "SPY" in mp_df.columns else None

# Daily EW index for regime
daily_df = pd.DataFrame(daily_series)
daily_df.index = pd.to_datetime(daily_df.index)
daily_df = daily_df.sort_index()
# Normalize to first available price so all stocks contribute equally
daily_norm = daily_df.divide(daily_df.ffill().bfill().iloc[0])
ew_daily = daily_norm.mean(axis=1)

# 200d MA regime (1=bull, 0=bear), shifted by 1 day (no lookahead)
ma200     = ew_daily.rolling(MA_WINDOW, min_periods=100).mean()
regime_d  = (ew_daily > ma200).astype(int)
regime_d_shift = regime_d.shift(1).fillna(0)

# Map daily regime to monthly: use regime at last trading day of prior month
regime_monthly = regime_d_shift.resample("ME").last()

# ── Monthly Backtest ─────────────────────────────────────────────────────
print("Running backtest...", flush=True)

# 12-1 momentum: 12-month return minus most-recent 1-month return
mom12 = nq_monthly.pct_change(periods=12)
mom1  = nq_monthly.pct_change(periods=1)
mom12_1 = mom12 - mom1

# Rank top-N each month (ascending=False → rank 1 = highest momentum)
# Shift by 1 month: signal at end of month t → position for month t+1
ranks = mom12_1.rank(axis=1, ascending=False)
top3_mask = (ranks <= TOP_N).shift(1)   # apply next month
top3_weights = top3_mask.div(TOP_N).fillna(0)   # equal weight

# Monthly SPY returns
spy_ret = spy_monthly.pct_change().fillna(0) if spy_monthly is not None else pd.Series(0, index=nq_monthly.index)

# Monthly NQ100 returns
nq_ret = nq_monthly.pct_change().fillna(0)

# Active sleeve monthly returns (momentum top-3)
mom_ret = (nq_ret * top3_weights).sum(axis=1)

# Regime: join daily regime_monthly to our backtest index
reg = regime_monthly.reindex(nq_monthly.index, method="ffill").fillna(0)

# Hybrid strategy return
active_ret  = reg * mom_ret
total_ret   = SPY_FLOOR * spy_ret + ACTIVE_PCT * active_ret

# Filter to backtest start
bt_start = pd.Timestamp(START_BT)
total_ret   = total_ret[total_ret.index >= bt_start]
spy_ret_bt  = spy_ret[spy_ret.index >= bt_start]
reg_bt      = reg[reg.index >= bt_start]
mom_ret_bt  = mom_ret[mom_ret.index >= bt_start]
ranks_bt    = ranks[ranks.index >= bt_start]
top3_w_bt   = top3_weights[top3_weights.index >= bt_start]

# Equity curves (start at 100)
hybrid_eq   = (1 + total_ret).cumprod() * 100
spy_eq      = (1 + spy_ret_bt).cumprod() * 100

# NQ100 EW monthly return for benchmark
nq_ew_ret   = nq_ret.mean(axis=1)
nq_ew_ret   = nq_ew_ret[nq_ew_ret.index >= bt_start]
nq_eq       = (1 + nq_ew_ret).cumprod() * 100

# ── Performance metrics ───────────────────────────────────────────────────
def calc_metrics(eq, rets):
    n_years = len(rets) / 12
    total   = eq.iloc[-1] / eq.iloc[0] - 1
    cagr    = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1
    ann_vol = rets.std() * math.sqrt(12)
    sharpe  = (rets.mean() * 12) / ann_vol if ann_vol > 0 else 0
    dd      = eq / eq.cummax() - 1
    max_dd  = dd.min()
    calmar  = cagr / abs(max_dd) if max_dd != 0 else 0
    down    = rets[rets < 0]
    sortino = (rets.mean() * 12) / (down.std() * math.sqrt(12)) if len(down) > 0 else 0
    return dict(
        total_return=round(total*100, 1),
        cagr=round(cagr*100, 1),
        ann_vol=round(ann_vol*100, 1),
        sharpe=round(sharpe, 2),
        sortino=round(sortino, 2),
        max_dd=round(max_dd*100, 1),
        calmar=round(calmar, 2),
        n_years=round(n_years, 1)
    )

m_hybrid = calc_metrics(hybrid_eq, total_ret)
m_spy    = calc_metrics(spy_eq,    spy_ret_bt)
m_nq     = calc_metrics(nq_eq,     nq_ew_ret)

print("Hybrid metrics:", m_hybrid)
print("SPY metrics:", m_spy)

# ── Drawdown series ────────────────────────────────────────────────────────
dd_hybrid = ((hybrid_eq / hybrid_eq.cummax()) - 1) * 100
dd_spy    = ((spy_eq    / spy_eq.cummax())    - 1) * 100
dd_nq     = ((nq_eq     / nq_eq.cummax())     - 1) * 100

# ── Monthly Returns Grid (for log/heatmap) ─────────────────────────────────
# Build year × month grid
monthly_log = []
for dt, r in total_ret.items():
    monthly_log.append({"year": dt.year, "month": dt.month, "ret": round(r*100, 2),
                         "regime": int(reg_bt.get(dt, 0))})

# Annual returns
yearly = total_ret.resample("YE").apply(lambda x: (1+x).prod()-1)
yearly_spy = spy_ret_bt.resample("YE").apply(lambda x: (1+x).prod()-1)

# ── Holdings Ledger ────────────────────────────────────────────────────────
ledger = []
for dt in total_ret.index:
    r = reg_bt.get(dt, 0)
    regime_str = "Bull" if r else "Bear→Cash"
    # Top-3 holdings (from previous month's signal)
    if r and dt in top3_w_bt.index:
        row = top3_w_bt.loc[dt]
        held = row[row > 0].index.tolist()
    else:
        held = []
    # Scores for held tickers
    scores = {}
    if dt in ranks_bt.index:
        for tk in held:
            if tk in ranks_bt.columns:
                scores[tk] = round(float(ranks_bt.loc[dt, tk]), 0)
    # Monthly returns of each holding
    ret_this = {}
    if dt in nq_ret.index:
        for tk in held:
            if tk in nq_ret.columns:
                ret_this[tk] = round(float(nq_ret.loc[dt, tk])*100, 1)
    ledger.append({
        "date": dt.strftime("%Y-%m-%d"),
        "holdings": held,
        "regime": regime_str,
        "strat_ret": round(float(total_ret.get(dt, 0))*100, 2),
        "spy_ret": round(float(spy_ret_bt.get(dt, 0))*100, 2),
        "holding_rets": ret_this
    })

# ── Annual summary ─────────────────────────────────────────────────────────
annual_summary = []
for dt in yearly.index:
    y = dt.year
    annual_summary.append({
        "year": y,
        "strat": round(float(yearly.get(dt, 0))*100, 1),
        "spy":   round(float(yearly_spy.get(dt, 0))*100, 1) if dt in yearly_spy.index else None
    })

# ── Assemble output ────────────────────────────────────────────────────────
dates_str   = [d.strftime("%Y-%m-%d") for d in hybrid_eq.index]
output = {
    "metrics": {"hybrid": m_hybrid, "spy": m_spy, "nq_ew": m_nq},
    "dates":   dates_str,
    "hybrid_eq":  [round(v,2) for v in hybrid_eq.values],
    "spy_eq":     [round(v,2) for v in spy_eq.values],
    "nq_eq":      [round(v,2) for v in nq_eq.values],
    "dd_hybrid":  [round(v,2) for v in dd_hybrid.values],
    "dd_spy":     [round(v,2) for v in dd_spy.values],
    "dd_nq":      [round(v,2) for v in dd_nq.values],
    "monthly_log": monthly_log,
    "annual":      annual_summary,
    "ledger":      ledger,
    "regime_dates": [d.strftime("%Y-%m-%d") for d in reg_bt.index],
    "regime_vals":  [int(v) for v in reg_bt.values],
    "n_tickers_avail": len(nq_monthly.columns)
}

out_path = "/sessions/tender-dreamy-mendel/mnt/outputs/backtest20y.json"
with open(out_path, "w") as f:
    json.dump(output, f)

print(f"\nSaved to {out_path}")
print(f"Period: {dates_str[0]} → {dates_str[-1]}  ({m_hybrid['n_years']} years)")
print(f"Tickers with data: {len(nq_monthly.columns)}")
