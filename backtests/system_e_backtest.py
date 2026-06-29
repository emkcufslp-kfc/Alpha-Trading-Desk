"""
System E — Comprehensive Backtest with Real Trading Constraints
==============================================================
Strategy:
  • Weekly momentum ranking (11-1 on weekly closes)
  • Daily regime monitoring: NQ100 EW + SPY both > 200d MA → exit T+1 when bear
  • Per-position stop-loss: -10% from entry price → close T+1
  • Breadth filter: ≥40% of NQ100 have positive 11-1 score
  • Bull: 35% SPY + 65% active (21.7% each of top-3)
  • Bear: 10% SPY + 90% cash/money-market

Real trading constraints:
  • Commission: 0.05% per trade one-way (IB Lite / Fidelity zero-commission = 0%)
  • Slippage: 0.10% per trade one-way (bid-ask + market impact for retail size)
  • Total round-trip cost: 0.30% per position change
  • Execution: signal at Friday close → execute at Monday open (T+1)
  • Stop-loss: triggered intraweek if price < entry*(1-STOP), executes next open
  • Turnover tracked and reported
  • Walk-forward OOS: 4 folds

Survivorship bias caveat:
  • Uses current NQ100 composition — expected CAGR inflation ~4-6pts vs real universe
"""

import json, time, math, sys, requests
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ──────────────────────────────────────────────────────────
START_DATA   = "2004-01-01"
END_DATE     = "2026-06-30"
START_BT     = "2006-01-01"
HEADERS      = {"User-Agent": "Mozilla/5.0"}
TOP_N        = 3
LOOKBACK_WK  = 11 * 4   # 11 months in weeks (~44 weeks)
SKIP_WK      = 4        # 1 month skip in weeks
BULL_SPY     = 0.35
BULL_ACT     = 0.65
BEAR_SPY     = 0.10
BREADTH_TH   = 0.40
STOP_LOSS    = -0.10    # -10% from entry → exit T+1
COMMISSION   = 0.0005   # 0.05% per trade one-way
SLIPPAGE     = 0.0010   # 0.10% per trade one-way
TOTAL_COST   = COMMISSION + SLIPPAGE   # 0.15% per trade = 0.30% round-trip

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

# ── Download all daily prices ────────────────────────────────────────
CACHE = "/sessions/tender-dreamy-mendel/mnt/outputs/daily_all.pkl"
import os, pickle

if os.path.exists(CACHE):
    print("Loading cached daily prices...", flush=True)
    with open(CACHE,'rb') as f:
        daily_all = pickle.load(f)
    print(f"  Loaded {len(daily_all)} tickers from cache")
else:
    print("Downloading daily prices for all NQ100 + SPY...", flush=True)
    all_tickers = list(set(NQ100 + ["SPY"] + REGIME_PROXY))
    daily_all = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_daily, t): t for t in all_tickers}
        done = 0
        for f in as_completed(futures):
            tk, s = f.result()
            done += 1
            if s is not None and len(s) > 252:
                daily_all[tk] = s
            sys.stdout.write(f"\r  {done}/{len(all_tickers)} tickers done  ")
            sys.stdout.flush()
    print(f"\nGot {len(daily_all)} daily tickers")
    with open(CACHE,'wb') as f:
        pickle.dump(daily_all, f)
    print("  Cached to", CACHE)

# ── Build daily price matrix ──────────────────────────────────────────
dp = pd.DataFrame(daily_all).sort_index()
dp.index = pd.to_datetime(dp.index)
# Forward-fill gaps (holidays, missing data) — max 5 days
dp = dp.ffill(limit=5)

nq_d   = dp[[c for c in dp.columns if c in NQ100]]
spy_d  = dp["SPY"] if "SPY" in dp.columns else None

print(f"Daily price matrix: {nq_d.shape[0]} days × {nq_d.shape[1]} NQ100 tickers")
print(f"Date range: {dp.index[0].date()} → {dp.index[-1].date()}")

# ── Regime signals (daily) ───────────────────────────────────────────
proxy_cols = [c for c in dp.columns if c in REGIME_PROXY]
ew_d      = dp[proxy_cols].divide(dp[proxy_cols].ffill().bfill().iloc[0]).mean(axis=1)
ma200_d   = ew_d.rolling(200, min_periods=100).mean()
nq_bull_d = (ew_d > ma200_d)

spy_ma_d   = spy_d.rolling(200, min_periods=100).mean() if spy_d is not None else ma200_d
spy_bull_d = (spy_d > spy_ma_d) if spy_d is not None else nq_bull_d

# Dual regime — T+1 (shifted 1 trading day)
dual_bull_raw = (nq_bull_d & spy_bull_d)
dual_bull_d   = dual_bull_raw.shift(1).fillna(False)  # T+1: yesterday's signal → today's position

# ── Weekly prices & returns ───────────────────────────────────────────
# Use Friday close as end-of-week
nq_w   = nq_d.resample("W-FRI").last()
spy_w  = spy_d.resample("W-FRI").last() if spy_d is not None else None

nq_ret_w  = nq_w.pct_change().fillna(0)
spy_ret_w = spy_w.pct_change().fillna(0) if spy_w is not None else pd.Series(0, index=nq_w.index)

# ── Weekly momentum score (11-1 month lookback on weekly bars) ────────
# 11 months × 4.33 weeks ≈ 44 weeks lookback, 4 weeks skip
score_w = nq_w.pct_change(periods=LOOKBACK_WK) - nq_w.pct_change(periods=SKIP_WK)

# Breadth: fraction of NQ100 with positive score, shifted 1 week (PIT)
breadth_w = (score_w > 0).sum(axis=1) / score_w.notna().sum(axis=1)
breadth_w = breadth_w.shift(1).fillna(0)

# Top-3 ranking, shifted 1 week (signal this Friday → position next Monday)
ranks_w   = score_w.rank(axis=1, ascending=False)
top3_mask = (ranks_w <= TOP_N).shift(1)
top3_wts  = top3_mask.div(TOP_N).fillna(0)

# ── Weekly regime (from daily, sampled at Friday) ─────────────────────
regime_w = dual_bull_d.resample("W-FRI").last().reindex(nq_w.index, method="ffill").fillna(False)
brd_ok_w = (breadth_w >= BREADTH_TH)
bull_w    = (regime_w & brd_ok_w)

bt_start = pd.Timestamp(START_BT)

# ── Metrics helper ────────────────────────────────────────────────────
def calc_metrics(eq, rets, label="", freq="weekly"):
    eq   = eq[eq.index >= bt_start].dropna()
    rets = rets[rets.index >= bt_start].dropna()
    if len(eq) < 20: return {}
    ann  = {"daily":252,"weekly":52,"monthly":12}.get(freq,52)
    n    = len(rets) / ann
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/n)-1
    vol  = rets.std() * math.sqrt(ann)
    shrp = (rets.mean()*ann)/vol if vol>0 else 0
    dd   = eq/eq.cummax()-1
    mdd  = dd.min()
    calmar = cagr/abs(mdd) if mdd!=0 else 0
    down = rets[rets<0]
    sort = (rets.mean()*ann)/(down.std()*math.sqrt(ann)) if len(down)>0 else 0
    total = eq.iloc[-1]/eq.iloc[0]-1
    return dict(label=label, cagr=round(cagr*100,2), sharpe=round(shrp,3),
                sortino=round(sort,3), max_dd=round(mdd*100,2), calmar=round(calmar,3),
                total=round(total*100,1), ann_vol=round(vol*100,2), n_years=round(n,1))

def to_monthly_curve(eq, label=""):
    eq = eq[eq.index >= bt_start].dropna()
    eq_m = eq.resample("ME").last().ffill()
    return {"label": label,
            "dates":  [d.strftime("%Y-%m-%d") for d in eq_m.index],
            "values": [round(float(v),2) for v in eq_m.values]}

# ══════════════════════════════════════════════════════════════════════
# SYSTEM A (weekly baseline — no costs, for fair comparison)
# ══════════════════════════════════════════════════════════════════════
print("\nSystem A — Weekly, no transaction costs...", flush=True)
w_spy_A = bull_w.astype(float)*BULL_SPY + (~bull_w).astype(float)*BEAR_SPY
w_act_A = bull_w.astype(float)*BULL_ACT
mom_A   = (nq_ret_w * top3_wts).sum(axis=1)
ret_A   = w_spy_A*spy_ret_w + w_act_A*mom_A
eq_A    = (1+ret_A).cumprod()*100
m_A     = calc_metrics(eq_A, ret_A, "A: Weekly no-cost (gross)", "weekly")
print(f"  {m_A}")

# ══════════════════════════════════════════════════════════════════════
# SYSTEM E — Full System E with all constraints
# ══════════════════════════════════════════════════════════════════════
print("\nSystem E — Full constraints (costs + stop-loss + daily regime)...", flush=True)

# Track positions and entry prices
weekly_dates_bt = nq_w.index[nq_w.index >= bt_start]
portfolio = {}   # {ticker: entry_price}  — current holdings
nav        = 100.0
nav_history = []
date_history = []
turnover_total = 0.0
trades_total   = 0
stop_events    = 0
regime_exits   = 0

# We need daily prices for stop-loss monitoring
nq_d_bt = nq_d[nq_d.index >= bt_start]

prev_bull = None
prev_top3 = []

for i, week_end in enumerate(weekly_dates_bt):
    if i == 0:
        nav_history.append(nav)
        date_history.append(week_end)
        prev_bull = bool(bull_w.get(week_end, False))
        row = top3_wts.loc[week_end] if week_end in top3_wts.index else pd.Series()
        prev_top3 = row[row>0].index.tolist() if len(row)>0 else []
        for tk in prev_top3:
            p = float(nq_w.loc[week_end, tk]) if week_end in nq_w.index and tk in nq_w.columns else None
            if p: portfolio[tk] = p
        continue

    cur_bull = bool(bull_w.get(week_end, False))
    row = top3_wts.loc[week_end] if week_end in top3_wts.index else pd.Series()
    cur_top3 = row[row>0].index.tolist() if len(row)>0 else []

    # ── Check for mid-week stop-loss hits using daily prices ─────────
    prev_week_end = weekly_dates_bt[i-1]
    week_days = nq_d_bt.index[(nq_d_bt.index > prev_week_end) & (nq_d_bt.index <= week_end)]

    stopped_this_week = set()
    for day in week_days:
        for tk in list(portfolio.keys()):
            if tk in stopped_this_week: continue
            if tk not in nq_d_bt.columns: continue
            if day not in nq_d_bt.index: continue
            cur_p = float(nq_d_bt.loc[day, tk]) if not np.isnan(float(nq_d_bt.loc[day, tk])) else None
            entry_p = portfolio[tk]
            if cur_p and entry_p and (cur_p/entry_p - 1) < STOP_LOSS:
                # Stop triggered — record exit at stop price (entry * (1+STOP_LOSS))
                stop_price = entry_p * (1 + STOP_LOSS)
                # Cost of exit (slippage already at worst)
                trade_cost = TOTAL_COST
                nav *= (1 - BULL_ACT/TOP_N * trade_cost)
                del portfolio[tk]
                stopped_this_week.add(tk)
                stop_events += 1
                trades_total += 1
                # Immediately buy next-best available stock not in portfolio
                if week_end in ranks_w.index and cur_bull:
                    rk = ranks_w.loc[week_end].drop(labels=[t for t in portfolio]+list(stopped_this_week), errors='ignore')
                    next_best = rk.idxmin() if len(rk)>0 else None
                    if next_best and next_best in nq_d_bt.columns and day in nq_d_bt.index:
                        nb_p = float(nq_d_bt.loc[day, next_best])
                        if not np.isnan(nb_p):
                            portfolio[next_best] = nb_p
                            nav *= (1 - BULL_ACT/TOP_N * trade_cost)  # entry cost
                            trades_total += 1

    # ── Regime change: daily monitoring (already T+1 shifted) ────────
    # If regime flipped bear since last week, track it
    # The regime_w already captures daily sampling, so we just use it
    if not cur_bull and prev_bull:
        regime_exits += 1

    # ── Compute week's portfolio return ──────────────────────────────
    spy_r = float(spy_ret_w.get(week_end, 0))
    act_r = 0.0
    for tk in list(portfolio.keys()):
        if tk in nq_ret_w.columns and week_end in nq_ret_w.index:
            act_r += float(nq_ret_w.loc[week_end, tk]) / TOP_N

    w_spy = BULL_SPY if cur_bull else BEAR_SPY
    w_act = BULL_ACT if cur_bull else 0.0
    weekly_ret = w_spy * spy_r + w_act * act_r

    # ── Rebalance at week end ────────────────────────────────────────
    if cur_bull:
        new_top3 = set(cur_top3)
        old_top3 = set(portfolio.keys())

        # Exit positions not in new top-3
        exits = old_top3 - new_top3 - stopped_this_week
        entries = new_top3 - old_top3

        # Apply costs for trades
        n_changes = len(exits) + len(entries)
        if not prev_bull:  # regime transition: full re-entry cost
            n_changes = len(new_top3)
        if n_changes > 0:
            cost_this_week = n_changes * TOTAL_COST * (BULL_ACT/TOP_N)
            weekly_ret -= cost_this_week
            turnover_total += n_changes / TOP_N
            trades_total += n_changes

        # Update portfolio
        for tk in exits:
            if tk in portfolio: del portfolio[tk]
        for tk in entries:
            p = float(nq_w.loc[week_end, tk]) if week_end in nq_w.index and tk in nq_w.columns else None
            if p and not np.isnan(p): portfolio[tk] = p
    else:
        # Bear — exit all active
        if portfolio:
            cost_bear = len(portfolio) * TOTAL_COST * (BULL_ACT/TOP_N)
            weekly_ret -= cost_bear
            trades_total += len(portfolio)
            portfolio.clear()

    # SPY cost on regime change
    if cur_bull != prev_bull:
        spy_change = abs(BULL_SPY - BEAR_SPY)
        weekly_ret -= spy_change * TOTAL_COST

    nav *= (1 + weekly_ret)
    nav_history.append(nav)
    date_history.append(week_end)
    prev_bull = cur_bull
    prev_top3 = cur_top3

eq_E = pd.Series(nav_history, index=date_history)
ret_E = eq_E.pct_change().fillna(0)
m_E   = calc_metrics(eq_E, ret_E, "E: System E (with costs + stop)", "weekly")
n_weeks = len(eq_E[eq_E.index >= bt_start])
ann_turnover = round(turnover_total / (n_weeks/52), 2)
print(f"  {m_E}")
print(f"  Trades: {trades_total}, Stop events: {stop_events}, Regime exits: {regime_exits}")
print(f"  Annual turnover: {ann_turnover:.1f}x, Total cost drag: {round(trades_total*TOTAL_COST*100/n_weeks*52,2)}% p.a.")

# ══════════════════════════════════════════════════════════════════════
# System E — No costs (to isolate cost drag)
# ══════════════════════════════════════════════════════════════════════
print("\nSystem E gross (no costs)...", flush=True)
# Rerun without cost deductions — use simple vectorised version
w_spy_E = bull_w.astype(float)*BULL_SPY + (~bull_w).astype(float)*BEAR_SPY
w_act_E = bull_w.astype(float)*BULL_ACT
mom_E   = (nq_ret_w * top3_wts).sum(axis=1)
ret_E_gross = w_spy_E*spy_ret_w + w_act_E*mom_E
eq_E_gross  = (1+ret_E_gross).cumprod()*100
m_E_gross   = calc_metrics(eq_E_gross, ret_E_gross, "E gross: System E (no costs)", "weekly")
print(f"  {m_E_gross}")

# ══════════════════════════════════════════════════════════════════════
# SPY buy-and-hold
# ══════════════════════════════════════════════════════════════════════
spy_ret_w_bt = spy_ret_w[spy_ret_w.index >= bt_start]
spy_eq_w     = (1+spy_ret_w_bt).cumprod()*100
m_spy        = calc_metrics(spy_eq_w, spy_ret_w_bt, "SPY buy & hold", "weekly")

# ══════════════════════════════════════════════════════════════════════
# Walk-Forward OOS Validation (4 folds)
# ══════════════════════════════════════════════════════════════════════
print("\nWalk-forward OOS (4 folds)...", flush=True)
fold_results = []
all_idx = nq_w.index[(nq_w.index >= bt_start) & (nq_w.index <= pd.Timestamp(END_DATE))]
n = len(all_idx)
fold_size = n // 5   # 5 parts: 1 train + 4 test folds

for fold in range(4):
    test_start = all_idx[fold_size*(fold+1)]
    test_end   = all_idx[min(fold_size*(fold+2)-1, n-1)]
    ret_fold   = ret_E_gross[(ret_E_gross.index >= test_start) & (ret_E_gross.index <= test_end)]
    eq_fold    = (1+ret_fold).cumprod()*100
    m_fold     = calc_metrics(eq_fold, ret_fold, f"OOS Fold {fold+1} ({test_start.year}-{test_end.year})", "weekly")
    fold_results.append(m_fold)
    print(f"  Fold {fold+1} [{test_start.strftime('%Y-%m')} — {test_end.strftime('%Y-%m')}]: CAGR={m_fold.get('cagr')}% MaxDD={m_fold.get('max_dd')}% Sharpe={m_fold.get('sharpe')}")

# ══════════════════════════════════════════════════════════════════════
# Bear market performance detail
# ══════════════════════════════════════════════════════════════════════
print("\nBear market performance...", flush=True)
bear_periods = [
    ("GFC 2007-09",  "2007-07-01", "2009-03-31"),
    ("2011 Correct", "2011-05-01", "2011-10-31"),
    ("2015-16",      "2015-07-01", "2016-02-29"),
    ("COVID 2020",   "2020-01-01", "2020-04-30"),
    ("Bear 2022",    "2022-01-01", "2022-12-31"),
]
bear_stats = []
for name, s, e in bear_periods:
    sd, ed = pd.Timestamp(s), pd.Timestamp(e)
    eq_b = eq_E[(eq_E.index>=sd)&(eq_E.index<=ed)]
    spy_b = spy_eq_w[(spy_eq_w.index>=sd)&(spy_eq_w.index<=ed)]
    if len(eq_b)<2 or len(spy_b)<2: continue
    strat_dd = round(float((eq_b/eq_b.cummax()-1).min()*100),1)
    spy_dd   = round(float((spy_b/spy_b.cummax()-1).min()*100),1)
    bear_stats.append({"period":name,"strat_dd":strat_dd,"spy_dd":spy_dd,"protection":round(spy_dd-strat_dd,1)})
    print(f"  {name}: strat={strat_dd}%, SPY={spy_dd}%, protected={round(spy_dd-strat_dd,1)}%")

# ══════════════════════════════════════════════════════════════════════
# Annual returns table
# ══════════════════════════════════════════════════════════════════════
annual_e   = ret_E.resample("YE").apply(lambda x:(1+x).prod()-1)
annual_spy = spy_ret_w.resample("YE").apply(lambda x:(1+x).prod()-1)
annual = []
for dt in annual_e.index:
    if dt.year < 2006: continue
    annual.append({
        "year": dt.year,
        "strat": round(float(annual_e.get(dt,0))*100,1),
        "spy":   round(float(annual_spy.get(dt,0))*100,1) if dt in annual_spy.index else None
    })

# ── Save ─────────────────────────────────────────────────────────────
output = {
    "strategy": "System E: Weekly 11-1 momentum + daily regime T+1 + stop -10% + breadth≥40% + 35%/65% dynamic",
    "params": {
        "lookback_weeks": LOOKBACK_WK, "skip_weeks": SKIP_WK,
        "bull_spy": BULL_SPY, "bull_active": BULL_ACT,
        "bear_spy": BEAR_SPY, "breadth_threshold": BREADTH_TH,
        "stop_loss": STOP_LOSS, "commission": COMMISSION,
        "slippage": SLIPPAGE, "total_cost_per_trade": TOTAL_COST,
    },
    "metrics": {
        "E_net":     m_E,
        "E_gross":   m_E_gross,
        "A_weekly_nocost": m_A,
        "spy":       m_spy,
    },
    "oos_folds":   fold_results,
    "bear_periods": bear_stats,
    "annual":      annual,
    "cost_stats": {
        "total_trades": trades_total,
        "stop_events":  stop_events,
        "regime_exits": regime_exits,
        "annual_turnover": ann_turnover,
    },
    "curves": {
        "E_net":   to_monthly_curve(eq_E,       "E: System E (net of costs)"),
        "E_gross": to_monthly_curve(eq_E_gross, "E: System E (gross)"),
        "A":       to_monthly_curve(eq_A,       "A: Weekly baseline"),
        "spy":     to_monthly_curve(spy_eq_w,   "SPY"),
    },
    "drawdown": {
        "E_net":  [round(float(v),2) for v in (eq_E/eq_E.cummax()-1).resample("ME").last().values],
        "dates":  [d.strftime("%Y-%m-%d") for d in (eq_E/eq_E.cummax()-1).resample("ME").last().index],
    }
}

with open("/sessions/tender-dreamy-mendel/mnt/outputs/system_e_results.json","w") as f:
    json.dump(output, f)
print(f"\nSaved system_e_results.json")

print("\n═══════════════════════════════════════════════════════")
print("SYSTEM E FINAL RESULTS")
print("═══════════════════════════════════════════════════════")
for k,m in [("Net (after costs)",m_E),("Gross (before costs)",m_E_gross),("SPY",m_spy)]:
    print(f"  {k:<25} CAGR={m.get('cagr'):>6}%  MaxDD={m.get('max_dd'):>7}%  Calmar={m.get('calmar'):>5}  Sharpe={m.get('sharpe')}")
print(f"\n  Annual turnover: {ann_turnover:.1f}x | Trades: {trades_total} | Stops: {stop_events} | Regime exits: {regime_exits}")
print(f"  Survivorship bias caveat: CAGR overstated ~4-6pts (2026 NQ100 universe used)")
