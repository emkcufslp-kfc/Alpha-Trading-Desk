import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =====================================================================
# 1. ARCHITECTURAL LAYOUT & ACCOUNT SESSION INITIALIZATION
# =====================================================================
st.set_page_config(layout="wide", page_title="Alpha Trading Desk")

# ── Safe defaults for cross-tab state ────────────────────────────────
if "xs_list" not in st.session_state:
    st.session_state.xs_list = []

if "ledger" not in st.session_state:
    st.session_state.ledger = {
        "cash": 100_000.0,
        "positions": {},   # {ticker: quantity}
        "history": [],     # audit trail
    }

if "circuit_breaker" not in st.session_state:
    st.session_state.circuit_breaker = False

if "nav_peak" not in st.session_state:
    st.session_state.nav_peak = 100_000.0

if "cost_basis" not in st.session_state:
    # {ticker: {"total_qty": int, "total_cost": float}}
    st.session_state.cost_basis = {}

if "last_regime" not in st.session_state:
    st.session_state.last_regime = None

if "regime_since" not in st.session_state:
    st.session_state.regime_since = None


def log_transaction(action: str, ticker: str, qty: int, price: float) -> bool:
    cost = qty * price
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if action == "BUY":
        if st.session_state.ledger["cash"] >= cost:
            st.session_state.ledger["cash"] -= cost
            st.session_state.ledger["positions"][ticker] = (
                st.session_state.ledger["positions"].get(ticker, 0) + qty
            )
            st.session_state.ledger["history"].append({
                "Timestamp": timestamp, "Action": "BUY",
                "Ticker": ticker, "Qty": qty,
                "Price": price, "Total Value": cost,
            })
            # Update cost basis
            cb = st.session_state.cost_basis.setdefault(ticker, {"total_qty": 0, "total_cost": 0.0})
            cb["total_qty"]  += qty
            cb["total_cost"] += cost
            return True

    elif action == "SELL":
        current_qty = st.session_state.ledger["positions"].get(ticker, 0)
        if current_qty >= qty:
            st.session_state.ledger["cash"] += cost
            st.session_state.ledger["positions"][ticker] -= qty
            if st.session_state.ledger["positions"][ticker] == 0:
                del st.session_state.ledger["positions"][ticker]
            st.session_state.ledger["history"].append({
                "Timestamp": timestamp, "Action": "SELL",
                "Ticker": ticker, "Qty": qty,
                "Price": price, "Total Value": cost,
            })
            # Reduce cost basis proportionally
            cb = st.session_state.cost_basis.get(ticker)
            if cb and cb["total_qty"] > 0:
                frac = qty / cb["total_qty"]
                cb["total_cost"] -= cb["total_cost"] * frac
                cb["total_qty"]  -= qty
                if cb["total_qty"] <= 0:
                    st.session_state.cost_basis.pop(ticker, None)
            return True

    return False


# =====================================================================
# 4J. LIVE STRATEGY SIGNAL ENGINE
# =====================================================================
@st.cache_data(ttl=900)   # refresh every 15 min
def compute_live_signal(
    tickers: tuple,
    spy_ticker: str = "SPY",
    spy_floor: float = 0.35,      # System E: 35% SPY bull, 10% bear
    top_n: int = 3,
    ma_window: int = 200,
    breadth_threshold: float = 0.40,  # ≥40% stocks bullish
    stop_loss: float = -0.10,          # per-position stop -10%
) -> dict:
    """
    System E live signal:
      - Regime    : BULL / BEAR (NQ100 EW + SPY both vs 200d MA, T+1)
      - Breadth   : ≥40% of NQ100 with positive 11-1 weekly score
      - Top-N     : weekly 11-1 momentum leaders (44-week - 4-week skip)
      - Target    : Bull: 35% SPY + 21.7% each top-3 | Bear: 10% SPY + 90% cash
      - Stop-loss : -10% per position → exit T+1, replace with next-best
    Uses ~2.5 years daily history (MA warmup + 44-week lookback).
    """
    lookback_start = (datetime.date.today() - datetime.timedelta(days=900)).isoformat()
    today          = datetime.date.today().isoformat()

    all_tickers = list(tickers) + ([spy_ticker] if spy_ticker not in tickers else [])
    raw = yf.download(all_tickers, start=lookback_start, end=today, progress=False)
    if raw.empty:
        return {}

    if isinstance(raw.columns, pd.MultiIndex):
        px = (raw.xs("Adj Close", axis=1, level=0)
              if "Adj Close" in raw.columns.levels[0]
              else raw.xs("Close", axis=1, level=0))
    else:
        px = raw[["Adj Close"]] if "Adj Close" in raw.columns else raw[["Close"]]

    px = px.dropna(how="all")
    nq_px  = px[[c for c in px.columns if c in tickers]]
    spy_px = px[spy_ticker] if spy_ticker in px.columns else None

    if nq_px.empty or len(nq_px) < ma_window:
        return {}

    # ── Regime ──────────────────────────────────────────────────────
    eq_idx  = nq_px.mean(axis=1)
    ma      = eq_idx.rolling(ma_window, min_periods=ma_window // 2).mean()
    current_eq  = float(eq_idx.iloc[-1])
    current_ma  = float(ma.iloc[-1])
    ma_margin   = (current_eq / current_ma - 1) * 100
    regime      = "BULL" if current_eq > current_ma else "BEAR"

    # ── SPY own MA regime ────────────────────────────────────────────
    spy_bull = False
    if spy_px is not None and len(spy_px) >= ma_window:
        spy_ma = spy_px.rolling(ma_window, min_periods=ma_window // 2).mean()
        spy_bull = float(spy_px.iloc[-1]) > float(spy_ma.iloc[-1])

    # Dual regime: both NQ100 EW AND SPY must be above 200d MA
    dual_bull = (regime == "BULL") and spy_bull

    # ── Weekly 11-1 momentum (System E: 44-week - 4-week skip) ───────
    wp = nq_px.resample("W-FRI").last()
    LOOKBACK_WK, SKIP_WK = 44, 4
    breadth = 0.0
    breadth_ok = False
    if len(wp) < LOOKBACK_WK + 2:
        top_tickers = []
        mom_table   = pd.Series(dtype=float)
    else:
        mom_lb  = wp.pct_change(periods=LOOKBACK_WK).iloc[-1]
        mom_sk  = wp.pct_change(periods=SKIP_WK).iloc[-1]
        mom_11_1 = (mom_lb - mom_sk).dropna().sort_values(ascending=False)
        # Breadth filter
        breadth = (mom_11_1 > 0).sum() / len(mom_11_1)
        breadth_ok = breadth >= breadth_threshold
        if dual_bull and breadth_ok:
            top_tickers = mom_11_1.head(top_n).index.tolist()
        else:
            top_tickers = []
        mom_table = mom_11_1

    # ── Latest prices ────────────────────────────────────────────────
    latest_prices = {t: float(nq_px[t].dropna().iloc[-1])
                     for t in nq_px.columns if t in nq_px.columns}
    if spy_px is not None:
        latest_prices[spy_ticker] = float(spy_px.dropna().iloc[-1])

    # ── Target weights (System E) ─────────────────────────────────────
    # Bull: 35% SPY + 65% active (21.7% each of top-3)
    # Bear: 10% SPY + 90% cash
    BULL_SPY, BULL_ACT = 0.35, 0.65
    BEAR_SPY            = 0.10
    bull_active = dual_bull and bool(top_tickers)
    if bull_active:
        per_stock = BULL_ACT / top_n
        target_weights = {spy_ticker: BULL_SPY}
        for t in top_tickers:
            target_weights[t] = target_weights.get(t, 0) + per_stock
    else:
        target_weights = {spy_ticker: BEAR_SPY}   # rest is cash

    # ── Recent drawdown of EW index ──────────────────────────────────
    peak   = float(eq_idx.cummax().iloc[-1])
    idx_dd = (current_eq / peak - 1) * 100

    # ── Days until next Friday (weekly rebalance) ─────────────────────
    today_dt = datetime.date.today()
    days_to_friday = (4 - today_dt.weekday()) % 7  # 4 = Friday
    if days_to_friday == 0: days_to_friday = 7       # already Friday → next Friday

    return {
        "regime":            "BULL" if bull_active else "BEAR",
        "dual_bull":         dual_bull,
        "spy_bull":          spy_bull,
        "breadth":           round(float(breadth) if top_tickers or len(mom_table)>0 else 0, 3),
        "breadth_ok":        dual_bull and (float(breadth) >= breadth_threshold if len(mom_table)>0 else False),
        "ma_margin":         round(ma_margin, 2),
        "eq_idx":            round(current_eq, 2),
        "ma_200":            round(current_ma, 2),
        "top_tickers":       top_tickers,
        "mom_table":         mom_table,
        "target_weights":    target_weights,
        "latest_prices":     latest_prices,
        "idx_dd":            round(idx_dd, 2),
        "days_to_rebalance": days_to_friday,
        "as_of":             str(nq_px.index[-1])[:10],
        "spy_ticker":        spy_ticker,
        "stop_loss":         stop_loss,
        "top_n":             top_n,
        "ma_window":         ma_window,
    }


# =====================================================================
# 2. DATA INGESTION & SANITIZATION LAYER
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_universe_clean(tickers: tuple, start, end) -> pd.DataFrame:
    """
    Multi-Index resilient download engine.
    Accepts tickers as a hashable tuple for st.cache_data key stability.
    Decouples Adj Close / Close across all yfinance MultiIndex shapes.
    """
    df = yf.download(list(tickers), start=start, end=end, progress=False)
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        price_df = (
            df.xs("Adj Close", axis=1, level=0)
            if "Adj Close" in df.columns.levels[0]
            else df.xs("Close", axis=1, level=0)
        )
    else:
        price_df = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

    if isinstance(price_df, pd.Series):
        price_df = price_df.to_frame()

    return price_df.dropna(how="all")


# =====================================================================
# 3. CORE STATISTICAL METRICS COMPILER
# =====================================================================
def compile_performance_metrics(
    equity_curve: pd.Series, daily_returns: pd.Series
) -> dict:
    """
    Risk-adjusted return analytics. Annualization constant: 252 trading days.

    Metrics:
      Total Return : (NAV_T / NAV_0) - 1
      CAGR         : NAV_T^(252/N) - 1
      Ann. Vol     : σ_daily × √252
      Sharpe       : (μ_daily / σ_daily) × √252
      Sortino      : (μ_daily × 252) / (σ_downside × √252)
                     where σ_downside uses only negative return observations
      Max Drawdown : min[(NAV_t - peak_t) / peak_t]
      Calmar       : CAGR / |Max Drawdown|
    """
    ANN = 252
    _empty = {
        "Total Return": "0.00%", "CAGR": "0.00%", "Ann. Vol": "0.00%",
        "Sharpe": "0.00", "Sortino": "0.00",
        "Max Drawdown": "0.00%", "Calmar": "0.00",
    }
    if len(daily_returns) == 0 or (daily_returns.std() == 0 or pd.isna(daily_returns.std())):
        return _empty

    total_ret = (equity_curve.iloc[-1] - 1) * 100
    n_days = len(equity_curve)
    cagr = (
        ((equity_curve.iloc[-1]) ** (ANN / n_days) - 1) * 100
        if equity_curve.iloc[-1] > 0 else -100.0
    )
    ann_vol = daily_returns.std() * np.sqrt(ANN) * 100
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(ANN)

    # Sortino: semi-deviation denominator — downside observations only
    neg_rets = daily_returns[daily_returns < 0]
    downside_std = neg_rets.std() * np.sqrt(ANN) if len(neg_rets) > 1 else np.nan
    sortino = (
        (daily_returns.mean() * ANN) / downside_std
        if (downside_std and downside_std > 0) else 0.0
    )

    running_max = equity_curve.cummax()
    max_dd = ((equity_curve - running_max) / running_max).min() * 100
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "Total Return": f"{total_ret:.2f}%",
        "CAGR": f"{cagr:.2f}%",
        "Ann. Vol": f"{ann_vol:.2f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd:.2f}%",
        "Calmar": f"{calmar:.2f}",
    }


# =====================================================================
# 4. ALPHA ENGINE COMPONENTS
# =====================================================================

# ─── 4A. Cross-Sectional Winsorizer ───────────────────────────────────
def _winsorize_panel(
    df: pd.DataFrame, q_lo: float = 0.025, q_hi: float = 0.975
) -> pd.DataFrame:
    """
    Vectorized row-wise (cross-sectional) winsorization.

    At each date t, clips each ticker's observation to the [q_lo, q_hi]
    quantile interval computed across the live cross-section of tickers.
    Prevents a single outlier ticker from dominating Z-score standardization.

    Implementation:
      lo_t = P_{q_lo}(X_{i,t}) over i       [Series, indexed by date]
      hi_t = P_{q_hi}(X_{i,t}) over i
      X̂_{i,t} = clip(X_{i,t}, lo_t, hi_t)  [axis=0 aligns on date index]
    """
    lo = df.quantile(q_lo, axis=1)
    hi = df.quantile(q_hi, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


# ─── 4B. Cross-Sectional Factor Engine ───────────────────────────────
def run_cross_sectional_pipeline(
    price_df: pd.DataFrame, lookback: int, top_n: int
) -> pd.DataFrame:
    """
    Cross-Sectional Relative Strength with Z-Score Standardization.

    Pipeline:
      1. Raw momentum panel  : M_{i,t} = P_{i,t} / P_{i,t-k} - 1
      2. Winsorize           : clip cross-section at [2.5%, 97.5%]
      3. Z-score             : Z_{i,t} = (M_{i,t} - μ_t) / σ_t   (row-wise)
      4. Rank Z-scores; long top-N, short bottom-N
      5. Equal-weight legs   : 0.5 long + 0.5 short → market-neutral

    Shift(1) on returns_df implicitly applies because pct_change() at date t
    represents the return earned *from* t-1 *to* t, while momentum signal
    ranked at t-1 drives the position taken at open of t.
    """
    returns_df = price_df.pct_change()
    momentum = price_df.pct_change(periods=lookback)

    mom_wins = _winsorize_panel(momentum)

    # Row-wise (cross-sectional) Z-score standardization
    row_mean = mom_wins.mean(axis=1)
    row_std = mom_wins.std(axis=1).replace(0.0, np.nan)
    z_scores = mom_wins.subtract(row_mean, axis=0).divide(row_std, axis=0)

    # Shift rank signal forward by 1 bar: rank computed at close of t
    # drives the position held from t to t+1, earning returns at t+1.
    # Without this shift, today's large-return tickers simultaneously
    # rank highest AND contribute that return — a classic lookahead bias.
    ranked = z_scores.rank(axis=1, ascending=False).shift(1)
    n_assets = len(price_df.columns)

    long_mask  = ranked <= top_n
    short_mask = ranked > (n_assets - top_n)

    strat_rets = (
        returns_df[long_mask].mean(axis=1) * 0.5
        - returns_df[short_mask].mean(axis=1) * 0.5
    )
    bench_rets = returns_df.mean(axis=1)

    return pd.DataFrame({
        "Strategy": (1 + strat_rets.fillna(0)).cumprod(),
        "Benchmark": (1 + bench_rets.fillna(0)).cumprod(),
        "Daily_Return": strat_rets.fillna(0),
    }, index=price_df.index)


# ─── 4C. CCF Significance Engine ─────────────────────────────────────
def compute_ccf_with_significance(
    cust_ret: pd.Series,
    supp_ret: pd.Series,
    max_lag: int = 5,
) -> dict:
    """
    Cross-Correlation Function with Bartlett Significance Bands.

    ρ(k) = Corr(supp_t, cust_{t+k})
           = sample correlation of supplier returns against
             customer returns shifted forward by k days.

    A positive k means customer returns at time (t+k) correlate with
    supplier returns at time t — i.e., customer movements LEAD the
    supplier by k days, which is the actionable hypothesis.

    Bartlett 95% confidence band (white-noise null):
      CI = ±2 / √N
    where N = number of non-null observations.

    Edge verdict:
      has_edge = True iff peak_lag > 0  AND  |ρ(peak_lag)| > CI
    """
    lags = list(range(-max_lag, max_lag + 1))
    corrs = [float(supp_ret.corr(cust_ret.shift(k))) for k in lags]
    n = int(cust_ret.dropna().shape[0])
    sig_band = 2.0 / np.sqrt(max(n, 1))

    abs_corrs = np.nan_to_num(np.abs(corrs))
    peak_idx = int(np.argmax(abs_corrs))
    peak_lag = lags[peak_idx]
    peak_corr = corrs[peak_idx]
    has_edge = (peak_lag > 0) and (abs(peak_corr) > sig_band)

    return {
        "lags": lags,
        "corrs": corrs,
        "sig_band": sig_band,
        "peak_lag": peak_lag,
        "peak_corr": peak_corr,
        "has_edge": has_edge,
    }


# ─── 4D. Lead-Lag Signal Engine ───────────────────────────────────────
def run_lead_lag_pipeline(
    price_df: pd.DataFrame,
    customer: str,
    supplier: str,
    lookback: int,
    threshold: float,
    signal_lag: int = 1,
) -> tuple:
    """
    Supply-Chain Lead-Lag Signal Generation.

    Signal mechanics:
      cust_mom_t  = P_cust,t / P_cust,t-k − 1       (rolling k-day momentum)
      signal_t    = 1  if cust_mom_{t − signal_lag} > θ,  else 0
      strat_ret_t = signal_t × supp_ret_t

    signal_lag ≥ 1 is the strict lookahead-bias guard:
    the momentum condition observed at bar (t - signal_lag) determines
    whether we hold the supplier at bar t. This ensures no future price
    information contaminates the signal vector.
    """
    if customer not in price_df.columns or supplier not in price_df.columns:
        return pd.DataFrame(), {}

    cust_ret = price_df[customer].pct_change()
    supp_ret = price_df[supplier].pct_change()
    cust_mom = price_df[customer].pct_change(periods=lookback)

    signal = pd.Series(
        np.where(cust_mom.shift(signal_lag) > threshold, 1, 0),
        index=price_df.index,
    )
    strat_ret = signal * supp_ret

    df_out = pd.DataFrame({
        "cust_ret": cust_ret,
        "supp_ret": supp_ret,
        "signal": signal,
        "Strategy": (1 + strat_ret.fillna(0)).cumprod(),
        "Benchmark": (1 + supp_ret.fillna(0)).cumprod(),
        "Daily_Return": strat_ret.fillna(0),
    }, index=price_df.index)

    ccf_data = compute_ccf_with_significance(cust_ret, supp_ret)

    return df_out, ccf_data


# ─── 4E. Quarterly Top-10 Selection Engine ───────────────────────────
def run_quarterly_top10_pipeline(
    price_df: pd.DataFrame, top_n: int = 10, ma_window: int = 200
) -> tuple:
    """
    Quarterly Momentum Selection — Long-Only Portfolio with MA Regime Filter.

    Mechanics:
      1. Resample prices to quarter-end: P_q = price at last trading day of quarter q
      2. Prior-quarter return: R_{i,q} = P_{i,q} / P_{i,q-1} − 1
      3. Cross-sectional rank at q; shift(1) so rank at q drives holdings for q+1
         (no lookahead: we only know q's return AFTER q ends)
      4. Top-N weight: w_{i,q+1} = 1/N if rank_{i,q} ≤ N, else 0
      5. Reindex weights to daily frequency via forward-fill
      6. Shift daily weights by 1 trading day (position effective next open)
      7. MA Regime Filter: zero all weights when equal-weight NQ100 < its ma_window-day MA
         (signal computed at t-1 to avoid lookahead)
      8. Strategy return: Σ_i  w_{i,t-1} × regime_{t-1} × r_{i,t}

    Benchmark: equal-weight NQ100 daily return.

    Returns:
      result_df  : DataFrame with Strategy / Benchmark / Daily_Return
      q_holdings : DataFrame — each row is a quarter, columns are top-N tickers
    """
    daily_rets = price_df.pct_change()

    # Quarter-end prices → prior-quarter return → rank
    q_prices = price_df.resample("QE").last()
    q_ret    = q_prices.pct_change()
    q_rank   = q_ret.rank(axis=1, ascending=False).shift(1)  # lookahead guard

    # Equal-weight within top-N
    q_weights = (q_rank <= top_n).astype(float).div(top_n)

    # Forward-fill to daily, then shift 1 trading day for execution lag
    daily_weights = (
        q_weights
        .reindex(price_df.index, method="ffill")
        .fillna(0.0)
        .shift(1)
        .fillna(0.0)
    )

    # MA Regime Filter: go to cash when NQ100 EW index is below its rolling MA
    eq_index = price_df.mean(axis=1)
    ma = eq_index.rolling(ma_window, min_periods=ma_window // 2).mean()
    regime = (eq_index > ma).astype(float).shift(1).fillna(0.0)
    daily_weights = daily_weights.multiply(regime, axis=0)

    strat_rets = (daily_rets * daily_weights).sum(axis=1)
    bench_rets  = daily_rets.mean(axis=1)

    result_df = pd.DataFrame({
        "Strategy":    (1 + strat_rets.fillna(0)).cumprod(),
        "Benchmark":   (1 + bench_rets.fillna(0)).cumprod(),
        "Daily_Return": strat_rets.fillna(0),
    }, index=price_df.index)

    # Build quarterly holdings table: which top-N stocks each quarter
    holdings_rows = []
    for qdate, row in q_rank.iterrows():
        valid = row.dropna()
        top_tickers = valid[valid <= top_n].sort_values().index.tolist()
        holdings_rows.append({
            "Quarter": str(qdate)[:10],
            "Holdings": ", ".join(top_tickers),
            "Count": len(top_tickers),
        })
    q_holdings = pd.DataFrame(holdings_rows).dropna()

    return result_df, q_holdings


# ─── 4F-alt. Quarterly Composite Lead-Lag Engine ─────────────────────
def run_quarterly_composite_leadlag(
    nq_prices: pd.DataFrame,
    supp_prices: pd.Series,
    supplier: str,
    top_n: int = 10,
    lookback: int = 3,
    threshold: float = 0.01,
    signal_lag: int = 1,
    ma_window: int = 200,
) -> tuple:
    """
    Dynamic Lead-Lag using a quarterly-refreshed NQ100 top-10 basket
    as the composite customer signal.

    Instead of fixing a single customer (e.g. NVDA), we:
      1. Each quarter, identify the top-10 NQ100 stocks by prior-quarter return
      2. Compute the equal-weight average momentum of that basket each day
      3. Use basket_momentum_{t − signal_lag} > θ as the entry signal into the supplier

    This avoids single-name bias and leverages the broadest demand signal
    from the dominant NQ100 companies as a forward indicator for the supplier.

    CCF is computed between the composite basket return and supplier return.
    """
    # Quarterly top-10 mask (reuse same logic as selection pipeline)
    q_prices = nq_prices.resample("QE").last()
    q_rank   = q_prices.pct_change().rank(axis=1, ascending=False).shift(1)
    q_top_mask = (q_rank <= top_n).reindex(nq_prices.index, method="ffill").fillna(False)

    # Daily composite basket momentum: mean momentum of current-quarter top-10
    basket_mom = nq_prices.pct_change(periods=lookback).where(q_top_mask)
    composite_mom = basket_mom.mean(axis=1)

    # Composite basket return (for CCF)
    basket_ret = nq_prices.pct_change().where(q_top_mask).mean(axis=1)

    supp_ret = supp_prices.pct_change()

    # MA Regime Filter applied to lead-lag signal
    eq_index = nq_prices.mean(axis=1)
    ma = eq_index.rolling(ma_window, min_periods=ma_window // 2).mean()
    regime = (eq_index > ma).astype(float).shift(1).fillna(0.0)

    signal = pd.Series(
        np.where((composite_mom.shift(signal_lag) > threshold) & (regime > 0), 1, 0),
        index=nq_prices.index,
    )
    strat_ret = signal * supp_ret

    df_out = pd.DataFrame({
        "basket_ret":  basket_ret,
        "supp_ret":    supp_ret,
        "composite_mom": composite_mom,
        "signal":      signal,
        "Strategy":    (1 + strat_ret.fillna(0)).cumprod(),
        "Benchmark":   (1 + supp_ret.fillna(0)).cumprod(),
        "Daily_Return": strat_ret.fillna(0),
    }, index=nq_prices.index)

    ccf_data = compute_ccf_with_significance(basket_ret, supp_ret)

    return df_out, ccf_data


# ─── 4G (prev 4E). Parameter Optimization Grid ───────────────────────
@st.cache_data(ttl=3600)
def run_optimization_grid(
    price_df: pd.DataFrame,
    customer: str,
    supplier: str,
) -> pd.DataFrame:
    """
    Exhaustive grid search over the (lookback, threshold) parameter space.
    Objective function: Annualized Sharpe Ratio.

    Grid:
      lookback  ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10}  (9 values)
      threshold ∈ {0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.      threshold in {0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%}  (6 values)
      Total: 54 backtests
    """
    lookbacks = range(2, 11)
    thresholds = [round(x * 0.005, 3) for x in range(1, 7)]
    records = []
    for lb in lookbacks:
        row = []
        for th in thresholds:
            df_r, _ = run_lead_lag_pipeline(price_df, customer, supplier, lb, th)
            if not df_r.empty and df_r["Daily_Return"].std() > 0:
                s = (df_r["Daily_Return"].mean() / df_r["Daily_Return"].std()) * np.sqrt(252)
            else:
                s = 0.0
            row.append(round(s, 3))
        records.append(row)
    return pd.DataFrame(records, index=list(lookbacks),
                        columns=[f"{t * 100:.1f}%" for t in thresholds])


# --- 4H. Walk-Forward OOS Validation ------------------------------------
def run_walk_forward(price_df, customer, supplier, n_splits=4):
    n = len(price_df)
    test_size = n // (n_splits + 1)
    mini_lookbacks = [3, 5, 7, 10]
    mini_thresholds = [0.010, 0.015, 0.020]
    oos_segments = []
    for i in range(n_splits):
        train_end = test_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + test_size, n)
        if (test_end - test_start) < 20:
            continue
        train_prices = price_df.iloc[:train_end]
        test_prices  = price_df.iloc[test_start:test_end]
        best_sharpe, best_lb, best_th = -np.inf, 5, 0.01
        for lb in mini_lookbacks:
            for th in mini_thresholds:
                df_tr, _ = run_lead_lag_pipeline(train_prices, customer, supplier, lb, th)
                if not df_tr.empty and df_tr["Daily_Return"].std() > 0:
                    s = (df_tr["Daily_Return"].mean() / df_tr["Daily_Return"].std()) * np.sqrt(252)
                    if s > best_sharpe:
                        best_sharpe, best_lb, best_th = s, lb, th
        df_test, _ = run_lead_lag_pipeline(test_prices, customer, supplier, best_lb, best_th)
        if not df_test.empty:
            oos_segments.append(df_test[["Strategy", "Benchmark", "Daily_Return"]].copy())
    if not oos_segments:
        return pd.DataFrame()
    chained_parts = []
    nav_scale = 1.0
    for seg in oos_segments:
        scaled = seg.copy()
        scaled["Strategy"] = seg["Strategy"] * nav_scale
        scaled["Benchmark"] = seg["Benchmark"] * nav_scale
        chained_parts.append(scaled)
        nav_scale = float(scaled["Strategy"].iloc[-1])
    return pd.concat(chained_parts)


# --- 4I. Kelly Fraction Sizing ------------------------------------------
def compute_kelly_fraction(daily_returns, half_kelly=True):
    wins   = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    W = len(wins) / max(len(daily_returns), 1)
    B = wins.mean() / abs(losses.mean())
    full_kelly = W - (1.0 - W) / B
    return max(0.0, round(full_kelly * (0.5 if half_kelly else 1.0), 4))


# =====================================================================
# 5. STREAMLIT UI FRAMEWORK
# =====================================================================
st.title("Alpha Engine Architecture & Backtest Desk")
st.markdown("---")

st.sidebar.header("Global Matrix Boundary")
start_input = st.sidebar.date_input("Start Boundary", datetime.date(2023, 1, 1))
end_input   = st.sidebar.date_input("End Boundary",   datetime.date.today())
st.sidebar.markdown("---")
st.sidebar.header("Risk Management")
dd_threshold = st.sidebar.slider("Drawdown Circuit Breaker (%)", 5, 40, 15, key="dd_cb") / 100.0

tab_cross, tab_supply, tab_ledger, tab_execution, tab_action = st.tabs([
    "Cross-Sectional Factor Matrix",
    "Hybrid Momentum Strategy",
    "Portfolio Account Ledger",
    "Execution Control Center",
    "⚡ Action Panel",
])

# ─── TAB 1: CROSS-SECTIONAL ──────────────────────────────────────────
with tab_cross:
    st.subheader("Cross-Sectional Z-Score Factor — Daily Rebalance")
    c_col1, c_col2 = st.columns([1, 3])
    with c_col1:
        _NQ100_DEFAULT = (
            "AAPL,ABNB,ACGL,ADBE,ADI,ADP,ADSK,ALGN,AMAT,AMD,"
            "AMGN,AMZN,ASML,AVGO,AZN,BIIB,BKNG,BKR,CDNS,CEG,"
            "CHTR,CINF,CMCSA,COST,CPRT,CRWD,CSCO,CSGP,CSX,CTAS,"
            "DASH,DDOG,DLTR,DXCM,EA,EBAY,ENPH,EXC,FANG,FAST,"
            "FTNT,GEHC,GFS,GILD,GOOG,GOOGL,HON,IDXX,ILMN,INTC,"
            "INTU,ISRG,KDP,KLAC,LRCX,LULU,MAR,MCHP,MDLZ,MELI,"
            "META,MNST,MRNA,MRVL,MSFT,MU,NFLX,NVDA,NXPI,ODFL,"
            "ON,ORLY,PANW,PAYX,PCAR,PDD,PEP,PYPL,QCOM,REGN,"
            "ROP,ROST,SBUX,SIRI,SNPS,TEAM,TMUS,TSLA,TTD,TTWO,"
            "TXN,VRSK,VRTX,WBD,WDAY,XEL,ZM,ZS"
        )
        xs_tickers_raw = st.text_area("Universe (NASDAQ-100)", _NQ100_DEFAULT, height=120)
        xs_list = [t.strip().upper() for t in xs_tickers_raw.split(",") if t.strip()]
        st.session_state.xs_list = xs_list
        xs_lookback = st.slider("Momentum Lookback (Bars)", 5, 60, 20, key="xs_lb")
        xs_top_n    = st.slider("Long/Short Tail N", 1, 10, 5, key="xs_tn")
    with c_col2:
        if xs_list:
            xs_prices = fetch_universe_clean(tuple(xs_list), start_input, end_input)
            if not xs_prices.empty:
                xs_results = run_cross_sectional_pipeline(xs_prices, xs_lookback, xs_top_n)
                xs_metrics = compile_performance_metrics(xs_results["Strategy"], xs_results["Daily_Return"])
                m_cols = st.columns(len(xs_metrics))
                for col, (k, v) in zip(m_cols, xs_metrics.items()):
                    col.metric(k, v)
                fig_xs = go.Figure()
                fig_xs.add_trace(go.Scatter(x=xs_results.index, y=xs_results["Strategy"],
                    name="Z-Score Factor Alpha", line=dict(color="#00FF66", width=2)))
                fig_xs.add_trace(go.Scatter(x=xs_results.index, y=xs_results["Benchmark"],
                    name="Equal-Weight Benchmark", line=dict(color="#FF3333", width=1.5, dash="dash")))
                fig_xs.update_layout(template="plotly_dark",
                    title="Cross-Sectional Growth Profile (Daily Z-Score Ranking)", height=450)
                st.plotly_chart(fig_xs, use_container_width=True)
            else:
                st.warning("No price data returned.")
        else:
            xs_list = []
            xs_prices = pd.DataFrame()

# ─── TAB 2: HYBRID MOMENTUM STRATEGY ─────────────────────────────────
with tab_supply:
    st.subheader("Hybrid Strategy: SPY Floor + NQ100 Monthly 12-1 Momentum (Legacy Comparison)")
    st.caption(
        "Legacy monthly 12-1 system for comparison. System E (Action Panel) upgrades this to weekly 11-1 + dual regime + stop-loss. "
        "Active sleeve rotates monthly into top-N NQ100 momentum stocks (bull) or cash (bear)."
    )
    h_col1, h_col2 = st.columns([1, 3])

    with h_col1:
        spy_floor_pct = st.slider("SPY Floor (%)", 10, 80, 35, step=5, key="spy_floor") / 100.0
        h_top_n       = st.slider("Active Top-N (NQ100)", 1, 10, 3, key="h_topn")
        h_ma_window   = st.slider("MA Regime Window (Days)", 50, 300, 200, step=25, key="h_ma")
        active_pct    = 1.0 - spy_floor_pct
        st.info(
            f"**Allocation:**\n"
            f"- {spy_floor_pct:.0%} SPY (always)\n"
            f"- {active_pct:.0%} Active sleeve\n"
            f"  - Bull → Top-{h_top_n} NQ100\n"
            f"  - Bear → Cash"
        )
        spy_ticker_input = st.text_input("SPY Proxy Ticker", "SPY").strip().upper()

    with h_col2:
        if "xs_prices" in locals() and not xs_prices.empty:
            spy_data = fetch_universe_clean(tuple([spy_ticker_input]), start_input, end_input)

            if not spy_data.empty and spy_ticker_input in spy_data.columns:
                spy_series = spy_data[spy_ticker_input]

                # ── Run hybrid strategy ─────────────────────────────
                dr_nq = xs_prices.pct_change()
                dr_spy = spy_series.pct_change().reindex(xs_prices.index).fillna(0)

                mp = xs_prices.resample("ME").last()
                mom_12_1 = mp.pct_change(periods=12) - mp.pct_change(periods=1)
                rank_m = mom_12_1.rank(axis=1, ascending=False).shift(1)
                w_m = (rank_m <= h_top_n).astype(float).div(h_top_n)
                dw_m = (w_m.reindex(xs_prices.index, method="ffill")
                         .fillna(0.0).shift(1).fillna(0.0))

                eq_idx = xs_prices.mean(axis=1)
                ma = eq_idx.rolling(h_ma_window, min_periods=h_ma_window // 2).mean()
                regime = (eq_idx > ma).astype(float).shift(1).fillna(0.0)

                mom_ret = (dr_nq * dw_m).sum(axis=1)
                active_ret = regime * mom_ret          # bear → 0 (cash)
                total_ret = spy_floor_pct * dr_spy + active_pct * active_ret

                hybrid_eq = (1 + total_ret.fillna(0)).cumprod()
                spy_eq_c  = (1 + dr_spy.fillna(0)).cumprod()

                hybrid_m = compile_performance_metrics(hybrid_eq, total_ret)
                spy_m_ui = compile_performance_metrics(spy_eq_c, dr_spy)

                # vs-SPY callouts
                h_cagr = float(hybrid_m["CAGR"].rstrip("%"))
                s_cagr = float(spy_m_ui["CAGR"].rstrip("%"))
                h_mdd  = float(hybrid_m["Max Drawdown"].rstrip("%"))
                s_mdd  = float(spy_m_ui["Max Drawdown"].rstrip("%"))
                c1, c2, c3 = st.columns(3)
                c1.metric("CAGR vs SPY", hybrid_m["CAGR"],
                          delta=f"{h_cagr - s_cagr:+.1f}%")
                c2.metric("MaxDD vs SPY", hybrid_m["Max Drawdown"],
                          delta=f"{h_mdd - s_mdd:+.1f}%",
                          delta_color="inverse")
                c3.metric("Sharpe vs SPY", hybrid_m["Sharpe"],
                          delta=f"{float(hybrid_m['Sharpe']) - float(spy_m_ui['Sharpe']):+.2f}")

                m_cols_h = st.columns(len(hybrid_m))
                for col, (k, v) in zip(m_cols_h, hybrid_m.items()):
                    col.metric(k, v)

                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(x=hybrid_eq.index, y=hybrid_eq,
                    name="Hybrid Strategy", line=dict(color="#00E5FF", width=2.5)))
                fig_h.add_trace(go.Scatter(x=spy_eq_c.index, y=spy_eq_c,
                    name=spy_ticker_input, line=dict(color="#FFD700", width=1.8, dash="dash")))
                # Shade bear periods
                bear_start = None
                for _i, (dt, r) in enumerate(regime.items()):
                    if r < 0.5 and bear_start is None:
                        bear_start = dt
                    elif r >= 0.5 and bear_start is not None:
                        fig_h.add_vrect(x0=bear_start, x1=dt,
                            fillcolor="rgba(255,51,51,0.07)", line_width=0)
                        bear_start = None

                fig_h.update_layout(
                    template="plotly_dark",
                    title="Hybrid Momentum — SPY Floor + Monthly NQ100 Active Sleeve",
                    height=420,
                    legend=dict(orientation="h", y=1.02),
                )
                st.plotly_chart(fig_h, use_container_width=True)

                # Quarterly holdings table
                st.markdown("##### 📋 Monthly Momentum Holdings")
                qh_rows = []
                for qd, row in rank_m.iterrows():
                    valid = row.dropna()
                    selected = valid[valid <= h_top_n].sort_values().index.tolist()
                    if selected:
                        qh_rows.append({"Date": str(qd)[:10], "Holdings": ", ".join(selected)})
                if qh_rows:
                    st.dataframe(pd.DataFrame(qh_rows).tail(12), use_container_width=True, hide_index=True)
            else:
                st.warning("SPY price data unavailable.")
        else:
            st.info("Load the NQ100 universe in the Cross-Sectional tab first.")

# ─── TAB 3: PORTFOLIO LEDGER ──────────────────────────────────────────
with tab_ledger:
    st.subheader("📒 Portfolio Account Ledger")
    st.caption("Track your paper-trading account. Transactions executed via the Action Panel are logged here.")

    l1, l2, l3 = st.columns(3)
    _lnav = st.session_state.ledger["cash"] + sum(
        st.session_state.ledger["positions"].get(t, 0) * 0
        for t in st.session_state.ledger["positions"]
    )
    l1.metric("Cash Balance", f"${st.session_state.ledger['cash']:,.0f}")
    l2.metric("Open Positions", len(st.session_state.ledger["positions"]))
    l3.metric("Transactions", len(st.session_state.ledger["history"]))

    st.markdown("##### Open Positions")
    if st.session_state.ledger["positions"]:
        pos_rows = [{"Ticker": t, "Qty": q} for t, q in st.session_state.ledger["positions"].items() if q > 0]
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

    st.markdown("##### Transaction History")
    if st.session_state.ledger["history"]:
        hist_df = pd.DataFrame(st.session_state.ledger["history"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No transactions recorded yet. Use the Action Panel to execute trades.")

    if st.button("🔄 Reset Account to $100,000 Cash"):
        st.session_state.ledger = {"cash": 100_000.0, "positions": {}, "history": []}
        st.session_state.nav_peak = 100_000.0
        st.session_state.cost_basis = {}
        st.success("Account reset.")
        st.rerun()

# ─── TAB 4: EXECUTION CONTROL CENTER ─────────────────────────────────
with tab_execution:
    st.subheader("🎛️ Execution Control Center")
    st.caption("Manual trade entry and account management. For strategy-driven signals, use the ⚡ Action Panel.")

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("##### Manual Trade Entry")
        mt_ticker = st.text_input("Ticker", key="mt_tkr").strip().upper()
        mt_action = st.radio("Action", ["BUY", "SELL"], horizontal=True, key="mt_act")
        mt_qty    = st.number_input("Quantity (shares)", min_value=1, value=1, key="mt_qty")
        mt_price  = st.number_input("Price per share ($)", min_value=0.01, value=100.0, step=0.01, key="mt_px")
        if st.button("Submit Trade", key="mt_submit", use_container_width=True):
            if mt_ticker:
                ok = log_transaction(mt_action, mt_ticker, int(mt_qty), float(mt_price))
                if ok:
                    st.success(f"{mt_action} {mt_qty}x {mt_ticker} @ ${mt_price:.2f} logged.")
                    st.rerun()
                else:
                    st.error("Trade rejected — insufficient cash or shares.")
            else:
                st.warning("Enter a ticker symbol.")

    with ec2:
        st.markdown("##### Cost Basis Tracker")
        if st.session_state.cost_basis:
            cb_rows = []
            for t, cb in st.session_state.cost_basis.items():
                if cb["total_qty"] > 0:
                    avg = cb["total_cost"] / cb["total_qty"]
                    cb_rows.append({"Ticker": t, "Qty": cb["total_qty"],
                                    "Avg Cost": f"${avg:.2f}",
                                    "Total Cost": f"${cb['total_cost']:,.0f}"})
            if cb_rows:
                st.dataframe(pd.DataFrame(cb_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No cost basis tracked yet.")

        st.markdown("##### Circuit Breaker Status")
        cb_status = "🔴 ACTIVE" if st.session_state.circuit_breaker else "🟢 Clear"
        st.metric("Circuit Breaker", cb_status)
        st.caption(f"Drawdown threshold: {dd_threshold:.0%}. Managed in Action Panel.")

# ─── TAB 5: LIVE ACTION PANEL ─────────────────────────────────────────
with tab_action:
    st.subheader("⚡ Live Action Panel — What to Do Right Now")
    st.caption("Signals refresh every 15 minutes. System E: 35% SPY + 65% active (top-3 weekly 11-1 momentum). Daily regime T+1 exit. Stop-loss −10% per position.")

    # Sidebar overrides for action panel
    ap_spy_floor = st.sidebar.slider("Action Panel: SPY Floor (%)", 10, 80, 35, step=5, key="ap_spy") / 100.0
    ap_top_n     = st.sidebar.slider("Action Panel: Top-N", 1, 7, 3, key="ap_topn")
    ap_ma        = st.sidebar.slider("Action Panel: MA Window", 50, 300, 200, step=25, key="ap_ma")
    ap_spy_ticker = st.sidebar.text_input("SPY Proxy", "SPY", key="ap_spy_tkr").strip().upper()

    # Fetch live signal
    _xs = st.session_state.get("xs_list", xs_list if "xs_list" in dir() else [])
    universe_for_signal = tuple(sorted(_xs)) if _xs else tuple()
    if not universe_for_signal:
        st.warning("Define the NQ100 universe in the Cross-Sectional tab first.")
    else:
        with st.spinner("Fetching live prices and computing signal …"):
            sig = compute_live_signal(
                universe_for_signal, ap_spy_ticker,
                ap_spy_floor, ap_top_n, ap_ma
            )

        if not sig:
            st.error("Could not fetch live price data. Check your internet connection.")
        else:
            prices     = sig["latest_prices"]
            regime     = sig["regime"]
            top_tickers= sig["top_tickers"]
            target_w   = sig["target_weights"]

            # ── Track regime change ──────────────────────────────────
            if st.session_state.last_regime != regime:
                st.session_state.regime_since = datetime.date.today()
                st.session_state.last_regime  = regime
            days_in_regime = (datetime.date.today() - st.session_state.regime_since).days \
                if st.session_state.regime_since else 0

            # ── Compute current NAV ──────────────────────────────────
            total_pos_val = sum(
                st.session_state.ledger["positions"].get(t, 0) * prices.get(t, 0.0)
                for t in st.session_state.ledger["positions"]
            )
            current_nav = st.session_state.ledger["cash"] + total_pos_val
            if current_nav > st.session_state.nav_peak:
                st.session_state.nav_peak = current_nav
            nav_dd = (current_nav / st.session_state.nav_peak - 1) * 100
            circuit_on = nav_dd < -abs(dd_threshold * 100)
            st.session_state.circuit_breaker = circuit_on

            # ── SECTION 1: REGIME STATUS ─────────────────────────────
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            if regime == "BULL":
                r_col1.success(f"### 🟢 BULL REGIME")
            else:
                r_col1.error(f"### 🔴 BEAR REGIME → CASH")
            r_col2.metric("NQ100 EW vs 200d MA", f"{sig['ma_margin']:+.2f}%",
                          help="How far above/below the 200d MA")
            spy_bull_flag = sig.get("spy_bull", True)
            breadth_val   = sig.get("breadth", 0)
            breadth_ok    = sig.get("breadth_ok", True)
            r_col3.metric(
                "SPY vs 200d MA",
                "✅ Above" if spy_bull_flag else "🔴 Below",
                help="SPY must also be above its 200d MA for bull regime"
            )
            r_col4.metric(
                "Breadth",
                f"{breadth_val*100:.0f}%",
                delta="✅ ≥40%" if breadth_ok else "🔴 <40% threshold",
                delta_color="normal" if breadth_ok else "inverse",
                help="% of NQ100 with positive 11-1 weekly score. Must be ≥40%."
            )
            st.caption(f"Signal as-of: {sig['as_of']} · Next rebalance: {sig['days_to_rebalance']}d (Friday close → Monday open)")

            st.divider()

            # ── SECTION 2: RISK GAUGES ───────────────────────────────
            st.markdown("#### 🛡️ Risk Dashboard")
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Portfolio NAV", f"${current_nav:,.0f}")
            g2.metric("NAV Drawdown", f"{nav_dd:.2f}%",
                      delta=f"{nav_dd - (-abs(dd_threshold*100)):.2f}% buffer",
                      delta_color="inverse")
            g3.metric("Peak NAV", f"${st.session_state.nav_peak:,.0f}")
            g4.metric("Cash Available", f"${st.session_state.ledger['cash']:,.0f}")
            g5.metric("Invested", f"${total_pos_val:,.0f}")

            if circuit_on:
                st.error(
                    f"⛔ CIRCUIT BREAKER ACTIVE — NAV down {nav_dd:.2f}% from peak "
                    f"(threshold: -{abs(dd_threshold*100):.0f}%). "
                    "All new execution is suspended. Review positions and consider reducing exposure."
                )
            elif nav_dd < -abs(dd_threshold * 100) * 0.7:
                st.warning(
                    f"⚠️ CAUTION — NAV at {nav_dd:.2f}% drawdown. "
                    f"Approaching circuit breaker at -{abs(dd_threshold*100):.0f}%. "
                    "Consider reducing position sizes."
                )
            else:
                st.success(f"✅ Risk nominal — {abs(nav_dd):.1f}% of {abs(dd_threshold*100):.0f}% circuit breaker used.")

            st.divider()

            # ── SECTION 3: TARGET PORTFOLIO ──────────────────────────
            st.markdown("#### 🎯 Strategy Target Portfolio (Right Now)")
            target_rows = []
            for tkr, w in target_w.items():
                target_val    = current_nav * w
                price_now     = prices.get(tkr, 0.0)
                target_shares = int(target_val / price_now) if price_now > 0 else 0
                current_qty   = st.session_state.ledger["positions"].get(tkr, 0)
                delta_qty     = target_shares - current_qty
                if delta_qty > 0:
                    action_str = f"🟢 BUY {delta_qty}"
                elif delta_qty < 0:
                    action_str = f"🔴 SELL {abs(delta_qty)}"
                else:
                    action_str = "⚪ HOLD"
                role = "SPY Floor (always-on)" if tkr == ap_spy_ticker else "Momentum (active sleeve)"
                target_rows.append({
                    "Ticker":        tkr,
                    "Role":          role,
                    "Target %":      f"{w:.0%}",
                    "Target Value":  f"${target_val:,.0f}",
                    "Current Price": f"${price_now:.2f}",
                    "Target Shares": target_shares,
                    "Current Shares":current_qty,
                    "Δ Shares":      delta_qty,
                    "Action":        action_str,
                })

            # Tickers currently held but NOT in target
            for tkr, qty in st.session_state.ledger["positions"].items():
                if tkr not in target_w and qty > 0:
                    price_now = prices.get(tkr, 0.0)
                    target_rows.append({
                        "Ticker":        tkr,
                        "Role":          "⚠️ Outside strategy",
                        "Target %":      "0%",
                        "Target Value":  "$0",
                        "Current Price": f"${price_now:.2f}",
                        "Target Shares": 0,
                        "Current Shares":qty,
                        "Δ Shares":      -qty,
                        "Action":        f"🔴 SELL {qty} (exit)",
                    })

            target_df = pd.DataFrame(target_rows)
            st.dataframe(target_df, use_container_width=True, hide_index=True)

            st.divider()

            # ── SECTION 4: ACTION CHECKLIST ──────────────────────────
            st.markdown("#### 📋 Action Checklist")
            st.caption(
                "System E: Weekly rebalance (Friday close → Monday open). "
                "Stop-loss: if any position drops >10% from entry price, sell at next open and replace with next-best ranked stock."
            )

            if circuit_on:
                st.error("⛔ Circuit breaker active — no new orders permitted. Exit positions to reduce risk.")
            else:
                buys  = [r for r in target_rows if r["Δ Shares"] > 0]
                sells = [r for r in target_rows if r["Δ Shares"] < 0]
                holds = [r for r in target_rows if r["Δ Shares"] == 0]

                if not buys and not sells:
                    st.success("✅ Portfolio is aligned with strategy. No action required.")
                else:
                    if regime == "BEAR":
                        st.warning(
                            "🔴 **BEAR REGIME ACTIVE** — The active momentum sleeve should be in cash (System E: 10% SPY + 90% cash). "
                            "Your only equity holding should be the SPY floor. "
                            "Sell all momentum positions and hold cash for the active sleeve."
                        )

                    ch1, ch2 = st.columns(2)
                    with ch1:
                        if buys:
                            st.markdown("**🟢 Buy Orders**")
                            for r in buys:
                                price_n = prices.get(r["Ticker"], 0.0)
                                cost_est = r["Δ Shares"] * price_n
                                pct_nav  = cost_est / current_nav * 100
                                caution  = " ⚠️ large" if pct_nav > 20 else ""
                                st.markdown(
                                    f"- **{r['Ticker']}** — BUY **{r['Δ Shares']} shares** "
                                    f"@ ~${price_n:.2f} = ${cost_est:,.0f} "
                                    f"({pct_nav:.1f}% NAV){caution}"
                                )
                        else:
                            st.markdown("**🟢 Buy Orders**\n- None")

                    with ch2:
                        if sells:
                            st.markdown("**🔴 Sell Orders**")
                            for r in sells:
                                price_n = prices.get(r["Ticker"], 0.0)
                                proceeds = abs(r["Δ Shares"]) * price_n
                                cb = st.session_state.cost_basis.get(r["Ticker"])
                                if cb and cb["total_qty"] > 0:
                                    avg_cost = cb["total_cost"] / cb["total_qty"]
                                    pnl = (price_n - avg_cost) * abs(r["Δ Shares"])
                                    pnl_str = f" | PnL: ${pnl:+,.0f}"
                                else:
                                    pnl_str = ""
                                st.markdown(
                                    f"- **{r['Ticker']}** — SELL **{abs(r['Δ Shares'])} shares** "
                                    f"@ ~${price_n:.2f} = ${proceeds:,.0f}{pnl_str}"
                                )
                        else:
                            st.markdown("**🔴 Sell Orders**\n- None")

                    # Total cash impact
                    total_buy_cost  = sum(r["Δ Shares"] * prices.get(r["Ticker"], 0) for r in buys)
                    total_sell_proc = sum(abs(r["Δ Shares"]) * prices.get(r["Ticker"], 0) for r in sells)
                    net_cash_impact = total_sell_proc - total_buy_cost
                    st.info(
                        f"Estimated net cash impact: **${net_cash_impact:+,.0f}** | "
                        f"Cash after trades: **${st.session_state.ledger['cash'] + net_cash_impact:,.0f}**"
                    )

            st.divider()

            # ── SECTION 5: FLOAT P&L ─────────────────────────────────
            st.markdown("#### 💰 Float P&L — Open Positions")
            pnl_rows = []
            total_unrealized = 0.0
            for tkr, qty in st.session_state.ledger["positions"].items():
                if qty <= 0:
                    continue
                price_now = prices.get(tkr, 0.0)
                mkt_val   = qty * price_now
                cb        = st.session_state.cost_basis.get(tkr)
                if cb and cb["total_qty"] > 0:
                    avg_cost  = cb["total_cost"] / cb["total_qty"]
                    cost_val  = qty * avg_cost
                    unrealized = mkt_val - cost_val
                    pct_gain   = (price_now / avg_cost - 1) * 100
                else:
                    avg_cost   = 0.0
                    cost_val   = 0.0
                    unrealized = 0.0
                    pct_gain   = 0.0
                total_unrealized += unrealized
                pnl_rows.append({
                    "Ticker":        tkr,
                    "Qty":           qty,
                    "Avg Cost":      f"${avg_cost:.2f}" if avg_cost else "—",
                    "Current Price": f"${price_now:.2f}",
                    "Market Value":  f"${mkt_val:,.0f}",
                    "Unrealized P&L":f"${unrealized:+,.0f}",
                    "Return %":      f"{pct_gain:+.2f}%",
                    "Status":        "🟢" if unrealized >= 0 else "🔴",
                })

            if pnl_rows:
                pnl_df = pd.DataFrame(pnl_rows)
                st.dataframe(pnl_df, use_container_width=True, hide_index=True)
                p1, p2, p3 = st.columns(3)
                p1.metric("Total Unrealized P&L", f"${total_unrealized:+,.0f}",
                          delta=f"{total_unrealized / (current_nav - st.session_state.ledger['cash']) * 100:+.2f}% on invested"
                          if total_pos_val > 0 else None)
                p2.metric("Total Market Value",   f"${total_pos_val:,.0f}")
                p3.metric("Realized + Unrealized",
                          f"${(current_nav - 100_000 + total_unrealized):+,.0f}",
                          help="vs. $100,000 starting NAV")
            else:
                st.info("No open positions. Portfolio is currently in cash.")

            st.divider()

            # ── SECTION 6: TOP-N MOMENTUM LEADERBOARD ────────────────
            st.markdown("#### 📊 NQ100 Momentum Leaderboard (Weekly 11-1)")
            if not sig["mom_table"].empty:
                mom_display = sig["mom_table"].head(15).reset_index()
                mom_display.columns = ["Ticker", "11-1 Wk Momentum"]
                mom_display["11-1 Wk Momentum"] = mom_display["11-1 Wk Momentum"].map("{:.2%}".format)
                mom_display["Rank"] = range(1, len(mom_display) + 1)
                mom_display["In Portfolio"] = mom_display["Ticker"].apply(
                    lambda t: "⭐ Selected" if t in top_tickers else
                              ("📍 Held" if t in st.session_state.ledger["positions"] else "—")
                )
                st.dataframe(mom_display[["Rank","Ticker","11-1 Wk Momentum","In Portfolio"]],
                             use_container_width=True, hide_index=True)

            st.divider()

            # ── SECTION 7: QUICK EXECUTE ─────────────────────────────
            st.markdown("#### ⚡ Quick Execute")
            if circuit_on:
                st.error("⛔ Execution suspended — circuit breaker is active.")
            else:
                qe_cols = st.columns(min(4, len(target_rows)))
                for i, row in enumerate(target_rows[:4]):
                    tkr = row["Ticker"]
                    with qe_cols[i % len(qe_cols)]:
                        dq = row["Δ Shares"]
                        st.markdown(f"**{tkr}** — {row['Action']}")
                        price_n = prices.get(tkr, 0.0)
                        adj_qty = st.number_input(
                            "Qty", min_value=0,
                            value=max(0, dq) if dq > 0 else max(0, abs(dq)),
                            step=1, key=f"qe_qty_{tkr}"
                        )
                        action_choice = "BUY" if dq >= 0 else "SELL"
                        action_choice = st.radio("Dir", ["BUY","SELL"],
                                                 index=0 if dq >= 0 else 1,
                                                 key=f"qe_dir_{tkr}", horizontal=True)
                        if st.button(f"Execute {tkr}", key=f"qe_btn_{tkr}",
                                     use_container_width=True):
                            if adj_qty > 0:
                                ok = log_transaction(action_choice, tkr, int(adj_qty), price_n)
                                if ok:
                                    st.success(f"{action_choice} {adj_qty}x {tkr} @ ${price_n:.2f}")
                                    st.rerun()
                                else:
                                    st.error("Rejected: insufficient cash or shares.")
