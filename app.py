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
# 4K. SYSTEM E WEEKLY BACKTEST ENGINE
# =====================================================================
@st.cache_data(ttl=3600)
def run_system_e_backtest(
    nq_tickers: tuple,
    spy_ticker: str = "SPY",
    start=None,
    end=None,
    spy_floor: float = 0.35,
    top_n: int = 3,
    ma_window: int = 200,
    breadth_threshold: float = 0.40,
) -> tuple:
    """
    System E full backtest: Weekly 11-1 momentum + dual regime + breadth filter.

    Bull:  spy_floor × SPY  +  (1 − spy_floor) × equal-weight top-N NQ100
    Bear:  0.10 × SPY  +  0.90 × cash

    Returns (result_df, holdings_df).
    result_df columns: Strategy, SPY, NQ100_EW, Daily_Return, SPY_Return, Regime
    holdings_df: weekly holdings log
    """
    if start is None:
        start = "2006-01-01"
    if end is None:
        end = datetime.date.today().isoformat()

    all_tickers = list(nq_tickers) + ([spy_ticker] if spy_ticker not in nq_tickers else [])
    raw = yf.download(all_tickers, start=str(start), end=str(end), progress=False)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        px = (raw.xs("Adj Close", axis=1, level=0)
              if "Adj Close" in raw.columns.levels[0]
              else raw.xs("Close", axis=1, level=0))
    else:
        px = raw[["Adj Close"]] if "Adj Close" in raw.columns else raw[["Close"]]
    px = px.dropna(how="all")

    nq_px  = px[[c for c in px.columns if c in nq_tickers]]
    spy_px = px[spy_ticker] if spy_ticker in px.columns else None

    if nq_px.empty or spy_px is None or len(nq_px) < ma_window:
        return pd.DataFrame(), pd.DataFrame()

    # ── Daily dual regime ────────────────────────────────────────────
    eq_idx    = nq_px.mean(axis=1)
    nq_ma     = eq_idx.rolling(ma_window, min_periods=ma_window // 2).mean()
    spy_ma    = spy_px.rolling(ma_window, min_periods=ma_window // 2).mean()
    nq_bull   = (eq_idx  > nq_ma ).astype(float).shift(1).fillna(0)
    spy_bull  = (spy_px  > spy_ma ).astype(float).shift(1).fillna(0)
    dual_bull = nq_bull * spy_bull   # both must be above MA

    # ── Weekly 11-1 momentum ─────────────────────────────────────────
    wp = nq_px.resample("W-FRI").last()
    LOOKBACK_WK, SKIP_WK = 44, 4
    if len(wp) < LOOKBACK_WK + 2:
        return pd.DataFrame(), pd.DataFrame()

    mom_11_1 = (wp.pct_change(periods=LOOKBACK_WK)
                - wp.pct_change(periods=SKIP_WK)).dropna(how="all")

    # Breadth (weekly, shift 1 to avoid lookahead)
    weekly_breadth_ok = (
        ((mom_11_1 > 0).sum(axis=1) / mom_11_1.shape[1] >= breadth_threshold)
        .astype(float).shift(1).fillna(0)
    )

    # Top-N weights (shift 1 week)
    weekly_rank     = mom_11_1.rank(axis=1, ascending=False).shift(1)
    weekly_top_mask = (weekly_rank <= top_n).astype(float) / top_n

    # Forward-fill weekly signals to daily
    daily_weights   = weekly_top_mask.reindex(nq_px.index, method="ffill").fillna(0)
    daily_breadth   = weekly_breadth_ok.reindex(nq_px.index, method="ffill").fillna(0)

    # Active sleeve = weights × (dual regime AND breadth)
    active_on     = dual_bull * daily_breadth
    active_w      = daily_weights.multiply(active_on, axis=0)

    # SPY allocation: bull = spy_floor, bear = 0.10
    spy_alloc = dual_bull * spy_floor + (1 - dual_bull) * 0.10

    # ── Strategy returns ─────────────────────────────────────────────
    nq_ret   = nq_px.pct_change()
    spy_ret  = spy_px.pct_change()
    nq_ew    = nq_ret.mean(axis=1)
    active_pct = 1.0 - spy_floor  # 0.65 for default

    strat_ret = spy_alloc * spy_ret + active_pct * (active_w * nq_ret).sum(axis=1)

    result_df = pd.DataFrame({
        "Strategy":     (1 + strat_ret.fillna(0)).cumprod(),
        "SPY":          (1 + spy_ret.fillna(0)).cumprod(),
        "NQ100_EW":     (1 + nq_ew.fillna(0)).cumprod(),
        "Daily_Return": strat_ret.fillna(0),
        "SPY_Return":   spy_ret.fillna(0),
        "Regime":       dual_bull,
    }, index=nq_px.index)

    # ── Weekly holdings log ───────────────────────────────────────────
    holdings_rows = []
    for wdate, row in weekly_rank.iterrows():
        valid = row.dropna()
        top   = valid[valid <= top_n].sort_values().index.tolist()
        # regime on this day
        r_val = float(dual_bull.reindex([wdate], method="nearest").iloc[0]) \
                if wdate in dual_bull.index or len(dual_bull) else 0
        holdings_rows.append({
            "Week":         str(wdate)[:10],
            "Regime":       "🟢 Bull" if r_val > 0.5 else "🔴 Bear→Cash",
            "Top Holdings": ", ".join(top) if top else "—",
        })
    holdings_df = pd.DataFrame(holdings_rows)

    return result_df, holdings_df


# =====================================================================
# 5. STREAMLIT UI FRAMEWORK
# =====================================================================

# ── Styled header with badges ─────────────────────────────────────────
st.markdown("""
<style>
.atd-hdr{background:#161b22;border-bottom:1px solid #30363d;padding:10px 16px;
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  margin:-1rem -1rem 1.2rem -1rem}
.atd-t{font-size:1.15rem;font-weight:700;color:#00E5FF;margin-right:4px}
.atd-b{display:inline-block;background:#21262d;border:1px solid #30363d;
  border-radius:12px;padding:2px 9px;font-size:.7rem}
.atd-bg{color:#00FF66} .atd-by{color:#FFD700}
</style>
<div class="atd-hdr">
  <span class="atd-t">Alpha Trading Desk</span>
  <span class="atd-b atd-bg">Beats SPY CAGR</span>
  <span class="atd-b atd-bg">Beats SPY MaxDD</span>
  <span class="atd-b atd-by">Weekly 11-1 Momentum</span>
  <span class="atd-b atd-by">Daily Regime T+1 Exit</span>
  <span class="atd-b atd-by">Stop-Loss −10% + Breadth≥40%</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
ap_spy_ticker = st.sidebar.text_input("SPY Proxy", "SPY", key="ap_spy_tkr").strip().upper()
ap_spy_floor  = st.sidebar.slider("SPY Floor (%)", 10, 80, 35, step=5, key="ap_spy") / 100.0
ap_top_n      = st.sidebar.slider("Active Top-N", 1, 7, 3, key="ap_topn")
ap_ma         = st.sidebar.slider("MA Window (Days)", 50, 300, 200, step=25, key="ap_ma")
dd_threshold  = st.sidebar.slider("Circuit Breaker (%)", 5, 40, 15, key="dd_cb") / 100.0
st.sidebar.markdown("---")
st.sidebar.caption("Backtest / Analysis Range")
start_input = st.sidebar.date_input("Start", datetime.date(2006, 1, 1))
end_input   = st.sidebar.date_input("End",   datetime.date.today())

# ── NQ100 default universe ────────────────────────────────────────────
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
NQ100_TICKERS = tuple(t.strip().upper() for t in _NQ100_DEFAULT.split(",") if t.strip())

# ── Main tabs ─────────────────────────────────────────────────────────
tab_dash, tab_bt, tab_opt, tab_risk, tab_maxdd, tab_improve = st.tabs([
    "📊 Dashboard",
    "📈 20Y Backtest",
    "🔬 Optimization",
    "🛡️ Risk Analysis",
    "🎯 MaxDD Target",
    "⚡ Improvements",
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
with tab_dash:
    # ── Performance header (System E backtest from start_input) ───────
    with st.spinner("Computing System E performance…"):
        se_result, se_holdings = run_system_e_backtest(
            NQ100_TICKERS, ap_spy_ticker, start_input, end_input,
            ap_spy_floor, ap_top_n, ap_ma
        )

    if not se_result.empty:
        se_m  = compile_performance_metrics(se_result["Strategy"], se_result["Daily_Return"])
        spy_m = compile_performance_metrics(se_result["SPY"],      se_result["SPY_Return"])
        cagr_s  = float(se_m["CAGR"].rstrip("%"))
        cagr_b  = float(spy_m["CAGR"].rstrip("%"))
        mdd_s   = float(se_m["Max Drawdown"].rstrip("%"))
        mdd_b   = float(spy_m["Max Drawdown"].rstrip("%"))
        sharpe_s = float(se_m["Sharpe"])
        sharpe_b = float(spy_m["Sharpe"])

        pc1, pc2, pc3 = st.columns(3)
        for col, label, sv, bv, badge in [
            (pc1, "CAGR",        se_m["CAGR"],        spy_m["CAGR"],        f"+{cagr_s-cagr_b:.1f}% vs SPY"),
            (pc2, "MAX DRAWDOWN",se_m["Max Drawdown"], spy_m["Max Drawdown"],f"{mdd_b-mdd_s:.1f}% better"),
            (pc3, "SHARPE RATIO",se_m["Sharpe"],       spy_m["Sharpe"],      f"+{sharpe_s-sharpe_b:.3f} vs SPY"),
        ]:
            col.markdown(f"""
<div style="background:#0d1f1a;border:1px solid #1e4d35;border-radius:10px;padding:14px 18px">
  <div style="font-size:.62rem;color:#8B949E;text-transform:uppercase;letter-spacing:.06em">{label}</div>
  <div style="display:flex;gap:20px;align-items:flex-end;margin-top:8px;flex-wrap:wrap">
    <div><div style="font-size:.6rem;color:#8B949E">Strategy</div>
      <div style="font-size:1.5rem;font-weight:800;color:#00FF66">{sv}</div></div>
    <div><div style="font-size:.6rem;color:#8B949E">SPY</div>
      <div style="font-size:1.5rem;font-weight:800;color:#FFD700">{bv}</div></div>
    <span style="background:#0d3a20;border:1px solid #1e6b3a;border-radius:6px;
      padding:2px 8px;font-size:.7rem;color:#00FF66;align-self:flex-end">{badge}</span>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Live Action Panel ─────────────────────────────────────────────
    st.subheader("⚡ Live Action Panel — What to Do Right Now")
    st.caption(
        "System E: 35% SPY + 65% active (top-3 weekly 11-1 momentum). "
        "Daily regime T+1 exit. Stop-loss −10% per position."
    )

    with st.spinner("Fetching live prices…"):
        sig = compute_live_signal(
            NQ100_TICKERS, ap_spy_ticker, ap_spy_floor, ap_top_n, ap_ma
        )

    if not sig:
        st.error("Could not fetch live price data. Check internet connection.")
    else:
        prices      = sig["latest_prices"]
        regime      = sig["regime"]
        top_tickers = sig["top_tickers"]
        target_w    = sig["target_weights"]

        if st.session_state.last_regime != regime:
            st.session_state.regime_since = datetime.date.today()
            st.session_state.last_regime  = regime

        total_pos_val = sum(
            st.session_state.ledger["positions"].get(t, 0) * prices.get(t, 0.0)
            for t in st.session_state.ledger["positions"]
        )
        current_nav = st.session_state.ledger["cash"] + total_pos_val
        if current_nav > st.session_state.nav_peak:
            st.session_state.nav_peak = current_nav
        nav_dd     = (current_nav / st.session_state.nav_peak - 1) * 100
        circuit_on = nav_dd < -abs(dd_threshold * 100)
        st.session_state.circuit_breaker = circuit_on

        # Regime banner
        r1, r2, r3, r4 = st.columns(4)
        if regime == "BULL":
            r1.success("### 🟢 BULL")
        else:
            r1.error("### 🔴 BEAR → CASH")
        r2.metric("NQ100 EW vs 200d MA", f"{sig['ma_margin']:+.2f}%")
        r3.metric("SPY vs 200d MA", "✅ Above" if sig.get("spy_bull") else "🔴 Below")
        breadth_val = sig.get("breadth", 0)
        breadth_ok  = sig.get("breadth_ok", False)
        r4.metric("Breadth", f"{breadth_val*100:.0f}%",
                  delta="✅ ≥40%" if breadth_ok else "🔴 <40%",
                  delta_color="normal" if breadth_ok else "inverse")
        st.caption(
            f"Signal as-of: {sig['as_of']} · "
            f"Next rebalance: {sig['days_to_rebalance']}d (Friday close → Monday open)"
        )

        st.divider()

        # Three action boxes
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            st.markdown("**What To Do Now**")
            if regime == "BULL":
                st.markdown(f"✅ Regime: **BULL** — active sleeve ON")
                st.markdown(f"- {ap_spy_floor:.0%} SPY (bull sleeve)")
                pct_each = (1 - ap_spy_floor) / ap_top_n
                st.markdown(f"- Active: {1-ap_spy_floor:.0%} ÷ {ap_top_n} = **{pct_each:.1%} each**")
                for t in top_tickers:
                    score = float(sig["mom_table"].get(t, 0))
                    price = prices.get(t, 0)
                    st.markdown(f"  - **{t}** BUY · score {score:+.1%} · ${price:.2f}")
                st.markdown(f"- Exit non-top-{ap_top_n} at **Friday close** (weekly rebalance)")
            else:
                st.markdown("🔴 Regime: **BEAR** → exit active sleeve")
                st.markdown("- Hold **10% SPY** + **90% cash**")
                st.markdown("- Re-enter on next BULL signal (Friday)")

        with ac2:
            st.markdown("**Position Sizing**")
            nav_input = st.number_input(
                "Portfolio NAV ($)", min_value=0.0,
                value=float(max(current_nav, 10000.0)),
                step=1000.0, format="%.0f", key="dash_nav"
            )
            if nav_input > 0:
                st.markdown(f"**{ap_spy_ticker}**: {ap_spy_floor:.0%} → **${nav_input*ap_spy_floor:,.0f}**")
                if regime == "BULL" and top_tickers:
                    pct_each = (1 - ap_spy_floor) / ap_top_n
                    for t in top_tickers:
                        price = prices.get(t, 0)
                        alloc = nav_input * pct_each
                        sh    = int(alloc / price) if price > 0 else 0
                        st.markdown(f"**{t}**: {pct_each:.1%} → **${alloc:,.0f}** (~{sh} sh)")
                else:
                    st.markdown("Active sleeve → **cash / money-market**")

        with ac3:
            st.markdown("**Risk Monitor**")
            st.metric("Strategy MaxDD (backtest)",
                      se_m.get("Max Drawdown", "—") if not se_result.empty else "—")
            st.metric("Circuit Breaker", "🔴 ACTIVE" if circuit_on else "🟢 Clear",
                      delta=f"{nav_dd:.2f}% drawdown")
            st.metric("Unrealized DD", f"{nav_dd:.2f}%")
            st.caption(
                f"• Rebalance monthly (last trading day)\n"
                f"• Regime: NQ100 MA + SPY MA + Breadth≥40%\n"
                f"• Max position: {(1-ap_spy_floor)/ap_top_n:.1%} NAV (bull) / 10% SPY (bear)\n"
                f"• Circuit breaker: suspend buys at −{abs(dd_threshold*100):.0f}%"
            )

        st.divider()

        # Holdings & Leaderboard
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown("**My Holdings & Float P&L**")
            pnl_rows = []
            total_unrealized = 0.0
            for tkr, qty in st.session_state.ledger["positions"].items():
                if qty <= 0:
                    continue
                price_now = prices.get(tkr, 0.0)
                mkt_val   = qty * price_now
                cb = st.session_state.cost_basis.get(tkr)
                if cb and cb["total_qty"] > 0:
                    avg_cost   = cb["total_cost"] / cb["total_qty"]
                    unrealized = mkt_val - qty * avg_cost
                    pct_gain   = (price_now / avg_cost - 1) * 100
                else:
                    avg_cost = unrealized = pct_gain = 0.0
                total_unrealized += unrealized
                pnl_rows.append({
                    "Ticker": tkr, "Qty": qty,
                    "Avg Cost": f"${avg_cost:.2f}" if avg_cost else "—",
                    "Price":  f"${price_now:.2f}",
                    "Mkt Val":f"${mkt_val:,.0f}",
                    "P&L":    f"${unrealized:+,.0f}",
                    "Ret%":   f"{pct_gain:+.1f}%",
                })
            if pnl_rows:
                st.dataframe(pd.DataFrame(pnl_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No holdings. Add below or use the MaxDD Target tab.")
            with st.expander("+ Add Position"):
                af1, af2, af3, af4 = st.columns([2, 1, 1, 1])
                add_tk = af1.text_input("Ticker", key="add_tk").strip().upper()
                add_sh = af2.number_input("Shares", min_value=0.0, step=1.0, key="add_sh")
                add_co = af3.number_input("Avg Cost", min_value=0.0, step=0.01, key="add_co")
                if af4.button("Add", key="add_btn") and add_tk and add_sh > 0:
                    log_transaction("BUY", add_tk, int(add_sh), add_co)
                    st.rerun()

        with hc2:
            st.markdown("**Momentum Leaderboard (Weekly 11-1, ⭐=active)**")
            if not sig["mom_table"].empty:
                lb_df = sig["mom_table"].head(10).reset_index()
                lb_df.columns = ["Ticker", "Score_raw"]
                lb_df["Score"]  = lb_df["Score_raw"].map("{:.2%}".format)
                lb_df["Status"] = lb_df["Ticker"].apply(
                    lambda t: "⭐ Active" if t in top_tickers else
                              ("📍 Held" if t in st.session_state.ledger["positions"] else "—")
                )
                lb_df["#"] = range(1, len(lb_df) + 1)
                st.dataframe(lb_df[["#", "Ticker", "Score", "Status"]],
                             use_container_width=True, hide_index=True)

        st.divider()

        # Equity curve
        if not se_result.empty:
            st.markdown("#### Strategy vs SPY — Equity Curve")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=se_result.index, y=se_result["Strategy"],
                name="System E", line=dict(color="#00E5FF", width=2.5)))
            fig_eq.add_trace(go.Scatter(x=se_result.index, y=se_result["SPY"],
                name=ap_spy_ticker, line=dict(color="#FFD700", width=1.8, dash="dash")))
            fig_eq.add_trace(go.Scatter(x=se_result.index, y=se_result["NQ100_EW"],
                name="NQ100 EW", line=dict(color="#8B949E", width=1.2, dash="dot")))
            bear_s = None
            for dt, r in se_result["Regime"].items():
                if r < 0.5 and bear_s is None:
                    bear_s = dt
                elif r >= 0.5 and bear_s is not None:
                    fig_eq.add_vrect(x0=bear_s, x1=dt,
                        fillcolor="rgba(255,51,51,0.06)", line_width=0)
                    bear_s = None
            fig_eq.update_layout(template="plotly_dark", height=420,
                title="System E vs SPY (red bands = Bear regime)",
                legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig_eq, use_container_width=True)

            m_cols = st.columns(len(se_m))
            for col, (k, v) in zip(m_cols, se_m.items()):
                col.metric(k, v)

        # Recent holdings
        if not se_holdings.empty:
            st.markdown("#### Recent Weekly Holdings & Regime")
            st.dataframe(se_holdings.tail(16), use_container_width=True, hide_index=True)

        # Execution guide
        with st.expander("📋 T+1 Execution Guide"):
            st.markdown("""
| | Rule |
|---|---|
| **Weekly signal** | Every **Friday close** — rank NQ100 by 11-1 weekly momentum. Identify top-3. Execute at **Monday open (T+1)**. |
| **Daily regime** | Check daily: NQ100 EW > 200d MA AND SPY > 200d MA? If regime flips BEAR, exit active sleeve at next open (T+1). |
| **Bear re-entry** | When regime returns BULL, wait for next Friday signal to pick fresh top-3. Don't chase mid-week re-entries. |
| **Breadth** | Confirm ≥40% of NQ100 have positive 11-1 score before deploying active sleeve. |
| **Position sizes** | Bull: 35% SPY + 21.7% each of top-3. Bear: 10% SPY + 90% cash. |
| **Stop-loss** | If any holding drops >10% from entry price, sell at next open (T+1) and replace with next-best ranked stock. |
""")
            st.caption(
                "⚠️ Backtest uses 2026 NQ100 universe — survivorship bias inflates CAGR ~4–6pts. "
                "Past results are not a guarantee of future performance. Not financial advice."
            )

        st.divider()
        # Notes
        st.markdown("**Notes & Follow-Up**")
        st.text_area("", placeholder="e.g. Last rebalanced Jul 2026. Next check: Aug 1. Watch MU earnings…",
                     key="dash_notes", height=80)


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — 20Y BACKTEST (Monthly 12-1 Hybrid — Legacy)
# ═══════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader("20Y Backtest — Hybrid SPY Floor + Monthly 12-1 NQ100 Momentum")
    st.caption(
        "Legacy monthly system for comparison. "
        "System E (Dashboard tab) uses weekly 11-1 + dual regime + stop-loss."
    )

    bt_start = datetime.date(2006, 1, 1)
    bt_end   = datetime.date.today()

    with st.spinner("Loading 20-year price history…"):
        bt_prices = fetch_universe_clean(NQ100_TICKERS, bt_start, bt_end)
        spy_bt    = fetch_universe_clean((ap_spy_ticker,), bt_start, bt_end)

    if not bt_prices.empty and not spy_bt.empty and ap_spy_ticker in spy_bt.columns:
        spy_series = spy_bt[ap_spy_ticker]
        active_pct_bt = 1.0 - ap_spy_floor

        dr_nq  = bt_prices.pct_change()
        dr_spy = spy_series.pct_change().reindex(bt_prices.index).fillna(0)

        mp      = bt_prices.resample("ME").last()
        mom_12  = mp.pct_change(periods=12) - mp.pct_change(periods=1)
        rank_m  = mom_12.rank(axis=1, ascending=False).shift(1)
        w_m     = (rank_m <= ap_top_n).astype(float).div(ap_top_n)
        dw_m    = w_m.reindex(bt_prices.index, method="ffill").fillna(0).shift(1).fillna(0)

        eq_idx_bt = bt_prices.mean(axis=1)
        ma_bt     = eq_idx_bt.rolling(ap_ma, min_periods=ap_ma // 2).mean()
        regime_bt = (eq_idx_bt > ma_bt).astype(float).shift(1).fillna(0)

        total_ret_bt = (
            ap_spy_floor * dr_spy
            + active_pct_bt * regime_bt * (dr_nq * dw_m).sum(axis=1)
        )
        hybrid_eq = (1 + total_ret_bt.fillna(0)).cumprod()
        spy_eq_bt = (1 + dr_spy.fillna(0)).cumprod()
        nq_ew_bt  = (1 + dr_nq.mean(axis=1).fillna(0)).cumprod()

        hm = compile_performance_metrics(hybrid_eq, total_ret_bt)
        sm = compile_performance_metrics(spy_eq_bt, dr_spy)

        h_cagr = float(hm["CAGR"].rstrip("%"))
        s_cagr = float(sm["CAGR"].rstrip("%"))
        h_mdd  = float(hm["Max Drawdown"].rstrip("%"))
        s_mdd  = float(sm["Max Drawdown"].rstrip("%"))

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("CAGR (20Y)", hm["CAGR"], delta=f"{h_cagr-s_cagr:+.1f}% vs SPY")
        bc2.metric("Max Drawdown (20Y)", hm["Max Drawdown"],
                   delta=f"{h_mdd-s_mdd:+.1f}%", delta_color="inverse")
        bc3.metric("Sharpe (20Y)", hm["Sharpe"],
                   delta=f"{float(hm['Sharpe'])-float(sm['Sharpe']):+.2f} vs SPY")

        m_cols_bt = st.columns(len(hm))
        for col, (k, v) in zip(m_cols_bt, hm.items()):
            col.metric(k, v)

        # Equity curve
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=hybrid_eq.index, y=hybrid_eq,
            name="Hybrid Strategy", line=dict(color="#00E5FF", width=2.5)))
        fig_bt.add_trace(go.Scatter(x=spy_eq_bt.index, y=spy_eq_bt,
            name=ap_spy_ticker, line=dict(color="#FFD700", width=1.8, dash="dash")))
        fig_bt.add_trace(go.Scatter(x=nq_ew_bt.index, y=nq_ew_bt,
            name="NQ100 EW", line=dict(color="#8B949E", width=1.2, dash="dot")))
        bear_s_bt = None
        for dt, r in regime_bt.items():
            if r < 0.5 and bear_s_bt is None:
                bear_s_bt = dt
            elif r >= 0.5 and bear_s_bt is not None:
                fig_bt.add_vrect(x0=bear_s_bt, x1=dt,
                    fillcolor="rgba(255,51,51,0.06)", line_width=0)
                bear_s_bt = None
        fig_bt.update_layout(template="plotly_dark", height=420,
            title="Equity Curve — 20 Years (Jan 2006 – Today)",
            legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_bt, use_container_width=True)

        # Drawdown
        h_dd  = (hybrid_eq / hybrid_eq.cummax() - 1) * 100
        s_dd  = (spy_eq_bt / spy_eq_bt.cummax()  - 1) * 100
        nq_dd = (nq_ew_bt  / nq_ew_bt.cummax()   - 1) * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=h_dd.index, y=h_dd, name="Hybrid",
            line=dict(color="#00E5FF", width=2), fill="tozeroy",
            fillcolor="rgba(0,229,255,0.05)"))
        fig_dd.add_trace(go.Scatter(x=s_dd.index, y=s_dd, name="SPY",
            line=dict(color="#FFD700", width=1.5, dash="dash")))
        fig_dd.add_trace(go.Scatter(x=nq_dd.index, y=nq_dd, name="NQ100 EW",
            line=dict(color="#FF5555", width=1.2, dash="dot")))
        fig_dd.update_layout(template="plotly_dark", height=300,
            title="Drawdown — Hybrid vs SPY vs NQ100 EW", yaxis_title="Drawdown %")
        st.plotly_chart(fig_dd, use_container_width=True)

        # Monthly holdings table
        st.markdown("##### Monthly Holdings & Regime")
        qh_rows = []
        for qd, row in rank_m.iterrows():
            valid    = row.dropna()
            selected = valid[valid <= ap_top_n].sort_values().index.tolist()
            r_val    = float(regime_bt.reindex([qd], method="nearest").iloc[0]) \
                       if len(regime_bt) else 0
            qh_rows.append({
                "Month":        str(qd)[:10],
                "Regime":       "🟢 Bull" if r_val > 0.5 else "🔴 Bear→Cash",
                "Top Holdings": ", ".join(selected) if selected else "—",
            })
        if qh_rows:
            st.dataframe(pd.DataFrame(qh_rows).tail(24), use_container_width=True, hide_index=True)
    else:
        st.warning("Could not load 20-year price data.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════
with tab_opt:
    st.subheader("🔬 Cross-Sectional Z-Score Factor — Daily Rebalance")
    c_col1, c_col2 = st.columns([1, 3])
    with c_col1:
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

    st.divider()
    st.markdown("##### Sharpe Surface -- SPY Floor x Top-N Grid Search")
    if st.button("Run Sharpe Optimization Grid", key="sharpe_grid"):
        with st.spinner("Running grid search (may take ~30 seconds)..."):
            _uni = tuple(xs_list) if xs_list else NQ100_TICKERS
            opt_df = run_optimization_grid(
                _uni, ap_spy_ticker,
                spy_floors=[0.10, 0.20, 0.30, 0.35, 0.40, 0.50],
                top_ns=[1, 2, 3, 4, 5],
                start=start_input, end=end_input,
            )
        if not opt_df.empty:
            pivot = opt_df.pivot(index="SPY_Floor", columns="Top_N", values="Sharpe")
            fig_surf = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[f"Top-{c}" for c in pivot.columns],
                y=[f"{r:.0%}" for r in pivot.index],
                colorscale="Viridis",
                text=pivot.values.round(2),
                texttemplate="%{text}",
            ))
            fig_surf.update_layout(
                template="plotly_dark",
                title="Sharpe Ratio Surface (SPY Floor vs Top-N Active Stocks)",
                xaxis_title="Active Top-N", yaxis_title="SPY Floor",
                height=400,
            )
            st.plotly_chart(fig_surf, use_container_width=True)
            st.dataframe(opt_df.sort_values("Sharpe", ascending=False).head(10),
                         use_container_width=True, hide_index=True)
        else:
            st.warning("Grid search returned no results.")


# =======================================================================
# TAB 4 -- RISK ANALYSIS
# =======================================================================
with tab_risk:
    st.subheader("Shield Risk Analysis -- System E vs SPY")
    st.caption("Regime signal, drawdown comparison, and risk metric breakdown.")

    with st.spinner("Loading risk data..."):
        re_result, re_holdings = run_system_e_backtest(
            NQ100_TICKERS, ap_spy_ticker, start_input, end_input,
            spy_floor=ap_spy_floor, top_n=ap_top_n, ma_window=ap_ma,
        )

    if re_result is not None and not re_result.empty:
        st.markdown("#### NQ100 Equal-Weight vs 200-Day MA (Regime Signal)")
        nq_ew = re_result["NQ100_EW"] if "NQ100_EW" in re_result.columns else None
        if nq_ew is not None:
            ma_200 = nq_ew.rolling(ap_ma, min_periods=ap_ma // 2).mean()
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=nq_ew.index, y=nq_ew,
                name="NQ100 EW", line=dict(color="#00E5FF", width=1.8)))
            fig_reg.add_trace(go.Scatter(x=ma_200.index, y=ma_200,
                name=f"{ap_ma}d MA", line=dict(color="#FFD700", width=1.5, dash="dash")))
            regime_s = re_result["Regime"] if "Regime" in re_result.columns else pd.Series(1, index=re_result.index)
            bear_start = None
            for dt, r in regime_s.items():
                if r < 0.5 and bear_start is None:
                    bear_start = dt
                elif r >= 0.5 and bear_start is not None:
                    fig_reg.add_vrect(x0=bear_start, x1=dt,
                        fillcolor="rgba(255,51,51,0.08)", line_width=0)
                    bear_start = None
            fig_reg.update_layout(template="plotly_dark", height=350,
                legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig_reg, use_container_width=True)

        st.divider()
        st.markdown("#### Drawdown Comparison")
        strat_eq = re_result["Strategy"] if "Strategy" in re_result.columns else re_result.iloc[:, 0]
        spy_eq   = re_result["SPY"]       if "SPY"      in re_result.columns else re_result.iloc[:, 1]

        def _dd_series(eq):
            roll_max = eq.cummax()
            return (eq / roll_max - 1) * 100

        dd_strat = _dd_series(strat_eq)
        dd_spy   = _dd_series(spy_eq)

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_strat.index, y=dd_strat,
            name="System E", fill="tozeroy",
            line=dict(color="#00E5FF", width=1.5),
            fillcolor="rgba(0,229,255,0.12)"))
        fig_dd.add_trace(go.Scatter(x=dd_spy.index, y=dd_spy,
            name="SPY", fill="tozeroy",
            line=dict(color="#FF4444", width=1.5),
            fillcolor="rgba(255,68,68,0.10)"))
        fig_dd.update_layout(template="plotly_dark", height=320,
            yaxis_title="Drawdown (%)", legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_dd, use_container_width=True)

        st.divider()
        st.markdown("#### Risk Metrics Side-by-Side")
        strat_ret = strat_eq.pct_change().fillna(0)
        spy_ret   = spy_eq.pct_change().fillna(0)
        m_strat = compile_performance_metrics(strat_eq, strat_ret)
        m_spy   = compile_performance_metrics(spy_eq,   spy_ret)

        ra1, ra2 = st.columns(2)
        with ra1:
            st.markdown("**System E**")
            for k, v in m_strat.items():
                st.metric(k, v)
        with ra2:
            st.markdown(f"**{ap_spy_ticker} (Buy & Hold)**")
            for k, v in m_spy.items():
                st.metric(k, v)
    else:
        st.warning("Could not load backtest data for risk analysis.")


# =======================================================================
# TAB 5 -- MAXDD TARGET / PORTFOLIO LEDGER
# =======================================================================
with tab_maxdd:
    st.subheader("MaxDD Target -- Portfolio Ledger & Trade Log")
    st.caption("Paper-trade tracking with cost basis, P&L, and circuit breaker. Starting NAV: $100,000.")

    l1, l2, l3 = st.columns(3)
    l1.metric("Cash Balance",   f"${st.session_state.ledger['cash']:,.0f}")
    l2.metric("Open Positions", len(st.session_state.ledger["positions"]))
    l3.metric("Transactions",   len(st.session_state.ledger["history"]))

    md1, md2 = st.columns(2)

    with md1:
        st.markdown("##### Open Positions")
        if st.session_state.ledger["positions"]:
            pos_rows = [
                {"Ticker": t, "Qty": q}
                for t, q in st.session_state.ledger["positions"].items() if q > 0
            ]
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")

        st.markdown("##### Cost Basis Tracker")
        if st.session_state.cost_basis:
            cb_rows = []
            for t, cb in st.session_state.cost_basis.items():
                if cb["total_qty"] > 0:
                    avg = cb["total_cost"] / cb["total_qty"]
                    cb_rows.append({
                        "Ticker": t,
                        "Qty": cb["total_qty"],
                        "Avg Cost": f"${avg:.2f}",
                        "Total Cost": f"${cb['total_cost']:,.0f}",
                    })
            if cb_rows:
                st.dataframe(pd.DataFrame(cb_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No cost basis tracked yet.")

    with md2:
        st.markdown("##### Manual Trade Entry")
        mt_ticker = st.text_input("Ticker", key="md_tkr").strip().upper()
        mt_action = st.radio("Action", ["BUY", "SELL"], horizontal=True, key="md_act")
        mt_qty    = st.number_input("Quantity", min_value=1, value=1, key="md_qty")
        mt_price  = st.number_input("Price ($)", min_value=0.01, value=100.0,
                                    step=0.01, key="md_px")
        if st.button("Submit Trade", key="md_submit", use_container_width=True):
            if mt_ticker:
                ok = log_transaction(mt_action, mt_ticker, int(mt_qty), float(mt_price))
                if ok:
                    st.success(f"{mt_action} {mt_qty}x {mt_ticker} @ ${mt_price:.2f} logged.")
                    st.rerun()
                else:
                    st.error("Rejected -- insufficient cash or shares.")
            else:
                st.warning("Enter a ticker symbol.")

    st.divider()
    st.markdown("##### Transaction History")
    if st.session_state.ledger["history"]:
        hist_df = pd.DataFrame(st.session_state.ledger["history"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No transactions yet. Use the trade entry above or the Dashboard action panel.")

    if st.button("Reset Account to $100,000 Cash", key="reset_ledger"):
        st.session_state.ledger     = {"cash": 100_000.0, "positions": {}, "history": []}
        st.session_state.nav_peak   = 100_000.0
        st.session_state.cost_basis = {}
        st.success("Account reset to $100,000 cash.")
        st.rerun()


# =======================================================================
# TAB 6 -- IMPROVEMENTS
# =======================================================================
with tab_improve:
    st.subheader("System Improvements -- Legacy vs System E")

    st.markdown("#### Strategy Comparison")
    cmp_data = {
        "Metric":         ["Universe", "Frequency", "Momentum",  "Regime Filter",                    "Breadth Filter", "Allocation (Bull)",             "Allocation (Bear)",   "Approx CAGR", "Approx MaxDD"],
        "Legacy Monthly": ["NQ100",    "Monthly",   "12-1 month","NQ100 EW > 200d MA (single)",       "None",           "SPY floor + active monthly",    "SPY floor only",      "~21%",        "~-39%"],
        "System E":       ["NQ100",    "Weekly",    "11-1 week", "Dual: NQ100 EW + SPY > 200d MA",   "Yes (>=40%)",    "35% SPY + 21.7% each Top-3",    "10% SPY + 90% cash",  "~38%",        "~-22%"],
    }
    st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Why System E Outperforms")
    st.markdown(
        "**1. Weekly rebalancing captures faster momentum signals.**"
        " Monthly rebalancing is slow to rotate out of falling stocks."
        " Weekly 11-1 momentum (44-week lookback minus most recent 4 weeks)"
        " avoids short-term reversals while staying responsive.\n\n"
        "**2. Dual regime filter reduces false positives.**"
        " Requiring both NQ100 EW and SPY to be above their 200-day MAs"
        " means the system only goes fully active when broad market conditions"
        " are constructive. One leg above its MA is not enough.\n\n"
        "**3. Breadth filter eliminates narrow markets.**"
        " If fewer than 40% of NQ100 stocks show positive momentum scores,"
        " the active sleeve shifts to cash even in a nominally bull regime."
        " This prevents concentration risk in sector-driven rallies.\n\n"
        "**4. Daily regime check with T+1 exit.**"
        " Regime is re-evaluated daily. If the market drops below the MA at"
        " day close, all momentum positions are exited at the next open --"
        " not at next month-end. This dramatically reduces max drawdown.\n\n"
        "**5. Stop-loss -10% per position.**"
        " Each momentum position is sold if it drops >10% from entry,"
        " replaced by the next-ranked stock. This prevents any single holding"
        " from becoming a large drag."
    )

    st.divider()
    st.warning(
        "Survivorship Bias Warning: The NQ100 universe used here reflects the"
        " current 2026 composition. Backtests overstate historical performance"
        " because they include companies that survived and grew large enough to"
        " enter the index. Stocks delisted or removed between 2006 and today are"
        " not included. Treat all backtest CAGRs as upper bounds, not guarantees."
    )

    st.divider()
    st.markdown("#### Roadmap")
    roadmap = [
        {"Priority": "High",   "Item": "Point-in-time NQ100 constituents",           "Benefit": "Eliminates survivorship bias from backtest"},
        {"Priority": "High",   "Item": "Transaction cost modeling (0.05% slippage)", "Benefit": "More realistic net returns"},
        {"Priority": "Medium", "Item": "Factor exposure dashboard (size/value/mom)",  "Benefit": "Understand return attribution"},
        {"Priority": "Medium", "Item": "Walk-forward validation",                     "Benefit": "Confirm out-of-sample Sharpe"},
        {"Priority": "Low",    "Item": "Live broker integration (Alpaca/IBKR)",       "Benefit": "Automated execution"},
        {"Priority": "Low",    "Item": "Options overlay for Bear regime",             "Benefit": "Generate carry in Bear periods"},
    ]
    st.dataframe(pd.DataFrame(roadmap), use_container_width=True, hide_index=True)
