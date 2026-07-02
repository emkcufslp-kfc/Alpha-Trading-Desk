# Alpha Trading Desk

## System E — Live Strategy

**Weekly 11-1 Momentum + Daily Regime T+1 Exit + Stop-Loss −10% + Breadth ≥40%**

| Metric | System E (net of costs) | SPY B&H |
|--------|------------------------|---------|
| CAGR (20Y) | **38.4%** | 10.9% |
| Max Drawdown | **−21.6%** | −54.6% |
| Calmar | **1.78** | 0.20 |
| Sharpe | **1.66** | 0.68 |
| Sortino | **2.65** | — |

*20-year backtest 2006–2026, net of 0.15%/trade transaction costs. Survivorship bias applies (2026 NQ100 universe used) — real-world CAGR likely 4–6pts lower.*

---

## Strategy Rules

### Allocation
| Regime | SPY | Active (top-3) | Cash |
|--------|-----|----------------|------|
| **Bull** | 35% | 21.7% each | 0% |
| **Bear** | 10% | 0% | 90% |

### Bull Regime Trigger (all three required — daily monitoring, T+1 execution)
1. NQ100 equal-weight index > its 200-day MA
2. SPY > its own 200-day MA
3. Breadth ≥ 40% (≥40% of NQ100 stocks have positive 11-1 weekly score)

### Momentum Signal (weekly)
- Score = 44-week return − 4-week return (≈ 11-month minus 1-month skip)
- Rank all NQ100 stocks cross-sectionally → buy top-3
- Signal: Friday close → execute Monday open (T+1)

### Stop-Loss
- Any position dropping >10% from entry price → exit next open (T+1)
- Immediately replace with next-best ranked stock not already held

---

## Backtest Validation

### Walk-Forward OOS (4 folds — out-of-sample)
| Period | CAGR | MaxDD | Sharpe |
|--------|------|-------|--------|
| 2010–2014 | 37.5% | −12.7% | 1.88 |
| 2014–2018 | 28.1% | −15.1% | 1.34 |
| 2018–2022 | 40.3% | −25.8% | 1.28 |
| 2022–2026 | 39.1% | −19.6% | 1.49 |

### Bear Market Protection
| Bear Period | Strategy | SPY | Protected |
|------------|----------|-----|-----------|
| GFC 2007–09 | −11.8% | −54.6% | **+42.8%** |
| COVID 2020 | −3.6% | −31.8% | **+28.2%** |
| Bear 2022 | −13.5% | −22.5% | **+9.0%** |

### Transaction Costs Modelled
- Commission: 0.05% per trade (one-way)
- Slippage: 0.10% per trade (one-way)
- Round-trip: 0.30% per position change
- 1,114 trades | 94 stop-loss triggers | 33 regime exits over 20Y
- Annual turnover: 13.9×

### Known Limitations
- **Survivorship bias**: Uses 2026 NQ100 composition. Real-world CAGR ~4–6pts lower.
- **No tax modelling**: All returns are pre-tax.

---

## Files

| File | Description |
|------|-------------|
| `dashboard.html` | Standalone live dashboard — open in any browser |
| `app.py` | Streamlit app with live System E signal computation |
| `backtests/system_e_backtest.py` | Full System E backtest with costs + stop-loss |
| `backtests/backtest20y.py` | 20Y monthly baseline backtest |
| `backtests/maxdd_optimize.py` | MaxDD optimisation grid (~540 configs) |
| `results/system_e_results.json` | System E metrics, OOS folds, bear periods |

## Running

```bash
pip install -r requirement.txt
streamlit run app.py
```

Or open `dashboard.html` directly — no server needed, works offline.
