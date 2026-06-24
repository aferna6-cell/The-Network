# Risk Metrics

Returns without risk context are meaningless. A 30% return from 60% drawdowns is
worse than 12% from 8% drawdowns. These are the metrics `src/backtest.py`
reports, always alongside a buy-and-hold benchmark.

## What we compute

- **Total return** — cumulative growth of $1 over the out-of-sample period.
- **CAGR** — geometric annual growth rate; comparable across periods of
  different length.
- **Annualised volatility** — std of returns × √252; the "size of the bumps".
- **Sharpe ratio** — mean / std of returns × √252 (risk-free assumed 0). Reward
  per unit of risk. Below ~1 is weak; above ~2 on daily equity signals is rare
  and usually a sign of [[Data Leakage]].
- **Max drawdown** — worst peak-to-trough decline. The number that actually makes
  people capitulate. Watch this more than return.
- **Directional accuracy / win rate** — sanity checks, not the goal. ~52% can be
  profitable; ~55%+ on daily bars is a red flag (see [[Backtesting Pitfalls]]).
- **Excess CAGR vs buy-and-hold** — did the model add anything over doing nothing?

## Why "beat buy-and-hold" is the real test

Holding [[Technical Indicators]] aside, an index fund returns ~7–10%/yr for zero
effort. A model from [[Gradient Boosting]] must clear that *after costs* to be
worth running. Most do not — which is the honest message of the [[Efficient
Market Hypothesis]].
