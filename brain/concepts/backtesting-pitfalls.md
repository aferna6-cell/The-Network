# Backtesting Pitfalls

A backtest is a simulation, and simulations lie when you let them. Even with
correct [[Walk-Forward Validation]], these traps remain.

## The big ones

- **Ignoring costs.** Commissions, spread, and slippage turn many "profitable"
  signals into losers. We charge `TRANSACTION_COST_BPS + SLIPPAGE_BPS` on every
  position change in `src/backtest.py`. A strategy that only works at zero cost
  does not work.
- **No benchmark.** Beating cash is easy; beating buy-and-hold is the real bar.
  Every report shows strategy vs benchmark side by side ([[Risk Metrics]]).
- **Overfitting via repeated trials.** Testing 500 indicator combos and keeping
  the best one fits noise. Mitigate with regularised [[Gradient Boosting]], few
  hyperparameters, and treating a single honest run as the result.
- **Look-ahead / survivorship.** Covered under [[Data Leakage]].
- **Tiny samples / one regime.** A strategy tuned on 2010–2020 may die in a new
  regime. More history and multiple tickers help but never fully cure this.
- **Overlapping labels.** With horizon > 1, daily returns overlap and inflate
  apparent Sharpe. We document this; the default horizon = 1 avoids it.

## The honest stance

If a result looks too good, it is leakage or a bug. The point of this repo is a
trustworthy pipeline that reports realistic — often unexciting — numbers, per the
[[Efficient Market Hypothesis]].
