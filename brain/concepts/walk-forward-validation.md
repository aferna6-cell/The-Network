# Walk-Forward Validation

The only honest way to evaluate a time-series trading model. Standard k-fold
cross-validation **shuffles rows**, which trains on the future to predict the
past — pure [[Data Leakage]]. Walk-forward never does that.

## How it works here (`src/backtest.py`)

1. Sort by date. Train on the first `INITIAL_TRAIN_FRACTION` (60%) of history.
2. Predict the next block of `RETRAIN_EVERY` (~21) days — strictly out of sample.
3. Expand the training window to include that block, retrain, predict the next.
4. Repeat to the end. Concatenate all out-of-sample predictions.

```
[==== train ====][test][== train ==][test] ...
                  ↑ predict here using only data to the left
```

## Purging / embargo

A label on day `t` uses price at `t+h`, so the last `h` rows before each cutoff
straddle the boundary. We drop them from training (`train_end = i - horizon`).
This is López de Prado's purging idea, minimal but correct.

## What you get

Out-of-sample probabilities for thousands of days, fed into [[Risk Metrics]] and
compared to buy-and-hold. This is the number that matters — in-sample accuracy
from [[Gradient Boosting]] is meaningless by comparison.

See also [[Backtesting Pitfalls]] for the costs and biases that walk-forward
alone does not fix.
