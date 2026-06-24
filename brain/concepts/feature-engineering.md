# Feature Engineering

Turning raw OHLCV bars into a model-ready matrix. In this repo it is the most
correctness-critical step and lives in `src/features.py`.

## Principles we follow

1. **Point-in-time only.** Feature on row `t` uses data ≤ `t`. The target uses
   data strictly after `t`. Violating this is [[Data Leakage]].
2. **Stationarity-ish.** Prefer ratios and returns over raw price levels. A
   model trained on $20 AAPL must still work at $200, so we feed `price/SMA - 1`,
   not `price`.
3. **Normalise level-dependent signals.** MACD is divided by price so its scale
   is comparable across time and tickers.
4. **Drop warm-up rows, not fill them.** The first ~50 rows lack a full window;
   we drop them rather than back-fill (which would leak).

## Target construction

`add_target()` builds `fwd_return = price[t+h]/price[t] - 1` and a binary
`target`. The last `h` rows have no future and are set NaN, then dropped — so the
model never trains on a fabricated label. See [[Walk-Forward Validation]] for how
the train/test boundary is additionally purged.

## Why it matters more than the model

Swapping [[Gradient Boosting]] for a [[Neural Networks]] model changes results at
the margin. A single leaked feature changes them by orders of magnitude — in the
wrong direction once real money is involved. Get this right first.
