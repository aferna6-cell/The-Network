# Cross-Sectional Momentum

Cross-sectional ranking asks a different question than single-stock timing.
Instead of "will AAPL go up tomorrow?" it asks "which stocks will *outperform the
others*?" and holds the top-ranked basket, rebalancing periodically. Most durable
equity anomalies — momentum, value, quality — are cross-sectional, which is why
this is a more legitimate place to look for an edge than single-name timing
against the [[Efficient Market Hypothesis]].

## How this repo implements it (`src/cross_sectional.py`)

- Stack every stock's [[Technical Indicators]] into one panel.
- Add **relative** features: each signal ranked (percentile) against the whole
  universe that day. This is the "different data" angle — relative strength, not
  absolute price.
- **Target:** did the stock beat the universe's *median* forward return? A
  relative label, not an up/down call.
- Each rebalance: rank by predicted probability, hold the top quantile
  equal-weight, charge turnover cost. Benchmark = equal-weight of the whole
  universe (the honest "did stock selection add anything?" test) plus SPY.

Leak-free per [[Data Leakage]]: features use only past bars, cross-sectional
ranks use only that day's data, the target uses only future bars as a label, and
[[Walk-Forward Validation]] retrains in time with a purge.

## Why we do NOT use fundamentals (yet)

Fundamentals (P/E, margins, growth) are the obvious "different data," but
**point-in-time** fundamental history — what was *known on each past date*, not
what we know now — is not available for free. Backtesting with *current*
fundamentals is severe look-ahead [[Data Leakage]]: you would be ranking 2016
stocks using 2024 financials. So we refuse to fake it. The honest free options
are relative price/volume features (here) and the point-in-time
[[News Sentiment]] log we are accumulating.

## Honest expectation

Cross-sectional momentum is real but **thin and regime-dependent**, and it gets
eaten by turnover costs (see [[Backtesting Pitfalls]]). Beating an equal-weight
basket after costs is a genuine — and hard — bar. Read [[Risk Metrics]], not just
return.
