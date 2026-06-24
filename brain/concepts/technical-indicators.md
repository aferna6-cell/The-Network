# Technical Indicators

Indicators are deterministic transforms of past price/volume meant to summarise
trend, momentum, mean-reversion, and volatility. They are the raw material for
[[Feature Engineering]] in this repo (`src/features.py`).

## The ones we compute (all backward-looking)

| Family | Indicator | Reads as |
|---|---|---|
| Returns | lagged % returns (1–20d), momentum/ROC | recent drift |
| Trend | SMA/EMA ratios, 10-vs-50 crossover | above/below trend |
| Momentum | RSI(14) | overbought/oversold |
| Trend+momentum | MACD, signal, histogram | trend acceleration |
| Mean-reversion | Bollinger %b | distance from band |
| Risk | rolling volatility (10/20d), daily range | turbulence |
| Flow | volume vs 20d average volume | participation |

## The non-negotiable rule

Every indicator on day `t` uses **only** data up to and including day `t`. A
rolling mean over the *next* 20 days, or normalising by a full-sample mean, is
[[Data Leakage]] and silently inflates the backtest. The implementation uses
`rolling(...)` and `shift(positive)` exclusively, never `shift(negative)` for a
feature.

## Honest expectation

No single indicator predicts returns reliably; that is the [[Efficient Market
Hypothesis]] at work. Their modest, combined signal is what [[Gradient Boosting]]
tries to extract — and why we never trust an indicator outside [[Walk-Forward
Validation]].
