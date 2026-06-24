# Efficient Market Hypothesis

The EMH holds that prices already reflect available information, so consistently
predicting future returns from public data is very hard. It is the single most
important prior for this whole project — the reason we expect modest results and
distrust impressive ones.

## Practical implications for us

- **Expect ~50–55% directional accuracy.** Anything much higher on daily bars is
  almost certainly [[Data Leakage]] or a [[Backtesting Pitfalls]] artifact, not
  alpha.
- **Costs eat thin edges.** Even a real small edge often vanishes after the
  frictions charged in [[Risk Metrics]] / `src/backtest.py`.
- **Edges decay.** A signal that worked historically gets arbitraged away. Any
  live edge needs continuous [[Walk-Forward Validation]], not a one-time fit.

## Why bother at all, then?

EMH is "mostly true," not "perfectly true." Markets are *near*-efficient:
short-lived inefficiencies exist (momentum, post-earnings drift, sentiment
overreactions — see [[News Sentiment]]). A disciplined pipeline that finds a
small, robust, cost-survivable edge — and honestly reports when it finds nothing
— is worth building. A system that promises to "beat the market" is not.

## Connected reading

[[Gradient Boosting]] · [[Feature Engineering]] · [[Technical Indicators]]
