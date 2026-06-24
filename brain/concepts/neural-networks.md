# Neural Networks

A neural network is a stack of layers that learn a function from inputs to
outputs by adjusting weights via gradient descent on a loss. Each layer applies
a linear transform followed by a non-linearity; depth lets the network compose
simple patterns into complex ones.

## Why this repo does NOT use one (yet)

For daily-bar stock prediction with a few dozen [[Technical Indicators]] as
inputs, neural nets are the wrong tool:

- **Data is small and noisy.** We have thousands of daily rows, not millions.
  Nets overfit badly in low signal-to-noise regimes — exactly the market case
  (see [[Efficient Market Hypothesis]]).
- **Tabular features favour trees.** On tabular data, gradient-boosted trees
  ([[Gradient Boosting]]) usually match or beat nets with far less tuning and no
  GPU. That is what we use.
- **Interpretability.** A recommendation tool should explain itself; trees give
  feature importances cheaply.

## When a net would earn its place

- Sequence models (LSTM/Temporal CNN/Transformer) on **intraday** or **tick**
  data where there is enough volume to learn temporal structure.
- Fusing heterogeneous inputs (price + text embeddings from [[News Sentiment]]).
- Only after the simple baseline is beaten honestly under [[Walk-Forward Validation]].

## Pitfall

A net will happily memorise leaked future information and post a fantastic
backtest. See [[Data Leakage]] and [[Backtesting Pitfalls]] — the model class
does not save you from a broken evaluation.
