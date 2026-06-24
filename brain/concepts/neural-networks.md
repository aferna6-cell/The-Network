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

## The honest experiments now live here (optional)

Rather than argue about it, the repo now *measures* it. Both are optional —
they need `requirements-deep.txt` (torch + transformers) and run under the exact
same leak-free walk-forward and cost model as the tree, so the comparison is
fair:

- **Sequence model** (`src/deep.py`): an LSTM fed the *same* features as the
  tree, only as an ordered `SEQ_WINDOW`-day window. `head_to_head()` scores it
  against the tree on a shared out-of-sample window.
  Run: `python scripts/deep_experiment.py`.
- **FinBERT sentiment** (`src/finbert.py` + `src/sentiment_features.py`):
  finance-tuned transformer scores headlines; a point-in-time trailing mean
  becomes a model feature. `scripts/sentiment_experiment.py --finbert` A/Bs it
  against the price-only baseline.

The prior stands until the numbers overturn it: on daily bars, after costs, the
tree is expected to win and the sentiment feature only matters once enough
**forward** news accumulates (today's log is weeks deep, not years). The
experiments exist to disprove the prior honestly, not to dress the repo in deep
learning. See [[Data Leakage]] — the scaler is fit on train rows only and a
prediction window for day *t* uses only rows ≤ *t*.

## Pitfall

A net will happily memorise leaked future information and post a fantastic
backtest. See [[Data Leakage]] and [[Backtesting Pitfalls]] — the model class
does not save you from a broken evaluation.
