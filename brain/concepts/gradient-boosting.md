# Gradient Boosting

Gradient boosting builds an ensemble of shallow decision trees sequentially:
each new tree fits the residual errors of the current ensemble. The sum of many
weak learners becomes a strong one. It is the workhorse for tabular prediction.

## What this repo uses

`sklearn.ensemble.HistGradientBoostingClassifier` (see `src/model.py`):

- **Histogram-based** splitting — fast, handles thousands of rows trivially.
- Native handling of missing values.
- Built-in early stopping via a validation slice.
- Outputs class probabilities we threshold into Buy / Hold / Sell.

Config lives in `config.MODEL_PARAMS`: a small `learning_rate` (0.05), shallow
trees (`max_depth=4`), and `l2_regularization` to fight overfitting on noisy
[[Technical Indicators]].

## Why boosting over a [[Neural Networks]] model here

- Strong out of the box on tabular [[Feature Engineering]] outputs.
- Cheap to retrain every step under [[Walk-Forward Validation]].
- Feature importances aid interpretability.

## The catch

Boosting is powerful enough to fit noise. Guardrails: regularisation, shallow
depth, early stopping, and — most importantly — an honest evaluation that
respects [[Data Leakage]] rules and charges costs ([[Backtesting Pitfalls]]).
A high in-sample accuracy means nothing; only out-of-sample [[Risk Metrics]] do.
