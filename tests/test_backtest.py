"""Backtest mechanics: cost accounting, metric math, walk-forward integrity."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import features, backtest


def _toy_predictions(probs, fwd_returns):
    idx = pd.bdate_range("2020-01-01", periods=len(probs), name="date")
    return pd.DataFrame(
        {"prob_up": probs,
         "target": (np.array(fwd_returns) > 0).astype(float),
         "fwd_return": fwd_returns},
        index=idx,
    )


def test_costs_reduce_returns():
    """A position change must cost money; holding flat must not."""
    # Alternate signal -> forced trades every step.
    probs = [0.9, 0.1, 0.9, 0.1, 0.9]
    fwd = [0.01, 0.01, 0.01, 0.01, 0.01]
    bt = backtest.run_backtest(_toy_predictions(probs, fwd), buy_threshold=0.5)
    assert (bt["trade_cost"] > 0).any()
    # On days we enter long, net return is gross minus the entry cost.
    entered = bt[(bt["position"] == 1) & (bt["position"].shift(1).fillna(0) == 0)]
    assert (entered["strategy_return"] < entered["fwd_return"]).all()


def test_flat_position_earns_nothing():
    probs = [0.2, 0.3, 0.1]  # never above buy threshold -> always flat
    fwd = [0.05, -0.05, 0.05]
    bt = backtest.run_backtest(_toy_predictions(probs, fwd), buy_threshold=0.5)
    assert (bt["position"] == 0).all()
    assert np.allclose(bt["strategy_return"], 0.0)


def test_perfect_signal_beats_benchmark():
    """A clairvoyant long-only signal should not underperform buy-and-hold."""
    rng = np.random.default_rng(1)
    fwd = rng.normal(0.0, 0.02, 200)
    probs = np.where(fwd > 0, 0.99, 0.01)  # perfectly aligned with the future
    bt = backtest.run_backtest(_toy_predictions(probs, fwd), buy_threshold=0.5)
    summary = backtest.summarize(bt)
    assert summary["strategy"]["total_return"] >= summary["benchmark"]["total_return"]
    # Perfect directional calls -> 100% accuracy.
    assert summary["directional_accuracy"] == 1.0


def test_equity_curve_is_cumulative_product():
    probs = [0.9, 0.9, 0.9]
    fwd = [0.10, -0.05, 0.02]
    bt = backtest.run_backtest(_toy_predictions(probs, fwd), buy_threshold=0.5)
    # Always long, entered once on day 0; later days have no extra cost.
    expected = np.cumprod(1 + bt["strategy_return"].values)
    assert np.allclose(bt["strategy_equity"].values, expected)


def test_walk_forward_is_out_of_sample(synthetic_prices):
    ds = features.make_dataset(synthetic_prices, horizon=1)
    preds = backtest.walk_forward_predict(
        ds, horizon=1, initial_train_fraction=0.6, retrain_every=21)
    # Predictions only exist past the initial training window.
    first_oos = ds.index[int(len(ds) * 0.6)]
    assert preds.index.min() >= first_oos
    assert preds["prob_up"].between(0, 1).all()
    assert len(preds) > 0


def test_walk_forward_rejects_tiny_history():
    tiny = pd.DataFrame({
        "open": [1, 2], "high": [1, 2], "low": [1, 2],
        "close": [1, 2], "adj_close": [1, 2], "volume": [1, 2],
        "target": [0.0, 1.0], "fwd_return": [0.1, -0.1],
        **{c: [0.0, 0.0] for c in features.FEATURE_COLUMNS},
    }, index=pd.bdate_range("2020-01-01", periods=2))
    try:
        backtest.walk_forward_predict(tiny)
        assert False, "expected ValueError on tiny history"
    except ValueError:
        pass
