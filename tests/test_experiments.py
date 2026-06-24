"""Experiment harness: position logic + leak-free evaluation on synthetic data."""
from __future__ import annotations

import pandas as pd

from src import experiments as ex


def test_binary_positions_threshold():
    cfg = ex.StrategyConfig("t", buy_threshold=0.55, sizing="binary")
    prob = pd.Series([0.4, 0.55, 0.56, 0.9])
    pos = ex._positions(prob, cfg)
    assert list(pos) == [0.0, 0.0, 1.0, 1.0]  # strictly above threshold


def test_confidence_positions_scale_and_clip():
    cfg = ex.StrategyConfig("t", sizing="confidence")
    prob = pd.Series([0.5, 0.6, 0.75, 1.0, 0.3])
    pos = ex._positions(prob, cfg)
    # size = (p-0.5)*2 clipped to [0,1]; below 0.5 -> 0.
    assert abs(pos.iloc[1] - 0.2) < 1e-9
    assert abs(pos.iloc[2] - 0.5) < 1e-9
    assert pos.iloc[3] == 1.0
    assert pos.iloc[0] == 0.0 and pos.iloc[4] == 0.0


def test_evaluate_returns_metrics(synthetic_prices):
    cfg = ex.StrategyConfig("baseline", horizon=1, buy_threshold=0.55)
    res = ex.evaluate(synthetic_prices, cfg)
    assert res is not None
    assert res["n_oos"] > 0
    for key in ("cagr", "sharpe", "max_drawdown"):
        assert key in res["strat"] and key in res["bench"]
    assert 0.0 <= res["avg_exposure"] <= 1.0


def test_regime_filter_reduces_exposure(synthetic_prices):
    base = ex.evaluate(synthetic_prices, ex.StrategyConfig("b", use_regime=False))
    gated = ex.evaluate(synthetic_prices, ex.StrategyConfig("g", use_regime=True))
    # Gating can only remove long days, never add them.
    assert gated["avg_exposure"] <= base["avg_exposure"] + 1e-9
