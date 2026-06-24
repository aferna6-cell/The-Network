"""Portfolio-aware recommendation logic."""
from __future__ import annotations

import pandas as pd

import config
from src import features, recommend
from src.model import train_model
from src.portfolio import Portfolio, Position


def _model(synthetic_prices):
    ds = features.make_dataset(synthetic_prices)
    return train_model(ds[features.FEATURE_COLUMNS], ds["target"])


def test_action_bands():
    assert recommend._action_from_prob(0.65) == "STRONG_BUY"
    assert recommend._action_from_prob(0.57) == "BUY"
    assert recommend._action_from_prob(0.50) == "HOLD"
    assert recommend._action_from_prob(0.43) == "SELL"
    assert recommend._action_from_prob(0.35) == "STRONG_SELL"


def test_recommend_shapes_output(synthetic_prices):
    model = _model(synthetic_prices)
    row = features.latest_feature_row(synthetic_prices)
    port = Portfolio(cash=0.0, positions={
        "AAPL": Position("AAPL", shares=10, cost_basis=100.0)})
    recs = recommend.recommend(
        {"AAPL": model}, {"AAPL": row}, {"AAPL": 120.0}, port)
    assert len(recs) == 1
    r = recs[0]
    assert r.held is True
    assert 0.0 <= r.confidence <= 1.0
    assert r.action in {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}


def test_suppress_forces_hold(synthetic_prices):
    model = _model(synthetic_prices)
    row = features.latest_feature_row(synthetic_prices)
    port = Portfolio(cash=1000.0, positions={})
    recs = recommend.recommend(
        {"AAPL": model}, {"AAPL": row}, {"AAPL": 120.0}, port, suppress=True)
    assert recs[0].action == "HOLD"
    assert "health degraded" in recs[0].rationale


def test_concentration_flag_in_rationale(synthetic_prices):
    model = _model(synthetic_prices)
    row = features.latest_feature_row(synthetic_prices)
    # Single holding = 100% weight -> concentrated.
    port = Portfolio(cash=0.0, positions={
        "AAPL": Position("AAPL", shares=10, cost_basis=100.0)})
    recs = recommend.recommend(
        {"AAPL": model}, {"AAPL": row}, {"AAPL": 120.0}, port)
    assert "concentrated" in recs[0].rationale
