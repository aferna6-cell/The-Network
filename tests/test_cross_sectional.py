"""Cross-sectional panel construction (leak-free) + backtest mechanics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import cross_sectional as xs
from src.features import FEATURE_COLUMNS


def _synth(seed, n=600):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n, name="date")
    rets = rng.normal(0.0004, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "adj_close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


def _panel(n_tickers=8):
    prices = {f"T{i}": _synth(i) for i in range(n_tickers)}
    return xs.build_panel(prices, horizon=21)


def test_panel_has_xs_features_and_target():
    panel = _panel()
    for col in xs.XS_FEATURES:
        assert col in panel.columns
    assert set(panel["target"].unique()) <= {0.0, 1.0}
    assert not panel[xs.XS_FEATURES].isna().any().any()


def test_target_is_cross_sectional_median():
    """target = beat the per-day median forward return."""
    panel = _panel()
    sample_date = panel["date"].mode().iloc[0]
    day = panel[panel["date"] == sample_date]
    med = day["fwd_return"].median()
    expected = (day["fwd_return"] > med).astype(float)
    assert (day["target"].values == expected.values).all()


def test_xs_ranks_are_percentiles():
    panel = _panel()
    ranks = panel["xs_rank_mom_20"]
    assert ranks.between(0, 1).all()


def test_backtest_runs_and_reports():
    panel = _panel(n_tickers=10)
    res = xs.backtest(panel, horizon=21, initial_fraction=0.6, top_quantile=0.3)
    assert res.metrics  # non-empty
    assert res.metrics["n_rebalances"] > 0
    for key in ("cagr", "sharpe", "max_drawdown"):
        assert key in res.metrics["strategy"]
    assert 0.0 <= res.metrics["avg_turnover"] <= 1.0
