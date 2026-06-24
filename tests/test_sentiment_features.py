"""The point-in-time contract for news sentiment features.

If these pass, a headline can never influence a feature on a day before it was
knowable — the single property that keeps the sentiment experiment honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sentiment_features import (
    SENTIMENT_FEATURE_COLUMNS,
    attach_sentiment,
    daily_sentiment_features,
)


def _news(rows):
    return pd.DataFrame(rows, columns=["published", "title", "sentiment"])


def test_empty_news_yields_neutral_zero():
    idx = pd.bdate_range("2024-01-01", periods=5)
    out = daily_sentiment_features(_news([]), idx)
    assert list(out.columns) == SENTIMENT_FEATURE_COLUMNS
    assert (out["sent_mean"] == 0.0).all()
    assert (out["sent_count"] == 0.0).all()


def test_future_headline_never_leaks_into_past():
    # A strongly positive headline published 2024-01-10 must not move any
    # feature dated on/before 2024-01-10 (lag=1 pushes first use to 01-11).
    idx = pd.bdate_range("2024-01-08", "2024-01-12")
    news = _news([["2024-01-10", "great news", 0.9]])
    out = daily_sentiment_features(news, idx, window=5, lag_days=1)
    assert out.loc[:"2024-01-10", "sent_mean"].eq(0.0).all()
    assert out.loc["2024-01-11", "sent_mean"] == 0.9
    assert out.loc["2024-01-11", "sent_count"] == 1.0


def test_window_and_lag_bound_the_lookback():
    idx = pd.to_datetime(["2024-01-20"])
    news = _news([
        ["2024-01-18", "in window", 1.0],   # within [13..19] with lag1/window5? hi=19 lo=15 -> in
        ["2024-01-10", "too old", -1.0],     # before lo -> excluded
    ])
    out = daily_sentiment_features(news, idx, window=5, lag_days=1)
    assert out.loc["2024-01-20", "sent_count"] == 1.0
    assert out.loc["2024-01-20", "sent_mean"] == 1.0


def test_mean_is_over_headlines_not_daily_means():
    idx = pd.to_datetime(["2024-01-20"])
    news = _news([
        ["2024-01-18", "a", 1.0],
        ["2024-01-18", "b", 0.0],
        ["2024-01-19", "c", -1.0],
    ])
    out = daily_sentiment_features(news, idx, window=5, lag_days=1)
    assert out.loc["2024-01-20", "sent_count"] == 3.0
    assert np.isclose(out.loc["2024-01-20", "sent_mean"], 0.0)


def test_attach_sentiment_fills_every_row():
    idx = pd.bdate_range("2024-01-01", periods=10)
    dataset = pd.DataFrame({"ret_1": np.arange(10.0)}, index=idx)
    out = attach_sentiment(dataset, _news([["2024-01-05", "x", 0.5]]))
    assert not out[SENTIMENT_FEATURE_COLUMNS].isna().any().any()
    assert len(out) == len(dataset)
