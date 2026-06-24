"""Pure runtime helpers: market-hours gate + retrain cadence."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src import runtime
from src.model import TrainedModel


def test_market_open_weekday_midday():
    # 2025-06-23 is a Monday. 14:00 UTC = 10:00 ET -> open.
    assert runtime.is_market_open(datetime(2025, 6, 23, 14, 0, tzinfo=timezone.utc))


def test_market_closed_weekend():
    # 2025-06-21 is a Saturday.
    assert not runtime.is_market_open(datetime(2025, 6, 21, 14, 0, tzinfo=timezone.utc))


def test_market_closed_after_hours():
    # 2025-06-23 Monday, 21:00 UTC = 17:00 ET -> closed.
    assert not runtime.is_market_open(datetime(2025, 6, 23, 21, 0, tzinfo=timezone.utc))


def _model(trained_at):
    return TrainedModel(estimator=None, feature_columns=[], horizon=1,
                        threshold=0.0, trained_at=trained_at)


def test_retrain_due_when_no_model():
    assert runtime.retrain_due(None)


def test_retrain_not_due_for_fresh_model():
    now = datetime(2025, 6, 23, 12, 0, tzinfo=timezone.utc)
    fresh = _model((now - timedelta(hours=1)).isoformat())
    assert not runtime.retrain_due(fresh, now=now)


def test_retrain_due_for_old_model():
    now = datetime(2025, 6, 23, 12, 0, tzinfo=timezone.utc)
    old = _model((now - timedelta(hours=48)).isoformat())
    assert runtime.retrain_due(old, now=now)
