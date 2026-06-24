"""Prediction ledger: append, reconcile outcomes, live accuracy."""
from __future__ import annotations

from datetime import datetime, timezone

from src import ledger


def _ts(day):
    return datetime(2025, 1, day, 15, 0, tzinfo=timezone.utc)


def test_append_and_load(tmp_path):
    path = tmp_path / "led.csv"
    ledger.append_prediction(
        ticker="AAPL", horizon_days=1, price_at_pred=100.0, prob_up=0.7,
        recommendation="BUY", model_version="v1", path=path, now=_ts(1))
    df = ledger.load_ledger(path)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["resolve_after"] == "2025-01-02"


def test_reconcile_marks_correct(tmp_path):
    path = tmp_path / "led.csv"
    # Bullish call at 100; price later 110 -> correct.
    ledger.append_prediction(
        ticker="AAPL", horizon_days=1, price_at_pred=100.0, prob_up=0.8,
        recommendation="BUY", model_version="v1", path=path, now=_ts(1))
    # Bearish call at 100; price later 110 -> wrong.
    ledger.append_prediction(
        ticker="MSFT", horizon_days=1, price_at_pred=100.0, prob_up=0.2,
        recommendation="SELL", model_version="v1", path=path, now=_ts(1))

    n = ledger.reconcile(lambda t: 110.0, path=path, now=_ts(3))
    assert n == 2
    df = ledger.load_ledger(path)
    by = {r["ticker"]: r for _, r in df.iterrows()}
    assert int(by["AAPL"]["correct"]) == 1   # leaned up, went up
    assert int(by["MSFT"]["correct"]) == 0   # leaned down, went up


def test_reconcile_skips_not_yet_due(tmp_path):
    path = tmp_path / "led.csv"
    ledger.append_prediction(
        ticker="AAPL", horizon_days=5, price_at_pred=100.0, prob_up=0.7,
        recommendation="BUY", model_version="v1", path=path, now=_ts(1))
    # Only 1 day later — horizon not elapsed.
    assert ledger.reconcile(lambda t: 110.0, path=path, now=_ts(2)) == 0


def test_has_open_prediction(tmp_path):
    path = tmp_path / "led.csv"
    assert not ledger.has_open_prediction("AAPL", path)
    ledger.append_prediction(
        ticker="AAPL", horizon_days=1, price_at_pred=100.0, prob_up=0.7,
        recommendation="BUY", model_version="v1", path=path, now=_ts(1))
    assert ledger.has_open_prediction("AAPL", path)      # open
    ledger.reconcile(lambda t: 110.0, path=path, now=_ts(3))
    assert not ledger.has_open_prediction("AAPL", path)  # now reconciled


def test_live_accuracy(tmp_path):
    path = tmp_path / "led.csv"
    for i, (prob, up) in enumerate([(0.8, True), (0.8, False), (0.2, False)]):
        ledger.append_prediction(
            ticker=f"T{i}", horizon_days=1, price_at_pred=100.0, prob_up=prob,
            recommendation="BUY", model_version="v1", path=path, now=_ts(1))
    final = lambda t: 110.0 if t in ("T0",) else 90.0
    ledger.reconcile(final, path=path, now=_ts(3))
    acc = ledger.live_accuracy(path=path)
    # T0 up/up correct, T1 up/down wrong, T2 down/down correct -> 2/3.
    assert acc["n"] == 3
    assert abs(acc["accuracy"] - 2 / 3) < 1e-9
