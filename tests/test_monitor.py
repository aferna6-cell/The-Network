"""Decay monitor: warming up, healthy, decayed, stale."""
from __future__ import annotations

from datetime import datetime, timezone

import config
from src import ledger, monitor


def _seed_ledger(tmp_path, monkeypatch, results):
    """results: list of (prob_up, went_up). Writes a reconciled ledger."""
    path = tmp_path / "led.csv"
    monkeypatch.setattr(config, "LEDGER_PATH", path)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i, (prob, up) in enumerate(results):
        ledger.append_prediction(
            ticker=f"T{i}", horizon_days=1, price_at_pred=100.0, prob_up=prob,
            recommendation="BUY", model_version="v1", path=path, now=now)
    final = lambda t, up=dict((f"T{i}", u) for i, (_, u) in enumerate(results)): \
        110.0 if up[t] else 90.0
    ledger.reconcile(final, path=path,
                     now=datetime(2025, 1, 5, tzinfo=timezone.utc))
    return path


def test_warming_up(tmp_path, monkeypatch):
    _seed_ledger(tmp_path, monkeypatch, [])
    rep = monitor.health_report(None)
    assert rep["live_accuracy"] is None
    assert "warming up" in rep["status"]
    assert not monitor.should_suppress(rep)


def test_healthy(tmp_path, monkeypatch):
    # 30 correct calls (prob up, went up).
    _seed_ledger(tmp_path, monkeypatch, [(0.8, True)] * 30)
    now = datetime(2025, 1, 1, 1, tzinfo=timezone.utc)
    rep = monitor.health_report("2025-01-01T00:00:00+00:00", now=now)
    assert rep["live_accuracy"] == 1.0
    assert not rep["decayed"] and not monitor.should_suppress(rep)


def test_decayed_suppresses(tmp_path, monkeypatch):
    # 40 calls all wrong (leaned up, went down) -> accuracy 0.
    _seed_ledger(tmp_path, monkeypatch, [(0.8, False)] * 40)
    rep = monitor.health_report("2025-01-01T00:00:00+00:00",
                                now=datetime(2025, 1, 1, 1, tzinfo=timezone.utc))
    assert rep["decayed"]
    assert monitor.should_suppress(rep)
    assert "DECAYED" in rep["status"]


def test_stale_flag(tmp_path, monkeypatch):
    _seed_ledger(tmp_path, monkeypatch, [(0.8, True)] * 30)
    # Model trained 100h before "now" -> stale.
    rep = monitor.health_report("2025-01-01T00:00:00+00:00",
                                now=datetime(2025, 1, 5, 4, tzinfo=timezone.utc))
    assert rep["stale"]
