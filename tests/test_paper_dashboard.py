"""Tests for the static paper-trading dashboard renderer (no network, no deps)."""
from __future__ import annotations

import pandas as pd

from src import paper_dashboard


def _snapshot(**over):
    base = {
        "generated_at": "2026-06-24T21:08:37+00:00",
        "inception_date": "2026-06-24",
        "starting_equity": 500.0,
        "equity": 512.34,
        "cash": 4.65,
        "total_return": 0.02468,
        "benchmark_equity": 505.0,
        "benchmark_return": 0.01,
        "vs_benchmark": 7.34,
        "holdings": [
            {"ticker": "AAPL", "shares": 0.3334, "value": 100.0, "unrealized_pct": 0.05},
            {"ticker": "XOM", "shares": 0.7289, "value": 99.0, "unrealized_pct": -0.02},
        ],
    }
    base.update(over)
    return base


def _equity(n=4):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-06-24", periods=n, freq="D"),
        "equity": [500 + i for i in range(n)],
        "benchmark_equity": [500 - i for i in range(n)],
    })


def test_renders_key_metrics_and_chart():
    html = paper_dashboard.render_dashboard(_snapshot(), _equity())
    assert html.startswith("<!doctype html>")
    assert "$512.34" in html          # equity
    assert "+2.47%" in html           # total return
    assert "$+7.34" in html           # vs SPY (positive)
    assert "<svg" in html and "<polyline" in html
    assert "AAPL" in html and "XOM" in html
    assert "Fake money" in html       # honesty banner survives


def test_behind_benchmark_marks_negative():
    html = paper_dashboard.render_dashboard(
        _snapshot(vs_benchmark=-3.2, equity=490.0, total_return=-0.02), _equity())
    assert "$-3.20" in html
    assert "behind" in html


def test_single_point_shows_message_not_chart():
    html = paper_dashboard.render_dashboard(_snapshot(), _equity(n=1))
    assert "appears once" in html
    assert "<polyline" not in html


def test_no_holdings_renders_all_cash():
    html = paper_dashboard.render_dashboard(_snapshot(holdings=[]), _equity())
    assert "All cash" in html


def test_build_writes_file(tmp_path, monkeypatch):
    import json
    import config
    snap = tmp_path / "snap.json"
    eq = tmp_path / "eq.csv"
    snap.write_text(json.dumps(_snapshot()))
    _equity().to_csv(eq, index=False)
    monkeypatch.setattr(config, "PAPER_SNAPSHOT_PATH", snap)
    monkeypatch.setattr(config, "PAPER_EQUITY_PATH", eq)
    out = paper_dashboard.build(tmp_path / "index.html")
    assert out.exists() and out.read_text().startswith("<!doctype html>")
