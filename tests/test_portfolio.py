"""Portfolio loading, valuation, weights, concentration."""
from __future__ import annotations

import json

from src import portfolio as pf


def _write(tmp_path, data):
    p = tmp_path / "portfolio.json"
    p.write_text(json.dumps(data))
    return p


def test_load_and_value(tmp_path):
    path = _write(tmp_path, {
        "cash": 1000.0,
        "positions": {"AAPL": {"shares": 10, "cost_basis": 100.0}},
    })
    port = pf.load_portfolio(path)
    prices = {"AAPL": 150.0}
    assert port.holds("AAPL")
    assert port.total_value(prices) == 1000.0 + 10 * 150.0
    assert port.positions["AAPL"].unrealized_pnl(150.0) == 500.0


def test_weights_and_concentration(tmp_path):
    path = _write(tmp_path, {
        "cash": 0.0,
        "positions": {
            "AAPL": {"shares": 9, "cost_basis": 100.0},   # 900
            "MSFT": {"shares": 1, "cost_basis": 100.0},   # 100
        },
    })
    port = pf.load_portfolio(path)
    prices = {"AAPL": 100.0, "MSFT": 100.0}
    weights = port.weights(prices)
    assert abs(weights["AAPL"] - 0.9) < 1e-9
    assert "AAPL" in port.concentrated(prices, limit=0.25)
    assert "MSFT" not in port.concentrated(prices, limit=0.25)


def test_missing_file_is_empty_portfolio(tmp_path):
    port = pf.load_portfolio(tmp_path / "nope.json")
    assert port.cash == 0.0 and port.positions == {}


def test_zero_share_positions_dropped(tmp_path):
    path = _write(tmp_path, {
        "cash": 0.0, "positions": {"AAPL": {"shares": 0, "cost_basis": 1.0}},
    })
    assert pf.load_portfolio(path).positions == {}
