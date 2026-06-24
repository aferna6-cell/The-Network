"""Paper account: execution, valuation, benchmark, rebalancing, persistence."""
from __future__ import annotations

from datetime import datetime, timezone

import config
from src import paper, paper_runtime


def _acct(cash=500.0):
    return paper.new_account(cash=cash, now=datetime(2025, 1, 1, tzinfo=timezone.utc))


def test_buy_reduces_cash_and_adds_shares():
    a = _acct()
    cost = a.trade("AAPL", 2.0, 100.0)          # buy 2 @ 100
    assert a.positions["AAPL"]["shares"] == 2.0
    assert a.positions["AAPL"]["cost_basis"] == 100.0
    # cash = 500 - 200 - cost
    assert abs(a.cash - (500.0 - 200.0 - cost)) < 1e-9
    assert cost > 0


def test_sell_increases_cash_and_closes():
    a = _acct()
    a.trade("AAPL", 2.0, 100.0)
    cash_after_buy = a.cash
    a.trade("AAPL", -2.0, 110.0)                 # sell all @ 110
    assert "AAPL" not in a.positions
    assert a.cash > cash_after_buy               # sold higher than the remaining cash


def test_equity_and_benchmark():
    a = _acct()
    a.trade("AAPL", 1.0, 100.0)
    assert abs(a.equity({"AAPL": 150.0}) - (a.cash + 150.0)) < 1e-9
    a.ensure_benchmark({"SPY": 50.0})
    assert a.benchmark_shares == 10.0            # 500 / 50
    assert a.benchmark_equity({"SPY": 60.0}) == 600.0


def test_rebalance_reaches_equal_weight():
    a = _acct(500.0)
    prices = {t: 100.0 for t in ["A", "B", "C", "D", "E"]}
    trades = paper_runtime.rebalance(a, ["A", "B", "C", "D", "E"], prices)
    assert len(trades) == 5
    vals = [p["shares"] * 100.0 for p in a.positions.values()]
    # Each name ~ 99% / 5 of equity, equal-weighted.
    assert all(abs(v - vals[0]) < 1e-6 for v in vals)
    assert abs(sum(vals) - 500.0 * config.PAPER_INVEST_FRACTION) < 1.0
    assert a.cash >= 0


def test_rebalance_rotates_out_old_names():
    a = _acct(500.0)
    prices = {t: 100.0 for t in ["A", "B", "C", "D", "E", "F"]}
    paper_runtime.rebalance(a, ["A", "B", "C", "D", "E"], prices)
    paper_runtime.rebalance(a, ["B", "C", "D", "E", "F"], prices)  # drop A, add F
    assert "A" not in a.positions
    assert "F" in a.positions


def test_persistence_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PAPER_ACCOUNT_PATH", tmp_path / "acct.json")
    a = _acct()
    a.trade("AAPL", 1.5, 200.0)
    a.last_rebalance = "2025-01-01T00:00:00+00:00"
    paper.save_account(a)
    b = paper.load_account()
    assert b.cash == a.cash
    assert b.positions["AAPL"]["shares"] == 1.5
    assert b.last_rebalance == a.last_rebalance
