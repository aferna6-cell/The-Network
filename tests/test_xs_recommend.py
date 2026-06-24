"""Cross-sectional recommender: action mapping + end-to-end on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import cross_sectional as xs, xs_recommend
from src.portfolio import Portfolio, Position


def _synth(seed, n=500):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-01", periods=n, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    return pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "adj_close": close,
        "volume": rng.integers(1e6, 5e6, n).astype(float),
    }, index=dates)


def test_decide_actions():
    d = xs_recommend._decide
    # not held, top quintile -> BUY
    assert d(0.9, held=False, concentrated=False, top_q=0.2, sell_pct=0.4) == "BUY"
    # not held, mediocre -> AVOID
    assert d(0.5, held=False, concentrated=False, top_q=0.2, sell_pct=0.4) == "AVOID"
    # held, bottom -> SELL
    assert d(0.2, held=True, concentrated=False, top_q=0.2, sell_pct=0.4) == "SELL"
    # held, decent -> HOLD
    assert d(0.7, held=True, concentrated=False, top_q=0.2, sell_pct=0.4) == "HOLD"
    # held + concentrated + not top -> TRIM (risk overrides)
    assert d(0.5, held=True, concentrated=True, top_q=0.2, sell_pct=0.4) == "TRIM"


def test_recommend_end_to_end():
    prices = {f"T{i}": _synth(i) for i in range(12)}
    model = xs.train_full_model(prices, horizon=21)
    port = Portfolio(cash=1000.0, positions={
        "T0": Position("T0", shares=10, cost_basis=100.0)})
    prices_now = {t: float(p["adj_close"].iloc[-1]) for t, p in prices.items()}

    recs = xs_recommend.recommend(prices, port, model, prices_now)
    assert len(recs) == 12
    ranks = sorted(r.rank for r in recs)
    assert ranks == list(range(1, 13))            # unique 1..N
    assert all(r.action in {"BUY", "SELL", "TRIM", "HOLD", "AVOID"} for r in recs)
    assert all(0.0 <= r.confidence <= 1.0 for r in recs)


def test_snapshot_writes_actions_and_track(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "XS_LEDGER_PATH", tmp_path / "xsl.csv")
    monkeypatch.setattr(config, "XS_RECOMMENDATIONS_PATH", tmp_path / "xsr.json")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)

    prices = {f"T{i}": _synth(i) for i in range(12)}
    model = xs.train_full_model(prices, horizon=21)
    port = Portfolio(cash=1000.0, positions={})
    prices_now = {t: float(p["adj_close"].iloc[-1]) for t, p in prices.items()}
    recs = xs_recommend.recommend(prices, port, model, prices_now)
    snap = xs_recommend.log_and_snapshot(recs, model, port, prices_now)

    assert "live_track_record" in snap
    assert (tmp_path / "xsr.json").exists()
    # First run has no graded calls yet.
    assert snap["live_track_record"]["accuracy"] is None


def test_frequent_runs_do_not_duplicate_ledger(tmp_path, monkeypatch):
    import config
    import pandas as pd
    from src import ledger
    monkeypatch.setattr(config, "XS_LEDGER_PATH", tmp_path / "xsl.csv")
    monkeypatch.setattr(config, "XS_RECOMMENDATIONS_PATH", tmp_path / "xsr.json")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)

    prices = {f"T{i}": _synth(i) for i in range(12)}
    model = xs.train_full_model(prices, horizon=21)
    port = Portfolio(cash=1000.0, positions={})
    prices_now = {t: float(p["adj_close"].iloc[-1]) for t, p in prices.items()}
    recs = xs_recommend.recommend(prices, port, model, prices_now)

    # Run three times in a row (simulating an always-on loop).
    for _ in range(3):
        xs_recommend.log_and_snapshot(recs, model, port, prices_now)

    led = ledger.load_ledger(config.XS_LEDGER_PATH)
    # No ticker should appear more than once while its call is still open.
    assert led["ticker"].is_unique
