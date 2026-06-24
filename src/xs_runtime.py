"""Always-on loop for the cross-sectional recommender.

One pass = fetch data + live prices, retrain the cross-sectional model on a daily
cadence (frozen snapshot between), rank the universe, map to portfolio-relative
Buy/Sell/Trim/Hold calls, reconcile + log to the ledger (deduped), and write the
snapshot the dashboard reads.

The signal is slow (~1-month horizon), so running this every few minutes keeps
prices and the dashboard fresh but will NOT churn the recommendations — and the
ledger dedup (one open call per name) keeps the live track record honest no
matter how often the loop runs. Places no trades.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import config
from src import cross_sectional as xs, data, runtime, xs_recommend
from src.model import load_model, save_model
from src.portfolio import load_portfolio


def _get_model(prices_by_ticker: dict, *, retrain: bool):
    path = config.XS_MODEL_PATH
    model = load_model(path) if path.exists() else None
    if retrain or runtime.retrain_due(model):
        model = xs.train_full_model(prices_by_ticker, horizon=config.XS_HORIZON)
        save_model(model, path)
    return model


def run_once(
    tickers: list[str] | None = None,
    *,
    start: str = "2015-01-01",
    retrain: bool = False,
    refresh: bool = True,
    now: datetime | None = None,
) -> dict:
    """One full ranking + recommendation pass. Returns the snapshot dict."""
    now = now or datetime.now(timezone.utc)
    portfolio = load_portfolio()
    universe = sorted(set(tickers or xs.UNIVERSE) | set(portfolio.tickers))

    prices_by_ticker, prices_now = {}, {}
    for t in universe:
        try:
            prices_by_ticker[t] = data.load_or_fetch(t, start=start, refresh=refresh)
            q = data.latest_quote(t)
            prices_now[t] = q["price"] or float(prices_by_ticker[t]["adj_close"].iloc[-1])
        except Exception as err:
            print(f"  skip {t}: {err}")

    model = _get_model(prices_by_ticker, retrain=retrain)
    recs = xs_recommend.recommend(prices_by_ticker, portfolio, model, prices_now)
    return xs_recommend.log_and_snapshot(recs, model, portfolio, prices_now, now=now)


def run_loop(tickers: list[str] | None = None, *, once: bool = False) -> None:
    """Run forever (or a single pass with once=True)."""
    retrained_today = False
    while True:
        now = datetime.now(timezone.utc)
        if config.RUN_OUTSIDE_MARKET_HOURS or runtime.is_market_open(now):
            # Refresh prices every tick; the daily-cadence retrain is handled
            # inside run_once via retrain_due.
            snap = run_once(tickers, now=now)
            n_actions = len(snap["actions"])
            tr = snap["live_track_record"]
            acc = (f"{tr['accuracy']:.0%}/{tr['n']}" if tr["accuracy"] is not None
                   else "unproven")
            print(f"[{now.isoformat()}] {n_actions} actions · track {acc}")
        else:
            print(f"[{now.isoformat()}] market closed — idle")
        if once:
            return
        time.sleep(config.LOOP_INTERVAL_SECONDS)
