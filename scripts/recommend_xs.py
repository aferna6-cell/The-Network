"""Generate concrete Buy / Sell / Trim recommendations from cross-sectional ranks.

Usage:
    python scripts/recommend_xs.py                 # use default universe + portfolio.json
    python scripts/recommend_xs.py --retrain       # force a fresh model
    python scripts/recommend_xs.py --tickers AAPL MSFT NVDA ...

Reads your holdings (portfolio.json), ranks the universe, and prints what to do —
each call with confidence, rank, and a rationale — plus the model's LIVE track
record so you know how much to trust it. Places no trades.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import cross_sectional as xs, data, runtime, xs_recommend  # noqa: E402
from src.model import load_model, save_model  # noqa: E402
from src.portfolio import load_portfolio  # noqa: E402


def _get_model(prices_by_ticker, retrain: bool):
    path = config.XS_MODEL_PATH
    model = load_model(path) if path.exists() else None
    if retrain or runtime.retrain_due(model):
        print("Training cross-sectional model ...", flush=True)
        model = xs.train_full_model(prices_by_ticker, horizon=config.XS_HORIZON)
        save_model(model, path)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=xs.UNIVERSE)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    portfolio = load_portfolio()
    universe = sorted(set(args.tickers) | set(portfolio.tickers))
    print(f"Ranking {len(universe)} names ...", flush=True)

    prices_by_ticker, prices_now = {}, {}
    for t in universe:
        try:
            prices_by_ticker[t] = data.load_or_fetch(t, start=args.start,
                                                     refresh=args.refresh)
            q = data.latest_quote(t)
            prices_now[t] = q["price"] or float(prices_by_ticker[t]["adj_close"].iloc[-1])
        except Exception as err:
            print(f"  skip {t}: {err}")

    model = _get_model(prices_by_ticker, args.retrain)
    recs = xs_recommend.recommend(prices_by_ticker, portfolio, model, prices_now)
    snap = xs_recommend.log_and_snapshot(
        recs, model, portfolio, prices_now, now=datetime.now(timezone.utc))

    track = snap["live_track_record"]
    tr = (f"{track['accuracy']:.0%} over {track['n']} graded calls"
          if track["accuracy"] is not None else "no graded calls yet — unproven")
    print(f"\nPortfolio value: ${snap['portfolio_value']:,.2f}")
    print(f"Live track record: {tr}\n")

    def show(title, items):
        if not items:
            return
        print(title)
        for r in items:
            print(f"  {r['action']:5} {r['ticker']:6} "
                  f"#{r['rank']}/{r['universe_size']:<3} "
                  f"conf={r['confidence']:.2f}  — {r['rationale']}")
        print()

    actions = snap["actions"]
    show("=== ACTIONS ===", actions)
    show("=== HOLD (keep) ===", snap["holds"])
    if not actions:
        print("No actionable Buy/Sell/Trim today.")

    print("Reminder: decision support, not an oracle. The edge is unproven and the "
          "research backtest is survivorship-biased — weight these by the live "
          "track record above, which only becomes meaningful after many calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
