"""Generate concrete Buy / Sell / Trim recommendations from cross-sectional ranks.

Usage:
    python scripts/recommend_xs.py                 # one pass, default universe
    python scripts/recommend_xs.py --retrain       # force a fresh model
    python scripts/recommend_xs.py --tickers AAPL MSFT NVDA ...

Reads your holdings (portfolio.json), ranks the universe, and prints what to do —
each call with confidence, rank, and a rationale — plus the model's LIVE track
record so you know how much to trust it. Places no trades. For the always-on
version, see scripts/run_live_xs.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import xs_runtime  # noqa: E402
from src.cross_sectional import UNIVERSE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=UNIVERSE)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    print(f"Ranking {len(args.tickers)} names ...", flush=True)
    snap = xs_runtime.run_once(args.tickers, start=args.start,
                               retrain=args.retrain, refresh=args.refresh)

    tr = snap["live_track_record"]
    track = (f"{tr['accuracy']:.0%} over {tr['n']} graded calls"
             if tr["accuracy"] is not None else "no graded calls yet — unproven")
    print(f"\nPortfolio value: ${snap['portfolio_value']:,.2f}")
    print(f"Live track record: {track}\n")

    def show(title, items):
        if not items:
            return
        print(title)
        for r in items:
            print(f"  {r['action']:5} {r['ticker']:6} "
                  f"#{r['rank']}/{r['universe_size']:<3} "
                  f"conf={r['confidence']:.2f}  — {r['rationale']}")
        print()

    show("=== ACTIONS ===", snap["actions"])
    show("=== HOLD (keep) ===", snap["holds"])
    if not snap["actions"]:
        print("No actionable Buy/Sell/Trim today.")

    print("Reminder: decision support, not an oracle. The edge is UNPROVEN and the "
          "research backtest is survivorship-biased — weight these by the live "
          "track record above, which only becomes meaningful after many calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
