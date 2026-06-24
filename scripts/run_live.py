"""Entry point for the always-on recommendation loop.

Usage:
    python scripts/run_live.py            # run forever (for a persistent host)
    python scripts/run_live.py --once     # single pass (testing / cron)
    python scripts/run_live.py --tickers AAPL MSFT NVDA

Reads your holdings from portfolio.json (see portfolio.example.json), refreshes
data + prices, retrains on cadence, and writes recommendations to
state/recommendations.json. Places no trades.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS)
    parser.add_argument("--once", action="store_true",
                        help="run a single pass instead of looping forever")
    args = parser.parse_args()

    if args.once:
        snap = runtime.run_once(args.tickers)
        print(f"\nHealth: {snap['health']['status']}")
        print(f"Portfolio value: ${snap['portfolio_value']:,.2f}")
        for r in snap["recommendations"]:
            held = "held" if r["held"] else "watch"
            print(f"  {r['ticker']:6} {r['action']:11} "
                  f"p(up)={r['prob_up']:.2f} conf={r['confidence']:.2f} "
                  f"[{held}] — {r['rationale']}")
    else:
        runtime.run_loop(args.tickers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
