"""Always-on loop for the cross-sectional recommender.

Usage:
    python scripts/run_live_xs.py            # run forever (persistent host)
    python scripts/run_live_xs.py --once     # single pass (cron / testing)

Refreshes prices every tick (config.LOOP_INTERVAL_SECONDS), retrains the model
once per day, and keeps state/xs_recommendations.json current for the dashboard.
The signal is slow (~1-month horizon) so recommendations won't churn each tick;
the ledger is deduped (one open call per name) so the live track record stays
honest. Places no trades.
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
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    xs_runtime.run_loop(args.tickers, once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
