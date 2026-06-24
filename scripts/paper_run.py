"""Paper-trading demo — hand the model $500 of fake money and let it run.

Usage:
    python scripts/paper_run.py --reset           # start fresh with $500 cash
    python scripts/paper_run.py --reset --cash 1000
    python scripts/paper_run.py --once            # one tick (mark + rebalance if due)
    python scripts/paper_run.py                    # always-on loop

On --reset the account is wiped and re-funded; the next tick buys the model's
top-K picks. The benchmark buys the same dollars of SPY and holds. No real money.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import paper, paper_runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="wipe + re-fund the account")
    parser.add_argument("--cash", type=float, default=config.PAPER_START_CASH)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.reset:
        acct = paper.new_account(cash=args.cash, now=datetime.now(timezone.utc))
        paper.save_account(acct)
        # Wipe the old equity curve so the chart starts clean.
        config.PAPER_EQUITY_PATH.unlink(missing_ok=True)
        print(f"Reset paper account with ${args.cash:,.2f} cash.")

    if paper.load_account() is None:
        print("No account. Run with --reset first.")
        return 1

    if args.once or args.reset:
        snap = paper_runtime.run_once()
        print(f"\nEquity: ${snap['equity']:,.2f} ({snap['total_return']:+.2%})  |  "
              f"SPY benchmark: ${snap['benchmark_equity']:,.2f} "
              f"({snap['benchmark_return']:+.2%})  |  cash ${snap['cash']:,.2f}")
        if snap["trades_this_tick"]:
            print("\nTrades this tick:")
            for t in snap["trades_this_tick"]:
                print(f"  {t['action']:4} {t['ticker']:6} {t['shares']} @ ${t['price']}")
        if snap["holdings"]:
            print("\nHoldings:")
            for h in snap["holdings"]:
                print(f"  {h['ticker']:6} {h['shares']:>8} sh  "
                      f"${h['value']:>8.2f}  ({h['unrealized_pct']:+.1%})")
        print("\nFake money. The model decides; we watch the forward curve vs SPY.")
    else:
        paper_runtime.run_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
