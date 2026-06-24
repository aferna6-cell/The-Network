"""Sequence model (LSTM) vs gradient-boosted tree — honest head-to-head.

Both models see the SAME leak-free features under the SAME walk-forward and the
SAME costs; only the model class differs. We run once and print whatever
happens — including "the tree still wins", which on daily bars is the expected
result and not a failure to hide.

Usage:
    pip install -r requirements.txt -r requirements-deep.txt   # needs torch
    python scripts/deep_experiment.py
    python scripts/deep_experiment.py --tickers AAPL MSFT SPY NVDA --refresh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import data, deep  # noqa: E402


def _pct(x: float) -> str:
    return f"{x * 100:+6.1f}%"


def _row(name: str, m: dict) -> str:
    s, b = m["strat"], m["bench"]
    return (f"{name:14}{_pct(s['cagr']):>11}{s['sharpe']:>9.2f}"
            f"{_pct(s['max_drawdown']):>10}{_pct(m['avg_exposure']):>10}"
            f"{_pct(s['cagr'] - b['cagr']):>10}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    print(f"LSTM vs tree on {', '.join(args.tickers)} "
          f"(window={config.SEQ_WINDOW}, retrain every {config.RETRAIN_EVERY}d)\n")
    hdr = (f"{'model':14}{'CAGR':>11}{'Sharpe':>9}{'maxDD':>10}"
           f"{'exposure':>10}{'vs B&H':>10}")

    tree_wins = lstm_wins = 0
    for ticker in args.tickers:
        prices = data.load_or_fetch(ticker, refresh=args.refresh)
        try:
            res = deep.head_to_head(prices)
        except ValueError as err:
            print(f"{ticker}: skipped ({err})")
            continue
        print(f"== {ticker}  ({res['n_shared_oos']} shared OOS days) ==")
        print(hdr)
        print("-" * len(hdr))
        print(_row("tree (GBM)", res["tree"]))
        print(_row("lstm", res["lstm"]))
        # Buy-and-hold is identical for both; compare strat CAGR head to head.
        if res["lstm"]["strat"]["cagr"] > res["tree"]["strat"]["cagr"]:
            lstm_wins += 1
        else:
            tree_wins += 1
        print()

    print(f"Head-to-head CAGR: tree ahead on {tree_wins}, lstm ahead on {lstm_wins}.")
    print("Note: a higher-CAGR model that also carries more drawdown has not "
          "necessarily 'won' — read Sharpe and maxDD too. After costs, daily-bar "
          "sequence models usually do NOT beat the tree; that is the honest prior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
