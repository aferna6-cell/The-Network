"""Run the pre-specified strategy experiments and print an honest comparison.

Usage:
    python scripts/experiment.py
    python scripts/experiment.py --tickers AAPL MSFT SPY NVDA --refresh

Each config is fixed up front (src/experiments.py). We run once and report
whatever happens — including "still loses to buy-and-hold".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import experiments  # noqa: E402


def _pct(x: float) -> str:
    return f"{x * 100:+6.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    print(f"Running {len(experiments.EXPERIMENTS)} experiments on "
          f"{', '.join(args.tickers)} ...\n")
    out = experiments.compare(args.tickers, refresh=args.refresh)
    summary = out["summary"]

    hdr = (f"{'experiment':18}{'strat CAGR':>12}{'B&H CAGR':>11}"
           f"{'excess':>9}{'strat Shrp':>11}{'B&H Shrp':>10}"
           f"{'strat DD':>10}{'exposure':>10}{'beats B&H':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, s in summary.items():
        print(f"{name:18}"
              f"{_pct(s['mean_strat_cagr']):>12}"
              f"{_pct(s['mean_bench_cagr']):>11}"
              f"{_pct(s['mean_excess_cagr']):>9}"
              f"{s['mean_strat_sharpe']:>11.2f}"
              f"{s['mean_bench_sharpe']:>10.2f}"
              f"{_pct(s['mean_strat_maxdd']):>10}"
              f"{_pct(s['mean_exposure']):>10}"
              f"{s['beats_bh_count']:>6}/{s['n_tickers']}")

    print("\nNotes:")
    print("- Averages across tickers. 'excess' = strat CAGR − buy&hold CAGR.")
    print("- 'beats B&H' counts tickers where the strategy's CAGR exceeded buy&hold.")
    print("- Positive Sharpe with much lower drawdown can be valuable even if CAGR")
    print("  trails B&H (risk-adjusted). Read all columns, not just CAGR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
