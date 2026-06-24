"""Refresh the per-ticker news log. Run locally or by the scheduled GitHub Action.

Usage:
    python scripts/update_news.py                    # default tickers
    python scripts/update_news.py --tickers AAPL TSLA
    python scripts/update_news.py --universe         # the full ranked universe

Appends new, de-duplicated, sentiment-scored headlines to brain/news/<TICKER>.csv
and prints a short sentiment snapshot per ticker. This is the ONLY part of the
brain that runs on a recurring schedule.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import news  # noqa: E402
from src.cross_sectional import UNIVERSE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS)
    parser.add_argument(
        "--universe", action="store_true",
        help="refresh news for the full ranked universe (src/cross_sectional.UNIVERSE) "
             "— the candidate set the paper demo holds from, so forward news "
             "accumulates for every name it might trade.",
    )
    args = parser.parse_args()

    tickers = UNIVERSE if args.universe else args.tickers
    for ticker in tickers:
        try:
            news.update_news_log(ticker)
            snap = news.sentiment_summary(ticker)
        except Exception as err:
            print(f"  {ticker}: ERROR {err}")
            continue
        print(f"  {ticker}: {snap['n_headlines']} headlines stored, "
              f"mean sentiment {snap.get('mean_sentiment', 'n/a')} "
              f"(+{snap.get('positive', 0)}/~{snap.get('neutral', 0)}/"
              f"-{snap.get('negative', 0)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
