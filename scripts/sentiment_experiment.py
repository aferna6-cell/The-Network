"""Does news sentiment earn its keep as a model feature? Honest A/B.

Trains the SAME gradient-boosted tree twice under the SAME walk-forward: once on
price features alone (baseline), once on price features + a leak-free trailing
news-sentiment feature. Sentiment scores come from VADER by default or FinBERT
with --finbert (deep learning applied where it is strong — language).

The sentiment feature is strictly point-in-time (src/sentiment_features.py): it
only uses headlines published at least --lag days before each decision date.

Usage:
    python scripts/sentiment_experiment.py --tickers AAPL MSFT
    python scripts/sentiment_experiment.py --finbert      # needs requirements-deep.txt

Requires a committed news log at brain/news/<TICKER>.csv (build it with
scripts/update_news.py). Tickers without news are reported and skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import data, deep, features  # noqa: E402
from src.backtest import walk_forward_predict  # noqa: E402
from src.features import FEATURE_COLUMNS  # noqa: E402
from src.sentiment_features import SENTIMENT_FEATURE_COLUMNS, attach_sentiment  # noqa: E402


def _pct(x: float) -> str:
    return f"{x * 100:+6.1f}%"


def _load_news(ticker: str, use_finbert: bool) -> tuple[pd.DataFrame, str] | None:
    path = config.NEWS_DIR / f"{ticker.upper()}.csv"
    if not path.exists():
        return None
    news = pd.read_csv(path)
    if news.empty:
        return None
    if use_finbert:
        from src import finbert
        news = finbert.rescore_news(news)
        return news, "finbert_sentiment"
    return news, "sentiment"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS)
    parser.add_argument("--finbert", action="store_true",
                        help="score headlines with FinBERT instead of VADER")
    parser.add_argument("--window", type=int, default=config.SENT_WINDOW_DAYS)
    parser.add_argument("--lag", type=int, default=config.SENT_LAG_DAYS)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    scorer = "FinBERT" if args.finbert else "VADER"
    print(f"Sentiment A/B ({scorer}, window={args.window}d, lag={args.lag}d) on "
          f"{', '.join(args.tickers)}\n")
    hdr = f"{'ticker':8}{'model':14}{'CAGR':>11}{'Sharpe':>9}{'vs B&H':>10}{'OOS days':>10}"

    helped = 0
    tested = 0
    for ticker in args.tickers:
        loaded = _load_news(ticker, args.finbert)
        if loaded is None:
            print(f"{ticker}: no news log — run scripts/update_news.py first; skipped")
            continue
        news, sent_col = loaded

        prices = data.load_or_fetch(ticker, refresh=args.refresh)
        dataset = features.make_dataset(prices)
        dataset_s = attach_sentiment(dataset, news, window=args.window,
                                     lag_days=args.lag, sentiment_col=sent_col)
        try:
            base = walk_forward_predict(dataset)
            sent = walk_forward_predict(
                dataset_s, feature_columns=FEATURE_COLUMNS + SENTIMENT_FEATURE_COLUMNS)
        except ValueError as err:
            print(f"{ticker}: skipped ({err})")
            continue

        shared = base.index.intersection(sent.index)
        b = deep.economics(base.loc[shared, "prob_up"], prices)
        s = deep.economics(sent.loc[shared, "prob_up"], prices)
        print(hdr)
        print("-" * len(hdr))
        print(f"{ticker:8}{'baseline':14}{_pct(b['strat']['cagr']):>11}"
              f"{b['strat']['sharpe']:>9.2f}"
              f"{_pct(b['strat']['cagr'] - b['bench']['cagr']):>10}{b['n_oos']:>10}")
        print(f"{'':8}{'+sentiment':14}{_pct(s['strat']['cagr']):>11}"
              f"{s['strat']['sharpe']:>9.2f}"
              f"{_pct(s['strat']['cagr'] - s['bench']['cagr']):>10}{s['n_oos']:>10}\n")
        tested += 1
        if s["strat"]["cagr"] > b["strat"]["cagr"]:
            helped += 1

    if tested:
        print(f"Sentiment improved CAGR on {helped}/{tested} tickers. "
              "One slice of history; treat as indicative, not proof.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
