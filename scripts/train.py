"""Train + walk-forward backtest the model on historical data.

Usage:
    python scripts/train.py                       # default tickers
    python scripts/train.py --tickers AAPL MSFT   # custom universe
    python scripts/train.py --refresh             # force fresh download
    python scripts/train.py --horizon 5           # 5-day forward target

For each ticker it: downloads history, builds leak-free features, runs a
walk-forward out-of-sample backtest (with costs), prints an honest report vs
buy-and-hold, then trains a final model on all available data and saves it to
models/<TICKER>.joblib for the recommendation layer to use.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import data, features, model, backtest  # noqa: E402


def _pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def _report(ticker: str, summary: dict) -> str:
    s, b = summary["strategy"], summary["benchmark"]
    lines = [
        f"\n===== {ticker} =====",
        f"Out-of-sample days : {summary['n_oos_days']}",
        f"Directional acc.   : {_pct(summary['directional_accuracy'])}  "
        f"(50% = coin flip)",
        f"Trades             : {summary['n_trades']}  "
        f"(in market {summary['days_in_market']} days)",
        f"Win rate (in mkt)  : {_pct(summary['win_rate'])}",
        "",
        f"{'':18}{'Strategy':>12}{'Buy & Hold':>14}",
        f"{'Total return':18}{_pct(s['total_return']):>12}{_pct(b['total_return']):>14}",
        f"{'CAGR':18}{_pct(s['cagr']):>12}{_pct(b['cagr']):>14}",
        f"{'Ann. volatility':18}{_pct(s['ann_volatility']):>12}{_pct(b['ann_volatility']):>14}",
        f"{'Sharpe':18}{s['sharpe']:>12.2f}{b['sharpe']:>14.2f}",
        f"{'Max drawdown':18}{_pct(s['max_drawdown']):>12}{_pct(b['max_drawdown']):>14}",
        f"\nExcess CAGR vs B&H : {_pct(summary['excess_cagr'])}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS)
    parser.add_argument("--horizon", type=int, default=config.TARGET_HORIZON)
    parser.add_argument("--threshold", type=float, default=config.TARGET_THRESHOLD)
    parser.add_argument("--refresh", action="store_true",
                        help="force fresh download instead of using the cache")
    parser.add_argument("--long-short", action="store_true",
                        help="allow short positions (default is long-only)")
    args = parser.parse_args()

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_reports = []

    for ticker in args.tickers:
        print(f"\nLoading {ticker} ...", flush=True)
        try:
            prices = data.load_or_fetch(ticker, refresh=args.refresh)
        except Exception as err:
            print(f"  SKIP {ticker}: {err}")
            continue

        dataset = features.make_dataset(prices, horizon=args.horizon,
                                        threshold=args.threshold)
        print(f"  {len(prices)} bars -> {len(dataset)} usable rows", flush=True)

        try:
            preds = backtest.walk_forward_predict(
                dataset, horizon=args.horizon, threshold=args.threshold)
        except ValueError as err:
            print(f"  SKIP {ticker}: {err}")
            continue

        bt = backtest.run_backtest(preds, long_short=args.long_short)
        summary = backtest.summarize(bt)
        report = _report(ticker, summary)
        print(report)
        all_reports.append(report)

        # Train a final model on ALL data for live recommendations.
        final = model.train_model(
            dataset[features.FEATURE_COLUMNS], dataset["target"],
            horizon=args.horizon, threshold=args.threshold)
        path = model.save_model(final, config.MODELS_DIR / f"{ticker.upper()}.joblib")
        print(f"  saved model -> {path.relative_to(config.ROOT)}")

    if all_reports:
        report_path = config.REPORTS_DIR / "backtest_report.txt"
        report_path.write_text("\n".join(all_reports), encoding="utf-8")
        print(f"\nFull report written to {report_path.relative_to(config.ROOT)}")
    print("\nReminder: ~50-55% directional accuracy is normal. The framework's "
          "value is honesty, not a guaranteed edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
