"""Run the cross-sectional ranking backtest and report honestly vs benchmarks.

Usage:
    python scripts/cross_sectional.py
    python scripts/cross_sectional.py --start 2015-01-01 --horizon 21 --top 0.2 --refresh

Ranks the whole universe each rebalance, holds the top quantile, and compares to
(a) an equal-weight basket of the universe and (b) SPY buy-and-hold over the same
window. Run once; report whatever happens.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import cross_sectional as xs, data  # noqa: E402


def _pct(x: float) -> str:
    return f"{x * 100:+7.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--top", type=float, default=0.2)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    print(f"Fetching {len(xs.UNIVERSE)} tickers from {args.start} ...", flush=True)
    prices = {}
    for t in xs.UNIVERSE:
        try:
            prices[t] = data.load_or_fetch(t, start=args.start, refresh=args.refresh)
        except Exception as err:
            print(f"  skip {t}: {err}")
    print(f"Got {len(prices)} tickers. Building panel ...", flush=True)

    panel = xs.build_panel(prices, horizon=args.horizon)
    print(f"Panel: {len(panel):,} stock-days across "
          f"{panel['date'].nunique():,} days. Backtesting ...", flush=True)

    res = xs.backtest(panel, horizon=args.horizon, top_quantile=args.top)
    m = res.metrics
    if not m:
        print("No rebalances produced — try more history.")
        return 1

    s, b = m["strategy"], m["benchmark_equalweight"]

    # SPY benchmark over the same window.
    span = res.rebalances["date"]
    spy_line = ""
    try:
        spy = data.load_or_fetch("SPY", start=args.start, refresh=args.refresh)
        spy = spy.loc[(spy.index >= span.min()) & (spy.index <= span.max()), "adj_close"]
        spy_total = float(spy.iloc[-1] / spy.iloc[0] - 1)
        yrs = (span.max() - span.min()).days / 365.25
        spy_cagr = (1 + spy_total) ** (1 / yrs) - 1 if yrs > 0 else 0.0
        spy_line = (f"{'SPY buy & hold':22}{_pct(spy_total):>12}{_pct(spy_cagr):>11}"
                    f"{'—':>9}{'—':>11}")
    except Exception:
        pass

    print(f"\nRebalances: {m['n_rebalances']} (every {m['horizon_days']}d) · "
          f"avg turnover {m['avg_turnover']:.0%}\n")
    hdr = f"{'':22}{'total ret':>12}{'CAGR':>11}{'excess':>9}{'Sharpe':>11}{'maxDD':>11}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'XS top ' + str(int(args.top*100)) + '%':22}"
          f"{_pct(s['total_return']):>12}{_pct(s['cagr']):>11}"
          f"{_pct(m['excess_cagr']):>9}{s['sharpe']:>11.2f}{_pct(s['max_drawdown']):>11}")
    print(f"{'Equal-weight universe':22}"
          f"{_pct(b['total_return']):>12}{_pct(b['cagr']):>11}"
          f"{'—':>9}{b['sharpe']:>11.2f}{_pct(b['max_drawdown']):>11}")
    if spy_line:
        print(spy_line)

    print("\n'excess' = XS strategy CAGR − equal-weight-universe CAGR.")
    print("Beating equal-weight is the real test of stock-selection skill;")
    print("beating SPY can just mean the universe outperformed the index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
