"""Run the statistical gate on the cross-sectional strategy's out-of-sample
returns. PASS means the edge survived multiple-testing-robust checks; until it
does, the strategy stays on paper — no exceptions, no agent override.

    python scripts/verify_strategy.py --trials 8
    python scripts/verify_strategy.py --start 2015-01-01 --horizon 21 --trials 12

`--trials` is how many strategy variants you tried before picking this one (top
quantile, horizon, feature sets, model params...). Be honest: under-counting
trials inflates the Deflated Sharpe and defeats the whole point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import cross_sectional as xs, data, verify  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--top", type=float, default=0.2)
    p.add_argument("--trials", type=int, default=1,
                   help="number of strategy variants tried (multiple-testing N)")
    p.add_argument("--excess", action="store_true",
                   help="test strategy-minus-benchmark returns instead of raw")
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    print(f"Fetching {len(xs.UNIVERSE)} tickers from {args.start} ...", flush=True)
    prices = {}
    for t in xs.UNIVERSE:
        try:
            prices[t] = data.load_or_fetch(t, start=args.start, refresh=args.refresh)
        except Exception as err:
            print(f"  skip {t}: {err}")

    panel = xs.build_panel(prices, horizon=args.horizon)
    res = xs.backtest(panel, horizon=args.horizon, top_quantile=args.top)
    if res.rebalances.empty:
        print("No rebalances produced — try more history.")
        return 1

    reb = res.rebalances
    returns = (reb["strat_ret"] - reb["bench_ret"]) if args.excess else reb["strat_ret"]
    ppy = config.TRADING_DAYS_PER_YEAR / args.horizon
    verdict = verify.gate(returns, periods_per_year=ppy, n_trials=args.trials)

    label = "excess (strat - benchmark)" if args.excess else "strategy"
    print(f"\nVerifying {label} returns over {len(reb)} rebalances "
          f"(~{ppy:.1f} periods/yr), declaring {args.trials} trial(s):\n")
    print(verdict.render())
    print(f"\nexcess CAGR vs equal-weight: {res.metrics.get('excess_cagr', 0):+.2%}")

    # Factor-neutral alpha: strip market (equal-weight universe) + momentum.
    factors = {"market": reb["bench_ret"].to_numpy(),
               "momentum": xs.wml_factor(panel, reb["date"])}
    fa = verify.factor_alpha(reb["strat_ret"], factors, periods_per_year=ppy)
    print()
    print(fa.render())
    if not verdict.passed:
        print("\nNot approved to trade. This is the expected, honest outcome until "
              "a real out-of-sample edge shows up.")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
