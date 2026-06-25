"""Crowding / short-term-reversal experiment for the cross-sectional model.

The game-theory thesis: crowded names mean-revert — when everyone is already in,
the marginal trade comes from the other side. We can't see order flow on a free
daily feed, so we proxy crowding with the *cross-sectional rank* of short-term
returns and over-extension (how stretched is this name vs the universe today).

This adds those ranks to the model and runs BOTH the baseline and the augmented
strategy through the same statistical gate (src/verify.py). No faith, no
adoption — the numbers decide. Run once; report whatever happens.

    python scripts/crowding_experiment.py --start 2016-01-01 --trials 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import cross_sectional as xs, data, verify  # noqa: E402

# Already-existing per-stock features whose cross-sectional rank encodes
# "crowding": short-term reversal (ret_1, ret_5) + over-extension (boll_pctb,
# sma_ratio_20). The model learns the direction; we only hand it the signal.
CROWDING_RANKED = ["ret_1", "ret_5", "boll_pctb", "sma_ratio_20"]


def _run(panel, *, horizon, top, features, trials):
    res = xs.backtest(panel, horizon=horizon, top_quantile=top, features=features)
    ppy = config.TRADING_DAYS_PER_YEAR / horizon
    verdict = verify.gate(res.rebalances["strat_ret"], periods_per_year=ppy,
                          n_trials=trials)
    return res, verdict


def _line(name, res, verdict):
    m = res.metrics
    s = m["strategy"]
    return (f"{name:<12}sharpe {s['sharpe']:+.2f} | excessCAGR "
            f"{m['excess_cagr']:+.2%} | maxDD {s['max_drawdown']:+.1%} | "
            f"DSR {verdict.stats['deflated_sharpe']:.2f} | "
            f"gate {'PASS' if verdict.passed else 'FAIL'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--top", type=float, default=0.2)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    print(f"Fetching {len(xs.UNIVERSE)} tickers from {args.start} ...", flush=True)
    prices = {}
    for t in xs.UNIVERSE:
        try:
            prices[t] = data.load_or_fetch(t, start=args.start, refresh=args.refresh)
        except Exception as err:
            print(f"  skip {t}: {err}")

    print("Building panels (baseline + crowding) ...", flush=True)
    base_panel = xs.build_panel(prices, horizon=args.horizon)
    crowd_ranked = xs._RANKED + CROWDING_RANKED
    crowd_panel = xs.build_panel(prices, horizon=args.horizon, ranked=crowd_ranked)
    crowd_features = xs.FEATURE_COLUMNS + [f"xs_rank_{c}" for c in crowd_ranked]

    print("Backtesting ...", flush=True)
    base_res, base_v = _run(base_panel, horizon=args.horizon, top=args.top,
                            features=None, trials=args.trials)
    crowd_res, crowd_v = _run(crowd_panel, horizon=args.horizon, top=args.top,
                              features=crowd_features, trials=args.trials)

    n = len(base_res.rebalances)
    print(f"\n{n} rebalances · added ranks: {', '.join(CROWDING_RANKED)} · "
          f"{args.trials} trials declared\n")
    print(_line("baseline", base_res, base_v))
    print(_line("+crowding", crowd_res, crowd_v))

    bs, cs = base_res.metrics["strategy"]["sharpe"], crowd_res.metrics["strategy"]["sharpe"]
    verdict = ("Crowding ranks helped Sharpe." if cs > bs + 0.05 else
               "Crowding ranks hurt Sharpe." if cs < bs - 0.05 else
               "No meaningful difference.")
    print(f"\n{verdict} Neither version is approved unless its gate says PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
