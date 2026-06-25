"""Linear vs gradient-boosting baseline for the cross-sectional model.

The linear-regression article's claim: simple linear models, used with
discipline, rival the fancy ones — and complexity you can't justify is just
overfitting waiting to happen. So we put a scaled logistic regression head-to-
head against the gradient-boosting model on the SAME features, SAME walk-forward,
SAME gate. If linear ≈ GBM, prefer linear (interpretable, harder to overfit).

    python scripts/linear_baseline_experiment.py --start 2016-01-01 --trials 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import config  # noqa: E402
from src import cross_sectional as xs, data, verify  # noqa: E402


def _linear_model():
    # Scale first — logistic regression is scale-sensitive; the GBM isn't.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))


def _run(panel, *, horizon, top, trials, model_factory):
    res = xs.backtest(panel, horizon=horizon, top_quantile=top,
                      model_factory=model_factory)
    ppy = config.TRADING_DAYS_PER_YEAR / horizon
    v = verify.gate(res.rebalances["strat_ret"], periods_per_year=ppy,
                    n_trials=trials)
    return res, v


def _line(name, res, v):
    s = res.metrics["strategy"]
    return (f"{name:<18}sharpe {s['sharpe']:+.2f} | excessCAGR "
            f"{res.metrics['excess_cagr']:+.2%} | maxDD {s['max_drawdown']:+.1%} | "
            f"DSR {v.stats['deflated_sharpe']:.2f} | gate "
            f"{'PASS' if v.passed else 'FAIL'}")


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

    panel = xs.build_panel(prices, horizon=args.horizon)
    print(f"Backtesting GBM vs linear over the same panel ...", flush=True)
    gbm_res, gbm_v = _run(panel, horizon=args.horizon, top=args.top,
                          trials=args.trials, model_factory=None)
    lin_res, lin_v = _run(panel, horizon=args.horizon, top=args.top,
                          trials=args.trials, model_factory=_linear_model)

    print(f"\n{len(gbm_res.rebalances)} rebalances · {args.trials} trials declared\n")
    print(_line("gradient-boosting", gbm_res, gbm_v))
    print(_line("logistic (linear)", lin_res, lin_v))

    gs = gbm_res.metrics["strategy"]["sharpe"]
    ls = lin_res.metrics["strategy"]["sharpe"]
    if abs(gs - ls) < 0.1:
        msg = "Linear matches GBM — the model complexity isn't earning its keep."
    elif ls > gs:
        msg = "Linear beats GBM — prefer the simpler, more interpretable model."
    else:
        msg = "GBM beats linear — the non-linearity is doing real work."
    print(f"\n{msg} Neither is approved unless its gate says PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
