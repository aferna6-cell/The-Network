"""Pre-specified strategy experiments — honest, run-once comparisons.

Goal: test whether any of the candidate improvements beats buy-and-hold *after
costs*, without p-hacking. Configs are fixed up front; we run each once and
report whatever happens. Tuning until it looks good would just overfit this
particular slice of history (see brain/concepts/backtesting-pitfalls.md).

Design that keeps it overlap-free and leak-free:

* The model's *target* horizon can be long (predict the 10-day-ahead direction),
  but execution is always **daily long-or-flat**, and PnL always uses the real
  next-day return. So a long horizon changes only the SIGNAL, never the return
  accounting — no overlapping-return inflation.
* The regime filter and next-day return are computed from past prices only.

This module reuses the leak-free walk-forward engine in `src/backtest.py` and
does not modify it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from src import data, features
from src.backtest import _COST_RATE, _annualised_metrics, walk_forward_predict


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    horizon: int = 1            # signal horizon (target the model learns)
    buy_threshold: float = 0.55
    sizing: str = "binary"      # "binary" or "confidence"
    use_regime: bool = False
    regime_window: int = 200


# The fixed slate of experiments. Chosen a priori; not tuned to results.
EXPERIMENTS = [
    StrategyConfig("baseline_h1", horizon=1, buy_threshold=0.55),
    StrategyConfig("horizon_10", horizon=10, buy_threshold=0.55),
    StrategyConfig("regime_filter", horizon=1, buy_threshold=0.55, use_regime=True),
    StrategyConfig("confidence_sizing", horizon=1, buy_threshold=0.50,
                   sizing="confidence"),
    StrategyConfig("combined", horizon=10, buy_threshold=0.55,
                   sizing="confidence", use_regime=True),
]


def _positions(prob_up: pd.Series, cfg: StrategyConfig) -> pd.Series:
    if cfg.sizing == "confidence":
        # Long-only, size grows with conviction above 0.5.
        size = ((prob_up - 0.5) * 2).clip(lower=0.0, upper=1.0)
        return size.where(prob_up > 0.5, 0.0)
    return (prob_up > cfg.buy_threshold).astype(float)


def evaluate(prices: pd.DataFrame, cfg: StrategyConfig) -> dict | None:
    """Run one strategy on one ticker; return economic metrics vs buy-and-hold."""
    dataset = features.make_dataset(prices, horizon=cfg.horizon)
    try:
        preds = walk_forward_predict(dataset, horizon=cfg.horizon)
    except ValueError:
        return None

    adj = prices["adj_close"]
    next_day = adj.pct_change().shift(-1)          # real next-day return at row t
    regime_bull = (adj > adj.rolling(cfg.regime_window).mean()).astype(float)

    df = pd.DataFrame(index=preds.index)
    df["prob_up"] = preds["prob_up"]
    df["next_day"] = next_day.reindex(preds.index)
    df["regime"] = regime_bull.reindex(preds.index).fillna(0.0)
    df = df.dropna(subset=["next_day"])            # drop final row (no next day)

    pos = _positions(df["prob_up"], cfg)
    if cfg.use_regime:
        pos = pos * df["regime"]
    df["position"] = pos

    prev = df["position"].shift(1).fillna(0.0)
    cost = _COST_RATE * (df["position"] - prev).abs()
    df["strat_ret"] = df["position"] * df["next_day"] - cost
    df["strat_equity"] = (1 + df["strat_ret"]).cumprod()
    df["bench_equity"] = (1 + df["next_day"]).cumprod()

    strat = _annualised_metrics(df["strat_ret"], df["strat_equity"])
    bench = _annualised_metrics(df["next_day"], df["bench_equity"])
    turnover = float((df["position"] - prev).abs().sum())
    avg_exposure = float(df["position"].mean())

    return {
        "n_oos": int(len(df)),
        "strat": strat,
        "bench": bench,
        "excess_cagr": strat["cagr"] - bench["cagr"],
        "turnover": turnover,
        "avg_exposure": avg_exposure,
    }


def compare(tickers: list[str], *, refresh: bool = False) -> dict:
    """Run every experiment across every ticker; aggregate the results."""
    results: dict[str, list[dict]] = {cfg.name: [] for cfg in EXPERIMENTS}
    per_ticker: dict[str, dict] = {}

    for ticker in tickers:
        prices = data.load_or_fetch(ticker, refresh=refresh)
        per_ticker[ticker] = {}
        for cfg in EXPERIMENTS:
            res = evaluate(prices, cfg)
            if res is not None:
                results[cfg.name].append({"ticker": ticker, **res})
                per_ticker[ticker][cfg.name] = res

    summary = {}
    for name, runs in results.items():
        if not runs:
            continue
        summary[name] = {
            "mean_strat_cagr": float(np.mean([r["strat"]["cagr"] for r in runs])),
            "mean_bench_cagr": float(np.mean([r["bench"]["cagr"] for r in runs])),
            "mean_excess_cagr": float(np.mean([r["excess_cagr"] for r in runs])),
            "mean_strat_sharpe": float(np.mean([r["strat"]["sharpe"] for r in runs])),
            "mean_bench_sharpe": float(np.mean([r["bench"]["sharpe"] for r in runs])),
            "mean_strat_maxdd": float(np.mean([r["strat"]["max_drawdown"] for r in runs])),
            "mean_bench_maxdd": float(np.mean([r["bench"]["max_drawdown"] for r in runs])),
            "mean_exposure": float(np.mean([r["avg_exposure"] for r in runs])),
            "beats_bh_count": int(sum(r["excess_cagr"] > 0 for r in runs)),
            "n_tickers": len(runs),
        }
    return {"summary": summary, "per_ticker": per_ticker}
