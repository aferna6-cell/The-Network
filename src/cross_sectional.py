"""Cross-sectional ranking — rank a whole universe, hold the top names.

This is a genuinely different approach from single-stock timing. Instead of
asking "will AAPL go up tomorrow?", it asks "which stocks in the universe will
*outperform the others* over the next month?" and holds the top-ranked basket.
Most documented equity edges (momentum especially) live cross-sectionally, not
in single-name timing — see brain/concepts/cross-sectional-momentum.md.

"Different data" here means **relative / ranked features**: each stock's signals
ranked against every other stock on the same day. That is honestly buildable on
free price data. True point-in-time *fundamentals* are NOT free, and using
current fundamentals in a historical backtest is look-ahead leakage, so we do
not pretend to.

Leak-free by construction:
* Features at date t use only data <= t (same guarantees as src/features.py).
* Cross-sectional ranks at t use only contemporaneous (<= t) features.
* The target — "did this stock beat the universe median forward return?" — uses
  only future bars and is a LABEL, never a feature.
* Walk-forward in time with a purge so labels never straddle the train/test line.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import config
from src.features import FEATURE_COLUMNS, add_features

# A liquid, sector-diverse universe. Free Yahoo data; ragged history is fine
# (each date ranks whatever stocks already exist).
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "V", "MA", "UNH",
    "HD", "PG", "JNJ", "XOM", "CVX", "KO", "PEP", "WMT", "COST", "DIS",
    "NFLX", "AMD", "CSCO", "ORCL", "CRM", "BAC", "MRK", "PFE", "VZ", "NKE",
]

# Features whose cross-sectional rank (vs the universe, per day) we add.
_RANKED = ["mom_20", "ret_20", "vol_20", "rsi_14", "sma_ratio_50"]
XS_FEATURES = FEATURE_COLUMNS + [f"xs_rank_{c}" for c in _RANKED]


def build_panel(prices_by_ticker: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    """Stack per-stock features into one panel with cross-sectional ranks + target."""
    frames = []
    for ticker, prices in prices_by_ticker.items():
        f = add_features(prices)[FEATURE_COLUMNS].copy()
        adj = prices["adj_close"]
        f["fwd_return"] = adj.shift(-horizon) / adj - 1.0
        f["ticker"] = ticker
        f["date"] = f.index
        frames.append(f)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=FEATURE_COLUMNS + ["fwd_return"])

    # Cross-sectional rank (percentile within each day) — the relative signal.
    for col in _RANKED:
        panel[f"xs_rank_{col}"] = panel.groupby("date")[col].rank(pct=True)

    # Target: beat the universe's median forward return that day (relative).
    panel["median_fwd"] = panel.groupby("date")["fwd_return"].transform("median")
    panel["target"] = (panel["fwd_return"] > panel["median_fwd"]).astype(float)
    # Days with too few names to rank meaningfully add noise; require >= 5.
    counts = panel.groupby("date")["ticker"].transform("count")
    return panel[counts >= 5].dropna(subset=XS_FEATURES).reset_index(drop=True)


@dataclass
class XSResult:
    rebalances: pd.DataFrame   # per-period strat vs benchmark returns
    metrics: dict


def _metrics(returns: pd.Series, equity: pd.Series, periods_per_year: float) -> dict:
    n = len(returns)
    if n == 0:
        return {"total_return": 0, "cagr": 0, "sharpe": 0, "max_drawdown": 0}
    std = float(returns.std())
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (periods_per_year / n) - 1),
        "sharpe": float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def backtest(
    panel: pd.DataFrame,
    *,
    horizon: int = 21,
    initial_fraction: float = 0.6,
    top_quantile: float = 0.2,
    cost_bps: float = config.TRANSACTION_COST_BPS + config.SLIPPAGE_BPS,
) -> XSResult:
    """Walk-forward: each rebalance, rank the cross-section and hold the top names.

    Non-overlapping (rebalance == horizon), equal-weight top basket, charged for
    turnover. Benchmark = equal-weight of the whole universe that day.
    """
    dates = sorted(panel["date"].unique())
    n = len(dates)
    initial = int(n * initial_fraction)
    cost_rate = cost_bps / 10_000.0
    rows = []
    prev_top: set[str] = set()

    for i in range(initial, n, horizon):
        d = dates[i]
        purge_idx = i - horizon
        if purge_idx < 50:
            continue
        train = panel[panel["date"] <= dates[purge_idx]]
        if len(train) < 500:
            continue

        model = HistGradientBoostingClassifier(**config.MODEL_PARAMS)
        model.fit(train[XS_FEATURES], train["target"])
        up = list(model.classes_).index(1.0) if 1.0 in list(model.classes_) else -1

        cross = panel[panel["date"] == d].copy()
        if len(cross) < 5:
            continue
        cross["score"] = model.predict_proba(cross[XS_FEATURES])[:, up]
        k = max(1, int(round(len(cross) * top_quantile)))
        top = cross.nlargest(k, "score")
        top_set = set(top["ticker"])

        # Turnover-based cost: fraction of the basket replaced (sell + buy sides).
        turnover = 1.0 - (len(prev_top & top_set) / k if prev_top else 0.0)
        cost = 2 * cost_rate * turnover
        prev_top = top_set

        rows.append({
            "date": d,
            "n_universe": len(cross),
            "k": k,
            "turnover": turnover,
            "strat_ret": float(top["fwd_return"].mean()) - cost,
            "bench_ret": float(cross["fwd_return"].mean()),
            "top": ",".join(sorted(top_set)),
        })

    reb = pd.DataFrame(rows)
    if reb.empty:
        return XSResult(reb, {})
    ppy = config.TRADING_DAYS_PER_YEAR / horizon
    reb["strat_equity"] = (1 + reb["strat_ret"]).cumprod()
    reb["bench_equity"] = (1 + reb["bench_ret"]).cumprod()
    metrics = {
        "n_rebalances": int(len(reb)),
        "horizon_days": horizon,
        "strategy": _metrics(reb["strat_ret"], reb["strat_equity"], ppy),
        "benchmark_equalweight": _metrics(reb["bench_ret"], reb["bench_equity"], ppy),
        "avg_turnover": float(reb["turnover"].mean()),
    }
    metrics["excess_cagr"] = (
        metrics["strategy"]["cagr"] - metrics["benchmark_equalweight"]["cagr"]
    )
    return XSResult(reb, metrics)
