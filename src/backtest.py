"""Walk-forward backtest with realistic frictions.

Design choices that keep the numbers honest:

* **Walk-forward, expanding window.** We never train on data that comes after
  the bars we predict. The model is retrained every `RETRAIN_EVERY` days on all
  history up to the cutoff.
* **Purge / embargo.** Because a label on day `j` peeks `horizon` days into the
  future, the last `horizon` rows before each cutoff are dropped from training
  so no label leaks across the train/test boundary (López de Prado purging).
* **Transaction costs + slippage** are charged every time the position changes.
* **Everything is compared to buy-and-hold.** A strategy that "makes money" but
  trails buy-and-hold has earned you nothing but risk.

The default `horizon=1` (next-day) case produces non-overlapping daily returns,
which compound cleanly. For `horizon > 1` returns overlap; treat those metrics
as indicative only (documented, not silently wrong).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.features import FEATURE_COLUMNS
from src.model import train_model

_COST_RATE = (config.TRANSACTION_COST_BPS + config.SLIPPAGE_BPS) / 10_000.0


def walk_forward_predict(
    dataset: pd.DataFrame,
    *,
    horizon: int = config.TARGET_HORIZON,
    threshold: float = config.TARGET_THRESHOLD,
    initial_train_fraction: float = config.INITIAL_TRAIN_FRACTION,
    retrain_every: int = config.RETRAIN_EVERY,
    min_train_rows: int = 100,
) -> pd.DataFrame:
    """Produce out-of-sample P(up) for every bar past the initial train window.

    Returns a frame indexed by date with columns: prob_up, target, fwd_return.
    """
    dataset = dataset.sort_index()
    n = len(dataset)
    initial = int(n * initial_train_fraction)
    if initial < min_train_rows:
        raise ValueError(
            f"Not enough data: need >= {min_train_rows} train rows, "
            f"have {initial}. Provide more history."
        )

    chunks: list[pd.DataFrame] = []
    i = initial
    while i < n:
        block_end = min(i + retrain_every, n)
        train_end = i - horizon  # embargo the last `horizon` rows
        train = dataset.iloc[:train_end]
        if len(train) < min_train_rows:
            i = block_end
            continue

        model = train_model(
            train[FEATURE_COLUMNS], train["target"],
            horizon=horizon, threshold=threshold,
        )
        test = dataset.iloc[i:block_end]
        prob_up = model.predict_proba_up(test)
        chunks.append(pd.DataFrame({
            "prob_up": prob_up,
            "target": test["target"].values,
            "fwd_return": test["fwd_return"].values,
        }, index=test.index))
        i = block_end

    if not chunks:
        raise ValueError("Walk-forward produced no out-of-sample predictions")
    return pd.concat(chunks)


def run_backtest(
    predictions: pd.DataFrame,
    *,
    buy_threshold: float = config.BUY_THRESHOLD,
    long_short: bool = False,
) -> pd.DataFrame:
    """Turn predictions into positions, returns, and an equity curve.

    Long-only by default: hold (position 1) when P(up) > buy_threshold, else
    flat. With long_short=True, go short (-1) when P(up) < (1 - buy_threshold).
    """
    df = predictions.copy()
    if long_short:
        pos = np.where(df["prob_up"] > buy_threshold, 1.0,
                       np.where(df["prob_up"] < 1 - buy_threshold, -1.0, 0.0))
    else:
        pos = np.where(df["prob_up"] > buy_threshold, 1.0, 0.0)
    df["position"] = pos

    prev_pos = df["position"].shift(1).fillna(0.0)
    df["trade_cost"] = _COST_RATE * (df["position"] - prev_pos).abs()
    df["strategy_return"] = df["position"] * df["fwd_return"] - df["trade_cost"]

    df["strategy_equity"] = (1.0 + df["strategy_return"]).cumprod()
    df["benchmark_equity"] = (1.0 + df["fwd_return"]).cumprod()
    return df


def _annualised_metrics(returns: pd.Series, equity: pd.Series) -> dict:
    n = len(returns)
    periods = config.TRADING_DAYS_PER_YEAR
    total_return = float(equity.iloc[-1] - 1.0) if n else 0.0
    cagr = float(equity.iloc[-1] ** (periods / n) - 1.0) if n else 0.0
    std = float(returns.std())
    sharpe = float(returns.mean() / std * np.sqrt(periods)) if std > 0 else 0.0
    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1.0).min()) if n else 0.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": std * np.sqrt(periods),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def summarize(bt: pd.DataFrame) -> dict:
    """Compute headline metrics for the strategy and the buy-and-hold benchmark."""
    in_market = bt["position"] != 0
    pred_class = (bt["prob_up"] > 0.5).astype(float)
    directional_accuracy = float((pred_class == bt["target"]).mean())

    strat = _annualised_metrics(bt["strategy_return"], bt["strategy_equity"])
    bench = _annualised_metrics(bt["fwd_return"], bt["benchmark_equity"])

    n_trades = int((bt["position"].diff().fillna(bt["position"]).abs() > 0).sum())
    market_days = int(in_market.sum())
    win_rate = (
        float((bt.loc[in_market, "strategy_return"] > 0).mean())
        if market_days else 0.0
    )

    return {
        "n_oos_days": int(len(bt)),
        "directional_accuracy": directional_accuracy,
        "n_trades": n_trades,
        "days_in_market": market_days,
        "win_rate": win_rate,
        "strategy": strat,
        "benchmark": bench,
        "excess_cagr": strat["cagr"] - bench["cagr"],
    }
