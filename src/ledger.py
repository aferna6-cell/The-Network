"""Prediction ledger — the system's memory of every call it made.

This is what makes "learn from mistakes" real and honest. Every recommendation
is appended here with the price and probability at the time. Later, once the
horizon has elapsed, `reconcile()` fills in what actually happened and whether
the call was right. From that we can measure *live* hit rate — the only number
that proves whether the model is any good outside the backtest.

Stored as an append-only CSV so it survives restarts on a persistent host
(mount a volume at `state/`).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config

COLUMNS = [
    "timestamp",        # when the prediction was made (UTC ISO)
    "ticker",
    "horizon_days",
    "price_at_pred",    # price when we predicted
    "prob_up",
    "recommendation",   # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    "model_version",    # trained_at of the model that made the call
    "resolve_after",    # date (UTC) on/after which this can be reconciled
    "actual_return",    # filled by reconcile()
    "correct",          # filled by reconcile(): 1/0
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def append_prediction(
    *,
    ticker: str,
    horizon_days: int,
    price_at_pred: float,
    prob_up: float,
    recommendation: str,
    model_version: str,
    path: Path | None = None,
    now: datetime | None = None,
) -> None:
    """Record a single prediction (outcome columns left blank for later)."""
    path = path or config.LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    now = now or _now()
    resolve_after = (now + pd.Timedelta(days=horizon_days)).date().isoformat()

    row = {
        "timestamp": now.isoformat(),
        "ticker": ticker.upper(),
        "horizon_days": horizon_days,
        "price_at_pred": round(float(price_at_pred), 6),
        "prob_up": round(float(prob_up), 6),
        "recommendation": recommendation,
        "model_version": model_version,
        "resolve_after": resolve_after,
        "actual_return": "",
        "correct": "",
    }
    header = not path.exists()
    pd.DataFrame([row], columns=COLUMNS).to_csv(
        path, mode="a", header=header, index=False
    )


def load_ledger(path: Path | None = None) -> pd.DataFrame:
    path = path or config.LEDGER_PATH
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"correct": "object", "actual_return": "object"})


def reconcile(
    price_lookup,
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> int:
    """Fill in outcomes for predictions whose horizon has elapsed.

    `price_lookup(ticker) -> current price`. A prediction is "correct" when the
    realized return since prediction agrees in sign with the call (BUY-ish vs
    SELL-ish). HOLD rows are scored on whether prob_up direction matched.
    Returns the number of rows newly reconciled.
    """
    path = path or config.LEDGER_PATH
    df = load_ledger(path)
    if df.empty:
        return 0

    now = now or _now()
    today = now.date().isoformat()
    pending = df["correct"].isna() | (df["correct"].astype(str) == "")
    due = pending & (df["resolve_after"].astype(str) <= today)

    n = 0
    for idx in df.index[due]:
        ticker = df.at[idx, "ticker"]
        try:
            price_now = float(price_lookup(ticker))
        except Exception:
            continue
        entry = float(df.at[idx, "price_at_pred"])
        if entry <= 0:
            continue
        actual_return = price_now / entry - 1.0
        prob_up = float(df.at[idx, "prob_up"])
        # Did the model lean up, and did price go up? (sign agreement)
        leaned_up = prob_up >= 0.5
        went_up = actual_return > 0
        df.at[idx, "actual_return"] = round(actual_return, 6)
        df.at[idx, "correct"] = int(leaned_up == went_up)
        n += 1

    if n:
        df.to_csv(path, index=False)
    return n


def live_accuracy(window: int | None = None, path: Path | None = None) -> dict:
    """Hit rate over the most recently reconciled predictions."""
    df = load_ledger(path)
    done = df[df["correct"].astype(str).isin(["0", "1"])].copy()
    if done.empty:
        return {"n": 0, "accuracy": None}
    done["correct"] = done["correct"].astype(int)
    if window:
        done = done.tail(window)
    return {"n": int(len(done)), "accuracy": float(done["correct"].mean())}
