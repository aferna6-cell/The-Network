"""Model-health monitoring — always-on evaluation to match always-on learning.

A model that retrains itself on a schedule MUST be watched, or it will keep
emitting confident calls long after the regime it learned has gone. This module
turns the prediction ledger into a health verdict:

* live rolling accuracy vs a coin flip
* staleness (how long since the model was retrained)
* a decay flag the loop uses to downgrade / suppress recommendations

This is the "error correction" at the system level: we cannot stop the market
from changing, but we can detect when the model has stopped working and say so.
"""
from __future__ import annotations

from datetime import datetime, timezone

import config
from src import ledger


def _hours_since(iso_ts: str | None, now: datetime) -> float | None:
    if not iso_ts:
        return None
    try:
        ts = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 3600.0


def health_report(
    model_version: str | None = None,
    *,
    window: int = config.MONITOR_WINDOW,
    now: datetime | None = None,
) -> dict:
    """Assess live model health from the ledger + model age.

    Returns a dict with: n_evaluated, live_accuracy, model_age_hours, and the
    boolean flags `decayed` / `stale`, plus a human-readable `status`.
    """
    now = now or datetime.now(timezone.utc)
    acc = ledger.live_accuracy(window=window)

    decayed = (
        acc["accuracy"] is not None
        and acc["n"] >= max(10, window // 2)
        and acc["accuracy"] < config.MONITOR_MIN_ACCURACY
    )

    age_hours = _hours_since(model_version, now)
    stale = age_hours is not None and age_hours > config.MONITOR_STALE_AFTER_HOURS

    if acc["accuracy"] is None:
        status = "warming up — not enough reconciled predictions yet"
    elif decayed:
        status = (
            f"DECAYED — live accuracy {acc['accuracy']:.1%} over last {acc['n']} "
            f"below floor {config.MONITOR_MIN_ACCURACY:.0%}. Recommendations suppressed."
        )
    elif stale:
        status = f"STALE — model {age_hours:.0f}h old; retrain due."
    else:
        status = f"healthy — live accuracy {acc['accuracy']:.1%} over last {acc['n']}"

    return {
        "n_evaluated": acc["n"],
        "live_accuracy": acc["accuracy"],
        "model_age_hours": age_hours,
        "decayed": bool(decayed),
        "stale": bool(stale),
        "status": status,
    }


def should_suppress(report: dict) -> bool:
    """Whether the loop should withhold actionable recommendations."""
    return bool(report.get("decayed"))
