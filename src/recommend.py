"""Portfolio-aware recommendation engine.

Turns a model's probability into an *actionable* call relative to what you hold:

* not held + bullish  -> consider opening / adding
* held + bearish      -> trim or exit
* held + concentrated -> flag risk regardless of signal
* model decayed       -> downgrade everything to HOLD and say why

A recommendation is advice for a human, never an order. Nothing here places a
trade.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import config
from src.model import TrainedModel
from src.portfolio import Portfolio


@dataclass
class Recommendation:
    ticker: str
    action: str            # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    prob_up: float
    confidence: float      # |prob_up - 0.5| * 2, in [0, 1]
    held: bool
    weight: float          # current portfolio weight (0 if not held)
    price: float
    rationale: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["prob_up"] = round(self.prob_up, 4)
        d["confidence"] = round(self.confidence, 4)
        d["weight"] = round(self.weight, 4)
        return d


def _action_from_prob(prob_up: float) -> str:
    if prob_up >= config.STRONG_BUY_THRESHOLD:
        return "STRONG_BUY"
    if prob_up > config.BUY_THRESHOLD:
        return "BUY"
    if prob_up <= config.STRONG_SELL_THRESHOLD:
        return "STRONG_SELL"
    if prob_up < config.SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def _rationale(action: str, held: bool, weight: float, concentrated: bool) -> str:
    parts = []
    if action in ("STRONG_BUY", "BUY"):
        parts.append("opening or adding" if not held else "adding to position")
    elif action in ("STRONG_SELL", "SELL"):
        parts.append("trimming or exiting" if held else "avoid / not a buy")
    else:
        parts.append("no edge — hold")
    if held:
        parts.append(f"you hold {weight:.0%} of the portfolio here")
    if concentrated:
        parts.append("⚠ position is concentrated — consider trimming on risk grounds")
    return "; ".join(parts)


def recommend(
    models: dict[str, TrainedModel],
    feature_rows: dict[str, "object"],     # ticker -> latest feature Series
    prices: dict[str, float],
    portfolio: Portfolio,
    *,
    suppress: bool = False,
) -> list[Recommendation]:
    """Generate one recommendation per ticker we have a model + features for.

    `suppress=True` (from the decay monitor) forces every action to HOLD.
    """
    weights = portfolio.weights(prices)
    concentrated = set(portfolio.concentrated(prices))
    out: list[Recommendation] = []

    for ticker, model in models.items():
        row = feature_rows.get(ticker)
        price = prices.get(ticker)
        if row is None or price is None:
            continue
        prob_up = float(model.predict_proba_up(row.to_frame().T).iloc[0])
        action = "HOLD" if suppress else _action_from_prob(prob_up)
        held = portfolio.holds(ticker)
        weight = weights.get(ticker, 0.0)
        rationale = (
            "model health degraded — holding all positions until it recovers"
            if suppress else
            _rationale(action, held, weight, ticker in concentrated)
        )
        out.append(Recommendation(
            ticker=ticker,
            action=action,
            prob_up=prob_up,
            confidence=abs(prob_up - 0.5) * 2,
            held=held,
            weight=weight,
            price=price,
            rationale=rationale,
        ))

    # Strongest convictions first.
    out.sort(key=lambda r: r.confidence, reverse=True)
    return out
