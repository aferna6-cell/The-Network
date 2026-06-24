"""Cross-sectional recommender — concrete Buy / Sell / Trim / Hold calls.

Turns the cross-sectional ranking into the output you actually want: "buy X,
sell Y", relative to what you hold. Each call carries a confidence and a
plain-English rationale, and the whole thing is shown next to a LIVE track
record from the ledger so trust is earned by evidence, not asserted.

Honest by design:
* It logs every actionable call to its own ledger (src/ledger.py with
  config.XS_LEDGER_PATH) and reconciles past calls, so live accuracy accumulates.
* The edge is unproven and the research backtest is survivorship-biased
  (brain/concepts/cross-sectional-momentum.md). This is decision SUPPORT, not an
  oracle, and it places no trades.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import config
from src import cross_sectional as xs, ledger
from src.model import TrainedModel
from src.portfolio import Portfolio


@dataclass
class XSRecommendation:
    ticker: str
    action: str            # BUY / SELL / TRIM / HOLD / AVOID
    rank: int              # 1 = strongest in the universe
    universe_size: int
    percentile: float      # 0..1, 1 = best
    score: float           # model P(outperform universe)
    confidence: float      # |percentile - 0.5| * 2
    held: bool
    weight: float
    price: float
    rationale: str

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("percentile", "score", "confidence", "weight"):
            d[k] = round(getattr(self, k), 4)
        return d


def _decide(percentile: float, held: bool, concentrated: bool,
            top_q: float, sell_pct: float) -> str:
    buy_cut = 1.0 - top_q
    if held and concentrated and percentile < buy_cut:
        return "TRIM"                       # risk-based, independent of edge
    if held:
        return "SELL" if percentile < sell_pct else "HOLD"
    return "BUY" if percentile >= buy_cut else "AVOID"


def _rationale(rec_action: str, rank: int, n: int, pct: float,
               held: bool, weight: float) -> str:
    place = f"ranks #{rank} of {n} (top {(1 - pct) * 100:.0f}%)"
    own = f"you hold {weight:.0%}" if held else "you don't hold it"
    tail = {
        "BUY": "strong relative strength and not in your book — add candidate",
        "SELL": "weak relative strength among names you own — trim/exit candidate",
        "TRIM": "position is concentrated — reduce on risk grounds",
        "HOLD": "keep — still ranks acceptably",
        "AVOID": "not a buy at current rank",
    }[rec_action]
    return f"{place}; {own}; {tail}"


def recommend(
    prices_by_ticker: dict,
    portfolio: Portfolio,
    model: TrainedModel,
    prices_now: dict[str, float],
    *,
    top_q: float = config.XS_TOP_QUANTILE,
    sell_pct: float = config.XS_SELL_PERCENTILE,
) -> list[XSRecommendation]:
    """Rank the live universe and map each name to a portfolio-relative action."""
    cross = xs.latest_cross_section(prices_by_ticker)
    scores = model.predict_proba_up(cross).sort_values(ascending=False)
    percentile = scores.rank(pct=True)
    n = len(scores)
    weights = portfolio.weights(prices_now)
    concentrated = set(portfolio.concentrated(prices_now))

    recs: list[XSRecommendation] = []
    for rank, (ticker, score) in enumerate(scores.items(), start=1):
        pct = float(percentile[ticker])
        held = portfolio.holds(ticker)
        weight = weights.get(ticker, 0.0)
        action = _decide(pct, held, ticker in concentrated, top_q, sell_pct)
        price = prices_now.get(ticker, 0.0)
        recs.append(XSRecommendation(
            ticker=ticker, action=action, rank=rank, universe_size=n,
            percentile=pct, score=float(score), confidence=abs(pct - 0.5) * 2,
            held=held, weight=weight, price=price,
            rationale=_rationale(action, rank, n, pct, held, weight),
        ))
    return recs


def log_and_snapshot(
    recs: list[XSRecommendation],
    model: TrainedModel,
    portfolio: Portfolio,
    prices_now: dict[str, float],
    *,
    now: datetime | None = None,
) -> dict:
    """Reconcile past calls, log new actionable ones, and write a snapshot."""
    now = now or datetime.now(timezone.utc)
    ledger.reconcile(lambda t: prices_now.get(t),
                     path=config.XS_LEDGER_PATH, now=now)

    for r in recs:
        if r.action in ("BUY", "SELL") and r.price > 0:
            ledger.append_prediction(
                ticker=r.ticker, horizon_days=model.horizon, price_at_pred=r.price,
                prob_up=r.score, recommendation=r.action,
                model_version=model.trained_at, path=config.XS_LEDGER_PATH, now=now)

    track = ledger.live_accuracy(window=config.MONITOR_WINDOW,
                                 path=config.XS_LEDGER_PATH)
    snapshot = {
        "generated_at": now.isoformat(),
        "universe_size": len(recs),
        "portfolio_value": portfolio.total_value(prices_now),
        "live_track_record": track,
        "actions": [r.to_dict() for r in recs
                    if r.action in ("BUY", "SELL", "TRIM")],
        "holds": [r.to_dict() for r in recs if r.action == "HOLD"],
    }
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    config.XS_RECOMMENDATIONS_PATH.write_text(
        __import__("json").dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot
