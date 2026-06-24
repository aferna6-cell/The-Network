"""Portfolio state — what you actually hold.

Recommendations are only useful relative to your real position: "trim AAPL" or
"you're 40% in one name" beats an abstract up/down call. This module loads that
state from a JSON file (gitignored — it is personal financial data) and computes
current value, weights, and concentration.

Expected portfolio.json shape:

    {
      "cash": 5000.0,
      "positions": {
        "AAPL": {"shares": 10, "cost_basis": 180.0},
        "MSFT": {"shares": 5,  "cost_basis": 390.0}
      }
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass
class Position:
    ticker: str
    shares: float
    cost_basis: float  # average price paid per share

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.cost_basis) * self.shares


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position]

    @property
    def tickers(self) -> list[str]:
        return list(self.positions.keys())

    def holds(self, ticker: str) -> bool:
        return ticker.upper() in self.positions

    def total_value(self, prices: dict[str, float]) -> float:
        invested = sum(
            pos.market_value(prices[t]) for t, pos in self.positions.items()
            if prices.get(t) is not None
        )
        return self.cash + invested

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Fraction of total portfolio value in each position (cash excluded)."""
        total = self.total_value(prices)
        if total <= 0:
            return {t: 0.0 for t in self.positions}
        return {
            t: pos.market_value(prices[t]) / total
            for t, pos in self.positions.items()
            if prices.get(t) is not None
        }

    def concentrated(
        self, prices: dict[str, float], limit: float = config.MAX_POSITION_WEIGHT
    ) -> list[str]:
        """Tickers whose weight exceeds the concentration limit."""
        return [t for t, w in self.weights(prices).items() if w > limit]


def load_portfolio(path: Path | None = None) -> Portfolio:
    """Load and validate the portfolio JSON. Empty portfolio if file absent."""
    path = path or config.PORTFOLIO_PATH
    if not path.exists():
        return Portfolio(cash=0.0, positions={})

    raw = json.loads(path.read_text(encoding="utf-8"))
    cash = float(raw.get("cash", 0.0))
    positions: dict[str, Position] = {}
    for ticker, p in (raw.get("positions") or {}).items():
        t = ticker.upper()
        shares = float(p["shares"])
        if shares <= 0:
            continue
        positions[t] = Position(
            ticker=t, shares=shares, cost_basis=float(p.get("cost_basis", 0.0))
        )
    return Portfolio(cash=cash, positions=positions)
