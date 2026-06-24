"""Paper-trading account — fake money, real (simulated) execution.

The model is handed $500 of pretend cash and actually manages it: it buys, sells,
and rebalances, and we mark the account to market over time to get an HONEST
forward equity curve — the thing a backtest can never give you. Nothing here
touches a real broker or real money.

A parallel benchmark buys the same $500 of SPY at inception and holds, so the
demo always answers the only question that matters: did the model beat just
buying the index?
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import config

_COST_RATE = (config.TRANSACTION_COST_BPS + config.SLIPPAGE_BPS) / 10_000.0


@dataclass
class PaperAccount:
    cash: float
    starting_equity: float
    inception_date: str
    positions: dict[str, dict] = field(default_factory=dict)  # ticker -> {shares, cost_basis}
    last_rebalance: str | None = None
    benchmark_ticker: str = config.PAPER_BENCHMARK
    benchmark_shares: float = 0.0

    # --- valuation ---------------------------------------------------------
    def positions_value(self, prices: dict[str, float]) -> float:
        return sum(p["shares"] * prices[t]
                   for t, p in self.positions.items() if prices.get(t))

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.positions_value(prices)

    def benchmark_equity(self, prices: dict[str, float]) -> float:
        price = prices.get(self.benchmark_ticker)
        if not self.benchmark_shares or not price:
            return self.starting_equity
        return self.benchmark_shares * price

    # --- execution (simulated fills, costs charged) ------------------------
    def trade(self, ticker: str, delta_shares: float, price: float) -> float:
        """Buy (delta>0) or sell (delta<0) `delta_shares` at `price`. Returns cost."""
        notional = delta_shares * price
        cost = abs(notional) * _COST_RATE
        self.cash -= notional + cost
        cur = self.positions.get(ticker, {"shares": 0.0, "cost_basis": 0.0})
        new_shares = cur["shares"] + delta_shares
        if new_shares <= 1e-9:
            self.positions.pop(ticker, None)
        else:
            if delta_shares > 0:  # update average cost basis on buys
                spent = cur["shares"] * cur["cost_basis"] + delta_shares * price
                cur["cost_basis"] = spent / new_shares
            cur["shares"] = new_shares
            self.positions[ticker] = cur
        return cost

    def ensure_benchmark(self, prices: dict[str, float]) -> None:
        """Establish the buy-and-hold benchmark on the first chance we get a price."""
        price = prices.get(self.benchmark_ticker)
        if self.benchmark_shares == 0.0 and price:
            self.benchmark_shares = self.starting_equity / price

    # --- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "starting_equity": self.starting_equity,
            "inception_date": self.inception_date,
            "positions": self.positions,
            "last_rebalance": self.last_rebalance,
            "benchmark_ticker": self.benchmark_ticker,
            "benchmark_shares": self.benchmark_shares,
        }


def new_account(cash: float = config.PAPER_START_CASH,
                now: datetime | None = None) -> PaperAccount:
    now = now or datetime.now(timezone.utc)
    return PaperAccount(cash=cash, starting_equity=cash,
                        inception_date=now.date().isoformat())


def load_account(path: Path | None = None) -> PaperAccount | None:
    path = path or config.PAPER_ACCOUNT_PATH
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return PaperAccount(
        cash=d["cash"], starting_equity=d["starting_equity"],
        inception_date=d["inception_date"], positions=d.get("positions", {}),
        last_rebalance=d.get("last_rebalance"),
        benchmark_ticker=d.get("benchmark_ticker", config.PAPER_BENCHMARK),
        benchmark_shares=d.get("benchmark_shares", 0.0),
    )


def save_account(account: PaperAccount, path: Path | None = None) -> None:
    path = path or config.PAPER_ACCOUNT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(account.to_dict(), indent=2), encoding="utf-8")
