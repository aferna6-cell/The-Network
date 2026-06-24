"""Paper-trading engine + always-on loop.

The model autonomously manages the $500 paper account:

* **Every tick:** mark the account to market and append the equity point (so we
  get a live curve) — for the account and for the SPY buy-and-hold benchmark.
* **Monthly (PAPER_REBALANCE_DAYS):** re-rank the universe, pick the top-K names
  equal-weight, and trade the account to those targets (costs charged).

This is the honest forward test. It places no real trades.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone

import config
from src import cross_sectional as xs, data, paper, runtime, xs_recommend
from src.model import load_model, save_model


def _prices_now(universe: list[str]) -> tuple[dict, dict]:
    """Return (history_by_ticker, latest_price) for the universe + benchmark."""
    hist, now_px = {}, {}
    for t in sorted(set(universe) | {config.PAPER_BENCHMARK}):
        try:
            hist[t] = data.load_or_fetch(t, start="2015-01-01", refresh=True)
            q = data.latest_quote(t)
            now_px[t] = q["price"] or float(hist[t]["adj_close"].iloc[-1])
        except Exception as err:
            print(f"  skip {t}: {err}")
    return hist, now_px


def _get_model(hist: dict, *, retrain: bool):
    model = load_model(config.XS_MODEL_PATH) if config.XS_MODEL_PATH.exists() else None
    if retrain or runtime.retrain_due(model):
        model = xs.train_full_model(hist, horizon=config.XS_HORIZON)
        save_model(model, config.XS_MODEL_PATH)
    return model


def target_tickers(model, hist: dict, top_k: int = config.PAPER_TOP_K) -> list[str]:
    cross = xs.latest_cross_section(hist)
    scores = model.predict_proba_up(cross).sort_values(ascending=False)
    return list(scores.index[:top_k])


def rebalance(account: paper.PaperAccount, targets: list[str],
              prices: dict[str, float]) -> list[dict]:
    """Trade the account toward an equal-weight basket of `targets`."""
    equity = account.equity(prices)
    investable = equity * config.PAPER_INVEST_FRACTION
    per_name = investable / len(targets) if targets else 0.0
    trades = []

    # Exit names no longer in the target basket.
    for t in list(account.positions):
        if t not in targets and prices.get(t):
            shares = account.positions[t]["shares"]
            account.trade(t, -shares, prices[t])
            trades.append({"ticker": t, "action": "SELL", "shares": round(shares, 4),
                           "price": round(prices[t], 2)})

    # Move each target to its equal-weight allocation.
    for t in targets:
        price = prices.get(t)
        if not price:
            continue
        cur = account.positions.get(t, {"shares": 0.0})["shares"]
        desired = per_name / price
        delta = desired - cur
        if abs(delta * price) < 0.01:
            continue
        account.trade(t, delta, price)
        trades.append({"ticker": t, "action": "BUY" if delta > 0 else "SELL",
                       "shares": round(abs(delta), 4), "price": round(price, 2)})
    return trades


def _record_equity(now, equity, bench):
    path = config.PAPER_EQUITY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["timestamp", "equity", "benchmark_equity"])
        w.writerow([now.isoformat(), round(equity, 4), round(bench, 4)])


def _due_to_rebalance(account: paper.PaperAccount, now: datetime) -> bool:
    if not account.positions or account.last_rebalance is None:
        return True
    last = datetime.fromisoformat(account.last_rebalance)
    return (now.date() - last.date()).days >= config.PAPER_REBALANCE_DAYS


def run_once(*, retrain: bool = False, now: datetime | None = None) -> dict:
    """One tick: mark to market, rebalance if due, record + snapshot."""
    now = now or datetime.now(timezone.utc)
    account = paper.load_account()
    if account is None:
        raise RuntimeError("No paper account. Run `paper_run.py --reset` first.")

    hist, prices = _prices_now(xs.UNIVERSE + list(account.positions))
    account.ensure_benchmark(prices)

    trades: list[dict] = []
    if _due_to_rebalance(account, now):
        model = _get_model(hist, retrain=retrain)
        targets = target_tickers(model, hist)
        trades = rebalance(account, targets, prices)
        account.last_rebalance = now.isoformat()

    equity = account.equity(prices)
    bench = account.benchmark_equity(prices)
    _record_equity(now, equity, bench)
    paper.save_account(account)

    snapshot = {
        "generated_at": now.isoformat(),
        "inception_date": account.inception_date,
        "starting_equity": account.starting_equity,
        "equity": equity,
        "cash": account.cash,
        "total_return": equity / account.starting_equity - 1,
        "benchmark_equity": bench,
        "benchmark_return": bench / account.starting_equity - 1,
        "vs_benchmark": equity - bench,
        "holdings": [
            {"ticker": t, "shares": round(p["shares"], 4),
             "price": round(prices.get(t, 0.0), 2),
             "value": round(p["shares"] * prices.get(t, 0.0), 2),
             "cost_basis": round(p["cost_basis"], 2),
             "unrealized_pct": (prices.get(t, 0.0) / p["cost_basis"] - 1)
             if p["cost_basis"] else 0.0}
            for t, p in sorted(account.positions.items())
        ],
        "trades_this_tick": trades,
        "last_rebalance": account.last_rebalance,
    }
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    config.PAPER_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def run_loop(*, once: bool = False) -> None:
    while True:
        now = datetime.now(timezone.utc)
        if config.RUN_OUTSIDE_MARKET_HOURS or runtime.is_market_open(now):
            snap = run_once(now=now)
            print(f"[{now.isoformat()}] equity ${snap['equity']:.2f} "
                  f"({snap['total_return']:+.1%}) vs SPY ${snap['benchmark_equity']:.2f} "
                  f"({snap['benchmark_return']:+.1%})"
                  + (f" · {len(snap['trades_this_tick'])} trades" if snap['trades_this_tick'] else ""))
        else:
            print(f"[{now.isoformat()}] market closed — idle")
        if once:
            return
        time.sleep(config.LOOP_INTERVAL_SECONDS)
