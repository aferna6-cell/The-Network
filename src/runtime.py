"""The always-on loop.

Each tick (default every 5 min while the market is open):

    1. refresh data + current prices
    2. retrain if the model is older than the cadence (frozen snapshot between)
    3. recompute the latest feature row per ticker
    4. value the portfolio, generate portfolio-aware recommendations
    5. log each call to the prediction ledger
    6. reconcile past calls whose horizon elapsed -> live accuracy
    7. run the decay monitor; suppress recommendations if the model is failing
    8. write a recommendations snapshot for a dashboard / notifier to read

It retrains on a CADENCE, never per tick — see brain/concepts/online-learning.md
for why continuous self-training on noisy market data is a footgun.

Designed to run on a persistent host (Dockerfile provided). State lives under
`state/` — mount a volume there so the ledger survives restarts.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import config
from src import data, features, ledger, monitor, recommend
from src.model import TrainedModel, load_model, save_model, train_model
from src.portfolio import load_portfolio


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


def is_market_open(now: datetime | None = None) -> bool:
    """True during US equity regular hours (Mon–Fri, 09:30–16:00 ET)."""
    tz = ZoneInfo(config.MARKET_TZ)
    now = (now or datetime.now(timezone.utc)).astimezone(tz)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    return _parse_hhmm(config.MARKET_OPEN) <= now.timetz().replace(tzinfo=None) \
        <= _parse_hhmm(config.MARKET_CLOSE)


def retrain_due(model: TrainedModel | None, now: datetime | None = None) -> bool:
    """True if there is no model or it is older than the retrain cadence."""
    if model is None:
        return True
    now = now or datetime.now(timezone.utc)
    try:
        trained = datetime.fromisoformat(model.trained_at)
    except (ValueError, TypeError):
        return True
    if trained.tzinfo is None:
        trained = trained.replace(tzinfo=timezone.utc)
    age_h = (now - trained).total_seconds() / 3600.0
    return age_h >= config.RETRAIN_EVERY_HOURS


def _model_path(ticker: str):
    return config.MODELS_DIR / f"{ticker.upper()}.joblib"


def _load_or_train(ticker: str, *, horizon: int, threshold: float) -> TrainedModel | None:
    """Return a fresh model, retraining from daily history if the cadence is due."""
    path = _model_path(ticker)
    model = load_model(path) if path.exists() else None
    if not retrain_due(model):
        return model
    try:
        prices = data.load_or_fetch(ticker, refresh=True)
        ds = features.make_dataset(prices, horizon=horizon, threshold=threshold)
        model = train_model(ds[features.FEATURE_COLUMNS], ds["target"],
                            horizon=horizon, threshold=threshold)
        save_model(model, path)
        return model
    except Exception as err:
        print(f"  [retrain] {ticker}: failed ({err}); using existing model if any")
        return model


def run_once(
    tickers: list[str] | None = None,
    *,
    horizon: int = config.TARGET_HORIZON,
    threshold: float = config.TARGET_THRESHOLD,
    now: datetime | None = None,
) -> dict:
    """One full pass of the loop. Returns the recommendations snapshot dict."""
    tickers = tickers or config.DEFAULT_TICKERS
    now = now or datetime.now(timezone.utc)
    portfolio = load_portfolio()
    # Always consider held names plus the configured watchlist.
    universe = sorted(set(tickers) | set(portfolio.tickers))

    models: dict[str, TrainedModel] = {}
    feature_rows: dict[str, object] = {}
    prices: dict[str, float] = {}

    for ticker in universe:
        try:
            hist = data.load_or_fetch(ticker, refresh=True)
            feature_rows[ticker] = features.latest_feature_row(hist)
            quote = data.latest_quote(ticker)
            prices[ticker] = quote["price"] or float(hist["adj_close"].iloc[-1])
            model = _load_or_train(ticker, horizon=horizon, threshold=threshold)
            if model is not None:
                models[ticker] = model
        except Exception as err:
            print(f"  [ingest] {ticker}: skipped ({err})")

    # Reconcile past calls against the latest prices, then assess health.
    ledger.reconcile(lambda t: prices.get(t), now=now)
    newest_version = max((m.trained_at for m in models.values()), default=None)
    report = monitor.health_report(newest_version, now=now)
    suppress = monitor.should_suppress(report)

    recs = recommend.recommend(
        models, feature_rows, prices, portfolio, suppress=suppress)

    # Log every call so the ledger can later prove whether it was right.
    for r in recs:
        ledger.append_prediction(
            ticker=r.ticker, horizon_days=horizon, price_at_pred=r.price,
            prob_up=r.prob_up, recommendation=r.action,
            model_version=models[r.ticker].trained_at, now=now,
        )

    snapshot = {
        "generated_at": now.isoformat(),
        "market_open": is_market_open(now),
        "portfolio_value": portfolio.total_value(prices),
        "cash": portfolio.cash,
        "health": report,
        "recommendations": [r.to_dict() for r in recs],
    }
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    config.RECOMMENDATIONS_PATH.write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def run_loop(tickers: list[str] | None = None, *, once: bool = False) -> None:
    """Run forever (or a single pass with once=True)."""
    while True:
        now = datetime.now(timezone.utc)
        if config.RUN_OUTSIDE_MARKET_HOURS or is_market_open(now):
            snap = run_once(tickers, now=now)
            hp = snap["health"]["status"]
            print(f"[{now.isoformat()}] {len(snap['recommendations'])} recs · {hp}")
        else:
            print(f"[{now.isoformat()}] market closed — idle")
        if once:
            return
        time.sleep(config.LOOP_INTERVAL_SECONDS)
