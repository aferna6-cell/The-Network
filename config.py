"""Central configuration for the-network stock recommendation model.

Everything tunable lives here so experiments stay reproducible and there are no
magic numbers scattered through the codebase.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # cached price CSVs (gitignored)
MODELS_DIR = ROOT / "models"      # trained model artifacts (gitignored)
REPORTS_DIR = ROOT / "reports"    # backtest reports (gitignored)
NEWS_DIR = ROOT / "brain" / "news"  # structured news log (committed)

# --- Universe --------------------------------------------------------------
# Default tickers to train / track. SPY included as a market benchmark.
DEFAULT_TICKERS = ["AAPL", "MSFT", "SPY"]

# --- Data window -----------------------------------------------------------
HISTORY_START = "2010-01-01"   # how far back to pull daily bars

# --- Prediction target -----------------------------------------------------
# Horizon in trading days. 1 = predict tomorrow's direction.
TARGET_HORIZON = 1
# A move counts as "up" only if forward return exceeds this (in fraction).
# 0.0 = any positive move. Raising it filters out noise / tiny moves.
TARGET_THRESHOLD = 0.0

# --- Model -----------------------------------------------------------------
RANDOM_STATE = 42
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_depth": 4,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "random_state": RANDOM_STATE,
}

# --- Walk-forward backtest -------------------------------------------------
# Fraction of history used to train before the first out-of-sample prediction.
INITIAL_TRAIN_FRACTION = 0.6
# How often (trading days) to retrain as we walk forward. Larger = faster.
RETRAIN_EVERY = 21  # ~monthly

# --- Recommendation thresholds (probability of an up move) -----------------
# Used by the recommendation layer to turn a probability into an action.
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45

# --- Trading frictions (applied in backtest) -------------------------------
# Round-trip cost is charged whenever the position changes.
TRANSACTION_COST_BPS = 5.0   # commission, per side, in basis points
SLIPPAGE_BPS = 2.0           # assumed slippage, per side, in basis points
TRADING_DAYS_PER_YEAR = 252

# --- News --------------------------------------------------------------------
# Google News RSS is keyless and reliable. {query} is URL-encoded at call time.
NEWS_RSS_TEMPLATE = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)
# Per-ticker search query suffix to bias toward market-relevant news.
NEWS_QUERY_SUFFIX = "stock"
# Keep at most this many headlines per ticker per refresh.
NEWS_MAX_PER_TICKER = 25

# --- Always-on live loop ---------------------------------------------------
# The model trains on DAILY bars (next-day direction). The intraday loop
# re-fetches the latest data + current price and re-issues recommendations as
# the day moves — it does NOT retrain per tick (see brain/concepts/online-learning.md).
INTRADAY_INTERVAL = "5m"        # bar size for the live price refresh
INTRADAY_RANGE = "5d"           # how much intraday history to pull each tick
LOOP_INTERVAL_SECONDS = 300     # how often the live loop wakes (5 min)

# Retrain cadence: retrain once per new trading day (frozen snapshot between).
RETRAIN_EVERY_HOURS = 24

# Market hours (US equities), used to decide when the loop is "live".
MARKET_TZ = "America/New_York"
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
RUN_OUTSIDE_MARKET_HOURS = False  # if True, keep issuing recs when closed

# --- Portfolio + state (all gitignored — personal / regenerated) -----------
STATE_DIR = ROOT / "state"
PORTFOLIO_PATH = ROOT / "portfolio.json"       # your real holdings (gitignored)
LEDGER_PATH = STATE_DIR / "prediction_ledger.csv"
RECOMMENDATIONS_PATH = STATE_DIR / "recommendations.json"

# --- Portfolio-aware recommendation rules ----------------------------------
# Flag any single position above this share of the portfolio as concentrated.
MAX_POSITION_WEIGHT = 0.25
# Probability bands -> action label (BUY/SELL thresholds defined above).
STRONG_BUY_THRESHOLD = 0.60
STRONG_SELL_THRESHOLD = 0.40

# --- Live model-health / decay monitoring ----------------------------------
# Rolling window (number of reconciled predictions) for live accuracy.
MONITOR_WINDOW = 60
# If live rolling accuracy drops below this, flag the model as decayed.
MONITOR_MIN_ACCURACY = 0.48
# Warn that the model is stale once it is older than this (hours).
MONITOR_STALE_AFTER_HOURS = 48
