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
