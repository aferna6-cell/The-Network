"""Feature engineering — strictly leak-free.

The single most important rule in this whole project:

    Every feature on row `t` is computed using ONLY information available at the
    close of day `t`. The target uses ONLY days strictly after `t`.

If that rule is ever broken the backtest becomes fiction. The tests in
`tests/test_features.py` assert it explicitly (target alignment + no use of
future bars). All price-based features use the split/dividend-adjusted close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Columns produced by add_features(); the model trains on exactly these.
FEATURE_COLUMNS = [
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20",
    "mom_10", "mom_20",
    "sma_ratio_10", "sma_ratio_20", "sma_ratio_50",
    "ema_ratio_12", "ema_ratio_26",
    "sma_cross_10_50",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "boll_pctb",
    "vol_10", "vol_20",
    "volume_ratio_20",
    "range_pct",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing via EWM with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with all technical features appended.

    No row is dropped here; warm-up rows simply carry NaNs. Dropping happens in
    `make_dataset` so the caller controls it.
    """
    out = df.copy()
    price = out["adj_close"]
    log_ret = np.log(price / price.shift(1))

    # --- Lagged simple returns (all backward-looking) ----------------------
    for n in (1, 2, 5, 10, 20):
        out[f"ret_{n}"] = price.pct_change(n)

    # --- Momentum (price now vs price n days ago) --------------------------
    for n in (10, 20):
        out[f"mom_{n}"] = price / price.shift(n) - 1.0

    # --- Moving-average ratios (price relative to its own trend) -----------
    for n in (10, 20, 50):
        sma = price.rolling(n).mean()
        out[f"sma_ratio_{n}"] = price / sma - 1.0
    for n in (12, 26):
        ema = price.ewm(span=n, adjust=False, min_periods=n).mean()
        out[f"ema_ratio_{n}"] = price / ema - 1.0

    # Fast vs slow MA crossover (trend regime), normalised.
    sma10 = price.rolling(10).mean()
    sma50 = price.rolling(50).mean()
    out["sma_cross_10_50"] = sma10 / sma50 - 1.0

    # --- RSI ----------------------------------------------------------------
    out["rsi_14"] = _rsi(price, 14)

    # --- MACD ---------------------------------------------------------------
    ema12 = price.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = price.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd"] = macd / price            # normalise by price level
    out["macd_signal"] = signal / price
    out["macd_hist"] = (macd - signal) / price

    # --- Bollinger %b -------------------------------------------------------
    sma20 = price.rolling(20).mean()
    std20 = price.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["boll_pctb"] = (price - lower) / (upper - lower)

    # --- Realised volatility (rolling std of log returns) ------------------
    for n in (10, 20):
        out[f"vol_{n}"] = log_ret.rolling(n).std()

    # --- Volume relative to its own recent average -------------------------
    out["volume_ratio_20"] = out["volume"] / out["volume"].rolling(20).mean()

    # --- Intraday range as a share of price --------------------------------
    out["range_pct"] = (out["high"] - out["low"]) / price

    return out


def add_target(
    df: pd.DataFrame,
    horizon: int = config.TARGET_HORIZON,
    threshold: float = config.TARGET_THRESHOLD,
) -> pd.DataFrame:
    """Append `fwd_return` and binary `target` using ONLY future bars.

    target = 1 if the forward `horizon`-day return exceeds `threshold`, else 0.
    The last `horizon` rows get NaN (no future available) and are dropped later.
    """
    out = df.copy()
    price = out["adj_close"]
    fwd_return = price.shift(-horizon) / price - 1.0
    out["fwd_return"] = fwd_return
    out["target"] = (fwd_return > threshold).astype("float")
    # Re-mask the tail: rows without a real future must stay NaN, not 0.
    out.loc[fwd_return.isna(), "target"] = np.nan
    return out


def make_dataset(
    df: pd.DataFrame,
    horizon: int = config.TARGET_HORIZON,
    threshold: float = config.TARGET_THRESHOLD,
) -> pd.DataFrame:
    """Full pipeline: features + target, then drop warm-up / tail NaN rows.

    Returns a frame containing FEATURE_COLUMNS + ['fwd_return', 'target'] plus
    the original price columns, indexed by date.
    """
    framed = add_target(add_features(df), horizon=horizon, threshold=threshold)
    needed = FEATURE_COLUMNS + ["target"]
    return framed.dropna(subset=needed)


def latest_feature_row(df: pd.DataFrame) -> pd.Series:
    """Feature vector for the most recent bar (for live recommendations).

    Target is unknown here, so we only need features to be non-NaN.
    """
    framed = add_features(df).dropna(subset=FEATURE_COLUMNS)
    if framed.empty:
        raise ValueError("Not enough history to compute features")
    return framed.iloc[-1][FEATURE_COLUMNS]
