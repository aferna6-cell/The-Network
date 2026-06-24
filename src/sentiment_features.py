"""Turn a point-in-time news log into a leak-free daily sentiment feature.

This is the careful part of the "does news help the model?" experiment. The
sentiment *scores* can come from anywhere (VADER today, FinBERT in
`src/finbert.py`), but whatever produces them, the rule from `features.py`
still rules everything here:

    The sentiment value on row `t` may use ONLY headlines knowable at the close
    of day `t`.

Two safeguards enforce that:

* We filter on each headline's **published** date, never its fetched date.
* We apply a `lag_days` (default 1) so a headline published on day `t` — which
  may have hit the wire *after* the close — is not used until day `t + 1`.
  Conservative on purpose: a tiny bit of signal forgone beats a leak.

No heavy dependencies live here. `tests/test_sentiment_features.py` asserts the
point-in-time property directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns this module appends to a feature dataset.
SENTIMENT_FEATURE_COLUMNS = ["sent_mean", "sent_count"]


def daily_sentiment_features(
    news: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    window: int = 5,
    lag_days: int = 1,
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """Trailing-window sentiment for each date in `index`, strictly as-of.

    For each date `d`, averages the sentiment of every headline whose published
    date falls in `[d - lag_days - window + 1, d - lag_days]` (calendar days).
    Dates with no qualifying headline get `sent_mean = 0.0` (neutral),
    `sent_count = 0`.

    Returns a frame indexed by `index` with SENTIMENT_FEATURE_COLUMNS.
    """
    out = pd.DataFrame(
        {"sent_mean": 0.0, "sent_count": 0.0}, index=index, dtype="float64"
    )
    if news is None or news.empty or sentiment_col not in news.columns:
        return out

    pub = pd.to_datetime(news["published"], errors="coerce")
    sent = pd.to_numeric(news[sentiment_col], errors="coerce")
    ok = pub.notna() & sent.notna()
    if not ok.any():
        return out

    # Sort headlines by published date so we can binary-search windows.
    order = pub[ok].argsort(kind="stable")
    pub_sorted = pub[ok].to_numpy()[order].astype("datetime64[D]")
    sent_sorted = sent[ok].to_numpy()[order]
    # Prefix sums let us get any window's sum/count in O(log n).
    csum = np.concatenate([[0.0], np.cumsum(sent_sorted)])

    idx_days = index.to_numpy().astype("datetime64[D]")
    one = np.timedelta64(1, "D")
    hi = idx_days - np.timedelta64(lag_days, "D")          # latest usable pub date
    lo = hi - np.timedelta64(window - 1, "D")              # earliest usable pub date

    # Half-open search bounds: include lo..hi inclusive.
    left = np.searchsorted(pub_sorted, lo, side="left")
    right = np.searchsorted(pub_sorted, hi + one, side="left")
    count = (right - left).astype("float64")
    total = csum[right] - csum[left]
    mean = np.where(count > 0, total / np.where(count > 0, count, 1.0), 0.0)

    out["sent_mean"] = mean
    out["sent_count"] = count
    return out


def attach_sentiment(
    dataset: pd.DataFrame,
    news: pd.DataFrame,
    *,
    window: int = 5,
    lag_days: int = 1,
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """Return `dataset` with SENTIMENT_FEATURE_COLUMNS joined on by date.

    `dataset` is a feature frame indexed by date (as from `features.make_dataset`).
    Missing dates get neutral sentiment, so the model never sees a NaN.
    """
    feats = daily_sentiment_features(
        news, dataset.index, window=window, lag_days=lag_days,
        sentiment_col=sentiment_col,
    )
    out = dataset.copy()
    out[SENTIMENT_FEATURE_COLUMNS] = feats[SENTIMENT_FEATURE_COLUMNS]
    return out
