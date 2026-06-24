"""Shared test fixtures. All synthetic — tests never touch the network."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Deterministic geometric-random-walk OHLCV, ~3 years of trading days."""
    rng = np.random.default_rng(0)
    n = 750
    dates = pd.bdate_range("2018-01-01", periods=n, name="date")
    rets = rng.normal(0.0004, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": volume},
        index=dates,
    )
