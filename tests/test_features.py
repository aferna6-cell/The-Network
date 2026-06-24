"""Feature correctness + the all-important leak-free guarantees."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import features


def test_feature_columns_present(synthetic_prices):
    framed = features.add_features(synthetic_prices)
    for col in features.FEATURE_COLUMNS:
        assert col in framed.columns, f"missing feature {col}"


def test_make_dataset_has_no_nans(synthetic_prices):
    ds = features.make_dataset(synthetic_prices)
    cols = features.FEATURE_COLUMNS + ["target"]
    assert not ds[cols].isna().any().any(), "dataset still contains NaNs"
    assert len(ds) > 0


def test_target_alignment_is_forward_looking(synthetic_prices):
    """target on row t must reflect the move from t to t+horizon, nothing else."""
    horizon = 1
    framed = features.add_target(synthetic_prices, horizon=horizon, threshold=0.0)
    price = synthetic_prices["adj_close"]
    # Recompute the expected forward return independently.
    expected_up = (price.shift(-horizon) / price - 1.0) > 0.0
    aligned = framed["target"].dropna()
    assert (aligned == expected_up.loc[aligned.index].astype(float)).all()


def test_last_horizon_rows_have_no_target(synthetic_prices):
    horizon = 3
    framed = features.add_target(synthetic_prices, horizon=horizon)
    # The final `horizon` rows cannot have a real future label.
    assert framed["target"].iloc[-horizon:].isna().all()


def test_features_do_not_use_future_bars(synthetic_prices):
    """Mutating bars AFTER row t must not change any feature value AT row t.

    This is the core no-look-ahead property. We compute features, then corrupt
    the tail of the price series and recompute; every feature on early rows must
    be identical.
    """
    base = features.add_features(synthetic_prices)[features.FEATURE_COLUMNS]

    corrupted = synthetic_prices.copy()
    cut = 500
    # Replace everything from `cut` onward with garbage.
    corrupted.iloc[cut:, :] = corrupted.iloc[cut:, :] * 3.0 + 7.0
    recomputed = features.add_features(corrupted)[features.FEATURE_COLUMNS]

    # Rows strictly before the corruption must be untouched.
    pd.testing.assert_frame_equal(
        base.iloc[:cut], recomputed.iloc[:cut], check_exact=False, atol=1e-9
    )


def test_rsi_bounds(synthetic_prices):
    framed = features.add_features(synthetic_prices)
    rsi = framed["rsi_14"].dropna()
    assert rsi.between(0, 100).all()
