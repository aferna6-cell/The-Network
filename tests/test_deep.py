"""Sequence-model tests.

The pure-numpy windowing is always tested. The torch training/prediction paths
run only when the optional extras are installed (skip otherwise), so CI without
deep deps stays green.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import deep
from src.features import FEATURE_COLUMNS

torch = pytest.importorskip("torch", reason="optional deep-learning extra")


def test_make_windows_shapes_and_alignment():
    feats = np.arange(40, dtype="float64").reshape(10, 4)
    wins, ends = deep.make_windows(feats, window=3)
    assert wins.shape == (8, 3, 4)         # rows 2..9 can end a window
    assert ends.tolist() == list(range(2, 10))
    # The last window ends on the last row and contains rows 7,8,9.
    assert np.array_equal(wins[-1], feats[7:10])


def test_make_windows_too_short_is_empty():
    feats = np.zeros((2, 4))
    wins, ends = deep.make_windows(feats, window=5)
    assert len(wins) == 0 and len(ends) == 0


def _synthetic_dataset(n=500, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(size=n) for c in FEATURE_COLUMNS}
    df = pd.DataFrame(data, index=pd.bdate_range("2020-01-01", periods=n))
    # Target weakly depends on a feature so training has something to learn.
    df["target"] = (df["ret_1"] + rng.normal(scale=0.5, size=n) > 0).astype("float")
    df["fwd_return"] = rng.normal(scale=0.01, size=n)
    return df


def test_train_and_predict_returns_probabilities():
    ds = _synthetic_dataset()
    model = deep.train_sequence_model(ds.iloc[:400], window=10)
    feats = ds[FEATURE_COLUMNS].to_numpy()
    wins, _ = deep.make_windows(feats[400:], window=10)
    proba = model.predict_proba_up(wins)
    assert proba.shape[0] == len(wins)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_training_is_reproducible():
    ds = _synthetic_dataset()
    feats = ds[FEATURE_COLUMNS].to_numpy()
    wins, _ = deep.make_windows(feats[400:420], window=10)
    p1 = deep.train_sequence_model(ds.iloc[:400], window=10).predict_proba_up(wins)
    p2 = deep.train_sequence_model(ds.iloc[:400], window=10).predict_proba_up(wins)
    assert np.allclose(p1, p2)


def test_walk_forward_sequence_is_out_of_sample():
    ds = _synthetic_dataset(n=480)
    preds = deep.walk_forward_sequence(
        ds, window=10, initial_train_fraction=0.7,
        retrain_every=120, min_train_rows=200,
    )
    assert list(preds.columns) == ["prob_up", "target", "fwd_return"]
    assert ((preds["prob_up"] >= 0) & (preds["prob_up"] <= 1)).all()
    # Predictions begin only after the initial training window.
    assert preds.index.min() >= ds.index[int(len(ds) * 0.7)]
