"""Model training and persistence.

We use sklearn's HistGradientBoostingClassifier: strong on tabular data, no
native build headaches, handles the modest feature count well, and exposes
calibrated-ish probabilities we can threshold into recommendations.

This module knows nothing about time ordering — the caller (backtest.py) is
responsible for never handing it future data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import config
from src.features import FEATURE_COLUMNS


@dataclass
class TrainedModel:
    """A fitted classifier plus the metadata needed to use it safely later."""
    estimator: HistGradientBoostingClassifier
    feature_columns: list[str]
    horizon: int
    threshold: float
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    n_train_rows: int = 0

    def predict_proba_up(self, X: pd.DataFrame) -> pd.Series:
        """Probability of the 'up' class for each row of X."""
        X = X[self.feature_columns]
        proba = self.estimator.predict_proba(X)
        # Locate the column for class label 1.0 robustly.
        classes = list(self.estimator.classes_)
        up_idx = classes.index(1.0) if 1.0 in classes else len(classes) - 1
        return pd.Series(proba[:, up_idx], index=X.index, name="prob_up")


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    horizon: int = config.TARGET_HORIZON,
    threshold: float = config.TARGET_THRESHOLD,
    params: dict | None = None,
    feature_columns: list[str] | None = None,
) -> TrainedModel:
    """Fit a classifier on the given feature matrix and labels.

    `feature_columns` defaults to the canonical FEATURE_COLUMNS. Pass an
    extended list (e.g. baseline features + a sentiment column) to test whether
    an extra signal earns its keep — the model records exactly which columns it
    trained on so prediction uses the same set.
    """
    cols = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    params = {**config.MODEL_PARAMS, **(params or {})}
    estimator = HistGradientBoostingClassifier(**params)
    estimator.fit(X[cols], y)
    return TrainedModel(
        estimator=estimator,
        feature_columns=cols,
        horizon=horizon,
        threshold=threshold,
        n_train_rows=len(X),
    )


def save_model(model: TrainedModel, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> TrainedModel:
    return joblib.load(path)
