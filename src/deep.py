"""Sequence model (LSTM) — the deep-learning challenger to the tree baseline.

The honest question this answers: *does the temporal ordering of the features
carry signal that a gradient-boosted tree, which sees each day independently,
throws away?* So the LSTM is fed the **exact same** leak-free features
(`features.FEATURE_COLUMNS`) the tree uses — only as an ordered window of the
last `SEQ_WINDOW` days rather than a single flattened row. Same information,
same walk-forward, same costs, same metrics. The only thing that differs is the
model class. That is the only way the comparison means anything.

Leak discipline carried over verbatim from `backtest.py`:

* Expanding-window walk-forward; never train on bars after the test block.
* The last `horizon` rows before each cutoff are embargoed from training.
* The feature scaler is fit on TRAIN rows only.
* A prediction window for day `t` uses feature rows `t-W+1 .. t` — all <= t, all
  point-in-time. No future bar ever enters an input.

`torch` is an OPTIONAL dependency (requirements-deep.txt). The import is lazy so
the core suite runs without it; `tests/test_deep.py` skips when torch is absent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from src.backtest import _COST_RATE, _annualised_metrics
from src.features import FEATURE_COLUMNS, make_dataset


def _require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as err:
        raise ImportError(
            "The sequence model needs the optional deep-learning extras. Install:\n"
            "    pip install -r requirements-deep.txt\n"
            f"(original error: {err})"
        ) from err


# --------------------------------------------------------------------------- #
# Windowing (pure numpy — testable without torch)
# --------------------------------------------------------------------------- #
def make_windows(
    feats: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice `feats` (rows x features) into overlapping windows.

    Returns (windows, end_idx) where windows[k] = feats[end_idx[k]-W+1 : end_idx[k]+1]
    and end_idx[k] is the row each window *ends on* (the day being described).
    The first `window-1` rows cannot end a full window and are skipped.
    """
    n = len(feats)
    if n < window:
        return np.empty((0, window, feats.shape[1]), dtype="float32"), np.empty(0, dtype=int)
    ends = np.arange(window - 1, n)
    idx = ends[:, None] - (window - 1) + np.arange(window)[None, :]
    return feats[idx].astype("float32"), ends


@dataclass
class _Standardizer:
    """Column mean/std fit on training rows only (leak-free scaling)."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "_Standardizer":
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class SequenceModel:
    """A trained LSTM plus the metadata needed to use it safely later."""

    def __init__(self, net, scaler: _Standardizer, feature_columns: list[str],
                 window: int, horizon: int):
        self.net = net
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.window = window
        self.horizon = horizon

    def predict_proba_up(self, windows: np.ndarray) -> np.ndarray:
        """P(up) for a batch of pre-built windows (rows x window x features)."""
        torch = _require_torch()
        if len(windows) == 0:
            return np.empty(0, dtype="float32")
        scaled = self.scaler.transform(windows)
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.nan_to_num(scaled).astype("float32"))
            logits = self.net(x).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()


def _build_net(n_features: int):
    torch = _require_torch()
    import torch.nn as nn

    class _LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                n_features, config.SEQ_HIDDEN, num_layers=config.SEQ_LAYERS,
                batch_first=True,
                dropout=config.SEQ_DROPOUT if config.SEQ_LAYERS > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Dropout(config.SEQ_DROPOUT), nn.Linear(config.SEQ_HIDDEN, 1)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])   # last timestep -> logit

    return _LSTM()


def train_sequence_model(
    train: pd.DataFrame,
    *,
    feature_columns: list[str] = FEATURE_COLUMNS,
    window: int = config.SEQ_WINDOW,
    horizon: int = config.TARGET_HORIZON,
    seed: int = config.RANDOM_STATE,
) -> SequenceModel:
    """Fit the LSTM on one expanding-window training block."""
    torch = _require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)

    feats = train[feature_columns].to_numpy(dtype="float64")
    y = train["target"].to_numpy(dtype="float32")
    scaler = _Standardizer.fit(feats)

    windows, ends = make_windows(feats, window)
    labels = y[ends]
    keep = ~np.isnan(labels)
    windows, labels = windows[keep], labels[keep]
    if len(windows) < config.SEQ_BATCH:
        raise ValueError("Not enough windowed samples to train the sequence model")

    X = np.nan_to_num(scaler.transform(windows)).astype("float32")
    # Time-ordered validation tail for early stopping (no shuffle across the split).
    n_val = max(1, int(len(X) * config.SEQ_VAL_FRACTION))
    Xtr, ytr = X[:-n_val], labels[:-n_val]
    Xval, yval = X[-n_val:], labels[-n_val:]

    net = _build_net(len(feature_columns))
    opt = torch.optim.Adam(net.parameters(), lr=config.SEQ_LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    tx = torch.from_numpy(Xtr)
    ty = torch.from_numpy(ytr)
    vx = torch.from_numpy(Xval)
    vy = torch.from_numpy(yval)

    best_val, best_state, patience = float("inf"), None, 0
    gen = torch.Generator().manual_seed(seed)
    for _ in range(config.SEQ_EPOCHS):
        net.train()
        perm = torch.randperm(len(tx), generator=gen)
        for s in range(0, len(tx), config.SEQ_BATCH):
            bi = perm[s:s + config.SEQ_BATCH]
            opt.zero_grad()
            loss = loss_fn(net(tx[bi]).squeeze(-1), ty[bi])
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            vloss = float(loss_fn(net(vx).squeeze(-1), vy))
        if vloss < best_val - 1e-4:
            best_val, patience = vloss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= config.SEQ_PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return SequenceModel(net, scaler, list(feature_columns), window, horizon)


def walk_forward_sequence(
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str] = FEATURE_COLUMNS,
    window: int = config.SEQ_WINDOW,
    horizon: int = config.TARGET_HORIZON,
    initial_train_fraction: float = config.INITIAL_TRAIN_FRACTION,
    retrain_every: int = config.RETRAIN_EVERY,
    min_train_rows: int = 250,
) -> pd.DataFrame:
    """Out-of-sample P(up) from the LSTM, same protocol as walk_forward_predict.

    Returns a frame indexed by date with columns: prob_up, target, fwd_return.
    """
    dataset = dataset.sort_index()
    feats_all = dataset[feature_columns].to_numpy(dtype="float64")
    n = len(dataset)
    initial = int(n * initial_train_fraction)
    if initial < min_train_rows:
        raise ValueError(
            f"Not enough data: need >= {min_train_rows} train rows, have {initial}."
        )

    chunks: list[pd.DataFrame] = []
    i = initial
    while i < n:
        block_end = min(i + retrain_every, n)
        train_end = i - horizon            # embargo the last `horizon` rows
        train = dataset.iloc[:train_end]
        if len(train) < min_train_rows:
            i = block_end
            continue

        model = train_sequence_model(
            train, feature_columns=feature_columns, window=window, horizon=horizon,
        )
        # Build a window ending on each test row from point-in-time rows (<= t).
        test_pos = np.arange(i, block_end)
        valid = test_pos[test_pos >= window - 1]
        starts = valid - (window - 1)
        wins = np.stack([feats_all[s:e + 1] for s, e in zip(starts, valid)]) \
            if len(valid) else np.empty((0, window, len(feature_columns)))
        prob = model.predict_proba_up(wins)

        test = dataset.iloc[i:block_end]
        prob_series = pd.Series(np.nan, index=test.index)
        prob_series.iloc[valid - i] = prob
        chunks.append(pd.DataFrame({
            "prob_up": prob_series.values,
            "target": test["target"].values,
            "fwd_return": test["fwd_return"].values,
        }, index=test.index))
        i = block_end

    if not chunks:
        raise ValueError("Walk-forward produced no out-of-sample predictions")
    return pd.concat(chunks).dropna(subset=["prob_up"])


# --------------------------------------------------------------------------- #
# Fair economics — identical accounting applied to either model's prob_up
# --------------------------------------------------------------------------- #
def economics(
    prob_up: pd.Series, prices: pd.DataFrame, *, buy_threshold: float = config.BUY_THRESHOLD,
) -> dict:
    """Daily long-or-flat PnL vs buy-and-hold, using the real next-day return.

    Same friction model and metrics as src/experiments.py, so a tree and an LSTM
    are scored on a level field.
    """
    adj = prices["adj_close"]
    next_day = adj.pct_change().shift(-1)
    df = pd.DataFrame({"prob_up": prob_up})
    df["next_day"] = next_day.reindex(df.index)
    df = df.dropna(subset=["next_day", "prob_up"])

    pos = (df["prob_up"] > buy_threshold).astype(float)
    prev = pos.shift(1).fillna(0.0)
    cost = _COST_RATE * (pos - prev).abs()
    strat_ret = pos * df["next_day"] - cost
    strat_eq = (1 + strat_ret).cumprod()
    bench_eq = (1 + df["next_day"]).cumprod()
    return {
        "n_oos": int(len(df)),
        "avg_exposure": float(pos.mean()),
        "strat": _annualised_metrics(strat_ret, strat_eq),
        "bench": _annualised_metrics(df["next_day"], bench_eq),
    }


def head_to_head(prices: pd.DataFrame, *, horizon: int = config.TARGET_HORIZON) -> dict:
    """Tree vs LSTM on one ticker over a shared out-of-sample window."""
    from src.backtest import walk_forward_predict

    dataset = make_dataset(prices, horizon=horizon)
    tree_pred = walk_forward_predict(dataset, horizon=horizon)
    lstm_pred = walk_forward_sequence(dataset, horizon=horizon)
    shared = tree_pred.index.intersection(lstm_pred.index)

    tree = economics(tree_pred.loc[shared, "prob_up"], prices)
    lstm = economics(lstm_pred.loc[shared, "prob_up"], prices)
    return {"n_shared_oos": int(len(shared)), "tree": tree, "lstm": lstm}
