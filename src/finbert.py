"""FinBERT headline sentiment — the deep-learning half of the news experiment.

VADER (in `src/news.py`) is a 2014 lexicon: fast, keyless, but blind to finance
phrasing ("beats estimates", "cuts guidance", "headwinds"). FinBERT is a BERT
model fine-tuned on financial text, so it reads those the way an analyst would.
This is deep learning aimed where it is genuinely strong — language — rather
than bolted onto price bars where gradient-boosted trees already win.

`transformers` + `torch` are heavy and the model weights download from the
Hugging Face hub, so they are an OPTIONAL dependency (see requirements-deep.txt).
Import stays lazy and the error is explicit if the extras are missing — the core
test suite never needs them.

Output convention matches VADER's `compound`: a single signed score in [-1, 1],
computed as P(positive) - P(negative). That lets FinBERT scores flow through the
exact same `src/sentiment_features.py` plumbing with no special-casing.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

DEFAULT_MODEL = "ProsusAI/finbert"


@lru_cache(maxsize=2)
def _load_pipeline(model_name: str = DEFAULT_MODEL):
    """Lazily build a HF text-classification pipeline (cached per model)."""
    try:
        from transformers import pipeline
    except ImportError as err:  # extras not installed
        raise ImportError(
            "FinBERT needs the optional deep-learning extras. Install them with:\n"
            "    pip install -r requirements-deep.txt\n"
            f"(original error: {err})"
        ) from err
    return pipeline(
        "text-classification", model=model_name, top_k=None, truncation=True,
    )


def _signed_score(label_scores: list[dict]) -> float:
    """Map [{label, score}, ...] -> P(positive) - P(negative) in [-1, 1]."""
    by_label = {d["label"].lower(): float(d["score"]) for d in label_scores}
    return by_label.get("positive", 0.0) - by_label.get("negative", 0.0)


def score_texts(
    texts: list[str], *, model_name: str = DEFAULT_MODEL, batch_size: int = 32,
) -> list[float]:
    """Return a signed sentiment score in [-1, 1] for each text.

    Empty/whitespace texts score 0.0 without troubling the model.
    """
    results: list[float] = [0.0] * len(texts)
    todo = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    if not todo:
        return results
    clf = _load_pipeline(model_name)
    outputs = clf([t for _, t in todo], batch_size=batch_size)
    for (i, _), out in zip(todo, outputs):
        results[i] = round(_signed_score(out), 4)
    return results


def rescore_news(
    news: pd.DataFrame,
    *,
    model_name: str = DEFAULT_MODEL,
    out_col: str = "finbert_sentiment",
) -> pd.DataFrame:
    """Return a copy of a news log with FinBERT scores in a new column.

    Keeps the existing VADER `sentiment` column untouched so the two can be
    compared head-to-head as features. Title text drives the score.
    """
    out = news.copy()
    out[out_col] = score_texts(out["title"].fillna("").astype(str).tolist(),
                               model_name=model_name)
    return out
