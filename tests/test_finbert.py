"""FinBERT scorer tests.

The signed-score mapping and the empty-input short-circuit need no model and are
always tested. The real transformer path runs only when `transformers` is
installed (skip otherwise).
"""
from __future__ import annotations

import pytest

from src import finbert


def test_signed_score_positive_minus_negative():
    scores = [
        {"label": "positive", "score": 0.7},
        {"label": "negative", "score": 0.2},
        {"label": "neutral", "score": 0.1},
    ]
    assert finbert._signed_score(scores) == pytest.approx(0.5)


def test_signed_score_handles_missing_labels():
    assert finbert._signed_score([{"label": "neutral", "score": 1.0}]) == 0.0


def test_empty_texts_score_zero_without_model():
    # All-blank input must not attempt to load transformers.
    assert finbert.score_texts(["", "   ", None]) == [0.0, 0.0, 0.0]


def test_score_texts_real_model():
    pytest.importorskip("transformers", reason="optional deep-learning extra")
    scores = finbert.score_texts([
        "Company beats earnings estimates and raises guidance",
        "Company misses badly and cuts full-year outlook",
    ])
    assert len(scores) == 2
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[1]   # good news should outscore bad news
