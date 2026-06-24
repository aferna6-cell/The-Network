"""Market-news intelligence — structured, point-in-time, per ticker.

What this is: a keyless Google-News-RSS fetcher that records headlines for each
tracked ticker, scores their sentiment with VADER (a lexicon model — no network,
no API key, fast on headlines), and appends them to a per-ticker CSV log under
brain/news/.

What this is NOT (yet): an input to trade recommendations. It is deliberately
kept as *research context*. Every row stores the headline's **published** date
so a future backtest can filter to news that was knowable at decision time
(point-in-time correctness). Wiring sentiment into the model without that
discipline is how you build a leaking, self-deluding system — so we don't.
"""
from __future__ import annotations

import urllib.parse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; the-network-news/1.0)"}
_ANALYZER = SentimentIntensityAnalyzer()
_LOG_COLUMNS = [
    "published", "fetched_at", "ticker", "title", "source", "link",
    "sentiment", "sentiment_label",
]


def _label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _parse_pubdate(raw: str | None) -> str:
    """RFC-822 RSS date -> ISO date string (UTC). Falls back to today."""
    if raw:
        try:
            return parsedate_to_datetime(raw).astimezone(timezone.utc).date().isoformat()
        except (TypeError, ValueError):
            pass
    return datetime.now(timezone.utc).date().isoformat()


def fetch_headlines(
    ticker: str,
    *,
    max_items: int = config.NEWS_MAX_PER_TICKER,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch and sentiment-score recent headlines for one ticker."""
    query = urllib.parse.quote(f"{ticker} {config.NEWS_QUERY_SUFFIX}")
    url = config.NEWS_RSS_TEMPLATE.format(query=query)
    sess = session or requests.Session()
    resp = sess.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    fetched_at = datetime.now(timezone.utc).date().isoformat()
    rows = []
    for item in list(root.iterfind(".//item"))[:max_items]:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        compound = _ANALYZER.polarity_scores(title)["compound"]
        rows.append({
            "published": _parse_pubdate(item.findtext("pubDate")),
            "fetched_at": fetched_at,
            "ticker": ticker.upper(),
            "title": title,
            "source": source,
            "link": (item.findtext("link") or "").strip(),
            "sentiment": round(compound, 4),
            "sentiment_label": _label(compound),
        })
    return pd.DataFrame(rows, columns=_LOG_COLUMNS)


def update_news_log(
    ticker: str,
    *,
    news_dir: Path | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch headlines and merge into brain/news/<TICKER>.csv, de-duplicated.

    De-dup key is (ticker, published, title) so re-running the same day does not
    create duplicates but genuinely new headlines are added.
    """
    news_dir = news_dir or config.NEWS_DIR
    news_dir.mkdir(parents=True, exist_ok=True)
    path = news_dir / f"{ticker.upper()}.csv"

    fresh = fetch_headlines(ticker, session=session)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, fresh], ignore_index=True)
    else:
        combined = fresh

    combined = (
        combined.drop_duplicates(subset=["ticker", "published", "title"], keep="first")
        .sort_values("published")
        .reset_index(drop=True)
    )
    combined.to_csv(path, index=False)
    return combined


def sentiment_summary(ticker: str, *, news_dir: Path | None = None) -> dict:
    """Aggregate the stored news log for a ticker into a quick snapshot."""
    news_dir = news_dir or config.NEWS_DIR
    path = news_dir / f"{ticker.upper()}.csv"
    if not path.exists():
        return {"ticker": ticker.upper(), "n_headlines": 0}
    df = pd.read_csv(path)
    if df.empty:
        return {"ticker": ticker.upper(), "n_headlines": 0}
    counts = df["sentiment_label"].value_counts().to_dict()
    return {
        "ticker": ticker.upper(),
        "n_headlines": int(len(df)),
        "mean_sentiment": round(float(df["sentiment"].mean()), 4),
        "latest_published": str(df["published"].max()),
        "positive": int(counts.get("positive", 0)),
        "neutral": int(counts.get("neutral", 0)),
        "negative": int(counts.get("negative", 0)),
    }
