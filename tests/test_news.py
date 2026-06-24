"""News parsing + sentiment, using a canned RSS payload (no network)."""
from __future__ import annotations

from src import news

_SAMPLE_RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>AAPL surges to record high on strong earnings beat</title>
    <link>https://example.com/a</link>
    <pubDate>Mon, 02 Jun 2025 13:00:00 GMT</pubDate>
    <source url="https://example.com">Example News</source>
  </item>
  <item>
    <title>Apple faces lawsuit and plunges on weak guidance</title>
    <link>https://example.com/b</link>
    <pubDate>Tue, 03 Jun 2025 09:00:00 GMT</pubDate>
    <source url="https://example.com">Example News</source>
  </item>
</channel></rss>"""


class _FakeResp:
    content = _SAMPLE_RSS

    def raise_for_status(self):
        pass


class _FakeSession:
    def get(self, *args, **kwargs):
        return _FakeResp()


def test_fetch_headlines_parses_and_scores():
    df = news.fetch_headlines("AAPL", session=_FakeSession())
    assert len(df) == 2
    assert set(df.columns) == set(news._LOG_COLUMNS)
    # Positive headline should score positive, negative one negative.
    pos = df[df["title"].str.contains("surges")].iloc[0]
    neg = df[df["title"].str.contains("plunges")].iloc[0]
    assert pos["sentiment"] > 0 and pos["sentiment_label"] == "positive"
    assert neg["sentiment"] < 0 and neg["sentiment_label"] == "negative"
    assert pos["published"] == "2025-06-02"  # point-in-time date preserved


def test_update_news_log_dedupes(tmp_path):
    first = news.update_news_log("AAPL", news_dir=tmp_path, session=_FakeSession())
    second = news.update_news_log("AAPL", news_dir=tmp_path, session=_FakeSession())
    # Re-running the same payload must not duplicate rows.
    assert len(first) == len(second) == 2

    summary = news.sentiment_summary("AAPL", news_dir=tmp_path)
    assert summary["n_headlines"] == 2
    assert summary["positive"] == 1 and summary["negative"] == 1
