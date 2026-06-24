"""Historical price data ingestion.

We deliberately do NOT use the `yfinance` library. In sandboxed / proxied
environments its `curl_cffi` backend fails TLS handshakes against the egress
proxy, and Stooq now sits behind a JavaScript challenge. The raw Yahoo Finance
chart JSON API, hit with plain `requests`, honours standard CA-bundle env vars
(REQUESTS_CA_BUNDLE) and works reliably. So we talk to it directly.

Everything here returns a tidy pandas DataFrame indexed by date with columns:
    open, high, low, close, adj_close, volume

`adj_close` (split/dividend adjusted) is what every downstream feature uses.
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import config

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; the-network/1.0)"}
_REQUIRED_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def _to_epoch(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp())


def fetch_history(
    ticker: str,
    start: str = config.HISTORY_START,
    end: str | None = None,
    *,
    session: requests.Session | None = None,
    retries: int = 3,
) -> pd.DataFrame:
    """Download daily OHLCV for `ticker` from the Yahoo chart API.

    Raises ValueError if the API returns no usable data.
    """
    end_epoch = int(time.time()) if end is None else _to_epoch(end)
    start_epoch = _to_epoch(start)
    url = _CHART_URL.format(ticker=ticker)
    params = {
        "period1": start_epoch,
        "period2": end_epoch,
        "interval": "1d",
        "events": "div,splits",
    }
    sess = session or requests.Session()

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = sess.get(url, params=params, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            return _parse_chart(resp.json(), ticker)
        except Exception as err:  # network / parse — retry with backoff
            last_err = err
            time.sleep(2 ** attempt)
    raise ValueError(f"Failed to fetch {ticker} after {retries} tries: {last_err}")


def _parse_chart(payload: dict, ticker: str) -> pd.DataFrame:
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"{ticker}: API error {chart['error']}")
    results = chart.get("result")
    if not results:
        raise ValueError(f"{ticker}: empty result")

    res = results[0]
    timestamps = res.get("timestamp")
    if not timestamps:
        raise ValueError(f"{ticker}: no timestamps returned")

    quote = res["indicators"]["quote"][0]
    # adjclose is absent for some symbols; fall back to close.
    adj_block = res["indicators"].get("adjclose")
    adj_close = adj_block[0]["adjclose"] if adj_block else quote["close"]

    df = pd.DataFrame({
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "adj_close": adj_close,
        "volume": quote["volume"],
    }, index=pd.to_datetime(timestamps, unit="s").normalize())
    df.index.name = "date"

    # Drop rows where the core price is missing (market holidays / bad ticks).
    df = df.dropna(subset=["close", "adj_close"])
    if df.empty:
        raise ValueError(f"{ticker}: all rows missing price data")
    return df[_REQUIRED_COLS]


def fetch_intraday(
    ticker: str,
    interval: str = "5m",
    range_: str = "5d",
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch recent intraday bars (e.g. 1m/5m) for the live loop.

    Returns a DataFrame indexed by timestamp with the standard OHLCV columns
    (adj_close mirrors close intraday — Yahoo does not adjust intraday bars).
    """
    url = _CHART_URL.format(ticker=ticker)
    params = {"interval": interval, "range": range_, "includePrePost": "false"}
    sess = session or requests.Session()
    resp = sess.get(url, params=params, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    res = payload["chart"]["result"][0]
    timestamps = res.get("timestamp")
    if not timestamps:
        raise ValueError(f"{ticker}: no intraday data for {interval}/{range_}")
    quote = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "open": quote["open"], "high": quote["high"], "low": quote["low"],
        "close": quote["close"], "adj_close": quote["close"],
        "volume": quote["volume"],
    }, index=pd.to_datetime(timestamps, unit="s"))
    df.index.name = "datetime"
    return df.dropna(subset=["close"])


def latest_quote(
    ticker: str, *, session: requests.Session | None = None
) -> dict:
    """Return the current price snapshot from the chart API meta block.

    Keys: price, previous_close, currency, market_state, exchange_tz.
    `market_state` is e.g. PRE / REGULAR / POST / CLOSED.
    """
    url = _CHART_URL.format(ticker=ticker)
    params = {"interval": "1d", "range": "1d"}
    sess = session or requests.Session()
    resp = sess.get(url, params=params, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    meta = resp.json()["chart"]["result"][0]["meta"]
    return {
        "ticker": ticker.upper(),
        "price": meta.get("regularMarketPrice"),
        "previous_close": meta.get("chartPreviousClose") or meta.get("previousClose"),
        "currency": meta.get("currency"),
        "market_state": meta.get("marketState"),
        "exchange_tz": meta.get("exchangeTimezoneName"),
    }


def load_or_fetch(
    ticker: str,
    start: str = config.HISTORY_START,
    end: str | None = None,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return price history, using a local CSV cache to avoid re-downloading.

    Set `refresh=True` to force a fresh download.
    """
    cache_dir = cache_dir or config.DATA_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker.upper()}.csv"

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, index_col="date", parse_dates=True)
        return df[_REQUIRED_COLS]

    df = fetch_history(ticker, start=start, end=end)
    df.to_csv(cache_path)
    return df
