# the-network

A **leak-free, cost-aware research stack** for stock-market recommendations,
plus a lightweight **second brain** (knowledge graph + point-in-time news log).

> **Read this first.** This is a research framework, **not** a money printer.
> Markets are near-efficient (see [`brain/concepts/efficient-market-hypothesis.md`](brain/concepts/efficient-market-hypothesis.md)).
> Expect ~50–55% next-day directional accuracy, and expect most strategies to
> **trail buy-and-hold after costs**. The value here is an honest pipeline you
> can trust — one that reports realistic numbers instead of overfit fantasies.
> It places no real trades.

## What it does

1. **Ingests** daily price history from the Yahoo chart JSON API (`src/data.py`).
2. **Engineers** ~22 strictly backward-looking technical features (`src/features.py`).
3. **Trains** a gradient-boosted classifier to predict next-day direction (`src/model.py`).
4. **Backtests** it walk-forward, out-of-sample, with transaction costs + slippage,
   always compared to buy-and-hold (`src/backtest.py`).
5. **Tracks** sentiment-scored, point-in-time market news per ticker (`src/news.py`)
   — kept as research context, deliberately *not* wired into recommendations yet.
6. **Maintains** a curated knowledge graph of the methods used (`brain/`).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train + backtest on historical data (downloads + caches automatically)
python scripts/train.py --tickers AAPL MSFT SPY

# Refresh the point-in-time news log
python scripts/update_news.py --tickers AAPL MSFT SPY

# Regenerate the brain knowledge graph after editing notes
python scripts/build_brain.py

# Run the test suite (offline — synthetic data)
pytest -q
```

> In sandboxes behind a TLS-terminating proxy, point requests at the CA bundle:
> `export REQUESTS_CA_BUNDLE=/path/to/ca-bundle.crt`.

## Example backtest (real run, AAPL/SPY, ~1,600 OOS days)

|                | AAPL strat | AAPL B&H | SPY strat | SPY B&H |
|----------------|-----------:|---------:|----------:|--------:|
| CAGR           |      9.5%  |   25.8%  |    -3.0%  |  15.5%  |
| Sharpe         |      0.58  |    0.89  |    -0.09  |   0.81  |
| Max drawdown   |    -25.8%  |  -33.4%  |   -40.1%  | -33.7%  |
| Directional acc|     51.9%  |     —    |    54.7%  |    —    |

The honest read: this baseline **does not beat buy-and-hold**. That is the
expected result and exactly why the framework exists — to tell you the truth
before any money is on the line. Beating the benchmark would require a real,
cost-survivable edge we have not yet found.

## Project layout

```
config.py            # all tunables (tickers, horizon, costs, thresholds)
src/
  data.py            # Yahoo chart JSON API -> OHLCV, with CSV cache
  features.py        # leak-free technical features + target
  model.py           # HistGradientBoosting train + persistence
  backtest.py        # walk-forward, purged, cost-aware backtest + metrics
  news.py            # keyless RSS news + VADER sentiment, point-in-time
  brain.py           # parse [[wikilinks]] -> knowledge graph + Home.md
scripts/             # train.py, update_news.py, build_brain.py
brain/               # second brain: concepts/ (notes) + news/ (logs) + Maps/
tests/               # offline tests: leakage guards, cost/metric math, news
.github/workflows/   # tests CI + scheduled news refresh
```

## Design decisions worth knowing

- **No `yfinance`.** Its `curl_cffi` backend fails TLS through egress proxies;
  the raw chart API with `requests` is more robust. See `src/data.py`.
- **Fundamentals are written once, not re-researched on a schedule.** Re-scraping
  stable ML concepts weekly is wasted effort. Only the **news** log recurs.
- **Recurrence runs on GitHub Actions, not local cron** — ephemeral dev
  containers do not keep a scheduler alive.
- **News is research context, not a signal — yet.** Wiring it in without
  point-in-time discipline is textbook leakage. See
  [`brain/concepts/news-sentiment.md`](brain/concepts/news-sentiment.md).

## Deep learning (optional experiments)

Classical gradient-boosted trees are the right default for small, noisy,
tabular daily data — so they stay the baseline. But two **honest** deep-learning
experiments now live in the repo, each running under the *same* leak-free
walk-forward and cost model as the tree (the only fair way to compare):

```bash
pip install -r requirements.txt -r requirements-deep.txt   # torch + transformers

# (B) LSTM over a window of the SAME features vs the tree, head-to-head
python scripts/deep_experiment.py --tickers AAPL SPY

# (A) FinBERT headline sentiment as a point-in-time feature vs price-only baseline
python scripts/update_news.py --tickers AAPL          # build the news log first
python scripts/sentiment_experiment.py --tickers AAPL --finbert
```

The honest prior, which these exist to test rather than decorate: on daily bars
after costs the **tree is expected to win**, and the sentiment feature only
starts to matter once enough *forward* news accumulates (the committed log is
weeks deep, not years). Leak discipline carries over verbatim — the feature
scaler is fit on training rows only, and a prediction window for day *t* uses
only rows ≤ *t*. See
[`brain/concepts/neural-networks.md`](brain/concepts/neural-networks.md). These
extras are excluded from `requirements.txt` and the core test suite skips their
paths when torch/transformers are absent.

## Monitoring the paper-trading demo

Two ways to watch the $500 forward test:

- **Always-on web page** (no server). The paper-trade Action regenerates a
  self-contained `docs/index.html` every tick and commits it. Publish it once via
  **Settings → Pages → Source: `main` / `docs`**, then it lives at
  `https://aferna6-cell.github.io/The-Network/` and refreshes daily. Build it
  locally any time with `python scripts/build_paper_dashboard.py`.
- **Daily write-up** on [issue #2](https://github.com/aferna6-cell/The-Network/issues/2)
  — subscribe for a notification each trading day.
- **Local deep-dive:** `streamlit run dashboard/app.py` for the interactive view
  (recommendations + live track record alongside the paper demo).

## Roadmap

- **Phase 1 (this repo):** historical training + honest backtest. ✅
- **Phase 2:** dashboards reading these artifacts — static paper-trading monitor
  (`docs/index.html`) + Streamlit deep-dive. ✅
- **In progress:** sentiment as a backtested feature (FinBERT) + an LSTM
  sequence challenger — both wired under the honest harness, pending enough
  forward news + an out-of-sample edge that survives costs.
- **Later (validated separately):** richer universe; regime awareness.

## Disclaimer

For research and education only. Nothing here is financial advice. Past
backtested performance does not predict future results. Do not trade real money
based on this without independent validation and your own risk management.
