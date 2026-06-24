# Deploying the always-on loop

The live recommender (`scripts/run_live.py`) is a long-running process. It needs a
**persistent host** — this repo's dev container is ephemeral and will not keep it
alive. Below are the realistic options, cheapest first.

> It places **no trades**. It writes recommendations to `state/` for the
> dashboard to read; you decide what to act on.

## What to run (the recommended pair)
1. **The always-on monitor** — `python scripts/run_live_xs.py` — continuously
   refreshes prices, re-values your portfolio, reconciles past calls, recomputes
   the live track record, and re-ranks (retraining daily). Writes
   `state/xs_recommendations.json` + `state/xs_ledger.csv`.
2. **The dashboard** — `streamlit run dashboard/app.py` — reads that state and
   shows Buy/Sell/Trim, holds, portfolio value, and the live track record.

Run both on the persistent host (they share `state/`). On Railway/Fly, run the
loop as the main process and the dashboard as a second service (or a separate
small deploy) pointed at the same volume. `scripts/run_live.py` (single-stock
loop) remains available but the cross-sectional pair above is the current path.

## What you must provide
- **`portfolio.json`** — your holdings (copy `portfolio.example.json`). Never
  commit this; it is gitignored.
- **A volume mounted at `state/`** — so the prediction ledger survives restarts.
  Without persistence the model can never measure its own live accuracy.

## The $500 paper demo runs WITHOUT a host
The simplest "let it run" is already wired: the **`paper-trade` GitHub Action**
ticks the paper account every trading day and commits the equity curve to
`paper_state/` in the repo — no server, nothing for you to keep alive. View the
curve any time in `paper_state/paper_equity.csv` or the dashboard. Enable it once
in the repo's **Actions** tab (scheduled workflows need a manual first enable).

A persistent host is only needed if you want the **live dashboard** or
**intraday** marks. Options below.

## Option A — Railway (managed, ~$5/mo) — recommended for the live dashboard
Railway builds the `Dockerfile` automatically. Run the demo as two services off
the same repo:

1. **Service 1 — paper loop.** Override the start command to
   `python scripts/paper_run.py` and add a **volume mounted at `/app/paper_state`**
   (so the curve persists). This keeps the account ticking intraday.
2. **Service 2 — dashboard.** Start command
   `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`,
   mounted on the **same volume** `/app/paper_state`, exposed publicly. That's the
   live web view of the account.
3. (Optional) For the recommender loop too, a third service running
   `python scripts/run_live_xs.py` sharing `/app/state`.

Render and Fly.io work the same way (detect the `Dockerfile`, add a volume, set
the start command per service).

## Option B — Small VM (DigitalOcean/EC2/Hetzner, ~$4–6/mo)
```bash
git clone <repo> && cd The-Network
cp portfolio.example.json portfolio.json   # then edit with your real holdings
docker build -t the-network .
docker run -d --restart unless-stopped \
  -v "$PWD/portfolio.json:/app/portfolio.json" \
  -v networkstate:/app/state \
  --name the-network the-network
```

## Option C — No server (batch only)
If you don't want a 24/7 host, run a single pass on a schedule via GitHub Actions
or cron:
```bash
python scripts/run_live.py --once
```
This loses true intraday responsiveness but costs nothing and still retrains,
recommends, and updates the ledger once per run.

## Upgrading the data feed (recommended for serious intraday)
The default free Yahoo intraday feed is **delayed and rate-limited**. For
production intraday, swap `src/data.py`'s fetchers for a real-time provider
(Alpaca, Polygon, IEX). The rest of the loop is unchanged — only the data layer
needs new credentials.

## Environment notes
- Behind a TLS-terminating proxy: `export REQUESTS_CA_BUNDLE=/path/to/ca-bundle.crt`.
- Tune cadence/universe in `config.py` (`LOOP_INTERVAL_SECONDS`,
  `RETRAIN_EVERY_HOURS`, `DEFAULT_TICKERS`, market hours).
