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

## Option A — Railway / Render / Fly.io (managed, ~$5/mo)
1. Connect the repo; these platforms detect the `Dockerfile`.
2. Add a **persistent volume** mounted at `/app/state`.
3. Upload `portfolio.json` as a secret file or mount it at `/app/portfolio.json`.
4. Deploy. Logs show one line per tick.

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
