# brain/ — the second brain

A lightweight, source-controlled knowledge layer for this project. Two parts:

## `concepts/` — curated reference notes
Hand-written, leak-free notes on the methods this repo actually uses (ML,
backtesting, risk, market structure). They cross-reference each other with
Obsidian-style `[[wikilinks]]`, which `src/brain.py` parses into a knowledge
graph.

**These are written once and updated only when our methods change.** They are
*not* re-researched on a schedule — re-scraping stable fundamentals weekly is
wasted effort (the decision behind this design).

Start at [`Maps/Home.md`](Maps/Home.md) — the auto-generated map of content with
a concept graph. Regenerate it after editing notes:

```bash
python scripts/build_brain.py
```

## `news/` — point-in-time market news log
Per-ticker CSVs of sentiment-scored headlines (`<TICKER>.csv`), appended by
`scripts/update_news.py`. This is the **only** part that runs on a recurring
schedule (a GitHub Action — local cron does not survive ephemeral containers).

Every row stores the headline's **published** date so a future backtest can
filter to news knowable at decision time. News is kept as research context and is
deliberately **not** wired into trade recommendations yet — see
[`concepts/news-sentiment.md`](concepts/news-sentiment.md).

## Files
- `Maps/Home.md` — generated index + mermaid graph (do not edit by hand)
- `graph.json` — generated machine-readable nodes + edges
- `concepts/*.md` — the source notes (edit these)
- `news/*.csv` — generated news logs
