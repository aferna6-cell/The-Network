# News Sentiment

Markets react to news, and reactions can over- or under-shoot — a documented
source of short-lived inefficiency under the [[Efficient Market Hypothesis]].
This repo captures news as **research context**, not (yet) as a trade input.

## How it works (`src/news.py`)

- Pulls keyless Google News RSS per ticker.
- Scores each headline with **VADER**, a rule/lexicon sentiment model (no API
  key, no GPU, tuned for short social/news text). Output: a compound score in
  [-1, 1] and a positive/neutral/negative label.
- Appends to `brain/news/<TICKER>.csv`, de-duplicated, storing the **published**
  date of every headline.

## Why it is NOT wired into recommendations yet

Folding today's sentiment into a model and backtesting it is a leakage minefield:

- You must filter to news **published on or before** each decision date, or you
  leak the future — see [[Data Leakage]].
- Headline sentiment is noisy; VADER misreads sarcasm, finance jargon, and
  forward-looking language.
- It must clear [[Backtesting Pitfalls]] and beat buy-and-hold on [[Risk Metrics]]
  *after* costs before it earns a place in the signal.

So we store it point-in-time now and treat promoting it to a feature as a
deliberate, separately-validated experiment later — likely fused via
[[Feature Engineering]], possibly with a [[Neural Networks]] text encoder.
