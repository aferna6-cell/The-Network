# Online Learning

"Online" (or continual) learning means updating the model on each new datapoint
as it arrives, rather than retraining in batches. It sounds like the obvious way
to make a system "always learn from its mistakes." For noisy financial data it is
mostly a footgun, and this repo deliberately does **not** do it per tick.

## Why continuous self-training is dangerous here

1. **You train on noise.** A single daily/intraday bar is mostly randomness (the
   [[Efficient Market Hypothesis]]). A model that mutates on every tick chases
   that noise and gets *less* stable.
2. **You lose reproducibility.** If the model changes every minute you can never
   answer "why did it say sell on Tuesday?" — that model no longer exists.
3. **You cannot evaluate it honestly.** Continuous updating blurs the train/test
   boundary and quietly inflates apparent performance — a subtle [[Data Leakage]]
   that undoes everything [[Walk-Forward Validation]] protects.
4. **Catastrophic forgetting + feedback loops.** A few weird days can overwrite
   what the model learned, or send it into a spiral.

## What we do instead (the disciplined version)

The live loop (`src/runtime.py`) **ingests and re-evaluates continuously** but
**retrains on a cadence** (`RETRAIN_EVERY_HOURS`, default daily). Between retrains
the model is a **frozen, versioned snapshot** you can audit. Each retrain folds in
all new data — including the realized outcomes of recent mistakes recorded in the
prediction ledger (`src/ledger.py`). That is genuine learning, in safe,
inspectable increments.

Always-on *learning* is paired with always-on *evaluation*: the decay monitor
(`src/monitor.py`) watches live hit rate and suppresses recommendations when the
model stops working — the real "error correction" at the system level. See
[[Backtesting Pitfalls]] for why measuring live performance is non-negotiable.

## When true online learning earns its place

High-frequency settings with heavy infrastructure, strict validation, and
algorithms built for it (SGD/`River`). Not a solo daily/intraday recommender on
free data — scheduled retraining is the right tool.
