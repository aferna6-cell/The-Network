# Data Leakage

Leakage is when information that would not have been available at decision time
sneaks into training or features. It is the number-one cause of backtests that
look brilliant and lose money live. Treat every suspiciously good result as
leakage until proven otherwise.

## Forms we actively guard against

- **Look-ahead features.** Any feature using future bars (`shift(-n)`, centred
  rolling windows, full-sample scaling). Prevented in [[Feature Engineering]] by
  using only trailing windows.
- **Target bleed across the split.** A label on day `t` peeks `h` days ahead, so
  the last `h` training rows before a test cutoff partly live in the future. We
  **purge** them — see [[Walk-Forward Validation]] (`train_end = i - horizon`).
- **Survivorship bias.** Training only on tickers that still exist today. Noted
  as a limitation; our universe is small and explicit.
- **Normalisation leakage.** Scaling features with statistics computed over the
  whole dataset (including test). We avoid global scalers entirely.

## How we detect it

- The tests in `tests/test_features.py` assert target alignment and that
  shuffling future bars cannot change a past feature.
- A rule of thumb: directional accuracy comfortably above ~55% on daily bars is a
  red flag, not a triumph (see [[Efficient Market Hypothesis]]).

## Connected reading

[[Backtesting Pitfalls]] · [[Technical Indicators]] · [[Risk Metrics]]
