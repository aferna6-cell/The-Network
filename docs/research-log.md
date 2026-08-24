# Research log — paper-trading demo

Append-only record of every rebalance: what the model did and how the *previous*
basket actually performed. This is the compounding memory of the loop — each
entry is real data, not a forecast. Newest entries at the bottom.

## 2026-06-24 — inception
- **New basket:** AAPL, CRM, CVX, NFLX, XOM
- **Sold:** —
- **Bought:** AAPL 0.3334@$296.97, CRM 0.6405@$154.56, CVX 0.5785@$171.12, NFLX 1.3667@$72.44, XOM 0.7289@$135.81
- **Equity:** $500.00 (+0.00% since inception) · funded with $500 fake money

## 2026-07-24 — rebalance
- **Last basket realized:** +8.60% (SPY -0.02%, excess +8.62%)
- **New basket:** DIS, GOOGL, HD, ORCL, PEP
- **Sold:** NFLX 1.3667@$70.09, XOM 0.7289@$156.94, CRM 0.6405@$163.66, AAPL 0.3334@$333.02, CVX 0.5785@$194.79
- **Bought:** ORCL 0.9357@$114.99, GOOGL 0.3365@$319.74, DIS 1.1343@$94.85, PEP 0.7874@$136.64, HD 0.3231@$332.98
- **Equity:** $542.64 (+8.53% since inception) · vs SPY $499.92 (+42.72)

## 2026-08-24 — rebalance
- **Last basket realized:** +11.09% (SPY +3.32%, excess +7.77%)
- **New basket:** CRM, META, MRK, MSFT, ORCL
- **Sold:** GOOGL 0.3365@$348.06, DIS 1.1343@$110.61, PEP 0.7874@$144.67, HD 0.3231@$337.43, ORCL 0.0968@$142.45
- **Bought:** MSFT 0.2452@$487.31, META 0.2138@$559.02, CRM 0.5716@$209.06, MRK 0.7931@$150.66
- **Equity:** $602.83 (+20.57% since inception) · vs SPY $516.52 (+86.31)
