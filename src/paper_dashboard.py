"""Render the paper-trading demo as a single self-contained HTML page.

Why static HTML and not the Streamlit app: the demo runs untended on GitHub
Actions and commits its state to the repo. A static page the Action regenerates
each tick and publishes to GitHub Pages can be *watched from anywhere* — phone
included — with no server to keep alive. The Streamlit dashboard stays the local
deep-dive; this is the always-on monitor.

Everything is inline (CSS + an SVG equity chart, no JS, no CDN) so the file works
offline, renders on GitHub Pages, and never breaks because a script failed to
load. It only displays committed `paper_state/`; it never trades.
"""
from __future__ import annotations

import html
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config

_W, _H = 760, 300
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 56, 16, 16, 28


def _days_since(inception: str) -> int:
    try:
        d0 = date.fromisoformat(inception[:10])
        return max(0, (datetime.now(timezone.utc).date() - d0).days)
    except (TypeError, ValueError):
        return 0


def _daily_change(eq: pd.DataFrame | None) -> dict | None:
    """Day-over-day change for the model and the benchmark.

    Compares the latest calendar day's closing equity to the previous day's,
    so multiple ticks on the same day collapse to one honest daily number.
    """
    if eq is None or len(eq) < 2:
        return None
    df = eq.copy()
    df["d"] = df["timestamp"].dt.date
    by_day = df.groupby("d").last()
    if len(by_day) < 2:
        return None
    cur, prev = by_day.iloc[-1], by_day.iloc[-2]

    def chg(col: str) -> tuple[float, float]:
        a = cur[col] - prev[col]
        return a, (a / prev[col] if prev[col] else 0.0)

    m_abs, m_pct = chg("equity")
    b_abs, b_pct = chg("benchmark_equity")
    return {"m_abs": m_abs, "m_pct": m_pct, "b_abs": b_abs, "b_pct": b_pct}


def _next_rebalance(snapshot: dict) -> str:
    """Best-effort date of the next monthly rebalance, for the activity note."""
    anchor = snapshot.get("last_rebalance") or snapshot.get("inception_date")
    try:
        d = date.fromisoformat(str(anchor)[:10])
        return (d + timedelta(days=config.PAPER_REBALANCE_DAYS)).isoformat()
    except (TypeError, ValueError):
        return "—"


def _activity_section(trades: pd.DataFrame | None, snapshot: dict) -> str:
    """Recent buys/sells from the durable log; fall back to the inception buys.

    The trade log only starts accumulating from its first rebalance, so until
    then we reconstruct the opening portfolio from real holdings (shares bought
    at cost_basis) rather than show an empty table.
    """
    action_cls = {"BUY": "pos", "SELL": "neg", "TRIM": "warn"}
    rows, note = [], ""

    if trades is not None and len(trades):
        recent = trades.tail(8).iloc[::-1]
        for _, t in recent.iterrows():
            act = str(t["action"]).upper()
            rows.append(
                f'<tr><td>{html.escape(str(t["timestamp"])[:10])}</td>'
                f'<td class="{action_cls.get(act, "")}">{html.escape(act)}</td>'
                f'<td>{html.escape(str(t["ticker"]))}</td>'
                f'<td>{float(t["shares"]):.4f}</td>'
                f'<td>${float(t["price"]):,.2f}</td>'
                f'<td>${float(t["value"]):,.2f}</td></tr>')
        note = (f'Showing the {len(recent)} most recent trades. The model '
                f'rebalances ~every {config.PAPER_REBALANCE_DAYS} days; '
                f'next ≈ {html.escape(_next_rebalance(snapshot))}.')
    else:
        incep = html.escape(snapshot.get("inception_date", "—"))
        for h in snapshot.get("holdings", []):
            cb = h.get("cost_basis", 0.0)
            rows.append(
                f'<tr><td>{incep}</td><td class="pos">BUY</td>'
                f'<td>{html.escape(str(h["ticker"]))}</td>'
                f'<td>{h["shares"]:.4f}</td><td>${cb:,.2f}</td>'
                f'<td>${h["shares"] * cb:,.2f}</td></tr>')
        note = (f'No rebalance trades logged yet — these are the opening positions '
                f'bought at inception ({incep}). New buys/sells appear here after '
                f'the first rebalance (≈ {html.escape(_next_rebalance(snapshot))}).')

    if not rows:
        body = '<tr><td colspan="6" class="muted">No activity yet.</td></tr>'
    else:
        body = "\n".join(rows)
    return f'''<h2>Recent activity</h2>
  <table><thead><tr><th>Date</th><th>Action</th><th>Ticker</th>
    <th>Shares</th><th>Price</th><th>Value</th></tr></thead>
    <tbody>{body}</tbody></table>
  <p class="muted note">{note}</p>'''


def _svg_chart(eq: pd.DataFrame, start_equity: float) -> str:
    """Inline SVG line chart: model equity vs the SPY $500 benchmark."""
    if eq is None or len(eq) < 2:
        return ('<p class="muted">The equity curve appears once the account has '
                'ticked more than once. Check back tomorrow.</p>')

    model = eq["equity"].to_list()
    bench = eq["benchmark_equity"].to_list()
    lo = min(min(model), min(bench), start_equity)
    hi = max(max(model), max(bench), start_equity)
    span = (hi - lo) or 1.0
    n = len(model)

    def x(i: int) -> float:
        return _PAD_L + (_W - _PAD_L - _PAD_R) * (i / (n - 1))

    def y(v: float) -> float:
        return _PAD_T + (_H - _PAD_T - _PAD_B) * (1 - (v - lo) / span)

    def poly(vals: list[float]) -> str:
        return " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))

    base_y = y(start_equity)
    grid = (f'<line x1="{_PAD_L}" y1="{base_y:.1f}" x2="{_W - _PAD_R}" '
            f'y2="{base_y:.1f}" class="base"/>'
            f'<text x="{_PAD_L - 6}" y="{base_y + 4:.1f}" class="ylab" '
            f'text-anchor="end">${start_equity:,.0f}</text>')
    hilab = (f'<text x="{_PAD_L - 6}" y="{y(hi) + 4:.1f}" class="ylab" '
             f'text-anchor="end">${hi:,.0f}</text>')
    lolab = (f'<text x="{_PAD_L - 6}" y="{y(lo) + 4:.1f}" class="ylab" '
             f'text-anchor="end">${lo:,.0f}</text>')
    return f'''<svg viewBox="0 0 {_W} {_H}" class="chart" role="img"
     aria-label="Model equity versus SPY benchmark over time">
  {grid}{hilab}{lolab}
  <polyline class="bench" points="{poly(bench)}"/>
  <polyline class="model" points="{poly(model)}"/>
</svg>
<div class="legend"><span class="k model"></span>Model
  <span class="k bench"></span>SPY $500</div>'''


def _holdings_rows(holdings: list[dict]) -> str:
    if not holdings:
        return '<tr><td colspan="4" class="muted">All cash — no open positions.</td></tr>'
    out = []
    for h in holdings:
        up = h.get("unrealized_pct", 0.0)
        cls = "pos" if up >= 0 else "neg"
        out.append(
            f'<tr><td>{html.escape(str(h["ticker"]))}</td>'
            f'<td>{h["shares"]:.4f}</td><td>${h["value"]:,.2f}</td>'
            f'<td class="{cls}">{up:+.1%}</td></tr>')
    return "\n".join(out)


def render_dashboard(snapshot: dict, equity: pd.DataFrame | None,
                     trades: pd.DataFrame | None = None) -> str:
    """Return the full HTML document for the paper-trading monitor."""
    s = snapshot
    ahead = s["vs_benchmark"] >= 0
    vs_cls = "pos" if ahead else "neg"
    days = _days_since(s.get("inception_date", ""))
    ret_cls = "pos" if s["total_return"] >= 0 else "neg"
    updated = s.get("generated_at", "")[:19].replace("T", " ")

    dc = _daily_change(equity)
    if dc is None:
        today_val = '<div class="val muted">—</div><div class="muted">first day</div>'
    else:
        d_cls = "pos" if dc["m_abs"] >= 0 else "neg"
        today_val = (
            f'<div class="val {d_cls}">${dc["m_abs"]:+,.2f}</div>'
            f'<div class="{d_cls}">{dc["m_pct"]:+.2%}</div>'
            f'<div class="muted" style="font-size:12px">SPY {dc["b_pct"]:+.2%}</div>')

    return f'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta http-equiv="refresh" content="900"/>
<title>the-network · paper-trading monitor</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#0d1117; color:#e6edf3;
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:820px; margin:0 auto; padding:24px 16px 48px; }}
  h1 {{ font-size:22px; margin:0 0 2px; }}
  .sub {{ color:#8b949e; margin:0 0 18px; }}
  .banner {{ background:#221a00; border:1px solid #9e6a03; color:#e3b341;
    padding:10px 14px; border-radius:8px; font-size:13px; margin-bottom:20px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:12px; margin-bottom:22px; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:14px; }}
  .card .lab {{ color:#8b949e; font-size:12px; text-transform:uppercase;
    letter-spacing:.04em; }}
  .card .val {{ font-size:22px; font-weight:700; margin-top:4px; }}
  .pos {{ color:#3fb950; }} .neg {{ color:#f85149; }} .warn {{ color:#d29922; }}
  .note {{ font-size:12px; margin-top:8px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:6px; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #21262d; }}
  th {{ color:#8b949e; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  td:nth-child(n+2),th:nth-child(n+2) {{ text-align:right; }}
  .chart {{ width:100%; height:auto; background:#161b22; border:1px solid #30363d;
    border-radius:10px; margin-top:6px; }}
  .chart .model {{ fill:none; stroke:#3fb950; stroke-width:2; }}
  .chart .bench {{ fill:none; stroke:#58a6ff; stroke-width:1.5;
    stroke-dasharray:4 3; }}
  .chart .base {{ stroke:#30363d; stroke-width:1; stroke-dasharray:2 3; }}
  .chart .ylab {{ fill:#8b949e; font-size:11px; }}
  .legend {{ font-size:12px; color:#8b949e; margin-top:6px; }}
  .legend .k {{ display:inline-block; width:12px; height:3px; margin:0 4px 0 12px;
    vertical-align:middle; }}
  .legend .k.model {{ background:#3fb950; }} .legend .k.bench {{ background:#58a6ff; }}
  h2 {{ font-size:15px; color:#8b949e; text-transform:uppercase; letter-spacing:.04em;
    margin:26px 0 0; }}
  .muted {{ color:#8b949e; }}
  footer {{ color:#8b949e; font-size:12px; margin-top:30px;
    border-top:1px solid #21262d; padding-top:14px; }}
  a {{ color:#58a6ff; }}
</style></head>
<body><div class="wrap">
  <h1>📈 the-network — paper-trading monitor</h1>
  <p class="sub">$500 of fake money, managed autonomously. Day {days} since
    inception ({html.escape(s.get("inception_date", "—"))}).</p>

  <div class="banner">⚠️ <b>Fake money — not financial advice.</b> The equity
    curve is far too short to prove skill; judge it only after many weeks. The
    system places no real trades.</div>

  <div class="cards">
    <div class="card"><div class="lab">Equity</div>
      <div class="val">${s["equity"]:,.2f}</div>
      <div class="{ret_cls}">{s["total_return"]:+.2%}</div></div>
    <div class="card"><div class="lab">SPY $500 benchmark</div>
      <div class="val">${s["benchmark_equity"]:,.2f}</div>
      <div class="muted">{s["benchmark_return"]:+.2%}</div></div>
    <div class="card"><div class="lab">vs SPY</div>
      <div class="val {vs_cls}">${s["vs_benchmark"]:+,.2f}</div>
      <div class="muted">{"ahead of" if ahead else "behind"} the index</div></div>
    <div class="card"><div class="lab">Today's gain / loss</div>
      {today_val}</div>
    <div class="card"><div class="lab">Cash</div>
      <div class="val">${s["cash"]:,.2f}</div></div>
  </div>

  <h2>Equity vs SPY</h2>
  {_svg_chart(equity, s.get("starting_equity", config.PAPER_START_CASH))}

  <h2>Holdings</h2>
  <table><thead><tr><th>Ticker</th><th>Shares</th><th>Value</th>
    <th>Unrealized</th></tr></thead>
    <tbody>{_holdings_rows(s.get("holdings", []))}</tbody></table>

  {_activity_section(trades, s)}

  <footer>
    Last updated {html.escape(updated)} UTC · auto-refreshes every 15 min ·
    daily write-up on
    <a href="https://github.com/aferna6-cell/The-Network/issues/2">issue #2</a>.<br/>
    Research and education only. Past simulated performance does not predict
    future results.
  </footer>
</div></body></html>'''


def build(out_path: Path | None = None) -> Path:
    """Read committed paper_state/, render the page, write it to `out_path`."""
    out_path = out_path or (config.ROOT / "docs" / "index.html")
    snap_path = config.PAPER_SNAPSHOT_PATH
    if not snap_path.exists():
        raise FileNotFoundError(
            f"No paper snapshot at {snap_path}. Run scripts/paper_run.py first.")
    snapshot = json.loads(snap_path.read_text(encoding="utf-8"))

    equity = None
    if config.PAPER_EQUITY_PATH.exists():
        equity = pd.read_csv(config.PAPER_EQUITY_PATH, parse_dates=["timestamp"])

    trades = None
    if config.PAPER_TRADES_PATH.exists():
        trades = pd.read_csv(config.PAPER_TRADES_PATH)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_dashboard(snapshot, equity, trades), encoding="utf-8")
    return out_path
