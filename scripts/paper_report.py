"""Build a daily paper-trading summary and post it to the tracking GitHub issue.

Run by the paper-trade Action after each tick. Reads the committed snapshot +
equity curve and posts a human-readable comment to the issue named by
PAPER_REPORT_ISSUE, so you get a daily notification with no session or server.

Locally (no GITHUB_TOKEN), it just prints the report to stdout.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _read_equity_tail(n: int = 5) -> list[tuple[str, float, float]]:
    p = config.PAPER_EQUITY_PATH
    if not p.exists():
        return []
    rows = p.read_text(encoding="utf-8").strip().splitlines()[1:]  # skip header
    out = []
    for line in rows[-n:]:
        ts, eq, bench = line.split(",")
        out.append((ts[:10], float(eq), float(bench)))
    return out


def build_report() -> str:
    snap_path = config.PAPER_SNAPSHOT_PATH
    if not snap_path.exists():
        return "No paper snapshot yet — the account has not ticked."
    s = json.loads(snap_path.read_text(encoding="utf-8"))

    beating = s["equity"] >= s["benchmark_equity"]
    verdict = "ahead of" if beating else "behind"
    lines = [
        f"### Paper-trading daily report — {s['generated_at'][:10]}",
        "",
        f"- **Equity:** ${s['equity']:,.2f} ({s['total_return']:+.2%} since "
        f"{s['inception_date']})",
        f"- **SPY $500 benchmark:** ${s['benchmark_equity']:,.2f} "
        f"({s['benchmark_return']:+.2%})",
        f"- **vs SPY:** ${s['vs_benchmark']:+,.2f} — currently **{verdict}** the index",
        f"- **Cash:** ${s['cash']:,.2f}",
    ]

    if s.get("holdings"):
        lines += ["", "| Ticker | Shares | Value | Unrealized |", "|---|---|---|---|"]
        for h in s["holdings"]:
            lines.append(f"| {h['ticker']} | {h['shares']} | ${h['value']:,.2f} | "
                         f"{h['unrealized_pct']:+.1%} |")

    trades = s.get("trades_this_tick", [])
    if trades:
        lines += ["", "**Trades this tick:** " + ", ".join(
            f"{t['action']} {t['ticker']} {t['shares']}@${t['price']}" for t in trades)]
    else:
        lines += ["", "_No trades this tick (next rebalance is monthly)._"]

    tail = _read_equity_tail()
    if len(tail) > 1:
        lines += ["", "Recent equity vs SPY:"]
        for d, eq, bench in tail:
            lines.append(f"- {d}: ${eq:,.2f}  /  SPY ${bench:,.2f}")

    lines += ["", "_Fake money. The curve is still too short to prove skill — "
              "judge it after many weeks. Not financial advice._"]
    return "\n".join(lines)


def post_comment(report: str) -> bool:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    issue = os.environ.get("PAPER_REPORT_ISSUE")
    if not (token and repo and issue):
        return False
    url = f"https://api.github.com/repos/{repo}/issues/{issue}/comments"
    resp = requests.post(
        url, headers={"Authorization": f"Bearer {token}",
                      "Accept": "application/vnd.github+json"},
        json={"body": report}, timeout=30)
    resp.raise_for_status()
    return True


def main() -> int:
    report = build_report()
    if post_comment(report):
        print(f"Posted daily report to issue #{os.environ.get('PAPER_REPORT_ISSUE')}.")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
