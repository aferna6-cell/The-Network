"""Render the paper-trading monitor to docs/index.html (for GitHub Pages).

Run locally or by the paper-trade Action after each tick:
    python scripts/build_paper_dashboard.py

Reads committed paper_state/ only — it never trades. Publish by enabling GitHub
Pages on the repo (Settings -> Pages -> Source: main / docs).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import paper_dashboard  # noqa: E402


def main() -> int:
    out = paper_dashboard.build()
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
