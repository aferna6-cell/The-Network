"""Regenerate the brain knowledge graph (Home.md + graph.json) from notes.

Usage:
    python scripts/build_brain.py

Deterministic — no LLM, no network. Run after editing brain/concepts/*.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import brain  # noqa: E402


def main() -> int:
    graph = brain.rebuild()
    print(f"Rebuilt brain: {len(graph['nodes'])} concepts, "
          f"{len(graph['edges'])} links.")
    print("Wrote brain/Maps/Home.md and brain/graph.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
