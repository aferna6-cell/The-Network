"""the-network dashboard — view Buy/Sell/Trim calls, portfolio, and the LIVE
track record that tells you whether to trust any of it.

Run:
    streamlit run dashboard/app.py

Reads the snapshot written by the recommender (state/xs_recommendations.json) and
the prediction ledger. It only displays — it never trades and never overrides the
honesty framing: the edge is unproven until the live track record says otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

st.set_page_config(page_title="the-network", layout="wide")

ACTION_COLORS = {
    "BUY": "#1a7f37", "SELL": "#cf222e", "TRIM": "#bc4c00",
    "HOLD": "#57606a", "AVOID": "#8c959f",
}


def _load_snapshot() -> dict | None:
    p = config.XS_RECOMMENDATIONS_PATH
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _style_actions(df: pd.DataFrame):
    def color(val):
        return f"color: {ACTION_COLORS.get(val, '#000')}; font-weight: 700"
    return df.style.applymap(color, subset=["action"])


st.title("📈 the-network")
st.caption("Cross-sectional Buy / Sell / Trim recommendations for your portfolio.")

st.warning(
    "**Decision support, not an oracle.** The underlying edge is UNPROVEN and the "
    "research backtest is survivorship-biased. Weight every call by the live track "
    "record below — it only becomes meaningful after many graded calls. This places "
    "no trades; you decide.",
    icon="⚠️",
)

snap = _load_snapshot()
if snap is None:
    st.info("No recommendations yet. Run `python scripts/recommend_xs.py` "
            "(or the always-on loop `scripts/run_live_xs.py`) to generate them.")
    st.stop()

# --- KPI row ---------------------------------------------------------------
tr = snap.get("live_track_record", {})
acc = tr.get("accuracy")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio value", f"${snap['portfolio_value']:,.0f}")
c2.metric("Live hit rate",
          f"{acc:.0%}" if acc is not None else "—",
          help="Fraction of reconciled calls that were directionally right.")
c3.metric("Graded calls", tr.get("n", 0),
          help="How many calls have been scored against real outcomes yet.")
c4.metric("Universe", snap.get("universe_size", 0))
st.caption(f"Last updated: {snap.get('generated_at', 'n/a')}")

if acc is None:
    st.info("⏳ No graded calls yet — every recommendation below is an *unvalidated* "
            "guess until the ledger fills in over the coming weeks.")
elif acc < 0.5:
    st.error(f"⚠️ Live hit rate {acc:.0%} is below a coin flip over {tr.get('n')} "
             "calls. The model is not demonstrating skill — treat calls with caution.")

# --- Actions ---------------------------------------------------------------
st.subheader("Actions")
actions = pd.DataFrame(snap.get("actions", []))
if actions.empty:
    st.write("No actionable Buy/Sell/Trim right now.")
else:
    cols = ["action", "ticker", "rank", "universe_size", "confidence",
            "held", "weight", "rationale"]
    view = actions[[c for c in cols if c in actions.columns]]
    st.dataframe(_style_actions(view), use_container_width=True, hide_index=True)

# --- Holds -----------------------------------------------------------------
holds = pd.DataFrame(snap.get("holds", []))
if not holds.empty:
    st.subheader("Holds (keep)")
    st.dataframe(holds[["ticker", "rank", "universe_size", "weight", "rationale"]],
                 use_container_width=True, hide_index=True)

# --- Ledger / track record -------------------------------------------------
st.subheader("Prediction ledger")
if config.XS_LEDGER_PATH.exists():
    led = pd.read_csv(config.XS_LEDGER_PATH)
    open_n = int((led["correct"].astype(str).isin(["", "nan", "None"])).sum())
    st.caption(f"{len(led)} calls logged · {open_n} still open (awaiting outcome).")
    st.dataframe(led.tail(50).iloc[::-1], use_container_width=True, hide_index=True)
else:
    st.caption("No calls logged yet.")

if st.button("🔄 Refresh"):
    st.rerun()
st.caption("Tip: run the always-on loop so this updates itself, then just refresh.")
