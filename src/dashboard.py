"""Observability dashboard — a thin read-only view over the events store.

It only *queries and renders*: every number comes from a method on
``core.services.observability.Observer`` (which does any aggregation in SQL). No
business logic lives here — that's the week-5 rule, "the dashboard reads, it
doesn't compute." Run it as its own service:

    streamlit run src/dashboard.py

It reads the same ``EVENTS_DB`` the app writes, so in the container setup they
share one volume and this is a separate container from the chat app.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.services.observability import get_observer

st.set_page_config(page_title="Observability", page_icon="📊", layout="wide")

# The agent components whose events count as "agent usage".
_AGENTS = ("todo_agent", "planner_agent", "rag_agent")


def main() -> None:
    obs = get_observer()

    st.title("📊 Observability Dashboard")
    st.caption("Structured events from every agent, tool, and guardrail — one shared store.")
    if st.button("🔄 Refresh"):
        st.rerun()

    error = obs.error_rate()
    block = obs.guardrail_block_rate()

    # -- headline metrics ---------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Requests", error["total"])
    c2.metric("Error rate", f"{error['rate'] * 100:.1f}%", help=f"{error['errors']} failed of {error['total']}")
    c3.metric("Guardrail block rate", f"{block['rate'] * 100:.1f}%",
              help=f"{block['blocked']} blocked of {block['total']} checks")
    c4.metric("Avg RAG iterations", f"{obs.avg_rag_iterations():.2f}")
    tool_calls = sum(r["n"] for r in obs.counts_by_component(("tool",)))
    c5.metric("Tool calls", tool_calls)

    st.divider()

    # -- requests over time + per-agent split -------------------------------
    left, right = st.columns(2)
    with left:
        st.subheader("Requests over time")
        rows = obs.requests_over_time("minute")
        if rows:
            df = pd.DataFrame(rows).set_index("bucket")
            st.bar_chart(df, y="n", height=260)
        else:
            st.info("No requests logged yet.")

    with right:
        st.subheader("Per-agent usage")
        rows = [r for r in obs.counts_by_component(_AGENTS)]
        if rows:
            df = pd.DataFrame(rows).set_index("component")
            st.bar_chart(df, y="n", height=260)
        else:
            st.info("No agent activity yet.")

    # -- guardrail + latency ------------------------------------------------
    left, right = st.columns(2)
    with left:
        st.subheader("Guardrail checks")
        if block["total"]:
            df = pd.DataFrame(
                [{"outcome": "allowed", "n": block["total"] - block["blocked"]},
                 {"outcome": "blocked", "n": block["blocked"]}]
            ).set_index("outcome")
            st.bar_chart(df, y="n", height=240)
        else:
            st.info("No guardrail checks yet.")
    with right:
        st.subheader("Avg latency by component (ms)")
        rows = obs.avg_latency_by_component()
        if rows:
            df = pd.DataFrame(rows).set_index("component")
            st.bar_chart(df, y="avg_ms", height=240)
        else:
            st.info("No timed events yet.")

    st.divider()

    # -- evaluation trend ---------------------------------------------------
    st.subheader("Evaluation scores over time")
    latest = obs.latest_eval_scores()
    if latest:
        cols = st.columns(len(latest))
        for col, row in zip(cols, latest):
            col.metric(f"{row['eval_type']}", f"{row['score'] * 100:.0f}%",
                       help=f"{row['passed']}/{row['total']} — {row['ts'][:19]}")
    history = obs.eval_history()
    if history:
        df = pd.DataFrame(history)
        df["run"] = range(1, len(df) + 1)
        pivot = df.pivot_table(index="run", columns="eval_type", values="score", aggfunc="last")
        st.line_chart(pivot, height=280)
    else:
        st.info("No eval runs recorded yet — run `python eval/run_all.py`.")

    st.divider()

    # -- raw event feed -----------------------------------------------------
    st.subheader("Recent events")
    events = obs.recent_events(150)
    if events:
        df = pd.DataFrame(events)
        df["details"] = df["details"].apply(lambda d: ", ".join(f"{k}={v}" for k, v in d.items())[:80])
        st.dataframe(
            df[["ts", "component", "event_type", "ok", "latency_ms", "session_id", "details"]],
            use_container_width=True, hide_index=True, height=360,
        )
    else:
        st.info("No events yet. Use the chat app to generate some traffic.")


if __name__ == "__main__":
    main()
