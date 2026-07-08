"""Observability dashboard — a thin read-only view over the events store.

It only *queries and renders*: every number comes from a method on
``core.services.observability.Observer`` (which does any aggregation in SQL). No
business logic lives here — that's the week-5 rule, "the dashboard reads, it
doesn't compute."

:func:`render_dashboard` is the reusable body — the chat app calls it from its
"📊 Dashboard" view so the dashboard lives **inside the one app** (same port), not
a separate service. It reads the same ``EVENTS_DB`` the app writes.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.services.observability import get_observer

# The agent components whose events count as "agent usage".
_AGENTS = ("todo_agent", "planner_agent", "rag_agent")


def render_dashboard() -> None:
    """Render every panel. Call from inside the app (page config already set)."""
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

    # -- MCP servers registry (name · #tools · each tool's description) ------
    st.subheader("🔌 MCP servers")
    servers = obs.mcp_servers()
    if servers:
        cols = st.columns(len(servers))
        for col, s in zip(cols, servers):
            d = s["details"]
            status = "🟢 up" if s["ok"] else "🔴 down"
            col.metric(f"{s['server']}", f"{d.get('tool_count', 0)} tools", help=d.get("url", ""))
            col.caption(f"{status} · {d.get('url', '')}")
            if d.get("error"):
                col.error(d["error"])
            tools = d.get("tools", [])
            if tools:
                with col.expander(f"{len(tools)} tools"):
                    st.dataframe(
                        pd.DataFrame(tools)[["name", "description"]],
                        use_container_width=True, hide_index=True,
                    )
    else:
        st.info("No MCP server registry logged yet — start the chat app so it connects and records its tools.")

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

    # -- agent detail: what each agent was actually asked to do -------------
    st.header("🤖 Agent detail")

    left, right = st.columns(2)
    with left:
        st.subheader("Intent routing")
        rows = obs.intent_breakdown()
        if rows:
            df = pd.DataFrame(rows)
            st.bar_chart(df.set_index("intent"), y="n", height=240)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No routing decisions logged yet.")
    with right:
        st.subheader("Tool usage")
        rows = obs.tool_usage()
        if rows:
            df = pd.DataFrame(rows)
            df["avg_ms"] = df["avg_ms"].round(1)
            st.dataframe(
                df[["tool", "n", "avg_ms", "failures"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No tool calls logged yet.")

    left, right = st.columns(2)
    with left:
        st.subheader("Latency by operation (ms)")
        rows = obs.latency_by_operation()
        if rows:
            df = pd.DataFrame(rows)
            st.bar_chart(df.set_index("op"), y="avg_ms", height=260)
        else:
            st.info("No timed operations yet.")
    with right:
        st.subheader("DailyPlanner overload")
        p = obs.planner_summary()
        m1, m2, m3 = st.columns(3)
        m1.metric("Plans", p["plans"])
        m2.metric("Overloaded", p["overloads"], help=f"{p['rate'] * 100:.0f}% of plans")
        m3.metric("Tasks deferred", p["deferred"])
        overloads = obs.recent_overloads(8)
        if overloads:
            st.caption("Recent deferrals (what didn't fit):")
            df = pd.DataFrame([
                {"ts": o["ts"][:19], "deferred": ", ".join(o["details"].get("deferred", [])),
                 "scheduled": o["details"].get("scheduled"),
                 "avail_min": o["details"].get("available_minutes")}
                for o in overloads
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No overload events — every day has fit so far.")

    st.divider()

    # -- session inspector: drill into one conversation's full timeline ------
    st.subheader("🔎 Session inspector")
    sessions = obs.session_summary(50)
    if sessions:
        labels = {
            f"{s['session_id'][:8]}… · {s['events']} events · {s['requests']} req"
            f"{' · ⛔' + str(s['blocks']) if s['blocks'] else ''} · {s['last_ts'][:19]}": s["session_id"]
            for s in sessions
        }
        choice = st.selectbox("Pick a session", list(labels), index=0)
        events = obs.events_for_session(labels[choice])
        df = pd.DataFrame(events)
        df["details"] = df["details"].apply(lambda d: ", ".join(f"{k}={v}" for k, v in d.items())[:100])
        st.dataframe(
            df[["ts", "component", "event_type", "ok", "latency_ms", "details"]],
            use_container_width=True, hide_index=True, height=320,
        )
    else:
        st.info("No sessions with events yet.")

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

    # -- latest eval run: per-case results + criteria breakdown -------------
    if latest:
        st.markdown("**Latest run — per-case results**")
        etype = st.selectbox("Eval type", [r["eval_type"] for r in latest], key="eval_detail")
        run = obs.latest_eval_run(etype)
        cases = (run or {}).get("details", {}).get("cases", [])
        if run:
            st.caption(f"{run['passed']}/{run['total']} = {run['score'] * 100:.0f}%"
                       f"{' · ' + run['note'] if run['note'] else ''} · {run['ts'][:19]}")
        if cases:
            rows = []
            for c in cases:
                crit = c.get("criteria") or []
                rows.append({
                    "case": c.get("name", ""),
                    "result": "✅" if c.get("passed") else "❌",
                    # For the generation eval: show every criterion the answer was scored on.
                    "criteria": " · ".join(
                        f"{'✓' if x.get('met') else '✗'} {x.get('criterion', '')}" for x in crit
                    ),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=300)
        else:
            st.info("This run persisted no per-case detail.")

    st.divider()

    # -- agent debug trace (only populated when AGENT_DEBUG=1) ---------------
    st.subheader("🐞 Agent debug trace")
    st.caption("Fine-grained agent decisions — the classifier's verdict, each RAG "
               "reformulation, the planner's chosen workday. Run with `AGENT_DEBUG=1` to populate.")
    debug = obs.debug_events(100)
    if debug:
        df = pd.DataFrame(debug)
        df["message"] = df["details"].apply(lambda d: d.get("message", ""))
        df["detail"] = df["details"].apply(
            lambda d: ", ".join(f"{k}={v}" for k, v in d.items() if k != "message")[:100]
        )
        st.dataframe(
            df[["ts", "component", "session_id", "message", "detail"]],
            use_container_width=True, hide_index=True, height=280,
        )
    else:
        st.info("No debug events. Set `AGENT_DEBUG=1` on the chat app to record the agents' internal decisions.")

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


def main() -> None:
    """Standalone entry (kept for direct runs); the app uses render_dashboard()."""
    st.set_page_config(page_title="Observability", page_icon="📊", layout="wide")
    render_dashboard()


if __name__ == "__main__":
    main()
