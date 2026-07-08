"""Observability store — logging interface, dashboard queries, eval persistence."""

import time

from core.services.observability import (
    current_session,
    get_observer,
    log_event,
    record_eval_run,
    set_session,
    track,
)


def test_log_event_round_trips_with_shape_and_details():
    log_event("orchestrator", "request", session_id="s1", ok=True, latency_ms=12.5,
              details={"intent": "plan"})
    ev = get_observer().recent_events(1)[0]
    assert ev["component"] == "orchestrator" and ev["event_type"] == "request"
    assert ev["session_id"] == "s1" and ev["ok"] is True
    assert ev["details"]["intent"] == "plan"


def test_track_records_latency_and_success():
    with track("tool", "demo") as span:
        span["tool"] = "demo"
        time.sleep(0.005)
    ev = get_observer().recent_events(1)[0]
    assert ev["component"] == "tool" and ev["ok"] is True
    assert ev["latency_ms"] and ev["latency_ms"] >= 4


def test_track_marks_failure_and_reraises():
    import pytest

    with pytest.raises(ValueError):
        with track("tool", "boom"):
            raise ValueError("nope")
    ev = get_observer().recent_events(1)[0]
    assert ev["event_type"] == "boom" and ev["ok"] is False


def test_session_contextvar_defaults_the_session_id():
    set_session("ctx-session")
    try:
        assert current_session() == "ctx-session"
        log_event("guardrail", "input", ok=True)  # no explicit session_id
        ev = get_observer().recent_events(1)[0]
        assert ev["session_id"] == "ctx-session"
    finally:
        set_session(None)


def test_guardrail_block_rate_and_error_rate():
    log_event("guardrail", "input", ok=True)
    log_event("guardrail", "input", ok=False, details={"reason": "blocked"})
    log_event("orchestrator", "request", ok=True)
    log_event("orchestrator", "request", ok=False)

    block = get_observer().guardrail_block_rate()
    assert block["total"] == 2 and block["blocked"] == 1 and block["rate"] == 0.5
    err = get_observer().error_rate()
    assert err["total"] == 2 and err["errors"] == 1 and err["rate"] == 0.5


def test_avg_rag_iterations_reads_query_events():
    log_event("rag_agent", "query", details={"iterations": 1})
    log_event("rag_agent", "query", details={"iterations": 3})
    assert get_observer().avg_rag_iterations() == 2.0


def test_per_component_counts_filterable():
    for c in ("todo_agent", "todo_agent", "rag_agent", "guardrail"):
        log_event(c, "x")
    counts = {r["component"]: r["n"] for r in get_observer().counts_by_component(("todo_agent", "rag_agent"))}
    assert counts == {"todo_agent": 2, "rag_agent": 1}


def test_eval_runs_persist_and_read_back():
    record_eval_run("retrieval", 0.5, 4, 8, note="before")
    record_eval_run("retrieval", 0.9, 9, 10, note="after")
    hist = get_observer().eval_history("retrieval")
    assert [h["note"] for h in hist] == ["before", "after"]      # ordered, both persisted
    latest = {r["eval_type"]: r for r in get_observer().latest_eval_scores()}
    assert latest["retrieval"]["score"] == 0.9                    # most-recent per type


def test_intent_breakdown_reads_route_events():
    log_event("orchestrator", "route", details={"intent": "plan", "agent": "planner_agent"})
    log_event("orchestrator", "route", details={"intent": "plan", "agent": "planner_agent"})
    log_event("orchestrator", "route", details={"intent": "recall", "agent": "rag_agent"})
    rows = {r["intent"]: r for r in get_observer().intent_breakdown()}
    assert rows["plan"]["n"] == 2 and rows["plan"]["agent"] == "planner_agent"
    assert rows["recall"]["n"] == 1 and rows["recall"]["agent"] == "rag_agent"


def test_tool_usage_counts_and_flags_failures():
    log_event("tool", "list_tasks", ok=True, latency_ms=10.0)
    log_event("tool", "list_tasks", ok=True, latency_ms=20.0)
    log_event("tool", "add_task", ok=False, latency_ms=5.0)
    rows = {r["tool"]: r for r in get_observer().tool_usage()}
    assert rows["list_tasks"]["n"] == 2 and rows["list_tasks"]["avg_ms"] == 15.0
    assert rows["add_task"]["failures"] == 1


def test_latency_by_operation_groups_component_and_type():
    log_event("orchestrator", "classify", latency_ms=100.0)
    log_event("orchestrator", "classify", latency_ms=200.0)
    log_event("tool", "list_tasks", latency_ms=10.0)
    rows = {r["op"]: r for r in get_observer().latency_by_operation()}
    assert rows["orchestrator/classify"]["avg_ms"] == 150.0
    assert rows["orchestrator/classify"]["max_ms"] == 200.0
    assert rows["tool/list_tasks"]["n"] == 1


def test_planner_summary_and_recent_overloads():
    log_event("planner_agent", "plan", details={"overloaded": False})
    log_event("planner_agent", "plan", details={"overloaded": True})
    log_event("planner_agent", "overload", details={
        "deferred": ["taxes", "call bank"], "deferred_count": 2,
        "scheduled": 3, "available_minutes": 480,
    })
    p = get_observer().planner_summary()
    assert p["plans"] == 2 and p["overloads"] == 1 and p["rate"] == 0.5
    assert p["deferred"] == 2 and p["avg_available"] == 480

    recent = get_observer().recent_overloads()
    assert recent[0]["details"]["deferred"] == ["taxes", "call bank"]


def test_session_summary_and_events_timeline():
    set_session("sess-A")
    try:
        log_event("orchestrator", "request", ok=True)
        log_event("guardrail", "input", ok=False)  # a block in this session
        log_event("tool", "list_tasks", ok=True, latency_ms=5.0)
    finally:
        set_session(None)
    log_event("orchestrator", "request", session_id="sess-B", ok=True)

    summary = {s["session_id"]: s for s in get_observer().session_summary()}
    assert summary["sess-A"]["events"] == 3 and summary["sess-A"]["requests"] == 1
    assert summary["sess-A"]["blocks"] == 1
    assert summary["sess-B"]["events"] == 1

    timeline = get_observer().events_for_session("sess-A")
    assert [e["event_type"] for e in timeline] == ["request", "input", "list_tasks"]  # id-ordered


def test_mcp_servers_returns_latest_registry_per_server():
    log_event("mcp_server", "task_mcp", ok=True,
              details={"url": "http://x:8000/sse", "tool_count": 1, "tools": [{"name": "a", "description": "old"}]})
    log_event("mcp_server", "task_mcp", ok=True,  # a newer load — should win
              details={"url": "http://x:8000/sse", "tool_count": 2,
                       "tools": [{"name": "a", "description": "d"}, {"name": "b", "description": "d"}]})
    log_event("mcp_server", "memory_mcp", ok=False, details={"url": "http://y:8002/sse", "error": "down"})

    servers = {s["server"]: s for s in get_observer().mcp_servers()}
    assert servers["task_mcp"]["details"]["tool_count"] == 2          # latest, not the first
    assert servers["task_mcp"]["ok"] is True
    assert servers["memory_mcp"]["ok"] is False
    assert servers["memory_mcp"]["details"]["error"] == "down"


def test_log_debug_is_gated_by_the_agent_debug_flag(monkeypatch):
    from core.services.observability import debug_enabled, log_debug

    monkeypatch.delenv("AGENT_DEBUG", raising=False)
    assert debug_enabled() is False
    log_debug("orchestrator", "should be dropped", intent="plan")
    assert get_observer().debug_events() == []          # flag off → nothing recorded

    monkeypatch.setenv("AGENT_DEBUG", "1")
    assert debug_enabled() is True
    log_debug("orchestrator", "classified → plan", intent="plan")
    dbg = get_observer().debug_events()
    assert len(dbg) == 1 and dbg[0]["details"]["message"] == "classified → plan"
    assert dbg[0]["details"]["intent"] == "plan" and dbg[0]["event_type"] == "debug"


def test_latest_eval_run_returns_parsed_per_case_details():
    record_eval_run("generation", 0.5, 1, 2, note="old", details={"cases": [{"name": "q0", "passed": True}]})
    record_eval_run("generation", 1.0, 2, 2, note="new", details={"cases": [
        {"name": "q1", "passed": True, "criteria": [{"criterion": "states X", "met": True}]},
        {"name": "q2", "passed": True, "criteria": None},
    ]})
    run = get_observer().latest_eval_run("generation")
    assert run["note"] == "new" and run["passed"] == 2          # most-recent run
    cases = run["details"]["cases"]
    assert cases[0]["criteria"][0]["criterion"] == "states X"    # details parsed from JSON
    assert get_observer().latest_eval_run("nonexistent") is None


def test_logging_is_best_effort_and_never_raises():
    # A store failure must not surface as an exception to the caller.
    import core.services.observability as observability

    orig = observability.get_observer
    observability.get_observer = lambda: (_ for _ in ()).throw(RuntimeError("store down"))
    try:
        log_event("orchestrator", "request")  # must swallow the error, not raise
    finally:
        observability.get_observer = orig
