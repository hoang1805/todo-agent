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


def test_logging_is_best_effort_and_never_raises():
    # A store failure must not surface as an exception to the caller.
    import core.services.observability as observability

    orig = observability.get_observer
    observability.get_observer = lambda: (_ for _ in ()).throw(RuntimeError("store down"))
    try:
        log_event("orchestrator", "request")  # must swallow the error, not raise
    finally:
        observability.get_observer = orig
