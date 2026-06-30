"""Persisted multiturn history — store, sliding window, restart, orchestrator wiring."""

import tempfile
from pathlib import Path

import core.services.llm_client as llm_client
from agents.graph_orchestrator import build_graph_orchestrator
from core.services.history import History
from langchain_core.messages import AIMessage


def _tmp_db() -> str:
    return str(Path(tempfile.mkdtemp(prefix="hist-")) / "h.db")


# -- the store ---------------------------------------------------------------


def test_add_and_get_recent_turns_round_trip():
    h = History(_tmp_db())
    sid = h.create_session("Day plan")
    h.add_turn(sid, "user", "plan my day")
    h.add_turn(sid, "assistant", "here is your plan")

    turns = h.get_recent_turns(sid)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["content"] == "plan my day"


def test_recent_turns_window_is_bounded_and_ordered():
    h = History(_tmp_db())
    for i in range(10):
        h.add_turn("s1", "user", f"msg {i}")
    recent = h.get_recent_turns("s1", limit=3)
    assert [t["content"] for t in recent] == ["msg 7", "msg 8", "msg 9"]  # last 3, oldest-first


def test_sessions_are_isolated_and_listed():
    h = History(_tmp_db())
    a, b = h.create_session("A"), h.create_session("B")
    h.add_turn(a, "user", "in A")
    h.add_turn(b, "user", "in B")
    assert {s["id"] for s in h.list_sessions()} == {a, b}
    assert [t["content"] for t in h.get_all_turns(a)] == ["in A"]


def test_history_survives_restart_via_the_sqlite_file():
    db = _tmp_db()
    sid = History(db).create_session("persisted")
    History(db).add_turn(sid, "user", "remember this across restarts")

    fresh = History(db)  # a brand-new instance reading the same on-disk file
    turns = fresh.get_all_turns(sid)
    assert turns and turns[0]["content"] == "remember this across restarts"
    assert any(s["id"] == sid for s in fresh.list_sessions())


# -- orchestrator wiring -----------------------------------------------------


def test_orchestrator_persists_each_turn():
    h = History(_tmp_db())
    go = build_graph_orchestrator(tools=[], model_name="dummy", use_llm=False, history=h)
    go.start("what's on my list?", thread_id="s1")

    turns = h.get_all_turns("s1")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["content"] == "what's on my list?"


class _RecordingLLM:
    def __init__(self):
        self.seen = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.seen.append([getattr(m, "content", "") for m in messages])
        return AIMessage(content="noted")


def test_recent_history_is_injected_so_a_follow_up_resolves(monkeypatch):
    rec = _RecordingLLM()
    monkeypatch.setattr(llm_client, "create_ollama_model", lambda *a, **k: rec)

    h = History(_tmp_db())
    go = build_graph_orchestrator(
        tools=[], model_name="dummy", use_llm=True, history=h,
        classifier=lambda text: "unknown",  # force the agent path
    )
    go.start("apples are my favorite fruit", thread_id="hs")
    go.start("what did I just say?", thread_id="hs")

    # Turn 2's LLM call must see turn 1 (injected from the persisted history).
    assert any("apples are my favorite fruit" in m for m in rec.seen[1])
