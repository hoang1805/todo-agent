"""LangGraph orchestrator — deterministic paths + the agent⇄tools loop.

All of this runs without a real model: the deterministic paths need no LLM, and
the loop test injects a scripted fake LLM so we can prove the cycle
(agent → tools → agent → finalize) actually executes.
"""

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import core.services.llm_client as llm_client
from agents.graph_orchestrator import build_graph_orchestrator
from agents.planner_agent import Workday
from models.contract import CrudOp, Priority, TaskMutation


# -- deterministic paths (no LLM) -------------------------------------------


def _offline():
    # A tight, break-free window so the sample tasks overload the day.
    tight = Workday(start="09:00", end="11:00", breaks=())
    return build_graph_orchestrator(
        tools=[], model_name="dummy", workday=tight, use_llm=False
    )


def test_plan_path_through_graph_reports_overload():
    out = _offline().run("plan my day", date="2026-06-16")
    assert "Your plan for 2026-06-16" in out
    assert "overloaded" in out.lower()


def test_summary_path_through_graph():
    out = _offline().run("what's on my list?")
    assert "task(s)" in out


def test_unknown_path_gives_guidance_without_llm():
    out = _offline().run("good morning")
    assert "plan your day" in out.lower()


# -- the agent⇄tools loop ----------------------------------------------------


class _ScriptedLLM:
    """A fake chat model that replays a fixed list of AIMessages."""

    def __init__(self, scripted):
        self.scripted = scripted
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        msg = self.scripted[self.calls]
        self.calls += 1
        return msg


def test_multi_intent_prompt_routes_to_agent_loop(monkeypatch):
    # "plan ... and add ..." hits two intent families -> routed to the agent loop.
    # The fake answers immediately (no tool calls), proving the agent path ran.
    fake = _ScriptedLLM([AIMessage(content="handled both steps")])
    monkeypatch.setattr(llm_client, "create_ollama_model", lambda *a, **k: fake)

    go = build_graph_orchestrator(tools=[], model_name="dummy", use_llm=True)
    out = go.run("plan my day and add a task to call the bank")

    assert out == "handled both steps"
    assert fake.calls == 1


def test_agent_tools_loop_executes_a_tool_then_finishes(monkeypatch):
    @tool
    def echo(text: str) -> str:
        """Echo the given text back."""
        return text

    # First turn: the model asks to call the tool. Second turn: it answers.
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{"name": "echo", "args": {"text": "pong"}, "id": "call-1"}],
        ),
        AIMessage(content="All done: pong"),
    ]
    fake = _ScriptedLLM(scripted)
    monkeypatch.setattr(llm_client, "create_ollama_model", lambda *a, **k: fake)

    go = build_graph_orchestrator(
        tools=[echo], model_name="dummy", use_llm=True
    )
    # "echo ..." is not plan/summary, so it routes to the agent loop.
    out = go.run("please echo pong")

    assert out == "All done: pong"
    assert fake.calls == 2  # agent → (tools) → agent: looped exactly once


# -- human-in-the-loop -------------------------------------------------------


def test_mutating_tool_requires_approval_then_runs_on_accept(monkeypatch):
    executed = {}

    @tool
    def create_task(title: str) -> str:
        """Create a task."""
        executed["title"] = title
        return f"created: {title}"

    scripted = [
        AIMessage(
            content="",
            tool_calls=[{"name": "create_task", "args": {"title": "Email client"}, "id": "c1"}],
        ),
        AIMessage(content="Done — task created."),
    ]
    fake = _ScriptedLLM(scripted)
    monkeypatch.setattr(llm_client, "create_ollama_model", lambda *a, **k: fake)

    go = build_graph_orchestrator(tools=[create_task], model_name="dummy", use_llm=True)

    # Multi-intent prompt -> routed to the agent⇄tools loop (the CRUD intents on
    # their own now take the deterministic CRUD path, tested separately).
    pending = go.start("plan my day and add a task to email the client")
    assert pending.status == "interrupted"
    assert pending.interrupt["tool"] == "create_task"
    assert pending.interrupt["args"]["title"] == "Email client"
    assert "title" not in executed  # gated — not run yet

    done = go.resume(pending.thread_id, {"action": "accept"})
    assert done.status == "done"
    assert done.text == "Done — task created."
    assert executed["title"] == "Email client"  # ran only after approval


def test_mutating_tool_is_not_run_on_reject(monkeypatch):
    executed = {}

    @tool
    def delete_task(task_id: int) -> str:
        """Delete a task."""
        executed["id"] = task_id
        return "deleted"

    scripted = [
        AIMessage(
            content="",
            tool_calls=[{"name": "delete_task", "args": {"task_id": 5}, "id": "d1"}],
        ),
        AIMessage(content="Okay, I left it alone."),
    ]
    fake = _ScriptedLLM(scripted)
    monkeypatch.setattr(llm_client, "create_ollama_model", lambda *a, **k: fake)

    go = build_graph_orchestrator(tools=[delete_task], model_name="dummy", use_llm=True)

    # Multi-intent prompt -> routed to the agent⇄tools loop (see note above).
    pending = go.start("plan my day and delete task 5")
    assert pending.status == "interrupted"

    done = go.resume(pending.thread_id, {"action": "reject", "reason": "changed my mind"})
    assert done.status == "done"
    assert "id" not in executed  # never executed


# -- deterministic CRUD path (validate → confirm → execute → re-plan) ---------


def _crud_orchestrator(mutation, calls, *, result="Created task 'X'.", db=None):
    """Build a graph with the CRUD parser/executor stubbed (no LLM, no MCP)."""

    def parser(_user_input, _current_tasks):
        return mutation

    def executor(m):
        calls.append(m)
        return result

    tight = Workday(start="09:00", end="11:00", breaks=())
    return build_graph_orchestrator(
        tools=[], model_name="dummy", workday=tight, use_llm=False,
        mutation_parser=parser, mutation_executor=executor, checkpointer_db=db,
    )


def test_crud_create_interrupts_then_executes_and_replans():
    calls = []
    mutation = TaskMutation(op=CrudOp.create, title="Call the bank", priority=Priority.high)
    go = _crud_orchestrator(calls=calls, mutation=mutation, result="Created task 'Call the bank'.")

    pending = go.start("add a task to call the bank", date="2026-06-17")
    assert pending.status == "interrupted"
    assert pending.interrupt["tool"] == "create_task"
    assert pending.interrupt["args"]["title"] == "Call the bank"
    assert calls == []  # gated — not executed before approval

    done = go.resume(pending.thread_id, {"action": "accept"})
    assert done.status == "done"
    assert calls == [mutation]                      # executed exactly once, on accept
    assert "✅ Created task 'Call the bank'." in done.text
    assert "Your plan" in done.text                 # re-planned after the change


def test_crud_reject_does_not_execute():
    calls = []
    mutation = TaskMutation(op=CrudOp.delete, task_id="5")
    go = _crud_orchestrator(calls=calls, mutation=mutation, result="Deleted task 5.")

    pending = go.start("delete task 5")
    assert pending.status == "interrupted"
    assert pending.interrupt["tool"] == "delete_task"

    done = go.resume(pending.thread_id, {"action": "reject", "reason": "nope"})
    assert done.status == "done"
    assert calls == []                              # never executed
    assert "left your tasks unchanged" in done.text.lower()


def test_crud_invalid_request_is_surfaced_recoverably():
    def parser(_user_input, _current_tasks):
        raise ValueError("could not resolve which task")

    go = build_graph_orchestrator(
        tools=[], model_name="dummy", use_llm=False, mutation_parser=parser
    )
    out = go.run("delete the thing")  # routes to CRUD, parse fails → recoverable
    assert "rephrase" in out.lower()


class _RecordingLLM:
    """Fake chat model that records the messages it is invoked with."""

    def __init__(self):
        self.seen = []  # one list of message-contents per invoke

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.seen.append([getattr(m, "content", "") for m in messages])
        return AIMessage(content="ok")


def test_planner_remembers_conversation_across_turns(monkeypatch):
    rec = _RecordingLLM()
    monkeypatch.setattr(llm_client, "create_ollama_model", lambda *a, **k: rec)

    go = build_graph_orchestrator(tools=[], model_name="dummy", use_llm=True)
    # Two turns on the SAME thread_id -> the second must see the first turn.
    go.start("remember apples", thread_id="mem-1")
    go.start("what did I say", thread_id="mem-1")

    # The final invoke is the second turn's agent call (an unknown prompt also
    # triggers one classifier call, so don't rely on a fixed index).
    second_turn_agent = rec.seen[-1]
    assert "remember apples" in second_turn_agent  # prior user message retained
    assert "ok" in second_turn_agent               # prior assistant reply retained


def test_new_turn_does_not_re_emit_previous_turns_result():
    # Regression: with a stable thread, per-turn scratch (result/notice) must be
    # reset so a later turn never repeats the previous turn's confirmation/plan.
    calls = []
    mutation = TaskMutation(op=CrudOp.create, title="Call the bank")
    go = _crud_orchestrator(calls=calls, mutation=mutation, result="Created task 'Call the bank'.")

    pending = go.start("add a task to call the bank", thread_id="turns")
    done1 = go.resume(pending.thread_id, {"action": "accept"})
    assert "✅ Created task 'Call the bank'." in done1.text   # turn 1 confirms

    # Turn 2 on the SAME thread: a summary request must not carry the turn-1 notice.
    done2 = go.start("what's on my list?", thread_id="turns")
    assert done2.status == "done"
    assert "✅" not in done2.text                              # no stale notice
    assert "Created task 'Call the bank'." not in done2.text   # no stale result
    assert "task(s)" in done2.text                             # fresh summary


def test_crud_state_persists_across_orchestrator_instances(tmp_path):
    db = str(tmp_path / "ckpt.db")
    mutation = TaskMutation(op=CrudOp.create, title="Persisted task")

    go1 = _crud_orchestrator(calls=[], mutation=mutation, result="Created task 'Persisted task'.", db=db)
    pending = go1.start("add a task to persist this")
    assert pending.status == "interrupted"

    # Simulate a restart: a fresh orchestrator backed by the same SQLite file.
    calls2 = []
    go2 = _crud_orchestrator(calls=calls2, mutation=mutation, result="Created task 'Persisted task'.", db=db)
    done = go2.resume(pending.thread_id, {"action": "accept"})
    assert done.status == "done"
    assert len(calls2) == 1                         # resumed from disk and executed
    assert "Your plan" in done.text


# -- visualization + observer ------------------------------------------------


def test_draw_mermaid_includes_the_nodes():
    go = build_graph_orchestrator(tools=[], model_name="dummy", use_llm=False)
    mermaid = go.draw_mermaid()
    for node in ("classify", "todo", "planner", "extract_mutation", "finalize"):
        assert node in mermaid


def test_observer_logs_steps_without_breaking_the_run(caplog):
    import logging

    tight = Workday(start="09:00", end="11:00", breaks=())
    go = build_graph_orchestrator(
        tools=[], model_name="dummy", use_llm=False, workday=tight, observe=True
    )
    with caplog.at_level(logging.INFO, logger="agents.graph_orchestrator"):
        out = go.run("plan my day", date="2026-06-16")

    assert "Your plan" in out                                  # run still works
    traced = [r.message for r in caplog.records if "[trace]" in r.message]
    assert any("classify" in m for m in traced)                # steps were observed
    assert any("planner" in m for m in traced)
