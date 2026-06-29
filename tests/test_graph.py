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


def test_multi_intent_prompt_falls_back_to_agent_loop_without_decomposition(monkeypatch):
    # A multi-intent prompt is normally decomposed; with the decomposer returning
    # nothing it falls back to the agent loop. The fake answers immediately.
    fake = _ScriptedLLM([AIMessage(content="handled both steps")])
    monkeypatch.setattr(llm_client, "create_ollama_model", lambda *a, **k: fake)

    go = build_graph_orchestrator(tools=[], model_name="dummy", use_llm=True,
                                  classifier=lambda text: "complex", decomposer=lambda text: [])
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
        tools=[echo], model_name="dummy", use_llm=True, classifier=lambda text: "unknown"
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

    go = build_graph_orchestrator(tools=[create_task], model_name="dummy", use_llm=True,
                                  classifier=lambda text: "complex", decomposer=lambda text: [])

    # Multi-intent prompt; with decomposition unavailable it falls back to the
    # agent⇄tools loop, whose mutating tool is gated for approval.
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

    go = build_graph_orchestrator(tools=[delete_task], model_name="dummy", use_llm=True,
                                  classifier=lambda text: "complex", decomposer=lambda text: [])

    # Multi-intent prompt; decomposition unavailable → agent⇄tools loop fallback.
    pending = go.start("plan my day and delete task 5")
    assert pending.status == "interrupted"

    done = go.resume(pending.thread_id, {"action": "reject", "reason": "changed my mind"})
    assert done.status == "done"
    assert "id" not in executed  # never executed


# -- plan confirmation + lock ------------------------------------------------


def _plannable():
    """A graph with the deterministic planner, sample tasks, and a stub executor."""
    return build_graph_orchestrator(
        tools=[], model_name="dummy", use_llm=False,
        workday=Workday(start="09:00", end="18:00"),
        mutation_executor=lambda m: f"Updated {m.task_id}",
    )


def _lock(go, thread="lk"):
    go.start("plan my day", thread_id=thread, date="2026-06-17")
    return go.resume(thread, {"action": "accept"})


def test_plan_requires_confirmation_then_locks():
    go = _plannable()
    pending = go.start("plan my day", thread_id="lk", date="2026-06-17")
    assert pending.status == "interrupted"
    assert pending.interrupt["type"] == "plan_approval"
    assert "Your plan" in pending.interrupt["plan"]

    done = go.resume("lk", {"action": "accept"})
    assert done.status == "done" and "🔒" in done.text


def test_plan_reject_with_suggestion_replans_then_reconfirms():
    go = _plannable()
    go.start("plan my day", thread_id="rj", date="2026-06-17")
    again = go.resume("rj", {"action": "reject", "reason": "I can work from 7am to 11pm"})
    assert again.status == "interrupted" and again.interrupt["type"] == "plan_approval"
    done = go.resume("rj", {"action": "accept"})
    assert "🔒" in done.text


def test_locked_plan_mark_done_annotates_in_place():
    go = _plannable()
    _lock(go)
    pending = go.start("mark Finish Q3 report as done", thread_id="lk")
    assert pending.status == "interrupted" and pending.interrupt["type"] == "approval"
    done = go.resume("lk", {"action": "accept"})
    assert "✅" in done.text and "~~Finish Q3 report~~" in done.text


def test_plain_plan_shows_locked_plan_without_reconfirming():
    go = _plannable()
    _lock(go)
    res = go.start("plan my day", thread_id="lk")  # no "replan"
    assert res.status == "done"            # show_locked — no interrupt
    assert "Your plan" in res.text


def test_explicit_replan_regenerates_and_reconfirms():
    go = _plannable()
    _lock(go)
    res = go.start("replan my day", thread_id="lk")
    assert res.status == "interrupted" and res.interrupt["type"] == "plan_approval"


# -- recall branch + planning-log ingestion ----------------------------------


def test_recall_intent_routes_to_the_rag_agent():
    class _FakeRag:
        def run(self, query, on_step=None):
            if on_step:
                on_step("searching memory")
            return f"recalled: {query}"

    go = build_graph_orchestrator(tools=[], model_name="dummy", use_llm=False, rag_agent=_FakeRag())
    out = go.run("what do I usually defer when busy?")
    assert out.startswith("recalled:")


def test_recall_streams_rag_progress_steps():
    class _FakeRag:
        def run(self, query, on_step=None):
            if on_step:
                on_step("🔎 searching")
                on_step("✅ enough")
            return "recalled"

    go = build_graph_orchestrator(tools=[], model_name="dummy", use_llm=False, rag_agent=_FakeRag())
    seen = []
    go.start("what do I usually defer when busy?", thread_id="rag-prog",
             on_step=lambda node, delta: seen.extend((delta or {}).get("progress", [])))
    assert seen == ["🔎 searching", "✅ enough"]  # the loop's phases surface live


def test_rejecting_with_an_appointment_suggestion_blocks_that_time():
    # Regression: a reject suggestion that is a fixed-time event must register as
    # an appointment (block that slot) — not be parsed as the working-hours window
    # (which collapsed the whole day onto the event).
    go = build_graph_orchestrator(
        tools=[], model_name="dummy", use_llm=False,
        workday=Workday(start="09:00", end="20:00"),
    )
    go.start("plan my day", thread_id="appt", date="2026-06-23")
    r = go.resume("appt", {"action": "reject",
                           "reason": "i have lunch with my friend from 11am into 1.30pm."})
    plan = (r.interrupt or {}).get("plan", "")
    assert "11:00–13:30" in plan and "Lunch with my friend" in plan
    assert "of 0 available" not in plan  # the day wasn't shrunk onto the event


def test_complex_prompt_decomposes_runs_each_step_and_confirms_plan_last():
    # The decomposer returns the plan before the summary; the graph reorders it so
    # the (confirmed) plan runs LAST, and each step runs its real path.
    go = build_graph_orchestrator(
        tools=[], model_name="dummy", use_llm=False,
        workday=Workday(start="09:00", end="18:00"),
        decomposer=lambda text: [
            {"intent": "plan", "text": "plan my day"},
            {"intent": "summary", "text": "what's on my list"},
        ],
    )
    # multi-intent prompt → classify 'complex' → decompose
    r = go.start("summarize my tasks and plan my day", thread_id="cx", date="2026-06-24")
    # the plan step (ordered last) pauses for confirmation — proving it didn't skip
    # the confirm loop the way the old agent-loop path did.
    assert r.status == "interrupted"
    assert (r.interrupt or {}).get("type") == "plan_approval"

    done = go.resume("cx", {"action": "accept"})
    assert done.status == "done"
    assert "task(s)" in done.text        # the summary step ran
    assert "Your plan" in done.text      # the plan step ran
    assert "Plan locked" in done.text    # …and went through confirm + lock


def test_order_steps_puts_edits_first_plan_then_ask_last():
    from agents.orchestrator import order_steps

    ordered = order_steps([
        {"intent": "ask", "text": "k"},
        {"intent": "plan", "text": "p"},
        {"intent": "weather", "text": "w"},
        {"intent": "delete", "text": "d"},
        {"intent": "add", "text": "a"},
    ])
    # edits → neutral → plan → ask (the reasoning step sees everything before it)
    assert [s["intent"] for s in ordered] == ["delete", "add", "weather", "plan", "ask"]


def test_ask_step_receives_earlier_step_results_as_context(monkeypatch):
    # A recording fake LLM captures what the `ask` step's agent call actually sees.
    class _Recorder:
        def __init__(self):
            self.seen = ""
        def bind_tools(self, _tools):
            return self
        def invoke(self, messages):
            self.seen = "\n".join(str(getattr(m, "content", "")) for m in messages)
            return AIMessage(content="Do the report first.")

    rec = _Recorder()
    monkeypatch.setattr(llm_client, "create_ollama_model", lambda *a, **k: rec)

    class _FakeRag:
        def run(self, query, on_step=None):
            return "You usually defer reading tasks when busy."

    go = build_graph_orchestrator(
        tools=[], model_name="dummy", use_llm=True, rag_agent=_FakeRag(),
        classifier=lambda text: "complex",
        # recall (neutral) then ask (last); the ask must see the recall's output.
        decomposer=lambda text: [
            {"intent": "recall", "text": "what do I defer when busy?"},
            {"intent": "ask", "text": "so which should I do first?"},
        ],
    )
    # "recall … and plan …" → 2 intent families → complex → decompose.
    done = go.start("recall what I defer and plan my day", thread_id="ctx", date="2026-06-24")

    assert done.status == "done"
    assert "defer reading tasks" in rec.seen           # earlier result fed into the ask step
    assert "Do the report first." in done.text          # the ask step's answer is in the output
    assert "defer reading tasks" in done.text            # and the recall result too (joined)


def test_at_time_intent_answers_from_the_locked_plan():
    go = build_graph_orchestrator(
        tools=[], model_name="dummy", use_llm=False,
        workday=Workday(start="09:00", end="20:00"),
    )
    go.start("plan my day", thread_id="at", date="2026-06-23")
    go.resume("at", {"action": "accept"})  # lock it
    r = go.start("which task do I have at 18:30", thread_id="at", date="2026-06-23")
    assert r.status == "done"
    assert "18:30" in r.text and "Dinner" in r.text  # the meal block at that time


def test_cancelling_a_plan_discards_it_without_locking_or_replanning():
    go = build_graph_orchestrator(
        tools=[], model_name="dummy", use_llm=False,
        workday=Workday(start="09:00", end="20:00"),
    )
    go.start("plan my day", thread_id="cancel", date="2026-06-23")
    r = go.resume("cancel", {"action": "cancel"})
    assert r.status == "done"
    assert "discarded" in r.text.lower()
    # A follow-up plan still works (nothing was left locked).
    r2 = go.start("plan my day", thread_id="cancel", date="2026-06-23")
    assert r2.status == "interrupted"  # confirms again, not silently using a lock


def test_locking_a_plan_writes_a_planning_log():
    writes = []
    go = build_graph_orchestrator(
        tools=[], model_name="dummy", use_llm=False,
        workday=Workday(start="09:00", end="18:00"),
        memory_writer=lambda text, source_type: writes.append((source_type, text)),
    )
    go.start("plan my day", thread_id="ml", date="2026-06-17")
    go.resume("ml", {"action": "accept"})
    assert writes and writes[0][0] == "planning_log"
    assert "scheduled" in writes[0][1].lower()


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
