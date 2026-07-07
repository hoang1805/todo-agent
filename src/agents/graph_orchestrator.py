"""LangGraph orchestrator — the same hub-and-spoke routing, as a graph.

This expresses the orchestrator as a ``StateGraph``: each step is a **node**,
routing is **edges**, and the handoff contract (:class:`~models.contract.TaskList`)
is a typed field on the shared **state**. It reuses the exact same agents and
formatters as the plain-Python :class:`~agents.orchestrator.Orchestrator`; only
the control-flow shell differs.

Two things this shape makes explicit that a ``match`` statement cannot:

* **The loop lives in an edge.** The open-ended path is the classic ReAct cycle
  ``agent → tools → agent → …``. The decision to keep looping is the conditional
  edge :func:`_should_continue` (tool calls pending → go to ``tools``); the
  ``tools → agent`` edge is the loop-back. The loop's *memory* (the running
  message list, and an implicit step count) lives in state, bounded by
  ``recursion_limit``.
* **A tool-agent decides tools — but only on the open-ended path.** The ``agent``
  node is an LLM bound to the task tools; it chooses which tool and args each
  iteration. The planner's overload decision stays deterministic (a plain node),
  so the graded behaviour remains reliable and testable.

* **Multi-step prompts go through that loop.** The deterministic capabilities
  (planning, summarizing) are *also* exposed as tools (``plan_my_day``,
  ``summarize_tasks``) alongside the MCP CRUD tools, so a request that touches
  several things — "add a task, then re-plan, then show my list" — is sequenced
  by the loop, one tool per iteration. A prompt that matches more than one intent
  family is routed here automatically (see ``classify``).

* **Mutating tools are gated by a human.** Create/update/delete tools are wrapped
  with :func:`~agents.human_in_the_loop.add_human_in_the_loop`, which pauses the
  graph (LangGraph ``interrupt``) for approval before the change runs. The graph
  is compiled with a checkpointer so it can be resumed with the user's decision;
  see :meth:`GraphOrchestrator.start` / :meth:`GraphOrchestrator.resume`.

Graph
-----
::

    START → classify ─┬─(plan|summary)→ todo ─┬─(plan)→ planner ──┐
                      │                        └─(summary)→ summary┤
                      │                        └─(error)→─────────►finalize → END
                      ├─(add|update|delete)→ extract → confirm ⏸ → execute ─┐
                      │      (validate)      (approve?)   (MCP write) │      │
                      │                                   └─(reject/err)→────┤
                      │                          (success)→ todo → planner ──┤
                      └─(other/multi)→ agent ⇄ tools (loop) ──(done)─────────┘

The CRUD branch is the spec's write flow: **validate** the change into a
``TaskMutation`` (the contract checkpoint), **confirm** it with a human via
``interrupt``, **execute** it through the MCP write tools, then **re-plan** the
day so the user sees the effect.
"""

from __future__ import annotations

import functools
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Callable, NotRequired, TYPE_CHECKING, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage  # noqa: TC002 — runtime use
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from agents.human_in_the_loop import wrap_mutating_tools
from agents.orchestrator import (
    MutationExecutor,
    _run_coro,
    classify_intent,
    format_detail,
    format_summary,
    is_crud_intent,
    llm_classify_intent,
    wants_detail,
    make_appointment_parser,
    make_decomposer,
    make_mcp_fetcher,
    make_mutation_executor,
    make_mutation_parser,
    make_normalizer,
    make_planner,
    matched_intent_families,
    order_steps,
)
from agents.planner_agent import (
    DEFAULT_WORKDAY,
    Planner,
    Workday,
    apply_mutation_to_plan,
    asks_about_time,
    describe_plan_at_time,
    format_plan,
    parse_query_time,
    parse_workday,
    plan_day,
    wants_replan,
    workday_with_appointments,
)
from pydantic import ValidationError

from agents.rag_agent import RAGAgent, make_rag_agent
from agents.todo_agent import ContractError, MutationParser, TodoAgent
from core.services.checkpoint import make_checkpointer_opener
from core.services.guardrails import GuardrailError, check_input, check_output, sanity_check_schedule
from core.services.history import History, get_history
from core.services.observability import log_event, set_session, track
from core.services.prompts import load_prompt
from core.tools.common_tools import make_common_tools
from models.contract import CrudOp, DayPlan, TaskList, TaskMutation

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Safety bound on the agent⇄tools loop so a misbehaving model can't spin forever.
RECURSION_LIMIT = 25

_CONTRACT_ERROR_MSG = (
    "I couldn't prepare your tasks right now (the task data didn't validate). "
    "Please try again in a moment."
)
_UNKNOWN_MSG = (
    "I can plan your day or summarize your tasks. "
    "Try \"plan my day\" or \"what's on my list?\""
)
_MUTATION_PARSE_ERROR_MSG = (
    "I couldn't tell exactly what change you wanted (or which task it refers to). "
    "Please rephrase — e.g. \"add a task to call the bank\" or "
    "\"mark 'Finish Q3 report' as done\"."
)
_REJECTED_MSG = "Okay — I left your tasks unchanged."


def _in_sequence(state) -> bool:
    """True while a decomposed complex prompt is being run step by step."""
    return bool(state.get("steps"))


def _more_steps(state) -> bool:
    """True if the step queue has steps left to dispatch."""
    return state.get("step_index", 0) < len(state.get("steps") or [])


# Which agent handles each intent — for the dashboard's per-agent usage breakdown.
_AGENT_FOR_INTENT = {
    "plan": "planner_agent", "appointment": "planner_agent", "at_time": "planner_agent",
    "summary": "todo_agent", "detail": "todo_agent",
    "add": "todo_agent", "update": "todo_agent", "delete": "todo_agent",
    "recall": "rag_agent", "complex": "orchestrator", "weather": "orchestrator",
}


def _summarize_delta(delta: dict) -> str:
    """Render a node's state update compactly for the step observer."""
    parts = []
    for key, value in (delta or {}).items():
        if key == "messages":
            msgs = value if isinstance(value, list) else [value]
            parts.append(f"messages+={len(msgs)}")
        elif isinstance(value, str):
            flat = value.replace("\n", " ")
            parts.append(f"{key}={flat[:60]!r}" + ("…" if len(flat) > 60 else ""))
        elif isinstance(value, dict):
            parts.append(f"{key}={{{', '.join(value.keys())}}}")
        else:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts) if parts else "(no state change)"


def _trace_node(name: str, fn):
    """Wrap a node so each step logs which node ran and the state it produced."""

    @functools.wraps(fn)
    def wrapped(state):
        logger.info("[trace] ▶ %s", name)
        result = fn(state)
        logger.info("[trace] ✓ %s → %s", name, _summarize_delta(result))
        return result

    return wrapped


def _plan_summary(plan: DayPlan, date: str | None) -> str:
    """A short planning-log line written to memory when a plan is locked."""
    when = date or "today"
    deferred = ", ".join(t.title for t in plan.deferred) or "nothing"
    return (
        f"On {when}: scheduled {len(plan.blocks)} task(s) into "
        f"{plan.available_minutes} working minutes; deferred {len(plan.deferred)} "
        f"({deferred})."
    )


def _describe_mutation(mutation: TaskMutation) -> str:
    """A one-line, human-readable summary of a proposed change (for approval)."""
    if mutation.op is CrudOp.create:
        return f"Create a new task: '{mutation.title}'?"
    if mutation.op is CrudOp.delete:
        return f"Delete task {mutation.task_id}?"
    fields = ", ".join(
        f"{name}={getattr(value, 'value', value)}"
        for name, value in mutation.changed_fields.items()
    )
    return f"Update task {mutation.task_id} ({fields})?"


class PlannerState(TypedDict):
    """The shared state that flows along the graph's edges."""

    user_input: str
    session_id: NotRequired[str]  # = thread_id; carried so nodes can tag their events
    date: NotRequired[str | None]
    intent: NotRequired[str]
    # The validated task list as a JSON-native dict (so the checkpointer stores
    # only plain types, not TaskList/Priority). Reconstructed via TaskList on use.
    tasks: NotRequired[dict | None]
    # The validated CRUD change as a JSON-native dict (so the checkpointer stores
    # only plain types, not custom classes). Reconstructed via TaskMutation when used.
    mutation: NotRequired[dict | None]
    # The task the user last looked at / acted on, as ``{"id", "title"}``. Unlike
    # the other per-turn fields it is NOT reset between turns, so a follow-up like
    # "change its category" can resolve to it (see `extract_mutation`).
    focus_task: NotRequired[dict | None]
    # The appointment detected in this turn's prompt (``{"name", "start", "end"}``),
    # set by ``classify`` so ``register_appointment`` reuses it without re-parsing
    # (avoids a second LLM extraction). Reset each turn.
    detected_appointment: NotRequired[dict | None]
    # Fixed-time commitments for the day, each ``{"name", "start", "end"}``, and
    # the last stated working hours ``{"start", "end"}``. Both persist across turns
    # so "replan today" keeps honoring them (see `run_planner`).
    appointments: NotRequired[list[dict]]
    work_hours: NotRequired[dict | None]
    # The accepted plan (a DayPlan JSON dict). Persists across turns (NOT reset),
    # so once locked, task changes patch it in place instead of re-planning.
    locked_plan: NotRequired[dict | None]
    # A reject suggestion fed back into the planner during the confirm loop.
    plan_feedback: NotRequired[str]
    # The freshly generated plan awaiting confirmation (becomes locked_plan on accept).
    pending_plan: NotRequired[dict | None]
    approved: NotRequired[bool]  # set by the confirm node from the user's decision
    # The raw confirm decision ("accept" / "reject" / "cancel"), so routing can
    # tell a reject (→ re-plan) from a cancel (→ just finish). Reset each turn.
    confirm_action: NotRequired[str]
    notice: NotRequired[str]     # success line prepended to the re-planned result
    # Sliding window of prior turns from the persisted history store, injected by
    # the orchestrator so a follow-up ("the high-priority ones") resolves.
    history: NotRequired[list[dict]]
    # `add_messages` makes this an append-only log — the agent⇄tools loop's memory.
    messages: Annotated[list[AnyMessage], add_messages]
    result: NotRequired[str]
    error: NotRequired[str]
    # Per-turn, human-readable progress lines (e.g. the RAG loop's retrieve/judge/
    # reformulate phases). Surfaced live in the UI via streaming; reset each turn.
    progress: NotRequired[list[str]]
    # Complex-prompt decomposition: an ordered queue of single-intent steps
    # ({"intent", "text"}), a cursor into it, and each step's result. The loop runs
    # one step per pass through its real handler, then joins the results. Reset each turn.
    steps: NotRequired[list[dict]]
    step_index: NotRequired[int]
    step_results: NotRequired[list[str]]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def format_tool_roster(tools: list[BaseTool]) -> str:
    """Render the bound tools as a roster for the agent's system prompt."""
    lines = ["## Available tools", ""]
    for t in tools:
        first_line = (t.description or "").strip().splitlines()[0] if t.description else ""
        lines.append(f"- `{t.name}`: {first_line}")
    return "\n".join(lines)


def build_graph_orchestrator(
    tools: list[BaseTool],
    model_name: str,
    temperature: float = 0.0,
    workday: Workday | None = None,
    use_llm: bool = True,
    checkpointer_db: str | None = None,
    mutation_parser: MutationParser | None = None,
    mutation_executor: MutationExecutor | None = None,
    planner: Planner | None = None,
    appointment_parser: "Callable[[str], object] | None" = None,
    decomposer: "Callable[[str], list[dict]] | None" = None,
    classifier: "Callable[[str], str] | None" = None,
    memory_tools: list[BaseTool] | None = None,
    rag_agent: RAGAgent | None = None,
    memory_writer: "Callable[[str, str], None] | None" = None,
    history: History | None = None,
    observe: bool = False,
):
    """Wire and compile the LangGraph orchestrator.

    This is the single place agents are registered — adding a new specialist
    means adding a node and an edge here, exactly as with the plain orchestrator.
    Returns a :class:`GraphOrchestrator`.

    ``checkpointer_db`` enables a persistent (SQLite) checkpointer so interrupt/
    resume survives restarts; left ``None`` it uses an in-memory saver (tests).
    ``mutation_parser`` / ``mutation_executor`` / ``planner`` are injectable so
    the CRUD and planning paths are testable without an LLM or a live MCP server.
    """
    workday = workday or DEFAULT_WORKDAY
    mutation_parser = mutation_parser or make_mutation_parser(model_name, temperature, use_llm)
    mutation_executor = mutation_executor or make_mutation_executor(tools)
    planner = planner or make_planner(model_name, temperature, use_llm)

    def _safe_plan(task_list, wd, date=None):
        """Plan, then self-check (guardrail): on overlap/negative-duration, fall
        back to the deterministic planner so the agent never returns a broken schedule."""
        plan = planner(task_list, wd, date)
        issues = sanity_check_schedule(plan)
        if issues:
            logger.warning("planner self-check failed (%s); deterministic fallback.", issues)
            plan = plan_day(task_list, workday=wd, date=date)
        return plan

    appointment_parser = appointment_parser or make_appointment_parser(model_name, temperature, use_llm)
    decomposer = decomposer or make_decomposer(model_name, temperature, use_llm)
    # Memory tools (retrieve_*/remember) arrive in the same combined `tools` list,
    # so default to it; the retriever/writer pick the right tools by name.
    memory_tools = tools if memory_tools is None else memory_tools
    rag_agent = rag_agent or make_rag_agent(memory_tools, model_name, temperature, use_llm)
    if memory_writer is None:
        from agents.rag_agent import make_memory_writer
        memory_writer = make_memory_writer(memory_tools)
    todo_agent = TodoAgent(
        fetch_raw=make_mcp_fetcher(tools),
        normalize=make_normalizer(model_name, temperature, use_llm),
        parse=mutation_parser,
    )

    # The capabilities, exposed as tools so the agent loop can sequence them with
    # the MCP tools for multi-step prompts. Planning goes through the injected
    # `planner` (LLM with a validated deterministic fallback).
    def _plan_tool(day_start: str | None = None, day_end: str | None = None) -> str:
        try:
            tasks = todo_agent.run()
        except ContractError:
            return "ERROR: could not load tasks to plan (task data failed validation)."
        wd = workday
        if day_start or day_end:
            wd = Workday(
                start=day_start or workday.start,
                end=day_end or workday.end,
                breaks=workday.breaks,
            )
        return format_plan(planner(tasks, wd, None))

    def _summary_tool() -> str:
        try:
            tasks = todo_agent.run()
        except ContractError:
            return "ERROR: could not load tasks to summarize."
        return format_summary(tasks)

    plan_my_day = StructuredTool.from_function(
        func=_plan_tool,
        name="plan_my_day",
        description=(
            "Build a prioritized, time-blocked plan of today's tasks, fitted into "
            "the working hours around meal breaks; tasks that don't fit are "
            "deferred. Optional day_start/day_end (HH:MM) override the hours. "
            "Returns the formatted plan."
        ),
    )
    summarize_tasks = StructuredTool.from_function(
        func=_summary_tool,
        name="summarize_tasks",
        description=(
            "List today's tasks with priority, category, and estimated duration. "
            "Returns the formatted summary."
        ),
    )

    # Tools the agent loop can call: the deterministic capabilities, ALL the local
    # helper tools, and every live MCP task tool. Mutating MCP tools (create /
    # update / delete) are wrapped so they require human approval before running.
    agent_tools = [
        plan_my_day,
        summarize_tasks,
        *make_common_tools(),
        *wrap_mutating_tools(tools),
    ]

    # The tool-agent for the open-ended / multi-step path (only on the agent path).
    llm_with_tools = None
    if use_llm:
        from core.services.llm_client import create_ollama_model

        model = create_ollama_model(model_name, temperature, with_thinking=False)
        llm_with_tools = model.bind_tools(agent_tools)

    # -- nodes --------------------------------------------------------------

    def classify(state: PlannerState) -> dict:
        # Echo the user's prompt before any reasoning (for logs/observability).
        logger.info("USER PROMPT: %s", state["user_input"])
        text = state["user_input"]
        sid = state.get("session_id")
        # Classification is the LLM's job: a single call decides among every intent
        # (incl. `complex` for multi-step requests, `at_time`, `appointment`).
        # `llm_classify_intent` already falls back to the keyword classifier if the
        # model is unreachable or returns an invalid label, so this stays robust.
        with track("orchestrator", "classify", session_id=sid) as span:
            if classifier is not None:
                out = {"intent": classifier(text)}
            elif use_llm:
                out = {"intent": llm_classify_intent(text, model_name).value}
            elif wants_detail(text):
                out = {"intent": "detail"}
            elif asks_about_time(text):
                out = {"intent": "at_time"}
            elif (appt := appointment_parser(text)) is not None:  # regex offline
                out = {"intent": "appointment",
                       "detected_appointment": {"name": appt.name, "start": appt.start, "end": appt.end}}
            elif len(set(matched_intent_families(text))) > 1:
                out = {"intent": "complex"}
            else:
                out = {"intent": classify_intent(text).value}
            span["intent"] = out["intent"]
        # The routing decision: which intent, mapped to the agent that will handle it.
        log_event("orchestrator", "route", session_id=sid,
                  details={"intent": out["intent"], "agent": _AGENT_FOR_INTENT.get(out["intent"], "orchestrator")})
        return out

    def run_todo(state: PlannerState) -> dict:
        try:
            with track("todo_agent", "fetch", session_id=state.get("session_id")) as span:
                # Store as a JSON-native dict so the checkpointer persists plain types
                # (not TaskList/Priority, which trip the serde's unregistered-type warning).
                tasks = todo_agent.run()
                span["task_count"] = len(tasks.tasks)
            return {"tasks": tasks.model_dump(mode="json")}
        except ContractError as exc:
            logger.error("Contract failed: %s", exc.errors)
            return {"error": _CONTRACT_ERROR_MSG}

    def register_appointment(state: PlannerState) -> dict:
        """Record a fixed-time commitment so the re-plan schedules around it."""
        # Reuse what classify already parsed; re-parse only as a defensive fallback.
        entry = state.get("detected_appointment")
        if not entry:
            appt = appointment_parser(state["user_input"])
            if appt is None:  # defensive — classify already gated this
                return {}
            entry = {"name": appt.name, "start": appt.start, "end": appt.end}
        appointments = list(state.get("appointments") or [])
        if entry not in appointments:
            appointments.append(entry)
        return {
            "appointments": appointments,
            "notice": f"📌 Noted '{entry['name']}' from {entry['start']} to {entry['end']}.",
        }

    def run_planner(state: PlannerState) -> dict:
        # A reject suggestion (confirm loop) may be a working-hours change
        # ("actually I can work to 11pm") OR a fixed-time commitment ("I have lunch
        # 11am–1:30pm"). Parse the appointment first so its time range is never
        # mistaken for the day window (which would shrink the day onto the event).
        feedback = state.get("plan_feedback") or ""
        appt = appointment_parser(feedback) if feedback else None

        # Working hours: parse from the prompt when explicitly planning, and from a
        # reject suggestion only when it isn't a fixed-time event. Reuse the last
        # hours so "replan today" keeps them; fall back to the configured workday.
        work_hours = state.get("work_hours")
        hours_text = state["user_input"] if state.get("intent") == "plan" else ""
        if feedback and appt is None:
            hours_text = f"{hours_text} {feedback}".strip()
        parsed = parse_workday(hours_text) if hours_text else None
        if parsed is not None:
            wd = parsed
            work_hours = {"start": parsed.start, "end": parsed.end}
        elif work_hours:
            wd = Workday(start=work_hours["start"], end=work_hours["end"], breaks=workday.breaks)
        else:
            wd = workday

        # A suggestion may also add a fixed-time commitment to schedule around.
        appointments = list(state.get("appointments") or [])
        if appt is not None:
            entry = {"name": appt.name, "start": appt.start, "end": appt.end}
            if entry not in appointments:
                appointments.append(entry)

        # Fold in fixed-time commitments, then plan (LLM + safe fallback).
        wd = workday_with_appointments(wd, appointments)
        sid = state.get("session_id")
        with track("planner_agent", "plan", session_id=sid) as span:
            plan = _safe_plan(TaskList.model_validate(state["tasks"]), wd, state.get("date"))
            span["overloaded"] = plan.overloaded
            span["blocks"] = len(plan.blocks)
        # The overload branch: log what got deferred (and why) when the day can't fit.
        if plan.overloaded:
            log_event("planner_agent", "overload", session_id=sid, details={
                "deferred": [t.title for t in plan.deferred],
                "deferred_count": len(plan.deferred),
                "scheduled": len(plan.blocks),
                "available_minutes": plan.available_minutes,
            })
        out: dict = {
            "result": format_plan(plan),
            "pending_plan": plan.model_dump(mode="json"),
        }
        if work_hours:
            out["work_hours"] = work_hours
        if appointments != (state.get("appointments") or []):
            out["appointments"] = appointments
        return out

    def show_locked(state: PlannerState) -> dict:
        """Render the already-locked plan without re-planning."""
        return {"result": format_plan(DayPlan.model_validate(state["locked_plan"]))}

    def confirm_plan(state: PlannerState) -> dict:
        """Pause for the user to accept the plan, or reject with a suggestion.

        Accept → lock the plan for the day. Reject → fold the suggestion back in
        and re-plan (the confirm loop re-enters this node).
        """
        decision = interrupt(
            {
                "type": "plan_approval",
                "plan": state.get("result", ""),
                "message": "Accept this plan for today?",
            }
        )
        action = decision.get("action") if isinstance(decision, dict) else decision
        if action == "accept":
            plan_dict = state.get("pending_plan")
            # Ingestion (RAG): a locked plan writes a planning_log summary to memory,
            # creating the feedback loop between the planner and the recall agent.
            if memory_writer and plan_dict:
                try:
                    memory_writer(
                        _plan_summary(DayPlan.model_validate(plan_dict), state.get("date")),
                        "planning_log",
                    )
                except Exception as exc:  # noqa: BLE001 — memory is best-effort
                    logger.warning("planning_log write failed: %s", exc)
            return {
                "approved": True,
                "confirm_action": "accept",
                "locked_plan": plan_dict,
                "result": state.get("result", "") + "\n\n🔒 Plan locked for today.",
            }
        if action == "cancel":
            # Drop this proposal without locking it and without re-planning.
            return {
                "approved": False,
                "confirm_action": "cancel",
                "plan_feedback": "",
                "result": "🚫 Plan discarded — nothing was locked. Ask me to plan again whenever you like.",
            }
        suggestion = decision.get("reason") if isinstance(decision, dict) else None
        return {"approved": False, "confirm_action": "reject", "plan_feedback": suggestion or ""}

    def apply_to_locked(state: PlannerState) -> dict:
        """Patch the locked plan with the just-executed change (no re-plan)."""
        plan = DayPlan.model_validate(state["locked_plan"])
        mutation = TaskMutation.model_validate(state["mutation"])
        updated = apply_mutation_to_plan(plan, mutation)
        return {
            "locked_plan": updated.model_dump(mode="json"),
            "result": format_plan(updated),  # finalize prepends the ✅ notice
        }

    def run_summary(state: PlannerState) -> dict:
        return {"result": format_summary(TaskList.model_validate(state["tasks"]))}

    def run_detail(state: PlannerState) -> dict:
        from agents.orchestrator import _match_task

        tasks = TaskList.model_validate(state["tasks"])
        # Remember which task this was about so a follow-up edit can refer to it.
        task = _match_task(tasks, state["user_input"])
        if task is not None:
            return {
                "result": format_detail(tasks, state["user_input"]),
                "focus_task": {"id": task.id, "title": task.title},
            }
        # "Tell me about X" where X isn't a task: the user may mean something in
        # their documents/logs (e.g. an entity from an ingested file) — ask the
        # RAG agent before giving up, and only fall back to the task-list hint
        # when memory has nothing either.
        steps = ["ℹ️ No task matches — checking your documents and logs instead"]
        answer = rag_agent.run(state["user_input"], on_step=steps.append)
        if answer and not answer.startswith("I couldn't find anything relevant"):
            return {"result": answer, "progress": steps}
        return {"result": format_detail(tasks, state["user_input"]), "progress": steps}

    def run_at_time(state: PlannerState) -> dict:
        """Answer "which task do I have at <time>?" against the plan.

        Prefers the locked (or just-proposed) plan; if there isn't one yet, it
        builds a plan to answer against — honoring any stated hours/appointments —
        without locking it.
        """
        when = parse_query_time(state["user_input"])
        if when is None:
            return {"result": "Which time do you mean? e.g. \"what's scheduled at 7pm?\""}
        plan_dict = state.get("locked_plan") or state.get("pending_plan")
        if plan_dict:
            plan = DayPlan.model_validate(plan_dict)
        else:
            try:
                tasks = todo_agent.run()
            except ContractError as exc:
                logger.error("Contract failed: %s", exc.errors)
                return {"result": _CONTRACT_ERROR_MSG}
            wh = state.get("work_hours")
            wd = Workday(start=wh["start"], end=wh["end"], breaks=workday.breaks) if wh else workday
            wd = workday_with_appointments(wd, list(state.get("appointments") or []))
            plan = _safe_plan(tasks, wd, state.get("date"))
        return {"result": describe_plan_at_time(plan, when)}

    # -- complex path: decompose into single-intent steps, run each for real ----

    # Step intents whose handler reads `user_input`/state (set by `dispatch`).
    # Anything else (weather/unknown) routes to the agent loop, which reads
    # `messages`, so dispatch hands it the step text as the latest instruction.
    _DETERMINISTIC_STEP_INTENTS = {
        "plan", "summary", "detail", "at_time", "appointment", "add", "update", "delete", "recall",
    }

    def decompose(state: PlannerState) -> dict:
        """Split a multi-intent prompt into ordered single-intent steps.

        Empty (model off/failed) → fall back to the agent⇄tools loop.
        """
        steps = order_steps(decomposer(state["user_input"]))
        if not steps:
            return {"steps": []}
        return {"steps": steps, "step_index": 0, "step_results": []}

    def dispatch(state: PlannerState) -> dict:
        """Pop the next step: set its intent + sub-prompt and reset per-step scratch
        so its handler runs exactly as a standalone single-intent turn would."""
        steps = state["steps"]
        i = state.get("step_index", 0)
        step = steps[i]
        out: dict = {
            "intent": step["intent"],
            "user_input": step["text"],
            "step_index": i + 1,
            # Per-step scratch reset (cross-turn fields — locked_plan, appointments,
            # work_hours, focus_task — intentionally persist across steps).
            "tasks": None, "mutation": None, "approved": False, "confirm_action": "",
            "notice": "", "result": "", "error": "", "detected_appointment": None,
            "plan_feedback": "", "pending_plan": None,
        }
        if step["intent"] not in _DETERMINISTIC_STEP_INTENTS:
            # Agent-loop step (`ask`/`weather`): hand it the step text as the latest
            # instruction, prefixed with earlier steps' results so a reasoning step
            # ("…then tell me which to do first") can build on them.
            prior = state.get("step_results") or []
            instruction = step["text"]
            if prior:
                context = "\n\n".join(prior)
                instruction = (
                    f"Results from earlier steps of this request:\n\n{context}\n\n"
                    f"Using those where relevant, now: {step['text']}"
                )
            out["messages"] = [HumanMessage(content=instruction)]
        return out

    def run_rag(state: PlannerState) -> dict:
        """The 'recall' branch — the RAGAgent retrieves + reasons over memory.

        The loop's phases are collected into ``progress`` so the streaming UI can
        show what the recall agent actually did (which sources, reformulations).
        """
        steps: list[str] = []
        with track("rag_agent", "invoke", session_id=state.get("session_id")):
            answer = rag_agent.run(state["user_input"], on_step=steps.append)
        return {"result": answer, "progress": steps}

    # -- CRUD path: validate (contract checkpoint) → confirm → execute → replan --

    def extract_mutation(state: PlannerState) -> dict:
        """Parse + validate the write request into a TaskMutation (runs once).

        Stored as a JSON-native dict so the checkpointer persists only plain types.
        """
        focus_id = (state.get("focus_task") or {}).get("id")
        try:
            mutation = todo_agent.parse_mutation(state["user_input"], focus_task_id=focus_id)
            out: dict = {"mutation": mutation.model_dump(mode="json", exclude_none=True)}
            # Keep the focus on whichever task this change targets, so a further
            # follow-up ("and mark it done") chains to the same task.
            if mutation.task_id:
                out["focus_task"] = {"id": mutation.task_id, "title": mutation.title}
            return out
        except ContractError as exc:
            logger.error("Mutation parse failed: %s", exc.errors)
            return {"error": _MUTATION_PARSE_ERROR_MSG}

    def confirm_mutation(state: PlannerState) -> dict:
        """Pause for human approval before any task data is changed.

        This is the only node that calls ``interrupt`` — on resume it re-runs and
        ``interrupt`` returns the user's decision (the parse in ``extract`` is
        already in state, so it is not redone).
        """
        mutation = TaskMutation.model_validate(state["mutation"])
        decision = interrupt(
            {
                "type": "approval",
                "tool": f"{mutation.op.value}_task",
                "args": state["mutation"],
                "message": _describe_mutation(mutation),
            }
        )
        action = decision.get("action") if isinstance(decision, dict) else decision

        if action in ("accept", "edit"):
            if action == "edit" and isinstance(decision, dict) and decision.get("args"):
                try:
                    edited = TaskMutation.model_validate(decision["args"])
                except ValidationError:
                    return {"approved": False, "result": "The edited change wasn't valid; nothing was applied."}
                return {"approved": True, "mutation": edited.model_dump(mode="json", exclude_none=True)}
            return {"approved": True}

        reason = decision.get("reason") if isinstance(decision, dict) else None
        return {"approved": False, "result": _REJECTED_MSG + (f" ({reason})" if reason else "")}

    def execute_mutation(state: PlannerState) -> dict:
        """Apply the approved change via the MCP write tools."""
        result = mutation_executor(TaskMutation.model_validate(state["mutation"]))
        if result.startswith("ERROR"):
            return {"error": result}
        return {"notice": f"✅ {result}"}

    # The system prompt lists the live tools so the model knows exactly what it
    # can call (built once; the tool set is fixed for this orchestrator).
    system_text = load_prompt("complex_agent_system") + "\n\n" + format_tool_roster(agent_tools)

    def agent(state: PlannerState) -> dict:
        from langchain_core.messages import AIMessage, SystemMessage

        if llm_with_tools is None:
            return {"messages": [AIMessage(content=_UNKNOWN_MSG)]}
        # Inject the persisted-history window so a follow-up resolves against earlier turns.
        convo = state.get("history") or []
        system = system_text
        if convo:
            recent = "\n".join(f"{t['role']}: {t['content']}" for t in convo)
            system += f"\n\n## Recent conversation (for context)\n{recent}"
        response = llm_with_tools.invoke(
            [SystemMessage(content=system), *state["messages"]]
        )
        return {"messages": [response]}

    def finalize(state: PlannerState) -> dict:
        notice = state.get("notice")
        msgs = state.get("messages") or []

        if state.get("result"):
            # A successful CRUD change prepends its confirmation to the re-plan.
            final = f"{notice}\n\n{state['result']}" if notice else state["result"]
        elif state.get("error"):
            final = f"{notice}\n\n{state['error']}" if notice else state["error"]
        elif notice:
            # A standalone confirmation (e.g. a CRUD step in a sequence that skips
            # the auto-replan) — surface it on its own.
            final = notice
        elif state.get("intent") in (None, "unknown") and not msgs:
            final = _UNKNOWN_MSG
        else:
            last = msgs[-1] if msgs else None
            final = getattr(last, "content", "") or _UNKNOWN_MSG

        # Sequence mode: accumulate each step's result and emit one joined message
        # only on the last step (loop back to `dispatch` until the queue is empty).
        if state.get("steps"):
            results = list(state.get("step_results") or [])
            if final:
                results.append(final)
            if _more_steps(state):
                return {"step_results": results, "result": ""}
            joined = "\n\n".join(r for r in results if r) or _UNKNOWN_MSG
            return {"result": joined, "step_results": results, "messages": [AIMessage(content=joined)]}

        out: dict = {"result": final}
        # Record the assistant's reply in the conversation log so the next turn
        # remembers it (memory). The agent loop already leaves its final answer as
        # the last message, so only append for the deterministic paths.
        last = msgs[-1] if msgs else None
        loop_already_logged = (
            isinstance(last, AIMessage)
            and not getattr(last, "tool_calls", None)
            and (last.content or "") == final
        )
        if final and not loop_already_logged:
            out["messages"] = [AIMessage(content=final)]
        return out

    # -- edges (incl. the CRUD path and the loop) --------------------------

    def route_by_intent(state: PlannerState) -> str:
        # plan/summary → deterministic read pipeline; add/update/delete → CRUD
        # path; a multi-intent "complex" prompt → decompose into single-intent
        # steps; anything else → the agent loop. Shared by `classify` and `dispatch`.
        intent = state["intent"]
        if intent == "complex":
            return "decompose"
        if intent == "appointment":
            return "appointment"
        if intent == "plan":
            # A locked plan is shown as-is unless the user asks to "replan" — but a
            # plan step *inside a sequence* always regenerates, so it reflects the
            # edits the earlier steps just made.
            if state.get("locked_plan") and not wants_replan(state["user_input"]) and not _in_sequence(state):
                return "show_locked"
            return "todo"
        if intent in ("summary", "detail"):
            return "todo"
        if intent == "at_time":
            return "at_time"  # look up the plan at a time (no todo/planner pipeline)
        if intent in ("add", "update", "delete"):
            return "crud"
        if intent == "recall":
            return "rag"  # the one new branch for the memory/RAG agent
        return "agent"

    def route_after_todo(state: PlannerState) -> str:
        if state.get("error"):
            return "finalize"
        # summary lists all; detail shows one task; plan and any post-CRUD
        # re-plan go to the planner.
        if state["intent"] == "summary":
            return "summary"
        if state["intent"] == "detail":
            return "detail"
        return "planner"

    def route_after_planner(state: PlannerState) -> str:
        # A freshly (re)generated plan from an explicit plan/appointment request is
        # confirmed (accept → lock); CRUD-triggered re-plans (unlocked) just show.
        return "confirm_plan" if state["intent"] in ("plan", "appointment") else "finalize"

    def route_after_confirm_plan(state: PlannerState) -> str:
        # accept → done; cancel → done (discarded); reject → re-plan with the
        # suggestion (the confirm loop).
        if state.get("approved"):
            return "finalize"
        return "finalize" if state.get("confirm_action") == "cancel" else "planner"

    def route_after_extract(state: PlannerState) -> str:
        return "finalize" if state.get("error") else "confirm_mutation"

    def route_after_confirm(state: PlannerState) -> str:
        return "execute_mutation" if state.get("approved") else "finalize"

    def route_after_execute(state: PlannerState) -> str:
        if state.get("error"):
            return "finalize"
        # In a sequence, a later plan step (ordered last) does the planning, so
        # skip the per-mutation auto-replan and just surface the confirmation.
        if _in_sequence(state):
            return "finalize"
        # Locked plan → patch it in place; otherwise re-plan (unlocked).
        return "apply_to_locked" if state.get("locked_plan") else "todo"

    def route_after_decompose(state: PlannerState) -> str:
        # Steps found → run them; none (model off/failed) → the agent⇄tools loop.
        return "dispatch" if state.get("steps") else "agent"

    def route_after_finalize(state: PlannerState) -> str:
        # In a sequence with steps left, loop back for the next one; else end.
        return "dispatch" if _more_steps(state) else END

    def should_continue(state: PlannerState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "finalize"

    # Observer: when on (param or PLANNER_TRACE env), each node logs the step it
    # ran and the state it produced — so you can watch the agents work.
    observe = observe or bool(os.getenv("PLANNER_TRACE"))
    node = (lambda name, fn: _trace_node(name, fn)) if observe else (lambda name, fn: fn)

    # Shared intent → node map (used by both `classify` and the per-step `dispatch`).
    intent_routes = {
        "todo": "todo", "crud": "extract_mutation", "agent": "agent",
        "appointment": "register_appointment", "show_locked": "show_locked",
        "at_time": "at_time", "rag": "rag", "decompose": "decompose",
    }

    g = StateGraph(PlannerState)
    g.add_node("classify", node("classify", classify))
    g.add_node("decompose", node("decompose", decompose))
    g.add_node("dispatch", node("dispatch", dispatch))
    g.add_node("todo", node("todo", run_todo))
    g.add_node("planner", node("planner", run_planner))
    g.add_node("confirm_plan", node("confirm_plan", confirm_plan))
    g.add_node("show_locked", node("show_locked", show_locked))
    g.add_node("apply_to_locked", node("apply_to_locked", apply_to_locked))
    g.add_node("summary", node("summary", run_summary))
    g.add_node("detail", node("detail", run_detail))
    g.add_node("at_time", node("at_time", run_at_time))
    g.add_node("rag", node("rag", run_rag))
    g.add_node("register_appointment", node("register_appointment", register_appointment))
    g.add_node("extract_mutation", node("extract_mutation", extract_mutation))
    g.add_node("confirm_mutation", node("confirm_mutation", confirm_mutation))
    g.add_node("execute_mutation", node("execute_mutation", execute_mutation))
    g.add_node("agent", node("agent", agent))
    g.add_node("tools", ToolNode(agent_tools))
    g.add_node("finalize", node("finalize", finalize))

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_by_intent, intent_routes)
    # complex → decompose → (dispatch the step queue | agent loop fallback)
    g.add_conditional_edges(
        "decompose", route_after_decompose, {"dispatch": "dispatch", "agent": "agent"}
    )
    g.add_conditional_edges("dispatch", route_by_intent, intent_routes)
    g.add_edge("register_appointment", "todo")  # register, then re-plan the day
    g.add_conditional_edges(
        "todo", route_after_todo,
        {"planner": "planner", "summary": "summary", "detail": "detail",
         "finalize": "finalize"},
    )
    g.add_conditional_edges(
        "planner", route_after_planner,
        {"confirm_plan": "confirm_plan", "finalize": "finalize"},
    )
    g.add_conditional_edges(
        "confirm_plan", route_after_confirm_plan,
        {"finalize": "finalize", "planner": "planner"},  # reject → re-plan loop
    )
    g.add_conditional_edges(
        "extract_mutation", route_after_extract,
        {"confirm_mutation": "confirm_mutation", "finalize": "finalize"},
    )
    g.add_conditional_edges(
        "confirm_mutation", route_after_confirm,
        {"execute_mutation": "execute_mutation", "finalize": "finalize"},
    )
    g.add_conditional_edges(
        "execute_mutation", route_after_execute,
        {"todo": "todo", "apply_to_locked": "apply_to_locked", "finalize": "finalize"},
    )
    g.add_edge("show_locked", "finalize")
    g.add_edge("apply_to_locked", "finalize")
    g.add_edge("summary", "finalize")
    g.add_edge("detail", "finalize")
    g.add_edge("at_time", "finalize")
    g.add_edge("rag", "finalize")
    g.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "finalize": "finalize"}
    )
    g.add_edge("tools", "agent")  # <-- the loop back
    # In a sequence, loop back to dispatch for the next step; otherwise end.
    g.add_conditional_edges("finalize", route_after_finalize, {"dispatch": "dispatch", END: END})

    # The graph is left *uncompiled*: GraphOrchestrator compiles it per call with
    # a freshly-opened checkpointer (see its docstring) so a persistent SQLite
    # connection lives on the same event loop that runs the graph.
    history = history if history is not None else get_history()
    return GraphOrchestrator(g, make_checkpointer_opener(checkpointer_db), history=history)


@dataclass
class PlannerResult:
    """The outcome of a graph run.

    ``status`` is ``"done"`` (``text`` holds the answer) or ``"interrupted"``
    (``interrupt`` holds the proposed action awaiting approval; resume with the
    same ``thread_id``).
    """

    status: str
    thread_id: str
    text: str = ""
    interrupt: dict[str, Any] | None = None


class GraphOrchestrator:
    """Sync wrapper around the LangGraph, with human-in-the-loop support.

    Holds the *uncompiled* graph and a checkpointer opener. Each call opens the
    checkpointer, compiles the graph against it, and invokes — all in one
    coroutine — so a persistent SQLite connection is created and used on a single
    event loop (avoiding "connection bound to a closed loop" with ``asyncio.run``).
    """

    def __init__(self, graph_def, open_checkpointer, history: History | None = None) -> None:
        self._graph_def = graph_def
        self._open_checkpointer = open_checkpointer
        self._history = history  # canonical, persisted conversation log (may be None)

    # -- conversation history ----------------------------------------------

    def list_sessions(self) -> list[dict]:
        return self._history.list_sessions() if self._history else []

    def get_turns(self, session_id: str) -> list[dict]:
        return self._history.get_all_turns(session_id) if self._history else []

    def _record_assistant(self, thread_id: str, result: "PlannerResult") -> None:
        """Persist the assistant's reply once a turn completes (not while interrupted)."""
        if self._history and result.status == "done" and result.text:
            self._history.add_turn(thread_id, "assistant", result.text)

    # -- visualization ------------------------------------------------------

    def _drawable(self):
        """A compiled graph for drawing only (no checkpointer needed)."""
        return self._graph_def.compile()

    def draw_mermaid(self) -> str:
        """Return the graph as Mermaid text (paste into https://mermaid.live)."""
        return self._drawable().get_graph().draw_mermaid()

    def draw_ascii(self) -> str:
        """Return an ASCII rendering of the graph (needs the ``grandalf`` extra)."""
        return self._drawable().get_graph().draw_ascii()

    def save_visualization(self, path: str = "planner_graph") -> dict:
        """Write the graph to ``<path>.mmd`` and try ``<path>.png`` (best effort).

        Returns the paths actually written, e.g. ``{"mermaid": "...", "png": "..."}``.
        """
        from pathlib import Path

        graph = self._drawable().get_graph()
        written: dict[str, str] = {}

        mmd_path = Path(f"{path}.mmd")
        mmd_path.write_text(graph.draw_mermaid(), encoding="utf-8")
        written["mermaid"] = str(mmd_path)

        try:  # PNG needs mermaid.ink (network) or a local renderer — optional.
            png = graph.draw_mermaid_png()
            png_path = Path(f"{path}.png")
            png_path.write_bytes(png)
            written["png"] = str(png_path)
        except Exception as exc:  # noqa: BLE001 — PNG is a nice-to-have
            logger.info("PNG render skipped (%s); the .mmd file is available.", exc)

        return written

    def _config(self, thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }

    async def _ainvoke(self, payload, thread_id: str) -> dict:
        async with self._open_checkpointer() as saver:
            graph = self._graph_def.compile(checkpointer=saver)
            return await graph.ainvoke(payload, config=self._config(thread_id))

    async def _astream(self, payload, thread_id: str, on_step) -> dict:
        """Run the graph, reporting each node as it completes to *on_step(node, delta)*.

        Returns the final state in the same shape :meth:`_interpret` expects. Node
        functions run in worker threads, so progress is surfaced *here* — on the
        caller's thread, as each node finishes — which is what makes it safe to
        drive a UI from. Falls back to reading the final snapshot for the result
        and any pending interrupt.
        """
        interrupts = None
        async with self._open_checkpointer() as saver:
            graph = self._graph_def.compile(checkpointer=saver)
            config = self._config(thread_id)
            async for chunk in graph.astream(payload, config=config, stream_mode="updates"):
                for node, delta in chunk.items():
                    if node == "__interrupt__":
                        interrupts = delta
                    else:
                        on_step(node, delta if isinstance(delta, dict) else {})
            snapshot = await graph.aget_state(config)
        state = dict(snapshot.values)
        interrupts = interrupts or getattr(snapshot, "interrupts", None)
        if interrupts:
            state["__interrupt__"] = interrupts
        return state

    def _drive(self, payload, thread_id: str, on_step) -> dict:
        """Run *payload* through the graph, streaming progress when *on_step* is set."""
        runner = self._astream(payload, thread_id, on_step) if on_step else self._ainvoke(payload, thread_id)
        return _run_coro(runner)

    def start(
        self,
        user_input: str,
        date: str | None = None,
        thread_id: str | None = None,
        on_step: "Callable[[str, dict], None] | None" = None,
    ) -> PlannerResult:
        """Begin a run. May return an ``interrupted`` result awaiting approval.

        Pass *on_step(node, delta)* to receive each node as it completes (for a
        live progress display); leave it ``None`` for the plain invoke path.
        """
        thread_id = thread_id or str(uuid.uuid4())
        # Bind this turn's session so every deep call site (guardrails, tools, the
        # RAG loop) tags its events with it, without threading it through signatures.
        set_session(thread_id)
        t0 = time.perf_counter()
        # Guardrail checkpoint #1 — every turn's input passes through here before
        # any routing; a rejection short-circuits without invoking the graph.
        try:
            user_input = check_input(user_input)
        except GuardrailError as exc:
            log_event("orchestrator", "request", session_id=thread_id, ok=True,
                      latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                      details={"blocked": True, "stage": "input_guardrail"})
            return PlannerResult(status="done", thread_id=thread_id, text=exc.message)
        # History: fetch the recent window (prior turns) to inject, then record the
        # user's message. The orchestrator owns this, on every turn.
        history_window: list[dict] = []
        if self._history is not None:
            # Title a new session by its first message (INSERT-OR-IGNORE keeps it stable).
            self._history.ensure_session(thread_id, title=user_input[:48])
            history_window = self._history.get_recent_turns(thread_id)
            self._history.add_turn(thread_id, "user", user_input)
        payload = {
            "user_input": user_input,
            "session_id": thread_id,
            "history": history_window,
            "date": date,
            # `messages` is append-only — the conversation memory we keep.
            "messages": [HumanMessage(content=user_input)],
            # Everything else is per-turn scratch: reset it so a new turn
            # never re-emits the previous turn's plan/result/notice (the
            # checkpointer would otherwise carry them over on the thread).
            "intent": "",
            "tasks": None,
            "mutation": None,
            "detected_appointment": None,
            "approved": False,
            "confirm_action": "",
            "notice": "",
            "result": "",
            "error": "",
            "progress": [],
            # Complex-prompt step queue — reset so a new turn never re-runs a prior
            # turn's steps.
            "steps": [],
            "step_index": 0,
            "step_results": [],
            # Plan confirm-loop scratch (locked_plan persists, so it's NOT reset).
            "plan_feedback": "",
            "pending_plan": None,
        }
        try:
            result = self._interpret(self._drive(payload, thread_id, on_step), thread_id)
        except Exception:
            log_event("orchestrator", "request", session_id=thread_id, ok=False,
                      latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                      details={"stage": "run"})
            raise
        log_event("orchestrator", "request", session_id=thread_id, ok=True,
                  latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                  details={"status": result.status})
        self._record_assistant(thread_id, result)
        return result

    def resume(
        self,
        thread_id: str,
        decision: dict[str, Any],
        on_step: "Callable[[str, dict], None] | None" = None,
    ) -> PlannerResult:
        """Resume an interrupted run with the user's approval *decision*.

        ``decision`` is e.g. ``{"action": "accept"}``, ``{"action": "reject",
        "reason": "..."}``, or ``{"action": "edit", "args": {...}}``. Pass
        *on_step* to stream the post-approval steps (execute → re-plan, …).
        """
        set_session(thread_id)
        result = self._interpret(self._drive(Command(resume=decision), thread_id, on_step), thread_id)
        self._record_assistant(thread_id, result)
        return result

    def _interpret(self, state: dict, thread_id: str) -> PlannerResult:
        interrupts = state.get("__interrupt__")
        if interrupts:
            return PlannerResult(
                status="interrupted", thread_id=thread_id, interrupt=interrupts[0].value
            )
        # Guardrail checkpoint #2 — every final answer passes through here before
        # it reaches the user (basic safety pass at the orchestrator level; the
        # RAG-specific groundedness check runs inside RAGAgent where context exists).
        return PlannerResult(
            status="done", thread_id=thread_id, text=check_output(state.get("result", ""))
        )

    def run(self, user_input: str, date: str | None = None) -> str:
        """Convenience for the non-interactive path: returns the result text.

        If the run pauses for approval, returns a short note instead (callers that
        need the full approval flow should use :meth:`start` / :meth:`resume`).
        """
        result = self.start(user_input, date)
        if result.status == "interrupted":
            intr = result.interrupt or {}
            # No human in this non-interactive path: auto-accept a plan
            # confirmation (returns the locked plan), but never auto-approve a
            # data mutation — those still require explicit start()/resume().
            if intr.get("type") == "plan_approval":
                return self.resume(result.thread_id, {"action": "accept"}).text
            return f"[approval needed] {intr.get('message', 'Approval required.')}"
        return result.text
