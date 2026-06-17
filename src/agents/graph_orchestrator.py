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
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired, TYPE_CHECKING, TypedDict

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
    format_summary,
    is_crud_intent,
    make_mcp_fetcher,
    make_mutation_executor,
    make_mutation_parser,
    make_normalizer,
    matched_intent_families,
)
from agents.planner_agent import (
    DEFAULT_WORKDAY,
    DailyPlannerAgent,
    Workday,
    format_plan,
    plan_day,
)
from pydantic import ValidationError

from agents.todo_agent import ContractError, MutationParser, TodoAgent
from core.services.checkpoint import make_checkpointer_opener
from core.services.prompts import load_prompt
from core.tools.common_tools import make_common_tools
from models.contract import CrudOp, TaskList, TaskMutation

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
    date: NotRequired[str | None]
    intent: NotRequired[str]
    # The validated task list as a JSON-native dict (so the checkpointer stores
    # only plain types, not TaskList/Priority). Reconstructed via TaskList on use.
    tasks: NotRequired[dict | None]
    # The validated CRUD change as a JSON-native dict (so the checkpointer stores
    # only plain types, not custom classes). Reconstructed via TaskMutation when used.
    mutation: NotRequired[dict | None]
    approved: NotRequired[bool]  # set by the confirm node from the user's decision
    notice: NotRequired[str]     # success line prepended to the re-planned result
    # `add_messages` makes this an append-only log — the agent⇄tools loop's memory.
    messages: Annotated[list[AnyMessage], add_messages]
    result: NotRequired[str]
    error: NotRequired[str]


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
    observe: bool = False,
):
    """Wire and compile the LangGraph orchestrator.

    This is the single place agents are registered — adding a new specialist
    means adding a node and an edge here, exactly as with the plain orchestrator.
    Returns a :class:`GraphOrchestrator`.

    ``checkpointer_db`` enables a persistent (SQLite) checkpointer so interrupt/
    resume survives restarts; left ``None`` it uses an in-memory saver (tests).
    ``mutation_parser`` / ``mutation_executor`` are injectable so the CRUD path
    is testable without an LLM or a live MCP server.
    """
    workday = workday or DEFAULT_WORKDAY
    mutation_parser = mutation_parser or make_mutation_parser(model_name, temperature, use_llm)
    mutation_executor = mutation_executor or make_mutation_executor(tools)
    todo_agent = TodoAgent(
        fetch_raw=make_mcp_fetcher(tools),
        normalize=make_normalizer(model_name, temperature, use_llm),
        parse=mutation_parser,
    )
    planner_agent = DailyPlannerAgent(workday=workday)

    # The deterministic capabilities, exposed as tools so the agent loop can
    # sequence them with the MCP tools for multi-step prompts. The planning math
    # stays inside the tool — only the *ordering* is delegated to the LLM.
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
        return format_plan(plan_day(tasks, workday=wd))

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
        # A prompt that touches more than one intent family (e.g. "plan my day
        # AND add a task") is multi-step — send it to the agent loop, which can
        # sequence several tools, instead of a single deterministic path.
        if len(set(matched_intent_families(state["user_input"]))) > 1:
            return {"intent": "complex"}
        return {"intent": classify_intent(state["user_input"]).value}

    def run_todo(state: PlannerState) -> dict:
        try:
            # Store as a JSON-native dict so the checkpointer persists plain types
            # (not TaskList/Priority, which trip the serde's unregistered-type warning).
            return {"tasks": todo_agent.run().model_dump(mode="json")}
        except ContractError as exc:
            logger.error("Contract failed: %s", exc.errors)
            return {"error": _CONTRACT_ERROR_MSG}

    def run_planner(state: PlannerState) -> dict:
        plan = planner_agent.run(TaskList.model_validate(state["tasks"]), date=state.get("date"))
        return {"result": format_plan(plan)}

    def run_summary(state: PlannerState) -> dict:
        return {"result": format_summary(TaskList.model_validate(state["tasks"]))}

    # -- CRUD path: validate (contract checkpoint) → confirm → execute → replan --

    def extract_mutation(state: PlannerState) -> dict:
        """Parse + validate the write request into a TaskMutation (runs once).

        Stored as a JSON-native dict so the checkpointer persists only plain types.
        """
        try:
            mutation = todo_agent.parse_mutation(state["user_input"])
            return {"mutation": mutation.model_dump(mode="json", exclude_none=True)}
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
        response = llm_with_tools.invoke(
            [SystemMessage(content=system_text), *state["messages"]]
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
        elif state.get("intent") in (None, "unknown") and not msgs:
            final = _UNKNOWN_MSG
        else:
            last = msgs[-1] if msgs else None
            final = getattr(last, "content", "") or _UNKNOWN_MSG

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
        # path; everything else (incl. multi-intent "complex") → the agent loop.
        intent = state["intent"]
        if intent in ("plan", "summary"):
            return "todo"
        if intent in ("add", "update", "delete"):
            return "crud"
        return "agent"

    def route_after_todo(state: PlannerState) -> str:
        if state.get("error"):
            return "finalize"
        # summary just lists; plan and any post-CRUD re-plan go to the planner.
        return "summary" if state["intent"] == "summary" else "planner"

    def route_after_extract(state: PlannerState) -> str:
        return "finalize" if state.get("error") else "confirm_mutation"

    def route_after_confirm(state: PlannerState) -> str:
        return "execute_mutation" if state.get("approved") else "finalize"

    def route_after_execute(state: PlannerState) -> str:
        return "finalize" if state.get("error") else "todo"  # success → re-plan

    def should_continue(state: PlannerState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "finalize"

    # Observer: when on (param or PLANNER_TRACE env), each node logs the step it
    # ran and the state it produced — so you can watch the agents work.
    observe = observe or bool(os.getenv("PLANNER_TRACE"))
    node = (lambda name, fn: _trace_node(name, fn)) if observe else (lambda name, fn: fn)

    g = StateGraph(PlannerState)
    g.add_node("classify", node("classify", classify))
    g.add_node("todo", node("todo", run_todo))
    g.add_node("planner", node("planner", run_planner))
    g.add_node("summary", node("summary", run_summary))
    g.add_node("extract_mutation", node("extract_mutation", extract_mutation))
    g.add_node("confirm_mutation", node("confirm_mutation", confirm_mutation))
    g.add_node("execute_mutation", node("execute_mutation", execute_mutation))
    g.add_node("agent", node("agent", agent))
    g.add_node("tools", ToolNode(agent_tools))
    g.add_node("finalize", node("finalize", finalize))

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify", route_by_intent,
        {"todo": "todo", "crud": "extract_mutation", "agent": "agent"},
    )
    g.add_conditional_edges(
        "todo", route_after_todo,
        {"planner": "planner", "summary": "summary", "finalize": "finalize"},
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
        {"todo": "todo", "finalize": "finalize"},
    )
    g.add_edge("planner", "finalize")
    g.add_edge("summary", "finalize")
    g.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "finalize": "finalize"}
    )
    g.add_edge("tools", "agent")  # <-- the loop back
    g.add_edge("finalize", END)

    # The graph is left *uncompiled*: GraphOrchestrator compiles it per call with
    # a freshly-opened checkpointer (see its docstring) so a persistent SQLite
    # connection lives on the same event loop that runs the graph.
    return GraphOrchestrator(g, make_checkpointer_opener(checkpointer_db))


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

    def __init__(self, graph_def, open_checkpointer) -> None:
        self._graph_def = graph_def
        self._open_checkpointer = open_checkpointer

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

    def start(
        self, user_input: str, date: str | None = None, thread_id: str | None = None
    ) -> PlannerResult:
        """Begin a run. May return an ``interrupted`` result awaiting approval."""
        thread_id = thread_id or str(uuid.uuid4())
        state = _run_coro(
            self._ainvoke(
                {
                    "user_input": user_input,
                    "date": date,
                    # `messages` is append-only — the conversation memory we keep.
                    "messages": [HumanMessage(content=user_input)],
                    # Everything else is per-turn scratch: reset it so a new turn
                    # never re-emits the previous turn's plan/result/notice (the
                    # checkpointer would otherwise carry them over on the thread).
                    "intent": "",
                    "tasks": None,
                    "mutation": None,
                    "approved": False,
                    "notice": "",
                    "result": "",
                    "error": "",
                },
                thread_id,
            )
        )
        return self._interpret(state, thread_id)

    def resume(self, thread_id: str, decision: dict[str, Any]) -> PlannerResult:
        """Resume an interrupted run with the user's approval *decision*.

        ``decision`` is e.g. ``{"action": "accept"}``, ``{"action": "reject",
        "reason": "..."}``, or ``{"action": "edit", "args": {...}}``.
        """
        state = _run_coro(self._ainvoke(Command(resume=decision), thread_id))
        return self._interpret(state, thread_id)

    def _interpret(self, state: dict, thread_id: str) -> PlannerResult:
        interrupts = state.get("__interrupt__")
        if interrupts:
            return PlannerResult(
                status="interrupted", thread_id=thread_id, interrupt=interrupts[0].value
            )
        return PlannerResult(
            status="done", thread_id=thread_id, text=state.get("result", "")
        )

    def run(self, user_input: str, date: str | None = None) -> str:
        """Convenience for the non-interactive path: returns the result text.

        If the run pauses for approval, returns a short note instead (callers that
        need the full approval flow should use :meth:`start` / :meth:`resume`).
        """
        result = self.start(user_input, date)
        if result.status == "interrupted":
            msg = (result.interrupt or {}).get("message", "Approval required.")
            return f"[approval needed] {msg}"
        return result.text
