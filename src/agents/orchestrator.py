"""The Orchestrator — the hub-and-spoke coordinator.

It receives the user's request, classifies the intent, and routes to the right
agent(s). The agents are wired only to the orchestrator, never to each other —
so adding a third agent later is a change in **one place** (register it here and
add its route), not a re-threading of a pipeline.

For ``"plan"`` it runs the data specialist first (TodoAgent → validated
contract) and then the reasoning specialist (DailyPlannerAgent). The contract is
validated at the boundary inside ``TodoAgent``; a :class:`ContractError` there is
caught and surfaced recoverably instead of crashing the run.

Classification is plain Python (keyword rules) by default so it is deterministic
and testable; an optional LLM classifier is available for fuzzier phrasing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import ValidationError

from agents.planner_agent import (
    DEFAULT_AVAILABLE_MINUTES,
    PRIORITY_EMOJI,
    DailyPlannerAgent,
    format_plan,
)
from agents.todo_agent import (
    ContractError,
    Normalizer,
    RawFetcher,
    TodoAgent,
    heuristic_normalize,
    llm_normalize,
    sample_raw_tasks,
)
from core.contract import TaskList

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """What the user wants the system to do."""

    plan = "plan"
    summary = "summary"
    add = "add"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

# Ordered most-specific → least: the first bucket with a keyword hit wins.
_INTENT_KEYWORDS: list[tuple[Intent, tuple[str, ...]]] = [
    (Intent.add, ("add ", "create ", "new task", "remind me to", "schedule a")),
    (Intent.plan, ("plan", "time block", "time-block", "organize my day",
                   "schedule my day", "what should i do")),
    (Intent.summary, ("summary", "summarize", "what's on", "whats on",
                      "list", "show me", "what do i have", "overview")),
]


def classify_intent(user_input: str) -> Intent:
    """Classify intent with deterministic keyword rules (no LLM)."""
    text = user_input.lower()
    for intent, keywords in _INTENT_KEYWORDS:
        if any(kw in text for kw in keywords):
            return intent
    return Intent.unknown


def llm_classify_intent(user_input: str, model_name: str) -> Intent:
    """Classify intent with an LLM, falling back to keywords on any failure."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from core.llm_client import create_ollama_model
        from core.prompts import load_prompt

        llm = create_ollama_model(model_name, temperature=0.0, with_thinking=False)
        response = llm.invoke(
            [
                SystemMessage(content=load_prompt("intent_classifier_system")),
                HumanMessage(content=user_input),
            ]
        )
        label = (response.content or "").strip().lower()
        return Intent(label)
    except Exception as exc:  # noqa: BLE001 — fall back to the safe path
        logger.warning("LLM intent classification failed (%s); using keywords", exc)
        return classify_intent(user_input)


# ---------------------------------------------------------------------------
# Summary formatting (no LLM)
# ---------------------------------------------------------------------------


def format_summary(tasks: TaskList) -> str:
    """Render a plain-text summary of the validated task list."""
    if not tasks.tasks:
        return "You have no tasks. Enjoy the clear day! 🎉"

    total = sum(t.est_minutes for t in tasks.tasks)
    lines = [f"You have {len(tasks.tasks)} task(s) (~{total} min of work):", ""]
    for t in tasks.tasks:
        lines.append(
            f"  {PRIORITY_EMOJI[t.priority.value]} {t.title} — "
            f"{t.category}, {t.est_minutes} min"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Routes a user request to the registered agents.

    The agents are constructor dependencies — they appear here and nowhere else,
    which is the scaling property the hub-and-spoke design buys: a new agent is
    one new field plus one new branch in :meth:`run`.
    """

    def __init__(
        self,
        todo_agent: TodoAgent,
        planner_agent: DailyPlannerAgent,
        classifier=classify_intent,
    ) -> None:
        self.todo_agent = todo_agent
        self.planner_agent = planner_agent
        self.classifier = classifier

    def run(self, user_input: str, date: str | None = None) -> str:
        """Classify the request and route it, returning text for the user."""
        intent = self.classifier(user_input)
        logger.info("Orchestrator classified intent=%s", intent.value)

        match intent:
            case Intent.plan:
                tasks = self._get_tasks_or_message(intent)
                if isinstance(tasks, str):
                    return tasks
                plan = self.planner_agent.run(tasks, date=date)
                return format_plan(plan)

            case Intent.summary:
                tasks = self._get_tasks_or_message(intent)
                if isinstance(tasks, str):
                    return tasks
                return format_summary(tasks)

            case Intent.add:
                return (
                    "Adding tasks isn't wired into the orchestrator yet — "
                    "use the task-executor agent for create/update/delete."
                )

            case _:
                return (
                    "I can plan your day or summarize your tasks. "
                    "Try \"plan my day\" or \"what's on my list?\""
                )

    # -- helpers ------------------------------------------------------------

    def _get_tasks_or_message(self, intent: Intent) -> TaskList | str:
        """Run TodoAgent, converting a ContractError into a user message."""
        try:
            return self.todo_agent.run()
        except ContractError as exc:
            logger.error("Contract failed for intent=%s: %s", intent.value, exc.errors)
            return (
                "I couldn't prepare your tasks right now (the task data didn't "
                "validate). Please try again in a moment."
            )


# ---------------------------------------------------------------------------
# Runtime wiring (used by the Streamlit UI)
# ---------------------------------------------------------------------------

# MCP tool names that return the raw task records, in order of preference.
_RAW_TASK_TOOL_NAMES = ("get_today_task", "list_tasks", "get_tasks")


def _run_coro(coro):
    """Run an async coroutine from sync code, with or without a live loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running in this thread — the simple, clean path.
        return asyncio.run(coro)
    # A loop is already running (e.g. Streamlit's) — allow nested execution.
    import nest_asyncio  # type: ignore[import-untyped]

    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)


def _coerce_raw(result: object) -> list[dict]:
    """Turn an MCP tool result (str/list/dict) into a list of raw task dicts."""
    if isinstance(result, str):
        result = json.loads(result)
    if isinstance(result, dict):
        # Some servers wrap the list, e.g. {"tasks": [...]}.
        result = result.get("tasks", result.get("data", []))
    if not isinstance(result, list):
        raise ValueError(f"Unexpected task payload type: {type(result).__name__}")
    return [r for r in result if isinstance(r, dict)]


def make_mcp_fetcher(tools: list[BaseTool]) -> RawFetcher:
    """Build a raw-task fetcher backed by an MCP tool, falling back to samples.

    If no task tool is available (the MCP server is down or exposes nothing) or
    the call fails, returns :func:`sample_raw_tasks` so the planner mode still
    works as an offline demo instead of erroring out.
    """
    tool = next((t for t in tools if t.name in _RAW_TASK_TOOL_NAMES), None)

    def fetch() -> list[dict]:
        if tool is None:
            logger.warning("No task MCP tool found; using sample tasks.")
            return sample_raw_tasks()
        try:
            return _coerce_raw(_run_coro(tool.ainvoke({})))
        except Exception as exc:  # noqa: BLE001 — degrade to a working demo
            logger.warning("MCP task fetch failed (%s); using sample tasks.", exc)
            return sample_raw_tasks()

    return fetch


def make_normalizer(model_name: str, temperature: float, use_llm: bool) -> Normalizer:
    """Build a normalizer that prefers the LLM but degrades to heuristics.

    Validation errors are re-raised so TodoAgent can retry the model; only
    non-recoverable failures (e.g. Ollama unreachable) fall back to the
    deterministic :func:`heuristic_normalize`.
    """
    if not use_llm:
        return heuristic_normalize

    def normalize(raw: list[dict]) -> TaskList:
        try:
            return llm_normalize(raw, model_name, temperature)
        except (ValidationError, ValueError, json.JSONDecodeError):
            raise  # recoverable — let TodoAgent retry the model
        except Exception as exc:  # noqa: BLE001 — model down etc.
            logger.warning("LLM normalize failed (%s); using heuristics.", exc)
            return heuristic_normalize(raw)

    return normalize


def build_orchestrator(
    tools: list[BaseTool],
    model_name: str,
    temperature: float = 0.0,
    available_minutes: int = DEFAULT_AVAILABLE_MINUTES,
    use_llm: bool = True,
) -> Orchestrator:
    """Wire a ready-to-run Orchestrator from the live tools and model.

    This is the single place the UI calls. Adding a new agent would mean
    registering it here and giving it a route in :meth:`Orchestrator.run`.
    """
    todo_agent = TodoAgent(
        fetch_raw=make_mcp_fetcher(tools),
        normalize=make_normalizer(model_name, temperature, use_llm),
    )
    planner_agent = DailyPlannerAgent(available_minutes=available_minutes)
    classifier = (
        (lambda text: llm_classify_intent(text, model_name))
        if use_llm
        else classify_intent
    )
    return Orchestrator(todo_agent, planner_agent, classifier=classifier)
