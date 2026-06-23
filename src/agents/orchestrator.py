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
from typing import TYPE_CHECKING, Callable

from pydantic import ValidationError

from agents.planner_agent import (
    DEFAULT_WORKDAY,
    PRIORITY_EMOJI,
    Break,
    DailyPlannerAgent,
    Planner,
    Workday,
    format_plan,
    llm_extract_appointment,
    llm_plan_day,
    parse_appointment,
    plan_day,
)
from agents.todo_agent import (
    ContractError,
    MutationParser,
    Normalizer,
    RawFetcher,
    TodoAgent,
    heuristic_normalize,
    heuristic_parse_mutation,
    llm_normalize,
    llm_parse_mutation,
    sample_raw_tasks,
)
from models.contract import CrudOp, DayPlan, TaskList, TaskMutation

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """What the user wants the system to do."""

    plan = "plan"
    summary = "summary"
    detail = "detail"
    at_time = "at_time"  # "which task do I have at 7pm?" — look up the plan at a time
    appointment = "appointment"
    add = "add"
    update = "update"
    delete = "delete"
    weather = "weather"
    recall = "recall"
    unknown = "unknown"


#: The write intents — all routed to the CRUD (validate → confirm → execute) path.
_CRUD_INTENTS = (Intent.add, Intent.update, Intent.delete)


def is_crud_intent(intent: Intent) -> bool:
    """True for create/update/delete intents (the mutating, approval-gated path)."""
    return intent in _CRUD_INTENTS


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

# Ordered most-specific → least: the first bucket with a keyword hit wins.
# CRUD buckets come first; their keywords are multi-word/space-bounded phrases so
# they don't accidentally fire on plan/summary prompts (which would mis-route a
# request to the agent loop via `matched_intent_families`).
_INTENT_KEYWORDS: list[tuple[Intent, tuple[str, ...]]] = [
    (Intent.delete, ("delete ", "remove ", "cancel the", "cancel task")),
    (Intent.update, ("mark ", "complete ", "finish ", "rename ", "reschedule ",
                      "update task", "set priority", "set status",
                      "change priority", "change status", "as done", "is done",
                      "estimate", "duration", "categor", "move to",
                      "change the time", "set the time")),
    (Intent.add, ("add ", "create ", "new task", "remind me to", "schedule a")),
    (Intent.plan, ("plan", "time block", "time-block", "organize my day",
                   "schedule my day", "what should i do")),
    (Intent.summary, ("summary", "summarize", "what's on", "whats on",
                      "list", "show me", "what do i have", "overview")),
    (Intent.weather, ("weather", "forecast", "temperature", "how hot",
                      "how cold", "is it raining", "will it rain")),
    (Intent.recall, ("recall", "what do i usually", "according to my notes",
                     "remember when", "from my notes", "from my logs",
                     "my notes say", "what does my", "in my documents",
                     "based on my history", "what did i do")),
]


# A request for *one* task's details (e.g. "show me the detail of task X").
# Checked before the keyword table because such prompts usually also contain a
# summary trigger like "show me" / "list" — detail is the more specific intent.
_DETAIL_KEYWORDS = ("detail", "tell me about", "more about", "info about", "info on")


def wants_detail(user_input: str) -> bool:
    """True if the prompt asks for a single task's details (not the whole list)."""
    text = user_input.lower()
    return any(kw in text for kw in _DETAIL_KEYWORDS)


def classify_intent(user_input: str) -> Intent:
    """Classify intent with deterministic keyword rules (no LLM)."""
    if wants_detail(user_input):
        return Intent.detail
    text = user_input.lower()
    for intent, keywords in _INTENT_KEYWORDS:
        if any(kw in text for kw in keywords):
            return intent
    return Intent.unknown


def matched_intent_families(user_input: str) -> list[Intent]:
    """Return every intent family whose keywords appear in the input.

    Used to spot multi-step prompts: when more than one family matches (e.g.
    "plan my day and add a task"), a single intent isn't enough — the request
    should go to the agent loop, which can sequence several tools.
    """
    text = user_input.lower()
    return [
        intent
        for intent, keywords in _INTENT_KEYWORDS
        if any(kw in text for kw in keywords)
    ]


def llm_classify_intent(user_input: str, model_name: str) -> Intent:
    """Classify intent with an LLM, falling back to keywords on any failure."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from core.services.llm_client import create_ollama_model
        from core.services.prompts import load_prompt

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


# Words that carry no task-identifying signal — ignored when matching a prompt
# to a task title so "show me the detail of task X" matches on "X", not "task".
_DETAIL_STOPWORDS = frozenset({
    "show", "me", "the", "detail", "details", "of", "task", "tell", "about",
    "more", "info", "on", "a", "an", "for", "please", "what", "is", "are",
})


def _match_task(tasks: TaskList, user_input: str):
    """Find the task the prompt refers to (or ``None`` if nothing matches).

    Prefers a title that appears verbatim in the prompt (longest wins); otherwise
    falls back to the task sharing the most significant words with the prompt.
    """
    import re

    low = user_input.lower()
    verbatim = [t for t in tasks.tasks if t.title.lower() in low]
    if verbatim:
        return max(verbatim, key=lambda t: len(t.title))

    prompt_words = set(re.findall(r"\w+", low)) - _DETAIL_STOPWORDS
    best, best_score = None, 0
    for t in tasks.tasks:
        title_words = set(re.findall(r"\w+", t.title.lower())) - _DETAIL_STOPWORDS
        score = len(title_words & prompt_words)
        if score > best_score:
            best, best_score = t, score
    return best if best_score > 0 else None


def format_task_detail(task) -> str:
    """Render a single task's full details (used by the 'detail' intent)."""
    return "\n".join([
        f"**{task.title}**",
        "",
        f"- Priority: {PRIORITY_EMOJI[task.priority.value]} {task.priority.value}",
        f"- Estimate: {task.est_minutes} min",
        f"- Category: {task.category}",
        f"- Due: {task.due or '—'}",
        f"- ID: {task.id}",
    ])


def format_detail(tasks: TaskList, user_input: str) -> str:
    """Resolve the prompt to one task and render its details, or guide the user."""
    task = _match_task(tasks, user_input)
    if task is None:
        names = ", ".join(f"'{t.title}'" for t in tasks.tasks) or "(none)"
        return (
            "I couldn't find a task matching that. Your tasks are: " + names + "."
        )
    return format_task_detail(task)


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

            case Intent.detail:
                tasks = self._get_tasks_or_message(intent)
                if isinstance(tasks, str):
                    return tasks
                return format_detail(tasks, user_input)

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


# -- CRUD wiring: parse the request, then apply it via the MCP write tools -----

#: The task-mcp tool names this app drives for each kind of write.
MUTATION_TOOL_NAMES = {
    "create": "create_task",
    "fields": "update_task_by_id",      # name / description / due_date
    "status": "update_task_status",
    "priority": "update_task_priority",
    "delete": "delete_task_by_id",
}

#: A function that applies a validated mutation and returns a result message.
#: Messages beginning with ``"ERROR"`` signal a recoverable failure to the graph.
MutationExecutor = Callable[[TaskMutation], str]


def make_mutation_parser(
    model_name: str, temperature: float, use_llm: bool
) -> MutationParser:
    """Build a mutation parser that prefers the LLM but degrades to heuristics.

    Validation errors are re-raised so TodoAgent can retry the model; only
    non-recoverable failures (e.g. Ollama unreachable) fall back to the
    deterministic :func:`heuristic_parse_mutation`.
    """
    if not use_llm:
        return heuristic_parse_mutation

    def parse(
        user_input: str,
        current_tasks: list[dict],
        focus_task_id: str | None = None,
    ) -> TaskMutation:
        try:
            return llm_parse_mutation(
                user_input, current_tasks, model_name, temperature, focus_task_id
            )
        except (ValidationError, ValueError, json.JSONDecodeError):
            raise  # recoverable — let TodoAgent retry the model
        except Exception as exc:  # noqa: BLE001 — model down etc.
            logger.warning("LLM mutation parse failed (%s); using heuristics.", exc)
            return heuristic_parse_mutation(user_input, current_tasks, focus_task_id)

    return parse


def make_planner(
    model_name: str, temperature: float, use_llm: bool, max_retries: int = 2
) -> Planner:
    """Build a day planner that prefers the LLM but always returns a valid plan.

    With ``use_llm`` off it is the deterministic :func:`plan_day`. With it on, the
    LLM is asked for a schedule and validated (types + the semantic checks in
    ``_validate_plan``); on repeated failure or an unreachable model it falls back
    to :func:`plan_day`, so a usable plan is guaranteed either way.
    """
    if not use_llm:
        return lambda tasks, workday, date=None: plan_day(tasks, workday=workday, date=date)

    def plan(tasks: TaskList, workday: Workday, date: str | None = None) -> DayPlan:
        for attempt in range(1, max_retries + 1):
            try:
                return llm_plan_day(tasks, workday, model_name, temperature, date)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("LLM plan attempt %d failed (%s)", attempt, exc)
            except Exception as exc:  # noqa: BLE001 — model down etc.
                logger.warning("LLM planning failed (%s); using deterministic planner.", exc)
                break
        logger.info("Falling back to the deterministic planner.")
        return plan_day(tasks, workday=workday, date=date)

    return plan


def make_appointment_parser(
    model_name: str, temperature: float, use_llm: bool
) -> "Callable[[str], Break | None]":
    """Build an appointment detector: LLM extraction with the regex as fallback.

    Mirrors :func:`make_planner` / :func:`make_normalizer` — prefer the model
    (robust to natural phrasing like "11h to 13h30" or "lunch at noon for 2h"),
    but degrade to the deterministic :func:`parse_appointment` when Ollama is
    unavailable, so the planner still works offline. With ``use_llm`` off it *is*
    the regex. The LLM path self-gates on a cheap time cue, so it spends no model
    call on prompts that can't contain a fixed-time commitment.
    """
    if not use_llm:
        return parse_appointment

    def parse(text: str) -> "Break | None":
        text = (text or "").strip()
        if not text:
            return None
        try:
            return llm_extract_appointment(text, model_name, temperature)
        except Exception as exc:  # noqa: BLE001 — recoverable: fall back to the regex
            logger.warning("LLM appointment extraction failed (%s); using regex.", exc)
            return parse_appointment(text)

    return parse


def make_mutation_executor(tools: list[BaseTool]) -> MutationExecutor:
    """Build an executor that applies a :class:`TaskMutation` via the MCP tools.

    An ``update`` may touch more than one MCP tool (status, priority, and the
    name/description/due tool are separate endpoints), so a single confirmed
    change can issue several calls. Missing tools or a failed call degrade to a
    recoverable ``"ERROR…"`` message rather than raising into the graph.
    """
    by_name = {t.name: t for t in tools}

    def _call(kind: str, args: dict) -> None:
        tool = by_name.get(MUTATION_TOOL_NAMES[kind])
        if tool is None:
            raise LookupError(
                f"the task server does not expose '{MUTATION_TOOL_NAMES[kind]}'"
            )
        result = _run_coro(tool.ainvoke(args))
        # The MCP server returns a structured recoverable error on failure; treat
        # ``{"ok": false, "error": ...}`` (dict or JSON string) as a failed call.
        payload = result
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (ValueError, TypeError):
                payload = result
        if isinstance(payload, dict) and payload.get("ok") is False:
            raise RuntimeError(payload.get("error", "the task server rejected the change"))

    def execute(mutation: TaskMutation) -> str:
        try:
            if mutation.op is CrudOp.create:
                args: dict = {"name": mutation.title}
                if mutation.description is not None:
                    args["description"] = mutation.description
                if mutation.status is not None:
                    args["status"] = mutation.status.value
                if mutation.priority is not None:
                    args["priority"] = mutation.priority.value
                if mutation.est_minutes is not None:
                    args["est_minutes"] = mutation.est_minutes
                if mutation.category is not None:
                    args["category"] = mutation.category
                if mutation.due is not None:
                    args["due_date"] = mutation.due
                _call("create", args)
                return f"Created task '{mutation.title}'."

            if mutation.op is CrudOp.delete:
                _call("delete", {"task_id": mutation.task_id})
                return f"Deleted task {mutation.task_id}."

            # update — apply each changed field via its dedicated endpoint.
            applied: list[str] = []
            if mutation.priority is not None:
                _call("priority", {"task_id": mutation.task_id, "priority": mutation.priority.value})
                applied.append("priority")
            if mutation.status is not None:
                _call("status", {"task_id": mutation.task_id, "status": mutation.status.value})
                applied.append("status")
            field_args: dict = {"task_id": mutation.task_id}
            if mutation.title is not None:
                field_args["name"] = mutation.title
            if mutation.description is not None:
                field_args["description"] = mutation.description
            if mutation.est_minutes is not None:
                field_args["est_minutes"] = mutation.est_minutes
            if mutation.category is not None:
                field_args["category"] = mutation.category
            if mutation.due is not None:
                field_args["due_date"] = mutation.due
            if len(field_args) > 1:  # more than just task_id
                _call("fields", field_args)
                applied.append("details")
            return f"Updated task {mutation.task_id} ({', '.join(applied)})."

        except LookupError as exc:
            logger.warning("Mutation skipped: %s", exc)
            return f"ERROR: {exc}, so the change was not applied."
        except Exception as exc:  # noqa: BLE001 — surface a recoverable message
            logger.warning("Mutation execution failed (%s).", exc)
            return f"ERROR: the change could not be applied ({exc})."

    return execute


def build_orchestrator(
    tools: list[BaseTool],
    model_name: str,
    temperature: float = 0.0,
    workday: Workday | None = None,
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
    planner_agent = DailyPlannerAgent(workday=workday or DEFAULT_WORKDAY)
    classifier = (
        (lambda text: llm_classify_intent(text, model_name))
        if use_llm
        else classify_intent
    )
    return Orchestrator(todo_agent, planner_agent, classifier=classifier)
