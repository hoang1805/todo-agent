"""TodoAgent — the data specialist.

Its single job: turn whatever the task store returns (raw, messy dicts) into a
**schema-validated** :class:`~models.contract.TaskList` that the planner can trust.
Normalization means: fill in an estimated duration, assign a category, and a
priority — then validate.

The critical rule (objective 1b) lives in :meth:`TodoAgent.run`: **never pass
unvalidated data downstream.** If the normalizer emits something the contract
rejects, the step is retried; if it keeps failing, a *recoverable*
:class:`ContractError` is raised rather than letting garbage reach the planner.

Two normalizers are provided:

* :func:`llm_normalize` — uses Ollama structured output (JSON schema generated
  *from* the Pydantic contract) to steer the model toward valid output, then
  validates on the way in. Belt and suspenders.
* :func:`heuristic_normalize` — pure Python, no model. Used for the offline
  demo and the unit tests, so the whole pipeline runs without Ollama.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from pydantic import ValidationError

from models.contract import CrudOp, Priority, Status, Task, TaskList, TaskMutation

logger = logging.getLogger(__name__)


def _loads_jsonish(text: str):
    """Parse JSON from a model response, tolerating ```code fences``` and prose.

    Local models sometimes ignore structured-output constraints and wrap JSON in
    a markdown fence or add chatter. This strips a leading/trailing fence and, if
    needed, falls back to the outermost ``{...}``/``[...]`` block. Returns the
    decoded object (dict or list); raises ``json.JSONDecodeError`` if none found.
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        newline = cleaned.find("\n")
        cleaned = cleaned[newline + 1:] if newline != -1 else cleaned[3:]
        fence = cleaned.rfind("```")
        if fence != -1:
            cleaned = cleaned[:fence]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
        ends = [i for i in (cleaned.rfind("}"), cleaned.rfind("]")) if i != -1]
        if starts and ends and max(ends) > min(starts):
            return json.loads(cleaned[min(starts):max(ends) + 1])
        raise

#: A function that returns the raw task records from the store.
RawFetcher = Callable[[], list[dict]]
#: A function that turns raw records into a validated, normalized TaskList.
Normalizer = Callable[[list[dict]], TaskList]
#: A function that turns a request + current tasks into a validated TaskMutation.
MutationParser = Callable[[str, list[dict]], TaskMutation]


class ContractError(Exception):
    """Raised when normalization cannot produce a valid TaskList.

    Carries the failed attempts so the caller can log/inspect them. This is a
    *recoverable* signal: the orchestrator catches it and tells the user the
    data couldn't be prepared, rather than crashing the whole run.
    """

    def __init__(self, message: str, attempts: int, errors: list[str]) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.errors = errors


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------

# Default per-priority duration estimate when the store doesn't provide one.
_DEFAULT_EST = {Priority.high: 60, Priority.medium: 45, Priority.low: 30}

# Lightweight keyword → category inference for the heuristic path.
_CATEGORY_KEYWORDS = {
    "work": ("report", "meeting", "email", "review", "deploy", "ticket", "pr"),
    "health": ("gym", "run", "workout", "doctor", "walk"),
    "errand": ("buy", "shop", "pick up", "grocery", "bank"),
    "personal": ("call", "read", "study", "plan"),
}


def _infer_category(title: str) -> str:
    lowered = title.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return category
    return "general"


def heuristic_normalize(raw: list[dict]) -> TaskList:
    """Normalize raw records into a validated TaskList without an LLM.

    Fills gaps with simple rules (priority-based duration, keyword category),
    then constructs :class:`Task` objects — so Pydantic validation still runs and
    a bad record (e.g. missing title) is rejected exactly as in the LLM path.
    """
    tasks: list[Task] = []
    for index, record in enumerate(raw):
        priority = Priority(str(record.get("priority", "medium")).lower())
        est = int(record.get("est_minutes") or _DEFAULT_EST[priority])
        title = str(record.get("title") or record.get("name") or "").strip()
        category = record.get("category") or _infer_category(title)
        tasks.append(
            Task(
                id=record.get("id") or index,  # opaque id; fall back to position
                title=title,
                priority=priority,
                est_minutes=est,
                category=category,
                due=record.get("due") or record.get("due_date"),
            )
        )
    return TaskList(tasks=tasks)


def llm_normalize(
    raw: list[dict],
    model_name: str,
    temperature: float = 0.0,
) -> TaskList:
    """Normalize raw records via Ollama structured output, then validate.

    The JSON schema is generated *from* the contract, so the model is steered
    toward valid output and we still validate on the way in.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from core.services.llm_client import create_ollama_model
    from core.services.prompts import load_prompt

    schema = TaskList.model_json_schema()
    llm = create_ollama_model(
        model_name, temperature, format=schema, with_thinking=False
    )
    system = load_prompt("todo_agent_system")
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=f"Raw tasks:\n{json.dumps(raw, default=str)}"),
        ]
    )
    # Validate on the way IN. If this raises, run() retries the step.
    data = _loads_jsonish(response.content)
    if isinstance(data, list):  # model returned a bare task array
        data = {"tasks": data}
    return TaskList.model_validate(data)


# ---------------------------------------------------------------------------
# Mutation parsers (CRUD) — turn a request into a validated TaskMutation
# ---------------------------------------------------------------------------

# Leading phrases stripped off a "create" request to recover the bare title.
_CREATE_PREFIXES = (
    "add a task to ", "add a task ", "add task to ", "add task ", "add ",
    "create a task to ", "create a task ", "create task ", "create ",
    "new task to ", "new task ", "remind me to ", "schedule a ", "schedule ",
)
_DELETE_KEYWORDS = ("delete", "remove", "cancel", "drop")
_DONE_KEYWORDS = ("mark", "complete", "finish", " done")


def _resolve_task_id(text: str, current_tasks: list[dict]) -> str | None:
    """Best-effort: find a current task whose title appears in *text*."""
    low = text.lower()
    for record in current_tasks:
        title = str(record.get("title") or record.get("name") or "").strip()
        if title and title.lower() in low and record.get("id") is not None:
            return str(record["id"])
    return None


def heuristic_parse_mutation(
    user_input: str, current_tasks: list[dict], focus_task_id: str | None = None
) -> TaskMutation:
    """Parse a CRUD request without an LLM (offline demo / tests / fallback).

    Deterministic keyword rules: delete/remove → delete, mark/complete/done →
    update status, otherwise create. Constructs a :class:`TaskMutation`, so the
    contract validation still runs — an unresolved ``task_id`` for update/delete
    is rejected exactly as in the LLM path.

    ``focus_task_id`` is the task the user was last looking at; it is used only as
    a fallback when the text itself names no task (e.g. a follow-up like "mark it
    done"), so an explicit reference always wins.
    """
    low = user_input.lower()

    if any(kw in low for kw in _DELETE_KEYWORDS):
        return TaskMutation(
            op=CrudOp.delete,
            task_id=_resolve_task_id(user_input, current_tasks) or focus_task_id,
        )

    if any(kw in low for kw in _DONE_KEYWORDS):
        return TaskMutation(
            op=CrudOp.update,
            task_id=_resolve_task_id(user_input, current_tasks) or focus_task_id,
            status=Status.done,
        )

    title = user_input.strip()
    for prefix in _CREATE_PREFIXES:
        if low.startswith(prefix):
            title = user_input[len(prefix):].strip()
            break
    priority = (
        Priority.high if "high prio" in low
        else Priority.low if "low prio" in low
        else None
    )
    return TaskMutation(op=CrudOp.create, title=title, priority=priority)


def llm_parse_mutation(
    user_input: str,
    current_tasks: list[dict],
    model_name: str,
    temperature: float = 0.0,
    focus_task_id: str | None = None,
) -> TaskMutation:
    """Parse a CRUD request via Ollama structured output, then validate.

    The JSON schema is generated *from* :class:`TaskMutation`, and the current
    task list is supplied so the model can resolve a reference like "the gym
    task" to its ``task_id``. Validation runs on the way in; on failure
    :meth:`TodoAgent.parse_mutation` retries the step.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from core.services.llm_client import create_ollama_model
    from core.services.prompts import load_prompt

    schema = TaskMutation.model_json_schema()
    llm = create_ollama_model(
        model_name, temperature, format=schema, with_thinking=False
    )
    system = load_prompt("crud_agent_system")
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Current tasks:\n{json.dumps(current_tasks, default=str)}\n\n"
                    f"Request:\n{user_input}"
                )
            ),
        ]
    )
    # Validate on the way IN. If this raises, parse_mutation() retries the step.
    data = _loads_jsonish(response.content)
    if isinstance(data, list):  # model wrapped a single change in an array
        if not data:
            raise ValueError("the model returned no task change")
        data = data[0]
    # Follow-up like "change its category" names no task — fall back to the one
    # the user was last looking at (never overrides a task_id the model resolved).
    if (
        isinstance(data, dict)
        and focus_task_id
        and data.get("op") in ("update", "delete")
        and not data.get("task_id")
    ):
        data["task_id"] = focus_task_id
    return TaskMutation.model_validate(data)


# ---------------------------------------------------------------------------
# Sample data (offline demo / tests)
# ---------------------------------------------------------------------------


def sample_raw_tasks() -> list[dict]:
    """A representative raw task list for the offline demo and tests."""
    return [
        {"id": 1, "title": "Finish Q3 report", "priority": "high", "est_minutes": 120},
        {"id": 2, "title": "Review teammate's PR", "priority": "high", "est_minutes": 60},
        {"id": 3, "title": "Team sync meeting", "priority": "medium", "est_minutes": 60},
        {"id": 4, "title": "Reply to client emails", "priority": "medium", "est_minutes": 45},
        {"id": 5, "title": "Go for a run", "priority": "low", "est_minutes": 45},
        {"id": 6, "title": "Read research paper", "priority": "low", "est_minutes": 60},
    ]


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class TodoAgent:
    """Fetch raw tasks and emit a validated TaskList, retrying on bad output.

    Dependencies are injected so the agent is testable without MCP or an LLM:
    pass a fake *fetch_raw* and the deterministic :func:`heuristic_normalize`.
    """

    def __init__(
        self,
        fetch_raw: RawFetcher,
        normalize: Normalizer = heuristic_normalize,
        parse: MutationParser = heuristic_parse_mutation,
        max_retries: int = 2,
    ) -> None:
        self._fetch_raw = fetch_raw
        self._normalize = normalize
        self._parse = parse
        self.max_retries = max_retries

    def run(self) -> TaskList:
        """Produce a validated TaskList or raise a recoverable ContractError."""
        raw = self._fetch_raw()
        logger.info("TodoAgent fetched %d raw task(s)", len(raw))
        if not raw:
            return TaskList(tasks=[])

        errors: list[str] = []
        for attempt in range(1, self.max_retries + 1):
            try:
                task_list = self._normalize(raw)
                logger.info(
                    "TodoAgent normalized %d task(s) on attempt %d",
                    len(task_list.tasks),
                    attempt,
                )
                return task_list
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Normalization attempt %d failed: %s", attempt, exc)
                errors.append(str(exc))

        raise ContractError(
            "Could not normalize tasks into a valid TaskList.",
            attempts=self.max_retries,
            errors=errors,
        )

    def parse_mutation(
        self, user_input: str, focus_task_id: str | None = None
    ) -> TaskMutation:
        """Parse a CRUD request into a validated TaskMutation, retrying on bad output.

        Fetches the current tasks first (so a reference like "the gym task" can
        be resolved to an id), then runs the injected parser under the same
        validate-and-retry rule as :meth:`run`: a malformed or under-specified
        change never escapes downstream — it raises a recoverable
        :class:`ContractError` instead, which the caller surfaces to the user.

        ``focus_task_id`` (the last task the user viewed) lets a follow-up that
        omits the task — "change its category" — still resolve. It is forwarded
        only when set, so simpler two-argument parsers stay compatible.
        """
        current = self._fetch_raw()

        errors: list[str] = []
        for attempt in range(1, self.max_retries + 1):
            try:
                mutation = (
                    self._parse(user_input, current, focus_task_id)
                    if focus_task_id is not None
                    else self._parse(user_input, current)
                )
                logger.info(
                    "TodoAgent parsed a '%s' mutation on attempt %d",
                    mutation.op.value,
                    attempt,
                )
                return mutation
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Mutation parse attempt %d failed: %s", attempt, exc)
                errors.append(str(exc))

        raise ContractError(
            "Could not parse the request into a valid task change.",
            attempts=self.max_retries,
            errors=errors,
        )
