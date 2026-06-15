"""TodoAgent — the data specialist.

Its single job: turn whatever the task store returns (raw, messy dicts) into a
**schema-validated** :class:`~core.contract.TaskList` that the planner can trust.
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

from core.contract import Priority, Task, TaskList

logger = logging.getLogger(__name__)

#: A function that returns the raw task records from the store.
RawFetcher = Callable[[], list[dict]]
#: A function that turns raw records into a validated, normalized TaskList.
Normalizer = Callable[[list[dict]], TaskList]


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
                id=int(record.get("id", index)),
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

    from core.llm_client import create_ollama_model
    from core.prompts import load_prompt

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
    return TaskList.model_validate_json(response.content)


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
        max_retries: int = 2,
    ) -> None:
        self._fetch_raw = fetch_raw
        self._normalize = normalize
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
