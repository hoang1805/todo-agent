"""The inter-agent handoff contract (Pydantic).

This module is the *linchpin* of the multi-agent daily planner: the
``TodoAgent`` (data specialist) hands the ``DailyPlannerAgent`` (reasoning
specialist) a **schema-validated** ``TaskList`` — never free text and never
unvalidated JSON. Because the models are typed and bounded, a malformed task
**cannot** silently flow downstream; validation fails loudly at the boundary
so the producing step can be retried.

The contract has two halves:

* **Input to planning** — :class:`Task` / :class:`TaskList`. This is what
  crosses the agent boundary.
* **Output of planning** — :class:`TimeBlock` / :class:`DayPlan`. The planner's
  decision, expressed as a typed result instead of prose.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Input contract: the data TodoAgent hands to the planner
# ---------------------------------------------------------------------------


class Priority(str, Enum):
    """Task priority, highest urgency first."""

    high = "high"
    medium = "medium"
    low = "low"

    @property
    def rank(self) -> int:
        """Sort key — lower means more urgent (``high`` -> 0)."""
        return {Priority.high: 0, Priority.medium: 1, Priority.low: 2}[self]


class Task(BaseModel):
    """A single, fully-normalized task ready for planning.

    ``est_minutes`` and ``category`` are produced by TodoAgent's normalization
    step — the raw task store does not necessarily provide them. Bounding
    ``est_minutes`` gives free validation: an LLM that hallucinates a 9000-minute
    task is rejected at the contract boundary. ``extra="forbid"`` means an LLM
    that invents extra fields is rejected too, so the step gets retried instead
    of letting unexpected data through.
    """

    model_config = ConfigDict(extra="forbid")

    id: int
    title: str = Field(min_length=1)
    priority: Priority
    est_minutes: int = Field(gt=0, le=480)  # sane bounds = free validation
    category: str = Field(min_length=1)
    due: str | None = None  # ISO date string (YYYY-MM-DD), optional

    @field_validator("title", "category", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        # Strip before length checks so a whitespace-only value fails min_length.
        return value.strip() if isinstance(value, str) else value


class TaskList(BaseModel):
    """The validated collection of tasks that crosses the agent boundary."""

    tasks: list[Task]


# ---------------------------------------------------------------------------
# Output contract: the planner's decision, as typed data
# ---------------------------------------------------------------------------


class TimeBlock(BaseModel):
    """One scheduled task occupying a contiguous slot in the day."""

    task_id: int
    title: str
    priority: Priority
    start: str  # HH:MM, 24-hour
    end: str  # HH:MM, 24-hour
    est_minutes: int


class DayPlan(BaseModel):
    """The result of the planning decision.

    ``deferred`` is the heart of the "does it all fit?" branch: when the day is
    overloaded, the lowest-priority tasks land here instead of in ``blocks``,
    and ``overloaded`` is ``True``. A caller can render exactly what was dropped
    and why.
    """

    date: str | None = None  # human/ISO date the plan is for, if known
    available_minutes: int
    blocks: list[TimeBlock] = Field(default_factory=list)
    deferred: list[Task] = Field(default_factory=list)

    @property
    def scheduled_minutes(self) -> int:
        return sum(b.est_minutes for b in self.blocks)

    @property
    def overloaded(self) -> bool:
        """True when at least one task had to be deferred to fit the day."""
        return len(self.deferred) > 0
