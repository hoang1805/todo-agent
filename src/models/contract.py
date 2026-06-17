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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    id: str = Field(min_length=1)  # opaque identifier: UUID, code (e.g. "W001"), or number
    title: str = Field(min_length=1)
    priority: Priority
    est_minutes: int = Field(gt=0, le=480)  # sane bounds = free validation
    category: str = Field(min_length=1)
    due: str | None = None  # ISO date string (YYYY-MM-DD), optional

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, value: object) -> object:
        # IDs are opaque strings, but the store/LLM may emit a number — accept both.
        return str(value).strip() if value is not None else value

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

    task_id: str  # matches Task.id (opaque identifier)
    title: str
    priority: Priority
    start: str  # HH:MM, 24-hour
    end: str  # HH:MM, 24-hour
    est_minutes: int

    @field_validator("task_id", mode="before")
    @classmethod
    def _coerce_task_id(cls, value: object) -> object:
        return str(value) if value is not None else value


class MealBreak(BaseModel):
    """A fixed, non-task block in the day (lunch, dinner, …)."""

    name: str
    start: str  # HH:MM, 24-hour
    end: str  # HH:MM, 24-hour


class DayPlan(BaseModel):
    """The result of the planning decision.

    ``deferred`` is the heart of the "does it all fit?" branch: when the day is
    overloaded, the lowest-priority tasks land here instead of in ``blocks``,
    and ``overloaded`` is ``True``. A caller can render exactly what was dropped
    and why. ``breaks`` are the fixed meal blocks tasks are scheduled around.
    """

    date: str | None = None  # human/ISO date the plan is for, if known
    available_minutes: int  # working minutes = day window minus breaks
    blocks: list[TimeBlock] = Field(default_factory=list)
    deferred: list[Task] = Field(default_factory=list)
    breaks: list[MealBreak] = Field(default_factory=list)

    @property
    def scheduled_minutes(self) -> int:
        return sum(b.est_minutes for b in self.blocks)

    @property
    def overloaded(self) -> bool:
        """True when at least one task had to be deferred to fit the day."""
        return len(self.deferred) > 0


# ---------------------------------------------------------------------------
# Mutation contract: the validated CRUD change before it touches the store
# ---------------------------------------------------------------------------
# This is the *checkpoint* for write operations, the mirror of `Task` for reads:
# a create/update/delete request is parsed into a `TaskMutation` and validated
# before it is confirmed and sent to the MCP server. Because it is typed,
# bounded, and ``extra="forbid"``, a malformed or under-specified change (e.g. a
# delete with no task_id) is rejected at the boundary instead of hitting the API.


class Status(str, Enum):
    """Lifecycle state of a task (mirrors the task-mcp server's allowed values)."""

    pending = "pending"
    in_progress = "in_progress"
    done = "done"


class CrudOp(str, Enum):
    """The kind of write the user is requesting."""

    create = "create"
    update = "update"
    delete = "delete"


#: Fields a mutation may set on a task (everything except ``op``/``task_id``).
_MUTATION_FIELDS = ("title", "description", "priority", "status", "est_minutes", "category", "due")


class TaskMutation(BaseModel):
    """A single, validated create/update/delete request.

    The validator enforces the minimum each op needs to be actionable:

    * ``create`` — must carry a ``title`` (there is nothing to create otherwise).
    * ``update`` — must identify the task (``task_id``) **and** change at least
      one field (an update that changes nothing is a no-op, almost certainly a
      parsing miss worth bouncing back).
    * ``delete`` — must identify the task (``task_id``).
    """

    model_config = ConfigDict(extra="forbid")

    op: CrudOp
    task_id: str | None = None
    title: str | None = None
    description: str | None = None
    priority: Priority | None = None
    status: Status | None = None
    est_minutes: int | None = Field(default=None, gt=0, le=480)
    category: str | None = None
    due: str | None = None  # ISO date string (YYYY-MM-DD), optional

    @field_validator("title", "description", "task_id", "category", "due", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        # Normalize blank strings to None so they don't masquerade as set fields.
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @property
    def changed_fields(self) -> dict[str, object]:
        """The task fields this mutation actually sets (non-None), as a dict."""
        return {
            name: getattr(self, name)
            for name in _MUTATION_FIELDS
            if getattr(self, name) is not None
        }

    @model_validator(mode="after")
    def _check_op_requirements(self) -> "TaskMutation":
        if self.op is CrudOp.create:
            if not self.title:
                raise ValueError("create requires a 'title'")
        else:  # update / delete both need to identify the task
            if not self.task_id:
                raise ValueError(f"{self.op.value} requires a 'task_id'")
            if self.op is CrudOp.update and not self.changed_fields:
                raise ValueError("update requires at least one field to change")
        return self
