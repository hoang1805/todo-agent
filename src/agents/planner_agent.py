"""DailyPlannerAgent — the reasoning specialist.

Given a validated :class:`~models.contract.TaskList`, it ranks the tasks, fits them
into the user's working hours **around fixed meal breaks** (lunch, dinner, …),
and — the part that makes it an *agent* rather than a sorter — **decides what to
do when the day is overloaded**: it defers the lowest-priority tasks and reports
exactly what it dropped.

The decision (:func:`plan_day`) is deterministic, pure Python. That is on
purpose: it is the graded behaviour, so it must be reliable and unit-testable
without an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from models.contract import DayPlan, MealBreak, Task, TaskList, TimeBlock

#: Priority → emoji, shared by the plan and summary renderers.
PRIORITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}


# ---------------------------------------------------------------------------
# Workday configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Break:
    """A fixed block in the day that tasks must schedule around."""

    name: str
    start: str  # HH:MM, 24-hour
    end: str  # HH:MM, 24-hour


@dataclass(frozen=True)
class Workday:
    """The user's available hours and the meal breaks within them."""

    start: str = "09:00"
    end: str = "20:00"
    breaks: tuple[Break, ...] = field(
        default_factory=lambda: (
            Break("Lunch", "12:00", "13:00"),
            Break("Dinner", "18:00", "19:00"),
        )
    )


DEFAULT_WORKDAY = Workday()


def _t(hhmm: str) -> datetime:
    return datetime.strptime(hhmm, "%H:%M")


def _fmt(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def working_minutes(workday: Workday) -> int:
    """Minutes available for tasks = the day window minus break time inside it."""
    start, end = _t(workday.start), _t(workday.end)
    total = int((end - start).total_seconds() // 60)
    for brk in workday.breaks:
        bs, be = max(_t(brk.start), start), min(_t(brk.end), end)
        if be > bs:
            total -= int((be - bs).total_seconds() // 60)
    return max(total, 0)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _due_sort_key(due: str | None) -> str:
    """Sort key for the optional ISO due date — undated tasks sort last."""
    return due if due else "9999-12-31"


def rank_tasks(tasks: list[Task]) -> list[Task]:
    """Order tasks for scheduling: priority first, then due date, then size."""
    return sorted(
        tasks,
        key=lambda t: (t.priority.rank, _due_sort_key(t.due), t.est_minutes),
    )


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def plan_day(
    tasks: TaskList,
    workday: Workday = DEFAULT_WORKDAY,
    date: str | None = None,
) -> DayPlan:
    """Fit ranked tasks into the workday around its breaks; defer the overflow.

    Walks a clock cursor from the day's start. For each ranked task it skips the
    cursor past any break the task would straddle, then places the task if it
    still ends by the day's end — otherwise the task is **deferred** (and, since
    tasks are priority-ranked, the deferred set is always the least important
    work). The cursor only advances on a successful placement, so a single big
    task that doesn't fit doesn't block smaller tasks that still could.

    Returns a :class:`DayPlan`; ``plan.overloaded`` is ``True`` iff anything was
    deferred. Meal breaks within the window are returned in ``plan.breaks``.
    """
    start, end = _t(workday.start), _t(workday.end)
    ordered_breaks = sorted(workday.breaks, key=lambda b: _t(b.start))

    meal_blocks = [
        MealBreak(name=b.name, start=b.start, end=b.end)
        for b in ordered_breaks
        if _t(b.end) > start and _t(b.start) < end
    ]

    cursor = start
    blocks: list[TimeBlock] = []
    deferred: list[Task] = []

    for task in rank_tasks(tasks.tasks):
        place = cursor
        # Skip the placement cursor past any break this task would overlap.
        while True:
            duration = timedelta(minutes=task.est_minutes)
            straddled = next(
                (
                    b
                    for b in ordered_breaks
                    if _t(b.start) < place + duration and _t(b.end) > place
                ),
                None,
            )
            if straddled is None:
                break
            place = max(place, _t(straddled.end))

        block_end = place + timedelta(minutes=task.est_minutes)
        if block_end <= end:
            blocks.append(
                TimeBlock(
                    task_id=task.id,
                    title=task.title,
                    priority=task.priority,
                    start=_fmt(place),
                    end=_fmt(block_end),
                    est_minutes=task.est_minutes,
                )
            )
            cursor = block_end  # advance only when actually scheduled
        else:
            deferred.append(task)

    return DayPlan(
        date=date,
        available_minutes=working_minutes(workday),
        blocks=blocks,
        deferred=deferred,
        breaks=meal_blocks,
    )


# ---------------------------------------------------------------------------
# Formatting (no LLM) — deterministic rendering of the decision
# ---------------------------------------------------------------------------


def format_plan(plan: DayPlan) -> str:
    """Render a :class:`DayPlan` as readable text without an LLM."""
    header = f"🗓️  Your plan{f' for {plan.date}' if plan.date else ''}"
    lines = [header, ""]

    # Merge task blocks and meal breaks onto one timeline, ordered by start.
    timeline: list[tuple[str, object]] = [("task", b) for b in plan.blocks]
    timeline += [("break", m) for m in plan.breaks]
    timeline.sort(key=lambda item: item[1].start)

    if timeline:
        for kind, item in timeline:
            if kind == "task":
                lines.append(
                    f"  {item.start}–{item.end}  {PRIORITY_EMOJI[item.priority.value]} "
                    f"{item.title} ({item.est_minutes} min)"
                )
            else:
                lines.append(f"  {item.start}–{item.end}  🍽️ {item.name}")
    else:
        lines.append("  (nothing scheduled)")

    lines.append("")
    lines.append(
        f"Scheduled {plan.scheduled_minutes} of {plan.available_minutes} "
        f"available working minutes."
    )

    if plan.overloaded:
        lines.append("")
        lines.append("⚠️  Day is overloaded — deferred to keep it realistic:")
        for t in plan.deferred:
            lines.append(f"  • {t.title} ({t.priority.value}, {t.est_minutes} min)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class DailyPlannerAgent:
    """The reasoning specialist: turn a TaskList into a time-blocked DayPlan.

    The agent's *goal* is a realistic day; its *decision* is what to defer; its
    *loop* is rank-then-fit. The decision lives in :func:`plan_day` so it can be
    exercised in tests with no model running.
    """

    def __init__(self, workday: Workday = DEFAULT_WORKDAY) -> None:
        self.workday = workday

    def run(self, tasks: TaskList, date: str | None = None) -> DayPlan:
        """Make the planning decision and return the typed result."""
        return plan_day(tasks, workday=self.workday, date=date)
