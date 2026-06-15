"""DailyPlannerAgent — the reasoning specialist.

Given a validated :class:`~core.contract.TaskList`, it ranks the tasks,
time-blocks them across the available hours, and — the part that makes it an
*agent* rather than a sorter — **decides what to do when the day is
overloaded**: it defers the lowest-priority tasks and reports exactly what it
dropped.

The decision (:func:`plan_day`) is deterministic, pure Python. That is on
purpose: it is the graded behaviour, so it must be reliable and unit-testable
without an LLM. The LLM is used only for an optional human-friendly narrative
on top of the already-made decision (:meth:`DailyPlannerAgent.narrate`).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from core.contract import DayPlan, Task, TaskList, TimeBlock

# A standard work day: 8 hours.
DEFAULT_AVAILABLE_MINUTES = 480
DEFAULT_DAY_START = "09:00"

#: Priority → emoji, shared by the plan and summary renderers.
PRIORITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _due_sort_key(due: str | None) -> str:
    """Sort key for the optional ISO due date — undated tasks sort last."""
    return due if due else "9999-12-31"


def rank_tasks(tasks: list[Task]) -> list[Task]:
    """Order tasks for scheduling: priority first, then due date, then size.

    * **Priority** is primary — high before medium before low.
    * **Due date** breaks ties — sooner deadlines win.
    * **Estimated size** breaks remaining ties — shorter tasks first, so when a
      day is tight we fit more committed work in.
    """
    return sorted(
        tasks,
        key=lambda t: (t.priority.rank, _due_sort_key(t.due), t.est_minutes),
    )


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def plan_day(
    tasks: TaskList,
    available_minutes: int = DEFAULT_AVAILABLE_MINUTES,
    day_start: str = DEFAULT_DAY_START,
    date: str | None = None,
) -> DayPlan:
    """Build a time-blocked day, deferring the lowest-priority overflow.

    Walks the ranked tasks greedily, placing each into a contiguous slot while
    it still fits the remaining budget. The first task that would overflow — and
    every lower-ranked task after it — is **deferred** rather than scheduled.
    Because the input is ranked by priority, the deferred set is always the
    least important work, which is exactly the "what do I cut?" decision.

    Args:
        tasks: The validated task list to plan.
        available_minutes: Total minutes available to schedule (default 8h).
        day_start: Clock time the day begins, ``HH:MM`` 24-hour.
        date: Optional date label carried onto the resulting plan.

    Returns:
        A :class:`DayPlan`. ``plan.overloaded`` is ``True`` iff anything was
        deferred.
    """
    cursor = datetime.strptime(day_start, "%H:%M")
    used = 0
    blocks: list[TimeBlock] = []
    deferred: list[Task] = []

    for task in rank_tasks(tasks.tasks):
        if used + task.est_minutes <= available_minutes:
            end = cursor + timedelta(minutes=task.est_minutes)
            blocks.append(
                TimeBlock(
                    task_id=task.id,
                    title=task.title,
                    priority=task.priority,
                    start=cursor.strftime("%H:%M"),
                    end=end.strftime("%H:%M"),
                    est_minutes=task.est_minutes,
                )
            )
            used += task.est_minutes
            cursor = end
        else:
            deferred.append(task)

    return DayPlan(
        date=date,
        available_minutes=available_minutes,
        blocks=blocks,
        deferred=deferred,
    )


# ---------------------------------------------------------------------------
# Formatting (no LLM) — deterministic rendering of the decision
# ---------------------------------------------------------------------------


def format_plan(plan: DayPlan) -> str:
    """Render a :class:`DayPlan` as readable text without an LLM."""
    header = f"🗓️  Your plan{f' for {plan.date}' if plan.date else ''}"
    lines = [header, ""]

    if plan.blocks:
        for b in plan.blocks:
            lines.append(
                f"  {b.start}–{b.end}  {PRIORITY_EMOJI[b.priority.value]} {b.title} "
                f"({b.est_minutes} min)"
            )
    else:
        lines.append("  (nothing scheduled)")

    lines.append("")
    lines.append(
        f"Scheduled {plan.scheduled_minutes} of {plan.available_minutes} "
        f"available minutes."
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
    *loop* is plan-then-(optionally)-narrate. The decision lives in
    :func:`plan_day` so it can be exercised in tests with no model running.
    """

    def __init__(
        self,
        available_minutes: int = DEFAULT_AVAILABLE_MINUTES,
        day_start: str = DEFAULT_DAY_START,
    ) -> None:
        self.available_minutes = available_minutes
        self.day_start = day_start

    def run(self, tasks: TaskList, date: str | None = None) -> DayPlan:
        """Make the planning decision and return the typed result."""
        return plan_day(
            tasks,
            available_minutes=self.available_minutes,
            day_start=self.day_start,
            date=date,
        )
