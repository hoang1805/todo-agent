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

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from models.contract import DayPlan, MealBreak, Task, TaskList, TimeBlock

logger = logging.getLogger(__name__)

#: A function that turns a TaskList + Workday (+ optional date) into a DayPlan.
Planner = Callable[[TaskList, "Workday", "str | None"], DayPlan]

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


# A time like "7", "7:30", "7h30", "7am", "11 pm" — hour, optional minutes
# (``:`` or ``h`` separator), optional am/pm.
_TIME = r"(\d{1,2})(?:[:h](\d{2}))?\s*([ap]\.?m\.?)?"
# A range: "<time> to/until/till/through/- <time>" (e.g. "from 7am to 11pm").
_RANGE_RE = re.compile(
    rf"{_TIME}\s*(-|–|—|to|until|till|through)\s*{_TIME}", re.IGNORECASE
)


def _to_24h(hour: str | int, minute: str | None, ampm: str | None) -> str | None:
    """Render a parsed clock time as ``HH:MM`` (24-hour), or ``None`` if invalid."""
    hour = int(hour)
    minute = int(minute or 0)
    ap = (ampm or "").lower().replace(".", "")
    if ap == "pm" and hour != 12:
        hour += 12
    elif ap == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


def _range_from_match(match: re.Match) -> tuple[str, str] | None:
    """Turn a :data:`_RANGE_RE` match into a validated ``(start, end)`` pair."""
    sh, sm, sap, conn, eh, em, eap = match.groups()
    # A bare dash range with no am/pm or minutes ("06-18") is more likely a date
    # or plain number range than a clock range — ignore it to avoid false hits.
    if conn in ("-", "–", "—") and not (sap or eap or sm or em):
        return None
    start = _to_24h(sh, sm, sap)
    end = _to_24h(eh, em, eap)
    if not start or not end:
        return None
    # "9 to 5" (no am/pm, end ≤ start) almost certainly means a PM end.
    if end <= start and not eap and int(eh) < 12:
        end = _to_24h(int(eh) + 12, em, None)
    if not end or end <= start:
        return None
    return start, end


def parse_workday(text: str, base: Workday = DEFAULT_WORKDAY) -> Workday | None:
    """Extract working hours from free text (e.g. "I work from 7am to 11pm").

    Returns a :class:`Workday` with the parsed ``start``/``end`` and *base*'s meal
    breaks, or ``None`` if no hour range is found. Meal breaks are preserved so
    a wider window still schedules lunch/dinner correctly.
    """
    match = _RANGE_RE.search(text or "")
    if not match:
        return None
    rng = _range_from_match(match)
    if not rng:
        return None
    start, end = rng
    return Workday(start=start, end=end, breaks=base.breaks)


# Event words / lead-ins that mark a phrase as a *fixed-time commitment*
# ("I have a dinner from 17:30 to 19:30") rather than working hours.
_EVENT_WORDS = (
    "dinner", "lunch", "breakfast", "brunch", "meeting", "appointment", "call",
    "party", "event", "interview", "doctor", "dentist", "standup", "stand-up",
    "sync", "class", "gym", "workout", "date",
)
_APPT_LEADINS = (
    "i have", "i've got", "i ve got", "i got", "there's", "there is",
    "i'll have", "i will have", "i am going", "i'm going", "i'll be", "i will be",
)
_LABEL_LEADIN_RE = re.compile(
    r"^(?:please\s+)?"
    r"(?:i\s+(?:have|'?ve\s+got|got|will\s+have|am\s+having|'?m\s+having)|"
    r"there\s+(?:is|'s)|i\s*'?ll\s+(?:have|be)|i\s+will\s+be)\s+"
    r"(?:an?\s+)?",
    re.IGNORECASE,
)


def _appointment_label(text: str, upto: int) -> str:
    """Derive an event name from the words before the time range (best-effort)."""
    head = text[:upto].strip()
    head = re.sub(r"\b(from|at|on|starting|between|@)\s*$", "", head, flags=re.I).strip()
    head = _LABEL_LEADIN_RE.sub("", head).strip(" ,.-—–")
    if not head:
        return "Appointment"
    return head[0].upper() + head[1:]


def parse_appointment(text: str) -> Break | None:
    """Parse a fixed-time commitment ("dinner with family 17:30–19:30") to a Break.

    Returns ``None`` for working-hours statements ("I work 7am–11pm") and for any
    text without both a time range *and* an event cue, so it doesn't fire on plain
    planning requests.
    """
    low = (text or "").lower()
    if "work" in low:  # "I work from 7 to 11" is the day window, not an event
        return None
    match = _RANGE_RE.search(text or "")
    if not match:
        return None
    rng = _range_from_match(match)
    if not rng:
        return None
    if not (any(w in low for w in _EVENT_WORDS) or any(p in low for p in _APPT_LEADINS)):
        return None
    start, end = rng
    return Break(name=_appointment_label(text, match.start()), start=start, end=end)


def workday_with_appointments(workday: Workday, appointments: list[dict]) -> Workday:
    """Return *workday* with *appointments* merged in as fixed blocks.

    Each appointment (``{"name", "start", "end"}``) becomes a :class:`Break` the
    planner schedules around. A default meal break that overlaps an appointment is
    dropped in its favour (so a 17:30–19:30 dinner replaces the generic 18:00
    dinner slot rather than double-booking it).
    """
    if not appointments:
        return workday
    appts = [Break(name=a["name"], start=a["start"], end=a["end"]) for a in appointments]

    def overlaps(b1: Break, b2: Break) -> bool:
        return _t(b1.start) < _t(b2.end) and _t(b2.start) < _t(b1.end)

    kept = [b for b in workday.breaks if not any(overlaps(b, a) for a in appts)]
    merged = tuple(sorted(kept + appts, key=lambda b: _t(b.start)))
    return Workday(start=workday.start, end=workday.end, breaks=merged)


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


def _claim_meal_breaks(tasks: list[Task], breaks: list[Break]) -> tuple[dict, set]:
    """Match meal-named tasks to meal breaks.

    A task whose title contains a break's name (e.g. "Dinner with CEO" → the
    ``Dinner`` break) *claims* that break: it occupies the slot and is never
    deferred, replacing the generic placeholder. Each break is claimed by at most
    one task and each task claims at most one break.

    Returns ``(claims, claimed_ids)`` where ``claims`` maps a :class:`Break` to
    its claiming :class:`Task` and ``claimed_ids`` is the set of claimed task ids.
    """
    claims: dict[Break, Task] = {}
    claimed_ids: set = set()
    for brk in breaks:
        keyword = brk.name.lower()
        for task in tasks:
            if task.id in claimed_ids:
                continue
            if keyword in task.title.lower():
                claims[brk] = task
                claimed_ids.add(task.id)
                break
    return claims, claimed_ids


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

    A task that *is* a meal (its title matches a break's name, e.g. "dinner with
    CEO") claims that break: it fills the slot and is never deferred, replacing
    the generic label. Other tasks schedule around every break's time as usual.

    Returns a :class:`DayPlan`; ``plan.overloaded`` is ``True`` iff anything was
    deferred. Meal breaks within the window are returned in ``plan.breaks``.
    """
    start, end = _t(workday.start), _t(workday.end)
    ordered_breaks = sorted(workday.breaks, key=lambda b: _t(b.start))
    breaks_in_window = [b for b in ordered_breaks if _t(b.end) > start and _t(b.start) < end]

    # A meal task claims its matching break (occupies the slot, never deferred).
    claims, claimed_ids = _claim_meal_breaks(tasks.tasks, breaks_in_window)

    meal_blocks = [
        MealBreak(
            name=claims[b].title if b in claims else b.name,
            start=b.start,
            end=b.end,
        )
        for b in breaks_in_window
    ]

    cursor = start
    blocks: list[TimeBlock] = []
    deferred: list[Task] = []

    # Claimed meal tasks are placed in their break slot; schedule the rest.
    for task in rank_tasks([t for t in tasks.tasks if t.id not in claimed_ids]):
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
# LLM planning — the model decides the schedule; the contract keeps it honest
# ---------------------------------------------------------------------------


def _validate_plan(plan: DayPlan, tasks: TaskList, workday: Workday) -> DayPlan:
    """Reject a model-produced plan that isn't a usable schedule.

    Type validity is already guaranteed by :class:`DayPlan`; this adds the
    *semantic* checks the contract can't express — blocks inside the working
    window, no overlaps (including meal breaks), real task ids, and every task
    accounted for (scheduled or deferred). A failure raises ``ValueError`` so the
    caller can retry the model or fall back to the deterministic planner.
    """
    win_start, win_end = _t(workday.start), _t(workday.end)
    valid_ids = {t.id for t in tasks.tasks}

    intervals: list[tuple[datetime, datetime, str]] = []
    scheduled: set[str] = set()
    for b in plan.blocks:
        if b.task_id not in valid_ids:
            raise ValueError(f"plan references unknown task_id {b.task_id!r}")
        if b.task_id in scheduled:
            raise ValueError(f"task {b.task_id!r} is scheduled more than once")
        scheduled.add(b.task_id)
        s, e = _t(b.start), _t(b.end)
        if not (win_start <= s < e <= win_end):
            raise ValueError(f"block {b.title!r} ({b.start}-{b.end}) is outside working hours")
        intervals.append((s, e, b.title))

    # Breaks and fixed appointments occupy the timeline too — nothing may overlap
    # them. Use the *workday's* breaks (the source of truth) so a task can't be
    # placed over an appointment even if the model dropped it from its output.
    for brk in workday.breaks:
        bs, be = _t(brk.start), _t(brk.end)
        if be > win_start and bs < win_end:
            intervals.append((bs, be, brk.name))

    intervals.sort()
    for (s1, e1, t1), (s2, e2, t2) in zip(intervals, intervals[1:]):
        if s2 < e1:
            raise ValueError(f"overlapping blocks: {t1!r} and {t2!r}")

    for t in plan.deferred:
        if t.id not in valid_ids:
            raise ValueError(f"deferred references unknown task {t.id!r}")

    accounted = scheduled | {t.id for t in plan.deferred}
    if accounted != valid_ids:
        raise ValueError(f"tasks neither scheduled nor deferred: {valid_ids - accounted}")
    return plan


def llm_plan_day(
    tasks: TaskList,
    workday: Workday,
    model_name: str,
    temperature: float = 0.0,
    date: str | None = None,
) -> DayPlan:
    """Plan the day with an LLM, returning a validated :class:`DayPlan`.

    The JSON schema is generated *from* ``DayPlan`` so the model is constrained to
    the contract's shape; the result is then validated (types **and** the
    semantic checks in :func:`_validate_plan`). On any failure this raises, so the
    caller (see :func:`make_planner`) can retry or fall back to :func:`plan_day`.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from agents.todo_agent import _loads_jsonish
    from core.services.llm_client import create_ollama_model
    from core.services.prompts import load_prompt

    schema = DayPlan.model_json_schema()
    llm = create_ollama_model(model_name, temperature, format=schema, with_thinking=False)
    payload = {
        "date": date,
        "working_hours": {"start": workday.start, "end": workday.end},
        "available_minutes": working_minutes(workday),
        "breaks": [{"name": b.name, "start": b.start, "end": b.end} for b in workday.breaks],
        "tasks": [t.model_dump(mode="json") for t in tasks.tasks],
    }
    response = llm.invoke(
        [
            SystemMessage(content=load_prompt("planner_agent_system")),
            HumanMessage(content=json.dumps(payload, default=str)),
        ]
    )
    data = _loads_jsonish(response.content)
    if isinstance(data, list):  # model wrapped the plan in an array
        if not data:
            raise ValueError("the model returned no plan")
        data = data[0]
    plan = DayPlan.model_validate(data)
    return _validate_plan(plan, tasks, workday)


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
