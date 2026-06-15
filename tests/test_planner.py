"""The planner's decision — exercised with no LLM running."""

from agents.planner_agent import DailyPlannerAgent, plan_day, rank_tasks
from core.contract import Task, TaskList


def _tl(*specs) -> TaskList:
    """Build a TaskList from (id, priority, est_minutes[, due]) tuples."""
    tasks = []
    for spec in specs:
        tid, priority, est = spec[0], spec[1], spec[2]
        due = spec[3] if len(spec) > 3 else None
        tasks.append(
            Task(id=tid, title=f"task-{tid}", priority=priority, est_minutes=est,
                 category="general", due=due)
        )
    return TaskList(tasks=tasks)


def test_ranking_orders_by_priority_then_due_then_size():
    tl = _tl(
        (1, "low", 30),
        (2, "high", 60, "2026-06-20"),
        (3, "high", 60, "2026-06-10"),  # sooner due -> before task 2
        (4, "medium", 15),
    )
    ranked = [t.id for t in rank_tasks(tl.tasks)]
    assert ranked == [3, 2, 4, 1]


def test_normal_day_schedules_everything():
    tl = _tl((1, "high", 120), (2, "medium", 60), (3, "low", 45))
    plan = plan_day(tl, available_minutes=480, day_start="09:00")

    assert plan.overloaded is False
    assert plan.deferred == []
    assert len(plan.blocks) == 3
    # Blocks are contiguous from the start time.
    assert plan.blocks[0].start == "09:00"
    assert plan.blocks[0].end == "11:00"
    assert plan.blocks[1].start == "11:00"
    assert plan.scheduled_minutes == 225


def test_overloaded_day_defers_lowest_priority():
    # Total 390 min but only 180 available -> the day cannot fit.
    tl = _tl(
        (1, "high", 120),
        (2, "high", 60),
        (3, "medium", 60),
        (4, "low", 60),
        (5, "low", 90),
    )
    plan = plan_day(tl, available_minutes=180)

    assert plan.overloaded is True
    # High-priority work is scheduled; low-priority is what gets dropped.
    scheduled_ids = {b.task_id for b in plan.blocks}
    deferred_ids = {t.id for t in plan.deferred}
    assert scheduled_ids == {1, 2}
    assert deferred_ids == {3, 4, 5}
    assert plan.scheduled_minutes <= plan.available_minutes


def test_planner_agent_carries_date_and_uses_its_budget():
    agent = DailyPlannerAgent(available_minutes=60)
    plan = agent.run(_tl((1, "high", 60), (2, "low", 60)), date="2026-06-15")

    assert plan.date == "2026-06-15"
    assert len(plan.blocks) == 1
    assert plan.overloaded is True


def test_empty_tasklist_produces_empty_plan():
    plan = plan_day(TaskList(tasks=[]))
    assert plan.blocks == []
    assert plan.overloaded is False
