"""The planner's decision — exercised with no LLM running."""

from agents.planner_agent import (
    Break,
    DailyPlannerAgent,
    Workday,
    apply_mutation_to_plan,
    format_plan,
    plan_day,
    rank_tasks,
    wants_replan,
    working_minutes,
)
from models.contract import CrudOp, Status, TaskMutation
from models.contract import Task, TaskList

# A break-free workday makes the contiguous-scheduling assertions simple.
NO_BREAKS = Workday(start="09:00", end="18:00", breaks=())


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
    assert ranked == ["3", "2", "4", "1"]


def test_working_minutes_excludes_breaks():
    # 09:00–18:00 = 540 min, minus a 60-min lunch = 480.
    wd = Workday(start="09:00", end="18:00", breaks=(Break("Lunch", "12:00", "13:00"),))
    assert working_minutes(wd) == 480
    assert working_minutes(NO_BREAKS) == 540


def test_normal_day_schedules_everything():
    tl = _tl((1, "high", 120), (2, "medium", 60), (3, "low", 45))
    plan = plan_day(tl, workday=NO_BREAKS)

    assert plan.overloaded is False
    assert plan.deferred == []
    assert len(plan.blocks) == 3
    # Blocks are contiguous from the start time.
    assert plan.blocks[0].start == "09:00"
    assert plan.blocks[0].end == "11:00"
    assert plan.blocks[1].start == "11:00"
    assert plan.scheduled_minutes == 225


def test_tasks_schedule_around_a_break():
    # A 90-min task starting at 11:00 would cross a 12:00–13:00 lunch, so it is
    # pushed to after lunch.
    wd = Workday(start="11:00", end="18:00", breaks=(Break("Lunch", "12:00", "13:00"),))
    plan = plan_day(_tl((1, "high", 90)), workday=wd)

    assert [b.start for b in plan.blocks] == ["13:00"]
    assert plan.blocks[0].end == "14:30"
    # The meal block is reported so the UI can show it.
    assert [(m.name, m.start, m.end) for m in plan.breaks] == [("Lunch", "12:00", "13:00")]


def test_overloaded_day_defers_lowest_priority():
    # Total 390 min but only a 3-hour window -> the day cannot fit it all.
    tl = _tl(
        (1, "high", 120),
        (2, "high", 60),
        (3, "medium", 60),
        (4, "low", 60),
        (5, "low", 90),
    )
    plan = plan_day(tl, workday=Workday(start="09:00", end="12:00", breaks=()))

    assert plan.overloaded is True
    # High-priority work is scheduled; low-priority is what gets dropped.
    scheduled_ids = {b.task_id for b in plan.blocks}
    deferred_ids = {t.id for t in plan.deferred}
    assert scheduled_ids == {"1", "2"}
    assert deferred_ids == {"3", "4", "5"}
    assert plan.scheduled_minutes <= plan.available_minutes


def test_deferred_big_task_does_not_block_a_smaller_one():
    # A 90-min task can't fit a 60-min window, but a later 30-min task can.
    tl = _tl((1, "high", 90), (2, "high", 30))
    plan = plan_day(tl, workday=Workday(start="09:00", end="10:00", breaks=()))

    assert {b.task_id for b in plan.blocks} == {"2"}
    assert {t.id for t in plan.deferred} == {"1"}


def test_planner_agent_carries_date_and_defers_when_tight():
    agent = DailyPlannerAgent(workday=Workday(start="09:00", end="10:00", breaks=()))
    plan = agent.run(_tl((1, "high", 60), (2, "low", 60)), date="2026-06-15")

    assert plan.date == "2026-06-15"
    assert len(plan.blocks) == 1
    assert plan.overloaded is True


def test_empty_tasklist_produces_empty_plan():
    plan = plan_day(TaskList(tasks=[]), workday=NO_BREAKS)
    assert plan.blocks == []
    assert plan.overloaded is False


def test_meal_task_claims_the_matching_break():
    tasks = TaskList(tasks=[
        Task(id="1", title="Review PR", priority="high", est_minutes=60, category="work"),
        Task(id="2", title="Having a dinner with CEO", priority="medium", est_minutes=45, category="personal"),
    ])
    # Default workday has a Dinner break at 18:00–19:00.
    plan = plan_day(tasks)

    dinner = next(b for b in plan.breaks if b.start == "18:00")
    assert "dinner with ceo" in dinner.name.lower()            # break shows the task
    assert all(t.title != tasks.tasks[1].title for t in plan.deferred)  # never deferred
    assert all(b.title != tasks.tasks[1].title for b in plan.blocks)    # not a work block


def _two_task_plan():
    tasks = TaskList(tasks=[
        Task(id="1", title="Write report", priority="high", est_minutes=60, category="work"),
        Task(id="2", title="Call bank", priority="low", est_minutes=30, category="errand"),
    ])
    return plan_day(tasks, workday=NO_BREAKS)


def test_wants_replan_detects_explicit_regenerate():
    assert wants_replan("replan my day") and wants_replan("redo the plan please")
    assert not wants_replan("plan my day")
    assert not wants_replan("what's on my list?")


def test_apply_mutation_marks_block_done_without_touching_original():
    plan = _two_task_plan()
    updated = apply_mutation_to_plan(
        plan, TaskMutation(op=CrudOp.update, task_id="1", status=Status.done)
    )
    assert next(b for b in updated.blocks if b.task_id == "1").done is True
    assert next(b for b in plan.blocks if b.task_id == "1").done is False  # input untouched
    out = format_plan(updated)
    assert "✅" in out and "~~Write report~~" in out


def test_apply_mutation_delete_drops_the_block():
    updated = apply_mutation_to_plan(_two_task_plan(), TaskMutation(op=CrudOp.delete, task_id="2"))
    assert all(b.task_id != "2" for b in updated.blocks)


def test_apply_mutation_create_appends_a_block():
    plan = _two_task_plan()
    updated = apply_mutation_to_plan(
        plan, TaskMutation(op=CrudOp.create, title="New thing", est_minutes=20)
    )
    assert len(updated.blocks) == len(plan.blocks) + 1
    assert any(b.title == "New thing" for b in updated.blocks)


def test_apply_mutation_edit_updates_block_fields():
    updated = apply_mutation_to_plan(
        _two_task_plan(), TaskMutation(op=CrudOp.update, task_id="1", est_minutes=90)
    )
    assert next(b for b in updated.blocks if b.task_id == "1").est_minutes == 90


def test_meal_task_is_ordinary_without_a_matching_break():
    tasks = TaskList(tasks=[
        Task(id="1", title="Having a dinner with CEO", priority="high", est_minutes=45, category="personal"),
    ])
    # No Dinner break in this window -> it's just a normal task.
    plan = plan_day(tasks, workday=Workday(start="09:00", end="18:00", breaks=()))
    assert plan.breaks == []
    assert [b.title for b in plan.blocks] == ["Having a dinner with CEO"]
