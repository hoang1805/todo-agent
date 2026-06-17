"""Orchestrator routing — classification and end-to-end wiring, no LLM."""

import pytest

from agents.orchestrator import (
    Intent,
    Orchestrator,
    classify_intent,
    format_summary,
    is_crud_intent,
    matched_intent_families,
)
from agents.planner_agent import DailyPlannerAgent, Workday
from agents.todo_agent import ContractError, TodoAgent, heuristic_normalize, sample_raw_tasks


@pytest.mark.parametrize(
    "text, expected",
    [
        ("plan my day", Intent.plan),
        ("what should I do today?", Intent.plan),
        ("what's on my list?", Intent.summary),
        ("show me my tasks", Intent.summary),
        ("add a task to call the bank", Intent.add),
        ("remind me to buy milk", Intent.add),
        ("delete the gym task", Intent.delete),
        ("remove task 3", Intent.delete),
        ("mark the report as done", Intent.update),
        ("set priority of task 2 to high", Intent.update),
        ("change the estimate time of Fix JSON bug into 120 min", Intent.update),
        ("move the gym task to the home category", Intent.update),
        ("what's the weather in Hanoi?", Intent.weather),
        ("will it rain tomorrow?", Intent.weather),
        ("hello there", Intent.unknown),
    ],
)
def test_classify_intent(text, expected):
    assert classify_intent(text) == expected


def test_is_crud_intent():
    assert all(is_crud_intent(i) for i in (Intent.add, Intent.update, Intent.delete))
    assert not any(
        is_crud_intent(i) for i in (Intent.plan, Intent.summary, Intent.weather, Intent.unknown)
    )


def test_matched_intent_families_spans_tasks_and_weather():
    # A mixed prompt -> multiple families -> routed to the agent loop.
    families = set(matched_intent_families("create a task to call the bank and what's the weather"))
    assert {Intent.add, Intent.weather} <= families
    assert len(families) > 1


# -- the MCP mutation executor maps fields to the right write tools ----------


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return {"ok": True}


def test_mutation_executor_create_sends_category_and_est():
    from agents.orchestrator import make_mutation_executor
    from models.contract import CrudOp, Priority, TaskMutation

    create_tool = _FakeTool("create_task")
    execute = make_mutation_executor([create_tool])
    msg = execute(
        TaskMutation(op=CrudOp.create, title="Call bank", category="home",
                     est_minutes=20, priority=Priority.high)
    )
    assert "Created task" in msg
    args = create_tool.calls[0]
    assert args["name"] == "Call bank"
    assert args["category"] == "home"
    assert args["est_minutes"] == 20
    assert args["priority"] == "high"


def test_mutation_executor_update_routes_each_field_to_its_tool():
    from agents.orchestrator import make_mutation_executor
    from models.contract import CrudOp, Priority, TaskMutation

    fields = _FakeTool("update_task_by_id")
    priority = _FakeTool("update_task_priority")
    execute = make_mutation_executor([fields, priority])
    msg = execute(
        TaskMutation(op=CrudOp.update, task_id="7", est_minutes=30,
                     category="work", priority=Priority.low)
    )
    assert "Updated task 7" in msg
    assert priority.calls[0]["priority"] == "low"          # priority -> its own tool
    assert fields.calls[0]["est_minutes"] == 30            # est + category -> fields tool
    assert fields.calls[0]["category"] == "work"
    assert fields.calls[0]["task_id"] == "7"


def test_matched_intent_families_detects_single_and_multi():
    # Single family.
    assert set(matched_intent_families("plan my day")) == {Intent.plan}
    # Multi-step: both "add" and "plan" present -> two families.
    families = set(matched_intent_families("plan my day and add a task"))
    assert {Intent.add, Intent.plan} <= families
    assert len(families) > 1


def _orchestrator(workday=None, fetch_raw=sample_raw_tasks):
    todo = TodoAgent(fetch_raw=fetch_raw, normalize=heuristic_normalize)
    planner = DailyPlannerAgent(workday=workday) if workday else DailyPlannerAgent()
    return Orchestrator(todo, planner)


def test_plan_route_produces_schedule():
    out = _orchestrator().run("plan my day")
    assert "Your plan" in out
    assert "min" in out


def test_plan_route_reports_overload():
    tight = Workday(start="09:00", end="11:00", breaks=())
    out = _orchestrator(workday=tight).run("plan my day")
    assert "overloaded" in out.lower()
    assert "deferred" in out.lower()


def test_summary_route_lists_tasks():
    out = _orchestrator().run("what's on my list?")
    assert "task(s)" in out


def test_unknown_route_gives_guidance():
    out = _orchestrator().run("good morning")
    assert "plan your day" in out.lower()


def test_contract_error_is_surfaced_recoverably():
    def bad_fetch():
        return [{"id": 1}]  # missing title -> heuristic builds "", contract rejects

    todo = TodoAgent(fetch_raw=bad_fetch, normalize=heuristic_normalize, max_retries=1)
    orch = Orchestrator(todo, DailyPlannerAgent())
    out = orch.run("plan my day")
    assert "couldn't prepare your tasks" in out.lower()


def test_format_summary_empty():
    from models.contract import TaskList

    assert "no tasks" in format_summary(TaskList(tasks=[])).lower()
