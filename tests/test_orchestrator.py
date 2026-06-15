"""Orchestrator routing — classification and end-to-end wiring, no LLM."""

import pytest

from agents.orchestrator import Intent, Orchestrator, classify_intent, format_summary
from agents.planner_agent import DailyPlannerAgent
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
        ("hello there", Intent.unknown),
    ],
)
def test_classify_intent(text, expected):
    assert classify_intent(text) == expected


def _orchestrator(available_minutes=480, fetch_raw=sample_raw_tasks):
    todo = TodoAgent(fetch_raw=fetch_raw, normalize=heuristic_normalize)
    planner = DailyPlannerAgent(available_minutes=available_minutes)
    return Orchestrator(todo, planner)


def test_plan_route_produces_schedule():
    out = _orchestrator().run("plan my day")
    assert "Your plan" in out
    assert "min" in out


def test_plan_route_reports_overload():
    out = _orchestrator(available_minutes=120).run("plan my day")
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
    from core.contract import TaskList

    assert "no tasks" in format_summary(TaskList(tasks=[])).lower()
