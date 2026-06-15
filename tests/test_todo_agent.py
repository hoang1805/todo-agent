"""TodoAgent's validate-and-retry boundary — the 'never pass garbage' rule."""

import pytest

from agents.todo_agent import (
    ContractError,
    TodoAgent,
    heuristic_normalize,
    sample_raw_tasks,
)
from core.contract import TaskList


def test_heuristic_normalize_fills_gaps_and_validates():
    raw = [{"id": 1, "title": "Email the client", "priority": "high"}]
    tl = heuristic_normalize(raw)
    assert isinstance(tl, TaskList)
    task = tl.tasks[0]
    assert task.est_minutes == 60  # default for high priority
    assert task.category == "work"  # inferred from "email"


def test_heuristic_normalize_accepts_name_and_due_date_aliases():
    raw = [{"id": 2, "name": "Go for a run", "due_date": "2026-06-20"}]
    tl = heuristic_normalize(raw)
    assert tl.tasks[0].title == "Go for a run"
    assert tl.tasks[0].due == "2026-06-20"
    assert tl.tasks[0].category == "health"


def test_todo_agent_happy_path():
    agent = TodoAgent(fetch_raw=sample_raw_tasks, normalize=heuristic_normalize)
    tl = agent.run()
    assert len(tl.tasks) == len(sample_raw_tasks())


def test_todo_agent_empty_fetch_returns_empty_list():
    agent = TodoAgent(fetch_raw=lambda: [], normalize=heuristic_normalize)
    assert agent.run().tasks == []


def test_todo_agent_retries_then_succeeds():
    calls = {"n": 0}

    def flaky_normalize(raw):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("malformed JSON from the model")
        return heuristic_normalize(raw)

    agent = TodoAgent(fetch_raw=sample_raw_tasks, normalize=flaky_normalize, max_retries=2)
    tl = agent.run()
    assert calls["n"] == 2
    assert len(tl.tasks) > 0


def test_todo_agent_raises_recoverable_error_after_exhausting_retries():
    def always_bad(raw):
        raise ValueError("still malformed")

    agent = TodoAgent(fetch_raw=sample_raw_tasks, normalize=always_bad, max_retries=2)
    with pytest.raises(ContractError) as exc_info:
        agent.run()
    assert exc_info.value.attempts == 2
    assert len(exc_info.value.errors) == 2
