"""The contract is the boundary — these tests prove garbage can't pass it."""

import pytest
from pydantic import ValidationError

from models.contract import DayPlan, Priority, Task, TaskList, TimeBlock


def _task(**overrides):
    base = dict(id=1, title="Write report", priority="high", est_minutes=60, category="work")
    base.update(overrides)
    return Task(**base)


def test_valid_task_parses():
    task = _task(due="2026-06-15")
    assert task.priority is Priority.high
    assert task.due == "2026-06-15"


def test_string_and_numeric_ids_are_accepted_as_strings():
    # Real stores use codes / UUIDs; a number is coerced to its string form.
    assert _task(id="W001").id == "W001"
    assert _task(id="05b46062-3420-49c7-ab12-aa0e30fc3e68").id.startswith("05b4")
    assert _task(id=7).id == "7"
    payload = '{"tasks": [{"id": "AI-101", "title": "x", "priority": "low", "est_minutes": 30, "category": "general"}]}'
    assert TaskList.model_validate_json(payload).tasks[0].id == "AI-101"


def test_est_minutes_must_be_positive():
    with pytest.raises(ValidationError):
        _task(est_minutes=0)


def test_est_minutes_upper_bound_enforced():
    with pytest.raises(ValidationError):
        _task(est_minutes=9000)  # an LLM hallucination must be rejected


def test_empty_title_rejected():
    with pytest.raises(ValidationError):
        _task(title="   ")  # stripped to empty -> invalid


def test_invalid_priority_rejected():
    with pytest.raises(ValidationError):
        _task(priority="urgent")  # not a member of the enum


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        _task(notes="an invented field the LLM hallucinated")


def test_title_and_category_are_stripped():
    task = _task(title="  Plan ", category=" work ")
    assert task.title == "Plan"
    assert task.category == "work"


def test_tasklist_validates_from_json():
    payload = '{"tasks": [{"id": 1, "title": "x", "priority": "low", "est_minutes": 30, "category": "general"}]}'
    tl = TaskList.model_validate_json(payload)
    assert len(tl.tasks) == 1


def test_malformed_json_raises():
    with pytest.raises(ValidationError):
        TaskList.model_validate_json('{"tasks": [{"id": 1}]}')  # missing fields


def test_priority_rank_ordering():
    assert Priority.high.rank < Priority.medium.rank < Priority.low.rank


def test_dayplan_overloaded_flag_and_scheduled_minutes():
    block = TimeBlock(
        task_id=1, title="x", priority=Priority.high, start="09:00", end="10:00", est_minutes=60
    )
    fitted = DayPlan(available_minutes=480, blocks=[block], deferred=[])
    assert fitted.overloaded is False
    assert fitted.scheduled_minutes == 60

    overloaded = DayPlan(available_minutes=30, blocks=[], deferred=[_task()])
    assert overloaded.overloaded is True
