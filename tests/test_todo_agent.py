"""TodoAgent's validate-and-retry boundary — the 'never pass garbage' rule."""

import pytest
from pydantic import ValidationError

from agents.todo_agent import (
    ContractError,
    TodoAgent,
    heuristic_normalize,
    heuristic_parse_mutation,
    sample_raw_tasks,
)
from models.contract import CrudOp, Priority, Status, TaskList, TaskMutation


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


# -- CRUD: the mutation contract (the write-side checkpoint) -----------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(op=CrudOp.create),                                  # create needs a title
        dict(op=CrudOp.update, task_id="1"),                     # update needs ≥1 field
        dict(op=CrudOp.update, status=Status.done),              # update needs a task_id
        dict(op=CrudOp.delete),                                  # delete needs a task_id
        dict(op=CrudOp.create, title="x", est_minutes=0),        # est must be > 0
        dict(op=CrudOp.create, title="x", est_minutes=9000),     # est must be ≤ 480
    ],
)
def test_task_mutation_rejects_underspecified_changes(kwargs):
    with pytest.raises(ValidationError):
        TaskMutation(**kwargs)


def test_task_mutation_accepts_well_formed_changes():
    assert TaskMutation(op=CrudOp.create, title="Call bank").op is CrudOp.create
    assert TaskMutation(op=CrudOp.update, task_id="1", status=Status.done).changed_fields == {
        "status": Status.done
    }
    assert TaskMutation(op=CrudOp.delete, task_id="9").task_id == "9"
    # est_minutes and category are settable fields and count as a change.
    assert TaskMutation(op=CrudOp.update, task_id="1", est_minutes=30).changed_fields == {
        "est_minutes": 30
    }
    assert TaskMutation(op=CrudOp.update, task_id="1", category="home").changed_fields == {
        "category": "home"
    }


# -- CRUD: the heuristic parser + TodoAgent.parse_mutation -------------------

_CURRENT = [{"id": "5", "title": "Gym"}, {"id": "7", "title": "Finish Q3 report"}]


def test_heuristic_parse_create_strips_command_prefix():
    m = heuristic_parse_mutation("add a task to call the bank, high priority", [])
    assert m.op is CrudOp.create
    assert m.title == "call the bank, high priority"
    assert m.priority is Priority.high


def test_heuristic_parse_delete_and_update_resolve_task_id():
    deleted = heuristic_parse_mutation("delete the Gym task", _CURRENT)
    assert deleted.op is CrudOp.delete and deleted.task_id == "5"

    done = heuristic_parse_mutation("mark Finish Q3 report as done", _CURRENT)
    assert done.op is CrudOp.update and done.task_id == "7" and done.status is Status.done


def test_llm_parse_mutation_tolerates_fenced_json_array(monkeypatch):
    # Reproduces the reported failure: the model wrapped a single change in a
    # ```json fence as an array. The parser must still recover it.
    import core.services.llm_client as llm_client
    from langchain_core.messages import AIMessage
    from agents.todo_agent import llm_parse_mutation

    fenced = '```json\n[\n  {\n    "op": "delete",\n    "task_id": "P003"\n  }\n]\n```'

    class _Fake:
        def invoke(self, _messages):
            return AIMessage(content=fenced)

    monkeypatch.setattr(llm_client, "create_ollama_model", lambda *a, **k: _Fake())
    mutation = llm_parse_mutation("cancel the dentist task", [], "dummy", 0.0)
    assert mutation.op is CrudOp.delete and mutation.task_id == "P003"


def test_parse_mutation_raises_recoverable_error_when_unresolved():
    # "delete" with no matching current task -> task_id stays None -> invalid.
    agent = TodoAgent(fetch_raw=lambda: _CURRENT, max_retries=2)
    with pytest.raises(ContractError):
        agent.parse_mutation("delete some task that does not exist")
