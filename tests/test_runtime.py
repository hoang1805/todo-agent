"""Runtime wiring: raw-task coercion, MCP fetch fallback, normalizer fallback.

These exercise the client-side tool-handling code (the part of deliverable #4
this repo controls) without an LLM or a live MCP server.
"""

import json

import pytest

from agents.orchestrator import (
    _coerce_raw,
    make_mcp_fetcher,
    make_normalizer,
)
from agents.todo_agent import heuristic_normalize, sample_raw_tasks
from models.contract import TaskList


# -- _coerce_raw -------------------------------------------------------------


def test_coerce_raw_from_json_string():
    payload = json.dumps([{"id": 1, "title": "x"}])
    assert _coerce_raw(payload) == [{"id": 1, "title": "x"}]


def test_coerce_raw_unwraps_tasks_key():
    assert _coerce_raw({"tasks": [{"id": 1}]}) == [{"id": 1}]


def test_coerce_raw_passes_through_list():
    assert _coerce_raw([{"id": 1}, {"id": 2}]) == [{"id": 1}, {"id": 2}]


def test_coerce_raw_drops_non_dicts():
    assert _coerce_raw([{"id": 1}, "garbage", 5]) == [{"id": 1}]


def test_coerce_raw_rejects_unexpected_type():
    with pytest.raises(ValueError):
        _coerce_raw(42)


# -- make_mcp_fetcher --------------------------------------------------------


class _FakeTool:
    def __init__(self, name, result):
        self.name = name
        self._result = result

    async def ainvoke(self, _args):
        return self._result


def test_fetcher_uses_matching_tool():
    tool = _FakeTool("get_today_task", [{"id": 9, "title": "real task"}])
    fetch = make_mcp_fetcher([tool])
    assert fetch() == [{"id": 9, "title": "real task"}]


def test_fetcher_falls_back_to_sample_when_no_tool():
    fetch = make_mcp_fetcher([])
    assert fetch() == sample_raw_tasks()


def test_fetcher_falls_back_to_sample_on_tool_error():
    class _BoomTool:
        name = "list_tasks"

        async def ainvoke(self, _args):
            raise RuntimeError("connection refused")

    fetch = make_mcp_fetcher([_BoomTool()])
    assert fetch() == sample_raw_tasks()


# -- make_normalizer ---------------------------------------------------------


def test_normalizer_heuristic_when_llm_disabled():
    normalize = make_normalizer("any-model", 0.0, use_llm=False)
    assert normalize is heuristic_normalize


def test_normalizer_falls_back_to_heuristic_when_model_unreachable():
    # use_llm=True but llm_normalize will fail to connect -> heuristic fallback.
    normalize = make_normalizer("definitely-not-a-real-model", 0.0, use_llm=True)
    result = normalize(sample_raw_tasks())
    assert isinstance(result, TaskList)
    assert len(result.tasks) == len(sample_raw_tasks())


# -- get_weather tool (network stubbed) --------------------------------------


import core.tools.common_tools as common_tools


def test_get_weather_formats_a_reading(monkeypatch):
    def fake_http_json(url, timeout=6.0):
        if "geocoding" in url:
            return {"results": [{"latitude": 21.0, "longitude": 105.8,
                                 "name": "Hanoi", "country": "Vietnam"}]}
        return {
            "current": {"temperature_2m": 31, "apparent_temperature": 35,
                        "weather_code": 2, "wind_speed_10m": 10},
            "current_units": {"temperature_2m": "°C", "wind_speed_10m": "km/h"},
        }

    monkeypatch.setattr(common_tools, "_http_json", fake_http_json)
    out = common_tools.get_weather.invoke({"location": "Hanoi"})
    assert "Hanoi" in out and "partly cloudy" in out and "31" in out


def test_get_weather_handles_unknown_place(monkeypatch):
    monkeypatch.setattr(common_tools, "_http_json", lambda url, timeout=6.0: {"results": []})
    out = common_tools.get_weather.invoke({"location": "Nowhereville"})
    assert "couldn't find" in out.lower()


def test_get_weather_requires_a_location():
    assert "which place" in common_tools.get_weather.invoke({"location": "   "}).lower()
