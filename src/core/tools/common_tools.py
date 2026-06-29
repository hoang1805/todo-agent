"""Common, always-available native tools.

These are lightweight, dependency-free helpers defined directly in the agent
with LangChain's ``@tool`` decorator — unlike the task tools, which come from
the MCP server. They cover small bits of context the model cannot compute on
its own (such as the current date) and are bound to the agent on every turn.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime

from langchain_core.tools import BaseTool, tool

# Minimal WMO weather-code → description map (open-meteo's `weather_code`).
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def _http_json(url: str, timeout: float = 6.0) -> dict:
    """GET a URL and parse JSON (stdlib only, short timeout)."""
    req = urllib.request.Request(url, headers={"User-Agent": "todo-agent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@tool
def get_today_date() -> str:
    """Return today's local date as an ISO string (YYYY-MM-DD).

    Use this whenever you need to know what "today" is — for example to plan
    the day, resolve relative dates like "tomorrow"/"next week", or stamp a
    task. The model cannot know the current date on its own.
    """
    return datetime.now().date().isoformat()


@tool
def get_now() -> str:
    """Return the current local date and time as an ISO 8601 string.

    Includes the timezone offset, e.g. ``2026-06-15T14:30:00+07:00``. Use this
    when you need the time of day, not just the date.
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


@tool
def get_current_weekday() -> str:
    """Return the current day of the week (e.g. "Monday").

    Handy for planning around weekdays vs. weekends without parsing a date.
    """
    return datetime.now().strftime("%A")


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a place by name (city, town, etc.).

    Use this for any weather/forecast/temperature question. Pass the place the
    user mentioned (e.g. "Hanoi", "Paris, France"). Returns a short, plain-text
    description; if the place can't be found or the service is unreachable, it
    returns a message saying so rather than failing.
    """
    place = (location or "").strip()
    if not place:
        return "Please tell me which place you want the weather for."
    try:
        geo = _http_json(
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": place, "count": 1, "language": "en", "format": "json"})
        )
        results = geo.get("results") or []
        if not results:
            return f"I couldn't find a place called '{place}'."
        spot = results[0]
        lat, lon = spot["latitude"], spot["longitude"]
        label = ", ".join(p for p in (spot.get("name"), spot.get("country")) if p)

        data = _http_json(
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode({
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            })
        )
        cur = data.get("current", {})
        units = data.get("current_units", {})
        desc = _WMO.get(cur.get("weather_code"), "unknown conditions")
        temp = cur.get("temperature_2m")
        feels = cur.get("apparent_temperature")
        wind = cur.get("wind_speed_10m")
        t_unit = units.get("temperature_2m", "°C")
        w_unit = units.get("wind_speed_10m", "km/h")
        return (
            f"Weather in {label}: {desc}, {temp}{t_unit} "
            f"(feels like {feels}{t_unit}), wind {wind} {w_unit}."
        )
    except Exception as exc:  # noqa: BLE001 — network/parse errors degrade gracefully
        return f"Sorry, I couldn't fetch the weather right now ({exc})."


def make_common_tools() -> list[BaseTool]:
    """Return the list of always-available native tools.

    Returning a list keeps the call site uniform with the skill and MCP tools
    (``tools=[*common_tools, *skill_tools, *mcp_tools]``).
    """
    return [get_today_date, get_now, get_current_weekday, get_weather]
