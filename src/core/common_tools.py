"""Common, always-available native tools.

These are lightweight, dependency-free helpers defined directly in the agent
with LangChain's ``@tool`` decorator — unlike the task tools, which come from
the MCP server. They cover small bits of context the model cannot compute on
its own (such as the current date) and are bound to the agent on every turn.
"""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import BaseTool, tool


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


def make_common_tools() -> list[BaseTool]:
    """Return the list of always-available native tools.

    Returning a list keeps the call site uniform with the skill and MCP tools
    (``tools=[*common_tools, *skill_tools, *mcp_tools]``).
    """
    return [get_today_date, get_now, get_current_weekday]
