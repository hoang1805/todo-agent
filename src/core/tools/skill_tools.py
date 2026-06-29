"""The on-demand skill tool (progressive disclosure).

This is the *tool* half of the skill pattern: it builds the
``get_skill_detail`` tool the agent calls to load a skill's **full**
``SKILL.md`` instructions on demand. The loading and roster-formatting logic
lives in :mod:`core.services.skills`.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from models.skill import Skill

#: Name of the tool the agent calls to load a skill's full instructions.
GET_SKILL_DETAIL = "get_skill_detail"


def make_skill_tools(skills: list[Skill]) -> list[StructuredTool]:
    """Build the ``get_skill_detail`` tool bound to the loaded *skills*.

    The returned tool lets the agent fetch the full ``SKILL.md`` body for a
    skill by its exact name. Returning a list keeps the call site uniform with
    the MCP tools (``tools=[*skill_tools, *mcp_tools]``).
    """
    skill_map = {s.name: s for s in skills}

    def get_skill_detail(skill_name: str) -> str:
        """Load the complete instructions and workflow for a named skill."""
        skill = skill_map.get(skill_name)
        if skill is None:
            available = ", ".join(skill_map) or "(none)"
            return (
                f"No skill named '{skill_name}'. "
                f"Available skills: {available}."
            )
        return skill.detail

    tool = StructuredTool.from_function(
        func=get_skill_detail,
        name=GET_SKILL_DETAIL,
        description=(
            "Load the complete, detailed instructions and workflow for a skill "
            "by its exact name. Call this BEFORE acting on any request that "
            "matches a skill listed in your system prompt. Returns the full "
            "SKILL.md instructions, which you must then follow. "
            "Argument: skill_name — the exact skill name, e.g. 'daily_planner'."
        ),
    )
    return [tool]
