"""Skill loading, roster formatting, and the on-demand skill tool.

This module implements the *progressive-disclosure* skill pattern:

* Only each skill's **name + description** is placed in the agent's system
  prompt (see :func:`format_skill_roster`).  This keeps the context window
  lean — the heavy per-skill instructions are not loaded up front.
* The agent pulls a skill's **full instructions** on demand by calling the
  ``get_skill_detail`` tool built by :func:`make_skill_tools`.

The model therefore routes itself by deciding when to load a skill, replacing
the previous separate "planner" LLM call.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from core.models import Skill


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_skills(skill_dir: str | Path) -> tuple[list[Skill], list[str]]:
    """Scan *skill_dir* for ``*/SKILL.md`` files and parse each into a Skill.

    Args:
        skill_dir: Directory containing one sub-folder per skill, each holding
            a ``SKILL.md`` file.

    Returns:
        A ``(skills, errors)`` tuple. ``skills`` are the successfully parsed
        :class:`~core.models.Skill` objects (sorted by path); ``errors`` are
        human-readable messages for files that could not be parsed, so the
        caller can surface them however it likes (no UI dependency here).
    """
    skill_dir = Path(skill_dir)
    skills: list[Skill] = []
    errors: list[str] = []

    if not skill_dir.is_dir():
        errors.append(f"Skill directory not found: {skill_dir}")
        return skills, errors

    for skill_md in sorted(skill_dir.glob("*/SKILL.md")):
        try:
            skills.append(Skill.from_markdown(skill_md))
        except ValueError as exc:
            errors.append(f"Could not parse {skill_md}: {exc}")

    return skills, errors


# ---------------------------------------------------------------------------
# Roster (for the system prompt)
# ---------------------------------------------------------------------------


def format_skill_roster(skills: list[Skill]) -> str:
    """Render the compact ``- name: description`` roster for the system prompt."""
    if not skills:
        return "(no skills are currently available)"
    return "\n".join(f"- {s.name}: {s.description}" for s in skills)


# ---------------------------------------------------------------------------
# The on-demand skill tool
# ---------------------------------------------------------------------------

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
