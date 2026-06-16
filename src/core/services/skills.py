"""Skill loading and roster formatting.

This is the *loading* half of the progressive-disclosure skill pattern:

* Only each skill's **name + description** is placed in the agent's system
  prompt (see :func:`format_skill_roster`).  This keeps the context window
  lean — the heavy per-skill instructions are not loaded up front.
* The agent pulls a skill's **full instructions** on demand via the
  ``get_skill_detail`` tool built in :mod:`core.tools.skill_tools`.

The model therefore routes itself by deciding when to load a skill, replacing
the previous separate "planner" LLM call.
"""

from __future__ import annotations

from pathlib import Path

from models.skill import Skill


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
        :class:`~models.skill.Skill` objects (sorted by path); ``errors`` are
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
