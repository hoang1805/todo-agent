"""Prompt loader — reads prompt markdown files from the ``src/prompts/`` directory.

Usage::

    from core.services.prompts import load_prompt

    system = load_prompt("planner_system")          # no extension needed
    template = load_prompt("planner_skills_block")
"""

from __future__ import annotations

from pathlib import Path

# Prompts live under src/ (this file is src/core/services/prompts.py)
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt from ``src/prompts/<name>.md``.

    Args:
        name: Filename without extension, e.g. ``"planner_system"``.

    Returns:
        The prompt text with trailing whitespace stripped.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Available prompts: {[p.stem for p in _PROMPTS_DIR.glob('*.md')]}"
        )
    return path.read_text(encoding="utf-8").rstrip()
