"""Skill planner — selects which skills to invoke for a given user message.

The planner sends the user's message together with a compact roster of
available skills (name + description only — *not* the full detail) to the
LLM and asks it to return an ordered list of skill names to execute.

This keeps the planner's context window lean: the heavy per-skill prompts are
only injected at execution time by the executor in ``ollama_agent.py``.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from core.llm_client import create_ollama_model
from core.prompts import load_prompt

if TYPE_CHECKING:
    from core.models import Skill


# ---------------------------------------------------------------------------
# SkillPlanner
# ---------------------------------------------------------------------------


class SkillPlanner:
    """Routes a user message to the appropriate skill(s).

    Args:
        model: Ollama model name used for planning.
        temperature: Sampling temperature (low values = more deterministic).
    """

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self._model = model
        self._temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, user_message: str, skills: list[Skill], messages: list[BaseMessage] | None = None) -> list[str]:
        """Return an ordered list of skill names to execute.

        Args:
            user_message: The raw message typed by the user.
            skills: All loaded :class:`~core.models.Skill` objects.
            messages: Optional prior conversation messages from the checkpointer.

        Returns:
            Ordered list of skill *names* (matching ``Skill.name``) that
            should handle this request.  May be empty if no skill matches.
        """
        if not skills:
            return []

        skills_list = "\n".join(
            f"- {s.name}: {s.description}" for s in skills
        )

        # Build context block from the last few turns of history
        history_block = ""
        if messages:
            # We don't need a custom summarizer here anymore. The SummarizationMiddleware
            # compresses older messages into a SystemMessage automatically.
            # We just dump the last few messages for the planner to see context.
            recent = messages[-8:]
            lines = []
            for msg in recent:
                if isinstance(msg, HumanMessage):
                    lines.append(f"User: {str(msg.content)[:300]}")
                elif isinstance(msg, AIMessage):
                    lines.append(f"Assistant: {str(msg.content)[:300]}")
                elif isinstance(msg, SystemMessage):
                    lines.append(f"System: {str(msg.content)[:1000]}")
            if lines:
                prefix_template = load_prompt("planner_history_prefix")
                if not prefix_template.endswith("\n"):
                    prefix_template += "\n"
                history_block = prefix_template.format(history_text="\n".join(lines))

        user_block = load_prompt("planner_skills_block").format(
            skills_list=skills_list,
            user_message=f"{history_block}{user_message}",
        )

        llm = create_ollama_model(
            self._model,
            temperature=self._temperature,
            with_thinking=False,  # planner doesn't need CoT output
        )
        messages = [
            SystemMessage(content=load_prompt("planner_system")),
            HumanMessage(content=user_block),
        ]
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)

        return self._parse_skill_names(raw, skills)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_skill_names(raw: str, skills: list[Skill]) -> list[str]:
        """Extract a list of valid skill names from the LLM's raw output.

        Tries strict JSON parsing first, then falls back to a regex scan so
        that minor formatting quirks (e.g. markdown fences) don't break routing.
        """
        valid_names = {s.name for s in skills}

        # 1. Try to locate a JSON array in the output
        array_match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if array_match:
            try:
                candidates = json.loads(array_match.group())
                result = [c for c in candidates if c in valid_names]
                if result or candidates == []:
                    return result
            except json.JSONDecodeError:
                pass

        # 2. Fallback: pick any valid skill name mentioned in the output
        return [name for name in valid_names if name in raw]
