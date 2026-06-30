"""Shared guardrails — content/safety/policy checks, distinct from shape contracts.

A Pydantic **contract** validates shape (a valid ``Task``/``DataChunk``). A
**guardrail** decides whether content should pass at all. These are applied at the
orchestrator choke point (every turn: :func:`check_input` before routing,
:func:`check_output` before returning) and inside agents for agent-specific checks
(RAG prompt-injection defense via :func:`wrap_context`, the planner's
:func:`sanity_check_schedule`). All checks degrade deterministically — no LLM needed.
"""

from __future__ import annotations

import re

MAX_INPUT_CHARS = 4000

# Delimiter used to fence retrieved data in prompts (injection defense).
CONTEXT_TAG = "context"

# Phrases that signal an attempt to override instructions (prompt injection).
_INJECTION_PATTERNS = (
    "ignore previous instructions", "ignore all previous", "disregard the above",
    "disregard previous", "reveal the system prompt", "show me the system prompt",
    "you are now", "forget your instructions", "override your rules", "system prompt:",
)

_STOPWORDS = frozenset(
    "the a an of to and or is are was were be been do does did this that these those "
    "i you he she it we they my your what which who when where how why on in at for "
    "with as by from about into over after before me my our".split()
)


class GuardrailError(Exception):
    """A guardrail rejection carrying a user-facing message."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def check_input(text: str) -> str:
    """Validate a user message before routing; raise :class:`GuardrailError` to reject."""
    if not (text or "").strip():
        raise GuardrailError("Please type a message.")
    if len(text) > MAX_INPUT_CHARS:
        raise GuardrailError(
            f"That message is too long ({len(text)} chars; limit {MAX_INPUT_CHARS}). "
            "Please shorten it."
        )
    return text.strip()


def contains_injection(text: str) -> bool:
    """True if *text* looks like an instruction-override / prompt-injection attempt."""
    low = (text or "").lower()
    return any(pattern in low for pattern in _INJECTION_PATTERNS)


# ---------------------------------------------------------------------------
# Retrieval / RAG injection defense
# ---------------------------------------------------------------------------


def wrap_context(blocks: list[str]) -> str:
    """Fence retrieved chunks in delimited tags for the 'data, not instructions' prompt."""
    inner = "\n\n".join(b.strip() for b in blocks if b and b.strip())
    return f"<{CONTEXT_TAG}>\n{inner}\n</{CONTEXT_TAG}>"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower())) - _STOPWORDS


def is_grounded(answer: str, context: str, *, min_overlap: float = 0.3) -> bool:
    """Cheap groundedness: most of the answer's content words appear in the context."""
    answer_words = _tokens(answer)
    if not answer_words:
        return True
    overlap = len(answer_words & _tokens(context)) / len(answer_words)
    return overlap >= min_overlap


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def check_output(text: str, context: str | None = None) -> str:
    """Validate a final answer before returning.

    With *context* (a RAG answer), verify it traces back to the retrieved data — if
    not, append a caveat rather than presenting possible invention as fact. This
    *verifies* the grounding instruction instead of just hoping it was obeyed.
    """
    out = (text or "").strip()
    if context is not None and out and not is_grounded(out, context):
        out += "\n\n_(Note: this answer may not be fully grounded in your stored data.)_"
    return out


# ---------------------------------------------------------------------------
# Agent self-check — schedule sanity
# ---------------------------------------------------------------------------


def _minutes(hhmm: str) -> int | None:
    try:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def sanity_check_schedule(plan) -> list[str]:
    """Return a list of problems with a :class:`DayPlan` (empty list = sane).

    Catches an agent's *own* bad output: non-positive durations and overlapping
    time blocks (task blocks and meal breaks share one timeline).
    """
    issues: list[str] = []
    intervals: list[tuple[int, int, str]] = []
    for block in plan.blocks:
        start, end = _minutes(block.start), _minutes(block.end)
        if start is None or end is None:
            issues.append(f"invalid time on {block.title!r}")
            continue
        if end <= start:
            issues.append(f"non-positive duration: {block.title!r} ({block.start}-{block.end})")
        intervals.append((start, end, block.title))
    for brk in plan.breaks:
        start, end = _minutes(brk.start), _minutes(brk.end)
        if start is not None and end is not None:
            intervals.append((start, end, brk.name))
    intervals.sort()
    for (s1, e1, t1), (s2, e2, t2) in zip(intervals, intervals[1:]):
        if s2 < e1:
            issues.append(f"overlapping blocks: {t1!r} and {t2!r}")
    return issues
