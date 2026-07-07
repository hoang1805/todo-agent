"""Shared guardrails — content/safety/policy checks, distinct from shape contracts.

A Pydantic **contract** validates shape (a valid ``Task``/``DataChunk``). A
**guardrail** decides whether content should pass at all. These are applied at the
orchestrator choke point (every turn: :func:`check_input` before routing,
:func:`check_output` before returning) and inside agents for agent-specific checks
(RAG prompt-injection defense via :func:`wrap_context`, the planner's
:func:`sanity_check_schedule`).

Two layers:

* **Deterministic** (always on): length/empty checks, injection phrase list,
  token-overlap groundedness, schedule sanity. Work offline, run in tests.
* **LLM screening** (``GUARDRAIL_USE_LLM``, model ``GUARDRAIL_MODEL``): a small
  safety model reviews the user message (:func:`llm_check_input`) and verifies
  groundedness (:func:`llm_is_grounded`) — catching paraphrased injections and
  invented facts the deterministic layer can't. It *adds to* the deterministic
  layer, never replaces it, and any model failure silently falls back — Ollama
  being down never blocks the app.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 4000
MAX_DOC_CHARS = 200_000  # documents are far longer than chat messages, but still bounded

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
# LLM screening layer (optional; deterministic checks stay the baseline)
# ---------------------------------------------------------------------------

_INPUT_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"safe": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["safe"],
}

_GROUNDED_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"grounded": {"type": "boolean"}},
    "required": ["grounded"],
}


def _llm_enabled() -> bool:
    try:
        from configs import settings

        return bool(settings.GUARDRAIL_USE_LLM and settings.GUARDRAIL_MODEL)
    except Exception:  # noqa: BLE001 — no settings → deterministic only
        return False


def _parse_verdict(content: str) -> dict | None:
    """Extract the JSON verdict from a model reply.

    Cloud models don't reliably honour the structured-output format and may fence
    the JSON in markdown — take the first ``{…}`` span rather than trusting the
    whole reply to be JSON.
    """
    content = (content or "").strip()
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(content[start:end + 1])
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _llm_verdict(prompt_name: str, payload: str, schema: dict) -> dict | None:
    """One structured verdict from the guardrail model; ``None`` when unavailable.

    ``None`` (model down, bad JSON, unexpected shape) means "no opinion" — the
    caller then relies on the deterministic layer alone.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from configs import settings
        from core.services.llm_client import create_ollama_model
        from core.services.prompts import load_prompt

        llm = create_ollama_model(
            settings.GUARDRAIL_MODEL, temperature=0.0, format=schema, with_thinking=False
        )
        response = llm.invoke([
            SystemMessage(content=load_prompt(prompt_name)),
            HumanMessage(content=payload),
        ])
        verdict = _parse_verdict(response.content)
        if verdict is None:
            logger.warning("LLM guardrail returned no parseable verdict; ignoring it.")
        return verdict
    except Exception as exc:  # noqa: BLE001 — degrade to the deterministic layer
        logger.warning("LLM guardrail unavailable (%s); deterministic checks only.", exc)
        return None


def llm_check_input(text: str) -> None:
    """Ask the guardrail model to screen a user message; raise when it flags one."""
    if not _llm_enabled():
        return
    verdict = _llm_verdict("guardrail_input_system", text, _INPUT_VERDICT_SCHEMA)
    if verdict is not None and verdict.get("safe") is False:
        reason = (verdict.get("reason") or "it looks unsafe").strip().rstrip(".")
        logger.info("LLM guardrail blocked input: %s", reason)
        raise GuardrailError(f"That message was blocked by the safety check ({reason}).")


def llm_is_grounded(answer: str, context: str) -> bool | None:
    """The guardrail model's groundedness verdict, or ``None`` when unavailable."""
    if not _llm_enabled():
        return None
    verdict = _llm_verdict(
        "guardrail_grounded_system",
        f"Context:\n{context}\n\nAnswer:\n{answer}",
        _GROUNDED_VERDICT_SCHEMA,
    )
    if verdict is not None and isinstance(verdict.get("grounded"), bool):
        return verdict["grounded"]
    return None


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def _log_guardrail(event_type: str, ok: bool, details: dict) -> None:
    """Record a guardrail check into the shared observability store (best-effort)."""
    try:
        from core.services.observability import log_event

        log_event("guardrail", event_type, ok=ok, details=details)
    except Exception:  # noqa: BLE001 — logging must never affect the check
        pass


def check_input(text: str) -> str:
    """Validate a user message before routing; raise :class:`GuardrailError` to reject.

    Deterministic checks first (cheap, always on); then the LLM screen, which can
    catch paraphrased injection/abuse the phrase list misses. Every outcome — pass
    or block, with the reason — is logged to the observability store.
    """
    try:
        if not (text or "").strip():
            raise GuardrailError("Please type a message.")
        if len(text) > MAX_INPUT_CHARS:
            raise GuardrailError(
                f"That message is too long ({len(text)} chars; limit {MAX_INPUT_CHARS}). "
                "Please shorten it."
            )
        # Deterministic injection screen (defense in depth): blatant override
        # phrases are blocked even when the LLM screen is off/unreachable. The LLM
        # layer below then adds coverage for paraphrased attempts the list misses.
        if contains_injection(text):
            raise GuardrailError("That message looks like an attempt to override the assistant's instructions.")
        llm_check_input(text)
    except GuardrailError as exc:
        _log_guardrail("input", ok=False, details={"reason": exc.message, "chars": len(text or "")})
        raise
    _log_guardrail("input", ok=True, details={"chars": len(text)})
    return text.strip()


def check_document(text: str, name: str = "document") -> tuple[str, list[str]]:
    """Validate a document before ingestion; returns ``(text, warnings)``.

    Rejects (raises :class:`GuardrailError`) empty or oversize input. Instruction-like
    content only *warns* — it is stored anyway, because the read side already treats
    retrieved chunks as data, never instructions (:func:`wrap_context`); refusing the
    document would just lose legitimate text that happens to quote an attack.
    """
    text = (text or "").strip()
    if not text:
        raise GuardrailError(f"'{name}' is empty — nothing to ingest.")
    if len(text) > MAX_DOC_CHARS:
        raise GuardrailError(
            f"'{name}' is too large ({len(text)} chars; limit {MAX_DOC_CHARS}). "
            "Please split it into smaller files."
        )
    warnings: list[str] = []
    if contains_injection(text):
        warnings.append(
            f"'{name}' contains instruction-like text (e.g. \"ignore previous "
            "instructions\"). It was stored, but retrieved content is always "
            "treated as data — such instructions will not be followed."
        )
    return text, warnings


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
    The LLM verdict is preferred (it judges meaning, not word overlap); the
    token-overlap heuristic is the fallback when the model has no opinion.
    """
    out = (text or "").strip()
    caveated = False
    if context is not None and out:
        grounded = llm_is_grounded(out, context)
        used_llm = grounded is not None
        if grounded is None:
            grounded = is_grounded(out, context)
        if not grounded:
            out += "\n\n_(Note: this answer may not be fully grounded in your stored data.)_"
            caveated = True
        _log_guardrail("output", ok=True, details={
            "grounded": bool(grounded), "caveat_added": caveated,
            "judge": "llm" if used_llm else "overlap",
        })
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
