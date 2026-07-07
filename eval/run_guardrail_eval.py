"""Guardrail eval — does the input guardrail block the bad and allow the good?

Two classes in one dataset (both required): adversarial inputs that *should* be
blocked, and legitimate inputs that *should* pass. Testing only the adversarial
set can't reveal a guardrail that got so strict it blocks normal use — so the
score rewards getting *both* right (block-accuracy over all cases).

Runs fully offline against the deterministic guardrail layer by default; set
``GUARDRAIL_USE_LLM=1`` (with Ollama up) to also score the LLM screen.
"""

from __future__ import annotations

import os

os.environ.setdefault("GUARDRAIL_USE_LLM", "0")  # reproducible by default

from report import Result, Summary, read_jsonl, summarize  # noqa: E402


def _blocked(text: str) -> bool:
    from core.services.guardrails import GuardrailError, check_input

    try:
        check_input(text)
        return False
    except GuardrailError:
        return True


def run() -> list[Summary]:
    cases = read_jsonl("guardrail_cases.jsonl")
    results = []
    for c in cases:
        got = _blocked(c["text"])
        results.append(Result(
            name=(c["text"].strip() or "(empty)")[:40],
            passed=(got == c["should_block"]),
            info=f"want block={c['should_block']}, got={got}",
        ))
    note = "llm" if os.environ.get("GUARDRAIL_USE_LLM") in ("1", "true", "yes") else "deterministic"
    return [summarize("guardrail", results, note=note)]


def main() -> None:
    run()


if __name__ == "__main__":
    main()
