"""Eval harness — the offline runners (guardrail, loop) score and persist correctly.

The retrieval/chunking/generation runners need a live memory server / Ollama, so
they're exercised in the demo, not here; these two run fully in-process.
"""

import sys
from pathlib import Path

# The eval runners live in eval/ (beside src/), importing a sibling `report`.
_EVAL = Path(__file__).resolve().parent.parent / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))


def test_guardrail_eval_scores_both_classes_and_persists():
    import run_guardrail_eval
    from core.services.observability import get_observer

    (summary,) = run_guardrail_eval.run()
    # Deterministic layer blocks the phrase-list injections and empties, allows the
    # legitimate task prompts — so every case should be classified correctly.
    assert summary.eval_type == "guardrail"
    assert summary.score == 1.0, [r.info for r in summary.results if not r.passed]

    hist = get_observer().eval_history("guardrail")
    assert hist and hist[-1]["passed"] == summary.passed   # the run was persisted


def test_loop_eval_verifies_reformulation_and_termination():
    import run_loop_eval

    (summary,) = run_loop_eval.run()
    by_name = {r.name: r for r in summary.results}
    assert by_name["weak_then_strong"].passed        # retried after a weak pass
    assert by_name["never_satisfiable"].passed       # stopped at MAX_ITERATIONS
    assert by_name["strong_first"].passed            # answered on the first pass
    assert summary.score == 1.0
