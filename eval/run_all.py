"""Run all five evals; print a combined scoreboard and persist each run.

Each eval type is independent (its own dataset + runner), so any one can be run
alone; this just runs them all and summarizes. Evals that need a live dependency
(the memory server, Ollama) are skipped with a clear reason rather than failing
the whole batch, so the offline evals (guardrail, loop) always produce a score.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_chunking_eval          # noqa: E402
import run_generation_eval        # noqa: E402
import run_guardrail_eval         # noqa: E402
import run_loop_eval              # noqa: E402
import run_retrieval_eval         # noqa: E402

# (label, module) — order roughly cheapest/offline first.
_RUNNERS = [
    ("guardrail", run_guardrail_eval),
    ("loop", run_loop_eval),
    ("retrieval", run_retrieval_eval),
    ("chunking", run_chunking_eval),
    ("generation", run_generation_eval),
]


def main() -> None:
    scoreboard: list[tuple[str, str]] = []
    for label, module in _RUNNERS:
        try:
            for summary in module.run():
                tag = f"{label}:{summary.note}" if summary.note and label == "chunking" else label
                scoreboard.append((tag, f"{summary.score * 100:.1f}%  ({summary.passed}/{summary.total})"))
        except Exception as exc:  # noqa: BLE001 — one missing dependency shouldn't stop the rest
            scoreboard.append((label, f"skipped — {type(exc).__name__}: {str(exc)[:60]}"))

    print("\n" + "=" * 48)
    print("  EVAL SUMMARY")
    print("=" * 48)
    for name, score in scoreboard:
        print(f"  {name:<22} {score}")
    print("=" * 48)
    print("  (scores persisted to the events store — see the dashboard)")


if __name__ == "__main__":
    main()
