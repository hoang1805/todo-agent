"""Chunking eval — the retrieval eval, run once per chunking strategy.

Ingests the *same* fixture document under two strategies into separately **tagged**
data (so they never mix), runs the identical retrieval cases against each, and
reports the two hit-rates side by side. This turns week-4's manual "eyeball the
difference" into a measured number you can compare — and is the natural
before/after instrument for a chunking-strategy change.

Needs the memory MCP server reachable.
"""

from __future__ import annotations

from report import RUN_TAG, SAMPLE_DOC, Result, Summary, call_tool, read_jsonl, summarize

STRATEGIES = ("fixed_size", "recursive")
K = 6


def _seed(strategy: str) -> str:
    tag = f"eval_chunk_{strategy}_{RUN_TAG}"
    call_tool("remember", {
        "text": SAMPLE_DOC, "source_type": "document",
        "metadata": {"tag": tag}, "strategy": strategy,
    })
    return tag


def _results_for(tag: str, cases: list[dict]) -> list[Result]:
    from agents.rag_agent import _coerce_chunks

    out = []
    for c in cases:
        chunks = _coerce_chunks(call_tool("retrieve_document", {"query": c["query"], "k": K, "tag": tag}))
        hit = any(c["expect"].lower() in (ch.get("text", "").lower()) for ch in chunks)
        out.append(Result(c["query"][:48], hit, f"want '{c['expect']}'"))
    return out


def run() -> list[Summary]:
    cases = read_jsonl("chunking_cases.jsonl")
    summaries = []
    for strategy in STRATEGIES:
        tag = _seed(strategy)
        summaries.append(summarize("chunking", _results_for(tag, cases), note=strategy))

    # Side-by-side comparison line (the point of this eval).
    print("\n--- chunking strategy comparison (hit-rate) ---")
    for s in summaries:
        print(f"  {s.note:<12} {s.score * 100:5.1f}%  ({s.passed}/{s.total})")
    return summaries


def main() -> None:
    run()


if __name__ == "__main__":
    main()
