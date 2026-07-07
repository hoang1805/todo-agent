"""Agentic-loop eval — tests RAGAgent's *control flow*, not its final answer.

Each case scripts how the first retrieval behaves, then checks the loop did the
right thing by reading the **persisted trace** the loop emits (the week-5 events
store): did it reformulate and retry after a weak pass, and does ``MAX_ITERATIONS``
actually terminate a query that can never be satisfied? The RAGAgent loop code is
real; only the retriever is scripted, so the outcome is deterministic (no LLM).
"""

from __future__ import annotations

import os

os.environ.setdefault("GUARDRAIL_USE_LLM", "0")

from report import Result, Summary, read_jsonl, summarize  # noqa: E402


def _scripted_retriever(kind: str):
    """A retriever whose first pass is weak/empty/strong per *kind*."""
    from models.rag import DataChunk

    state = {"n": 0}
    strong = DataChunk(text="The Zephyr router supports 40 simultaneous devices.",
                       score=0.95, source_type="document")
    weak = DataChunk(text="unrelated filler text", score=0.05, source_type="document")

    def retrieve(source, query, k):
        state["n"] += 1
        if kind == "empty":
            return []
        if kind == "strong":
            return [strong]
        return [weak] if state["n"] == 1 else [strong]   # weak → strong

    return retrieve


def _latest_rag_query() -> dict:
    """The most recent RAG loop-summary event's details (iterations, reformulations)."""
    from core.services.observability import get_observer

    for e in get_observer().recent_events(50):
        if e["component"] == "rag_agent" and e["event_type"] == "query":
            return e["details"]
    return {}


def run() -> list[Summary]:
    from agents.rag_agent import RAGAgent, extractive_generate, heuristic_judge

    cases = read_jsonl("loop_cases.jsonl")
    results = []
    for c in cases:
        agent = RAGAgent(
            retrieve=_scripted_retriever(c["first"]),
            judge=heuristic_judge, generate=extractive_generate,
        )
        agent.run("what does the handbook say about the Zephyr router?")
        trace = _latest_rag_query()
        reformulated = trace.get("reformulations", 0) > 0
        outcome = trace.get("outcome")
        ok = (reformulated == c["expect_reformulate"]) and (outcome == c["expect_outcome"])
        # never-satisfiable must also hit the hard iteration cap
        if c["name"] == "never_satisfiable":
            ok = ok and trace.get("iterations") == RAGAgent.MAX_ITERATIONS
        results.append(Result(c["name"], ok, f"iters={trace.get('iterations')}, "
                              f"reformulations={trace.get('reformulations')}, outcome={outcome}"))
    return [summarize("loop", results, note="scripted retrieval")]


def main() -> None:
    run()


if __name__ == "__main__":
    main()
