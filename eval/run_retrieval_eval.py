"""Retrieval eval — tests ``retrieve_document`` directly, no LLM involved.

Ingests the shared handbook fixture, then for each case checks whether a top-k
result contains the expected fact. The score is a **hit-rate** — how often the
right chunk surfaced. This is the exact instrument that *measures* whether an
embedding-model swap or a chunking change actually helped, instead of guessing.

Needs the memory MCP server reachable (``MEMORY_MCP_URL``).
"""

from __future__ import annotations

from report import RUN_TAG, SAMPLE_DOC, Result, Summary, call_tool, read_jsonl, summarize

TAG = f"eval_retrieval_{RUN_TAG}"
K = 6


def _seed() -> None:
    call_tool("remember", {"text": SAMPLE_DOC, "source_type": "document", "metadata": {"tag": TAG}})


def _hit(query: str, expect: str) -> bool:
    from agents.rag_agent import _coerce_chunks

    chunks = _coerce_chunks(call_tool("retrieve_document", {"query": query, "k": K, "tag": TAG}))
    return any(expect.lower() in (c.get("text", "").lower()) for c in chunks)


def run() -> list[Summary]:
    _seed()
    cases = read_jsonl("retrieval_cases.jsonl")
    results = [Result(c["query"][:48], _hit(c["query"], c["expect"]), f"want '{c['expect']}'") for c in cases]
    return [summarize("retrieval", results, note=f"handbook@k={K}")]


def main() -> None:
    run()


if __name__ == "__main__":
    main()
