"""RAGAgent — the agentic retrieval specialist.

Not "retrieve once, stuff into a prompt." A real loop that **decides**: which
source to query, whether what came back is enough, and — if not — reformulates and
retrieves again, up to ``MAX_ITERATIONS``. Each tool result is validated against
:class:`~models.rag.DataChunk` and each judgment against
:class:`~models.rag.RetrievalDecision` before the loop branches on it.

Everything is injectable (``retrieve``/``judge``/``generate``) so the loop is
testable without an LLM or a live memory server; :func:`make_rag_agent` wires the
real ones (memory MCP tools + Ollama, each with a deterministic fallback).
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from pydantic import ValidationError

from models.rag import DataChunk, RetrievalDecision

logger = logging.getLogger(__name__)

#: (source, query, k) -> validated chunks. ``source`` is "logs"/"documents"/"both".
Retriever = Callable[[str, str, int], "list[DataChunk]"]
Judge = Callable[[str, "list[DataChunk]"], RetrievalDecision]
Generator = Callable[[str, "list[DataChunk]"], str]


class RAGAgent:
    """Retrieve → judge → (reformulate → retrieve)* → generate, with a hard cap."""

    MAX_ITERATIONS = 3

    def __init__(self, retrieve: Retriever, judge: Judge, generate: Generator, k: int = 3):
        self._retrieve = retrieve
        self._judge = judge
        self._generate = generate
        self.k = k

    def run(self, query: str) -> str:
        collected: list[DataChunk] = []   # accumulate — never discarded between passes
        current_query, source = query, "both"

        for i in range(self.MAX_ITERATIONS):
            chunks = self._retrieve(source, current_query, self.k)
            seen = {c.text for c in collected}
            collected.extend(c for c in chunks if c.text not in seen)

            decision = self._judge(query, collected)
            logger.info(
                "[rag] iter=%d query=%r source=%s got=%d sufficient=%s next_query=%r",
                i, current_query, source, len(chunks), decision.sufficient, decision.next_query,
            )
            if decision.sufficient:
                return self._generate(query, collected)

            current_query = decision.next_query or current_query
            source = decision.next_source or source

        # Out of iterations — answer honestly from whatever was collected.
        logger.info("[rag] stopped after %d iterations (%d chunks)", self.MAX_ITERATIONS, len(collected))
        if collected:
            return self._generate(query, collected)
        return "I couldn't find anything relevant in your logs or documents."


# ---------------------------------------------------------------------------
# Retrieval — drive the split memory tools and validate every result
# ---------------------------------------------------------------------------


def _coerce_chunks(result: object) -> list[dict]:
    """Turn a memory-tool result (list / JSON string / error dict) into dicts."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            return []
    if isinstance(result, dict):
        if result.get("ok") is False:  # structured recoverable error
            return []
        result = result.get("results") or result.get("data") or []
    return [r for r in result if isinstance(r, dict)] if isinstance(result, list) else []


def make_mcp_retriever(tools: list) -> Retriever:
    """Build a retriever over the memory MCP ``retrieve_log``/``retrieve_document`` tools."""
    from agents.orchestrator import _run_coro

    by_name = {t.name: t for t in tools}
    log_tool = by_name.get("retrieve_log")
    doc_tool = by_name.get("retrieve_document")

    def _call(tool, query: str, k: int) -> list[dict]:
        if tool is None:
            return []
        try:
            return _coerce_chunks(_run_coro(tool.ainvoke({"query": query, "k": k})))
        except Exception as exc:  # noqa: BLE001 — degrade, don't crash the loop
            logger.warning("retrieve via %s failed: %s", getattr(tool, "name", tool), exc)
            return []

    def retrieve(source: str, query: str, k: int) -> list[DataChunk]:
        raw: list[dict] = []
        if source in ("logs", "both"):
            raw += _call(log_tool, query, k)
        if source in ("documents", "both"):
            raw += _call(doc_tool, query, k)
        chunks: list[DataChunk] = []
        for r in raw:
            try:
                chunks.append(DataChunk.model_validate(r))  # the contract checkpoint
            except ValidationError as exc:
                logger.warning("dropping malformed retrieval result: %s", exc)
        return chunks

    return retrieve


# ---------------------------------------------------------------------------
# Judge + generate — LLM with deterministic fallbacks
# ---------------------------------------------------------------------------

_SUFFICIENT_THRESHOLD = 0.2


def heuristic_judge(query: str, collected: list[DataChunk]) -> RetrievalDecision:
    """No-LLM judge: sufficient if a reasonably-similar chunk exists, else reformulate.

    The reformulation (a *different* query + widened source) is what keeps the loop
    agentic when no model is available.
    """
    top = max((c.score for c in collected), default=0.0)
    if collected and top >= _SUFFICIENT_THRESHOLD:
        return RetrievalDecision(sufficient=True, reasoning=f"top score {top:.2f} ≥ {_SUFFICIENT_THRESHOLD}")
    return RetrievalDecision(
        sufficient=False,
        reasoning=f"weak match (top score {top:.2f}); broaden the search and reformulate",
        next_source="both",
        next_query=f"{query} — context details summary",
    )


def extractive_generate(query: str, collected: list[DataChunk]) -> str:
    """No-LLM answer: surface the most relevant stored chunks, grounded only in them."""
    if not collected:
        return "I couldn't find anything relevant in your logs or documents."
    top = sorted(collected, key=lambda c: c.score, reverse=True)[:3]
    lines = "\n".join(f"- {c.text}" for c in top)
    return f"Here's what I found in your memory:\n{lines}"


def llm_judge(query: str, collected: list[DataChunk], model_name: str, temperature: float = 0.0) -> RetrievalDecision:
    """Structured Ollama judge → validated :class:`RetrievalDecision` (raises on failure)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from agents.todo_agent import _loads_jsonish
    from core.services.llm_client import create_ollama_model
    from core.services.prompts import load_prompt

    schema = RetrievalDecision.model_json_schema()
    llm = create_ollama_model(model_name, temperature, format=schema, with_thinking=False)
    payload = {"query": query, "retrieved": [c.model_dump() for c in collected]}
    response = llm.invoke([
        SystemMessage(content=load_prompt("rag_judge_system")),
        HumanMessage(content=json.dumps(payload, default=str)),
    ])
    data = _loads_jsonish(response.content)
    if isinstance(data, list):
        data = data[0] if data else {}
    return RetrievalDecision.model_validate(data)


def llm_generate(query: str, collected: list[DataChunk], model_name: str, temperature: float = 0.0) -> str:
    """Grounded Ollama answer — instructed to use only the retrieved context."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from core.services.llm_client import create_ollama_model
    from core.services.prompts import load_prompt

    llm = create_ollama_model(model_name, temperature, with_thinking=False)
    context = "\n\n".join(f"[{c.source_type}] {c.text}" for c in collected) or "(nothing retrieved)"
    response = llm.invoke([
        SystemMessage(content=load_prompt("rag_generate_system")),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])
    return response.content or ""


def make_memory_writer(tools: list) -> "Callable[[str, str], None] | None":
    """Build a ``(text, source_type) -> None`` writer over the memory ``remember`` tool.

    Returns ``None`` when no memory server is connected (no ``remember`` tool), so
    callers simply skip writing — the planner still works without the memory server.
    """
    from agents.orchestrator import _run_coro

    remember = {t.name: t for t in tools}.get("remember")
    if remember is None:
        return None

    def write(text: str, source_type: str) -> None:
        try:
            _run_coro(remember.ainvoke({"text": text, "source_type": source_type}))
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning("remember(%s) failed: %s", source_type, exc)

    return write


def make_rag_agent(tools: list, model_name: str, temperature: float = 0.0, use_llm: bool = True) -> RAGAgent:
    """Wire a RAGAgent from the memory MCP tools, with LLM judge/generate + fallbacks."""
    retrieve = make_mcp_retriever(tools)
    if not use_llm:
        return RAGAgent(retrieve=retrieve, judge=heuristic_judge, generate=extractive_generate)

    def judge(query: str, collected: list[DataChunk]) -> RetrievalDecision:
        try:
            return llm_judge(query, collected, model_name, temperature)
        except Exception as exc:  # noqa: BLE001 — recoverable: fall back to heuristic
            logger.warning("LLM judge failed (%s); using heuristic.", exc)
            return heuristic_judge(query, collected)

    def generate(query: str, collected: list[DataChunk]) -> str:
        try:
            return llm_generate(query, collected, model_name, temperature) or extractive_generate(query, collected)
        except Exception as exc:  # noqa: BLE001 — recoverable: fall back to extractive
            logger.warning("LLM generate failed (%s); using extractive.", exc)
            return extractive_generate(query, collected)

    return RAGAgent(retrieve=retrieve, judge=judge, generate=generate)
