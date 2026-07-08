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

    def __init__(self, retrieve: Retriever, judge: Judge, generate: Generator, k: int = 6):
        self._retrieve = retrieve
        self._judge = judge
        self._generate = generate
        self.k = k

    def run(self, query: str, on_step: "Callable[[str], None] | None" = None) -> str:
        """Run the loop. *on_step* (optional) receives a human-readable line for
        each phase — retrieve, judge, reformulate — so a UI can show the agent's
        progress step by step instead of a single opaque wait."""
        def step(msg: str) -> None:
            if on_step is not None:
                on_step(msg)

        collected: list[DataChunk] = []   # accumulate — never discarded between passes
        current_query, source = query, "both"
        where = {"both": "logs & documents", "logs": "planning logs", "documents": "documents"}
        reformulations = 0

        for i in range(self.MAX_ITERATIONS):
            step(f"🔎 Searching {where.get(source, source)} (pass {i + 1}/{self.MAX_ITERATIONS}) — “{current_query}”")
            chunks = self._retrieve(source, current_query, self.k)
            seen = {c.text for c in collected}
            collected.extend(c for c in chunks if c.text not in seen)
            step(f"📚 Got {len(chunks)} result(s); {len(collected)} gathered so far")
            if _rag_debug():
                _debug_chunks(chunks, step)

            decision = self._judge(query, collected)
            logger.info(
                "[rag] iter=%d query=%r source=%s got=%d sufficient=%s next_query=%r",
                i, current_query, source, len(chunks), decision.sufficient, decision.next_query,
            )
            # One event per loop iteration — the persisted trace the agentic-loop
            # eval inspects to confirm the agent actually reformulated and retried.
            _log_rag("iteration", details={
                "pass": i + 1, "source": source, "query": current_query,
                "got": len(chunks), "gathered": len(collected), "sufficient": decision.sufficient,
            })
            # Debug trace: the judge's verdict and where it would look next.
            _debug_rag(
                f"pass {i + 1}: judged sufficient={decision.sufficient}",
                query=current_query, source=source, got=len(chunks),
                next_query=decision.next_query, next_source=decision.next_source,
            )
            if decision.sufficient:
                step("✅ Enough to answer — composing a grounded response")
                _log_rag("query", details={"iterations": i + 1, "reformulations": reformulations,
                                           "chunks": len(collected), "outcome": "sufficient"})
                return self._finish(query, collected)

            new_query = decision.next_query or current_query
            if new_query != current_query:
                reformulations += 1
            current_query = new_query
            source = decision.next_source or source
            if i < self.MAX_ITERATIONS - 1:
                step(f"↻ Not enough — reformulating to “{current_query}”")

        # Out of iterations — answer honestly from whatever was collected.
        logger.info("[rag] stopped after %d iterations (%d chunks)", self.MAX_ITERATIONS, len(collected))
        _log_rag("query", details={"iterations": self.MAX_ITERATIONS, "reformulations": reformulations,
                                   "chunks": len(collected),
                                   "outcome": "exhausted" if collected else "empty"})
        if collected:
            step("⏱️ Out of retries — answering from what I found (with a caveat)")
            return self._finish(query, collected)
        step("🚫 Nothing relevant found in your logs or documents")
        return "I couldn't find anything relevant in your logs or documents."

    def _finish(self, query: str, collected: list[DataChunk]) -> str:
        """Generate the answer, then verify it's grounded in the context (output guardrail)."""
        from core.services.guardrails import check_output

        answer = self._generate(query, collected)
        context = "\n".join(text for _, text in _generation_texts(collected))
        return check_output(answer, context)


def _log_rag(event_type: str, details: dict) -> None:
    """Record a RAG-loop event to the shared observability store (best-effort)."""
    try:
        from core.services.observability import log_event

        log_event("rag_agent", event_type, details=details)
    except Exception:  # noqa: BLE001 — logging must never break the loop
        pass


def _debug_rag(message: str, **details) -> None:
    """Fine-grained RAG debug trace (only persisted when AGENT_DEBUG is on)."""
    try:
        from core.services.observability import log_debug

        log_debug("rag_agent", message, **details)
    except Exception:  # noqa: BLE001 — debug logging must never break the loop
        pass


def _rag_debug() -> bool:
    try:
        from configs import settings

        return bool(settings.RAG_DEBUG)
    except Exception:  # noqa: BLE001 — no settings (bare unit test) → off
        return False


def _debug_chunks(chunks: "list[DataChunk]", step: "Callable[[str], None]") -> None:
    """RAG_DEBUG: emit every retrieved chunk's FULL text (and its parent context)
    into the step trace, instead of just the result count."""
    for c in chunks:
        step(f"🧩 `[{c.source_type} · score {c.score:.2f}]` {c.text}")
        parent = c.metadata.get("parent_text")
        if parent and parent != c.text:
            step(f"🪆 parent context → {parent}")


# ---------------------------------------------------------------------------
# Retrieval — drive the split memory tools and validate every result
# ---------------------------------------------------------------------------


def _unwrap_chunk(item: object) -> list[dict]:
    """Turn one result item into zero or more chunk dicts.

    Across the MCP boundary each chunk arrives as a text-content block —
    ``{"type": "text", "text": "<json of the chunk>", "id": ...}`` — so the real
    ``DataChunk``-shaped dict is JSON-encoded inside ``text``. Unwrap that; pass
    through items that are already chunk-shaped.
    """
    if not isinstance(item, dict):
        return []
    if "source_type" in item and "score" in item:  # already a chunk
        return [item]
    text = item.get("text")  # MCP text-content block: the chunk JSON lives here
    if isinstance(text, str):
        try:
            inner = json.loads(text)
        except (ValueError, TypeError):
            return []
        if isinstance(inner, dict):
            return [inner]
        if isinstance(inner, list):
            return [r for r in inner if isinstance(r, dict)]
    return []


def _coerce_chunks(result: object) -> list[dict]:
    """Turn a memory-tool result (list / JSON string / error / MCP blocks) into chunk dicts."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            return []
    if isinstance(result, dict):
        if result.get("ok") is False:  # structured recoverable error
            return []
        result = result.get("results") or result.get("data") or []
    if not isinstance(result, list):
        return []
    chunks: list[dict] = []
    for item in result:
        chunks.extend(_unwrap_chunk(item))
    return chunks


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
            from core.services.observability import timed_tool_call

            return _coerce_chunks(timed_tool_call(tool, {"query": query, "k": k}, _run_coro))
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


def _generation_texts(collected: list[DataChunk]) -> list[tuple[str, str]]:
    """``(source_type, text)`` for generation, ranked by score.

    For parent-child chunks the child matched but generation should see the whole
    parent, so prefer ``metadata['parent_text']``; de-duplicate so several children
    of one parent collapse to that parent once.
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for c in sorted(collected, key=lambda c: c.score, reverse=True):
        text = c.metadata.get("parent_text") or c.text
        if text in seen:
            continue
        seen.add(text)
        out.append((c.source_type, text))
    return out


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
    lines = "\n".join(f"- {text}" for _, text in _generation_texts(collected)[:3])
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

    from core.services.guardrails import wrap_context

    llm = create_ollama_model(model_name, temperature, with_thinking=False)
    # Injection defense: fence retrieved data in delimited tags; the system prompt
    # tells the model content inside them is data to reference, not instructions.
    blocks = [f"[{src}] {text}" for src, text in _generation_texts(collected)]
    context = wrap_context(blocks) if blocks else "(nothing retrieved)"
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
            from core.services.observability import timed_tool_call

            timed_tool_call(remember, {"text": text, "source_type": source_type}, _run_coro)
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning("remember(%s) failed: %s", source_type, exc)

    return write


def _coerce_remember_result(result: object) -> dict:
    """Normalize a ``remember`` tool result to its ``{"ok": ..., ...}`` dict.

    Like retrieval results, the dict may cross the MCP boundary as a JSON string
    or wrapped in a text-content block — unwrap both; anything unrecognizable is
    a structured failure, never an exception.
    """
    if isinstance(result, list) and result:  # MCP content blocks
        first = result[0]
        result = first.get("text") if isinstance(first, dict) else first
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            return {"ok": False, "error": f"unexpected remember result: {result!r:.80}"}
    if isinstance(result, dict):
        return result
    return {"ok": False, "error": f"unexpected remember result type: {type(result).__name__}"}


def make_document_ingestor(tools: list) -> "Callable[[str, dict | None], dict] | None":
    """Build a ``(text, metadata) -> result`` document ingestor over ``remember``.

    The heavy lifting (adaptive chunking, embedding, storage) happens server-side
    in the memory MCP; this just delivers the text and reports the outcome —
    ``{"ok": True, "ids": [...], "strategy": ...}`` or a recoverable error dict.
    Returns ``None`` when no memory server is connected, so the UI can hide the
    upload panel instead of offering a dead control.
    """
    from agents.orchestrator import _run_coro

    remember = {t.name: t for t in tools}.get("remember")
    if remember is None:
        return None

    def ingest(text: str, metadata: dict | None = None) -> dict:
        try:
            from core.services.observability import timed_tool_call

            raw = timed_tool_call(
                remember, {"text": text, "source_type": "document", "metadata": metadata or {}}, _run_coro
            )
            return _coerce_remember_result(raw)
        except Exception as exc:  # noqa: BLE001 — surface a recoverable error
            logger.warning("document ingest failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    return ingest


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
