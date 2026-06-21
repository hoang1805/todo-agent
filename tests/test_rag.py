"""RAGAgent — the agentic retrieve→judge→generate loop (no LLM / MCP)."""

from agents.rag_agent import RAGAgent, make_mcp_retriever
from models.rag import DataChunk, RetrievalDecision


def _chunk(text, score=0.9, src="document"):
    return DataChunk(text=text, score=score, source_type=src)


def test_generates_when_first_retrieval_is_sufficient():
    calls = []

    def retrieve(source, query, k):
        calls.append((source, query))
        return [_chunk("the answer is 42")]

    agent = RAGAgent(
        retrieve=retrieve,
        judge=lambda q, c: RetrievalDecision(sufficient=True, reasoning="ok"),
        generate=lambda q, c: f"answer: {c[0].text}",
    )
    out = agent.run("what is the answer?")
    assert "42" in out
    assert len(calls) == 1  # one retrieve pass, no reformulation


def test_weak_first_retrieval_triggers_a_reformulated_second_query():
    queries = []

    def retrieve(source, query, k):
        queries.append(query)
        return [_chunk("vague", score=0.02)] if len(queries) == 1 else [_chunk("the real answer", score=0.9)]

    decisions = iter([
        RetrievalDecision(sufficient=False, reasoning="weak", next_query="reformulated query", next_source="logs"),
        RetrievalDecision(sufficient=True, reasoning="good"),
    ])
    agent = RAGAgent(retrieve, lambda q, c: next(decisions), lambda q, c: "; ".join(x.text for x in c))
    out = agent.run("original query")

    assert queries[0] == "original query"
    assert queries[1] == "reformulated query"   # visibly different — the agentic part
    assert "real answer" in out


def test_source_selection_follows_the_judge():
    sources = []
    decisions = iter([
        RetrievalDecision(sufficient=False, reasoning="logs", next_source="logs", next_query="q2"),
        RetrievalDecision(sufficient=False, reasoning="docs", next_source="documents", next_query="q3"),
        RetrievalDecision(sufficient=True, reasoning="done"),
    ])

    def retrieve(source, query, k):
        sources.append(source)
        return [_chunk("x", score=0.0)]

    RAGAgent(retrieve, lambda q, c: next(decisions), lambda q, c: "ok").run("q1")
    assert sources == ["both", "logs", "documents"]  # first 'both', then judge-directed


def test_loop_terminates_at_max_iterations_and_answers_honestly():
    gen_calls = {"n": 0}

    def generate(q, c):
        gen_calls["n"] += 1
        return "best effort from what I have"

    agent = RAGAgent(
        retrieve=lambda s, q, k: [_chunk("nothing useful", score=0.0)],
        judge=lambda q, c: RetrievalDecision(sufficient=False, reasoning="never", next_query="more"),
        generate=generate,
    )
    out = agent.run("unanswerable")
    assert out == "best effort from what I have"
    assert gen_calls["n"] == 1  # generated once, after the cap — not looping forever


def test_retriever_validates_results_and_drops_malformed():
    class _Tool:
        name = "retrieve_document"

        async def ainvoke(self, args):
            return [
                {"text": "good", "score": 0.5, "source_type": "document", "metadata": {}},
                {"text": "bad", "score": "NOT_A_NUMBER", "source_type": "document"},
            ]

    chunks = make_mcp_retriever([_Tool()])("documents", "q", 3)
    assert len(chunks) == 1 and chunks[0].text == "good"  # malformed dropped at the contract
