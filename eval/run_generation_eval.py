"""Generation (end-to-end) eval — LLM-as-judge over real RAG answers.

"Is this a good answer" isn't a string match, so a case is a question plus a
**rubric**. The runner asks the real RAGAgent the question, then makes a *separate*
LLM call — the judge — with the question, rubric, and actual answer, for a
pass/fail. This is what catches prompt regressions: change a system prompt, re-run
this file, see immediately if quality dropped.

Needs the memory MCP server *and* Ollama (both the answering model and the judge).
"""

from __future__ import annotations

import json
import os

from report import SAMPLE_DOC, Result, Summary, call_tool, mcp_tools, read_jsonl, summarize

_JUDGE_SYSTEM = (
    "You grade whether an ANSWER satisfies a RUBRIC for a given QUESTION. "
    "Respond with JSON only: {\"pass\": true|false, \"reason\": \"...\"}. "
    "Pass only if the answer's substance meets the rubric and is not contradicted."
)


def _seed() -> None:
    call_tool("remember", {"text": SAMPLE_DOC, "source_type": "document", "metadata": {"tag": "eval_generation"}})


def _judge(model: str, question: str, rubric: str, answer: str) -> tuple[bool, str]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from core.services.llm_client import create_ollama_model

    llm = create_ollama_model(model, temperature=0.0, with_thinking=False)
    resp = llm.invoke([
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=f"QUESTION: {question}\n\nRUBRIC: {rubric}\n\nANSWER: {answer}"),
    ])
    content = (resp.content or "").strip()
    start, end = content.find("{"), content.rfind("}")
    try:
        verdict = json.loads(content[start:end + 1])
        return bool(verdict.get("pass")), str(verdict.get("reason", ""))[:80]
    except (ValueError, TypeError):
        return False, f"unparseable judge reply: {content[:60]}"


def run() -> list[Summary]:
    from configs.settings import MODEL
    from agents.rag_agent import make_rag_agent

    _seed()
    answer_model = os.getenv("OLLAMA_MODEL", MODEL)
    judge_model = os.getenv("EVAL_JUDGE_MODEL", answer_model)
    agent = make_rag_agent(list(mcp_tools().values()), answer_model, temperature=0.0, use_llm=True)

    cases = read_jsonl("generation_cases.jsonl")
    results = []
    for c in cases:
        answer = agent.run(c["query"])
        ok, reason = _judge(judge_model, c["query"], c["rubric"], answer)
        results.append(Result(c["query"][:48], ok, reason))
    return [summarize("generation", results, note=f"answer={answer_model}, judge={judge_model}")]


def main() -> None:
    run()


if __name__ == "__main__":
    main()
