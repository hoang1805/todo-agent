"""Generation (end-to-end) eval — LLM-as-judge over real RAG answers.

"Is this a good answer" isn't a string match, so a case is a question plus a set of
**criteria** a good answer must meet. The runner asks the real RAGAgent the
question, then makes a *separate* LLM call — the judge — with the question, the
criteria, and the actual answer, and grades **each criterion** for a pass/fail.
This is what catches prompt regressions: change a system prompt, re-run this file,
see immediately (and per-criterion) if quality dropped.

A case may give a ``criteria`` list (each graded independently — the case passes
only if all are met) or a single legacy ``rubric`` string.

Needs the memory MCP server *and* Ollama (both the answering model and the judge).
"""

from __future__ import annotations

import json
import os

from report import SAMPLE_DOC, Result, Summary, call_tool, mcp_tools, read_jsonl, summarize

_JUDGE_SYSTEM = (
    "You grade whether an ANSWER satisfies each CRITERION for a given QUESTION. "
    "Judge each criterion independently. Respond with JSON only: "
    "{\"criteria\": [{\"criterion\": \"<verbatim>\", \"met\": true|false}], \"reason\": \"...\"}. "
    "Mark a criterion met only if the answer's substance satisfies it and is not contradicted."
)


def _seed() -> None:
    call_tool("remember", {"text": SAMPLE_DOC, "source_type": "document", "metadata": {"tag": "eval_generation"}})


def _criteria_for(case: dict) -> list[str]:
    """A case may list explicit ``criteria`` or a single legacy ``rubric`` string."""
    if case.get("criteria"):
        return [str(c) for c in case["criteria"]]
    return [case["rubric"]]


def _judge(model: str, question: str, criteria: list[str], answer: str) -> tuple[bool, str, list[dict]]:
    """LLM-as-judge over a *list* of criteria — the case passes only if every
    criterion is met. Returns ``(passed, reason, per-criterion breakdown)``."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from core.services.llm_client import create_ollama_model

    llm = create_ollama_model(model, temperature=0.0, with_thinking=False)
    criteria_block = "\n".join(f"- {c}" for c in criteria)
    resp = llm.invoke([
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=f"QUESTION: {question}\n\nCRITERIA:\n{criteria_block}\n\nANSWER: {answer}"),
    ])
    content = (resp.content or "").strip()
    start, end = content.find("{"), content.rfind("}")
    try:
        verdict = json.loads(content[start:end + 1])
    except (ValueError, TypeError):
        # Unparseable → fail every criterion (visible in the breakdown, not silent).
        return False, f"unparseable judge reply: {content[:60]}", [
            {"criterion": c, "met": False} for c in criteria
        ]

    graded = verdict.get("criteria") or []
    # Align by position so a terse judge reply still maps onto our criteria list.
    breakdown = [
        {"criterion": criteria[i], "met": bool(graded[i].get("met")) if i < len(graded) else False}
        for i in range(len(criteria))
    ]
    passed = bool(breakdown) and all(cr["met"] for cr in breakdown)
    return passed, str(verdict.get("reason", ""))[:80], breakdown


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
        criteria = _criteria_for(c)
        ok, reason, breakdown = _judge(judge_model, c["query"], criteria, answer)
        results.append(Result(c["query"][:48], ok, reason, criteria=breakdown))
    return [summarize("generation", results, note=f"answer={answer_model}, judge={judge_model}")]


def main() -> None:
    run()


if __name__ == "__main__":
    main()
