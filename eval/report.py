"""Shared eval plumbing — load cases, score a run, persist it, print it.

The five runners each exercise a different slice of the *real* system and then
hand their per-case results here. This module owns the parts they'd otherwise
each reinvent: reading a JSONL dataset, turning results into a score, printing a
table, and persisting the run into the same observability store the dashboard
reads (so eval scores trend right next to live traffic).

Nothing here runs on the request path — eval is a manual, on-demand regression
check, run like a test suite (``python eval/run_all.py``).
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# A unique tag per eval process, so a run's seeded chunks are isolated from prior
# runs' (retrieval/chunking re-seed the fixture each time; without isolation the
# accumulated duplicates would crowd top-k and make the score depend on history).
RUN_TAG = uuid.uuid4().hex[:8]

# Make the app package importable (eval/ sits beside src/).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DATASETS = Path(__file__).resolve().parent / "datasets"

# A small fixture document shared by the retrieval / chunking / generation evals:
# distinctive, self-contained facts so a query's expected chunk is unambiguous.
SAMPLE_DOC = """# Acme Corp Handbook

## Products
Acme's flagship product is the Zephyr wireless router, designed for small offices.
The Zephyr supports up to 40 simultaneous devices and a mesh of three units.

## Support
The customer support hotline operates from 8am to 6pm Pacific time on weekdays.
Premium customers get a dedicated account manager and a four-hour response SLA.

## Expenses
Expense reports must be filed within 30 days of a purchase or the claim is void.
Meals while travelling are reimbursed up to 60 US dollars per day; alcohol is not.

## Security
The annual security audit is conducted every October by an external firm.
Every employee must rotate their account password at least once every 90 days.

## Benefits
Parental leave is 16 weeks fully paid for all full-time employees.
The company matches retirement contributions up to 6 percent of salary.
"""


@dataclass
class Result:
    """One case's outcome."""

    name: str
    passed: bool
    info: str = ""


@dataclass
class Summary:
    eval_type: str
    passed: int
    total: int
    note: str = ""
    results: list[Result] = field(default_factory=list)

    @property
    def score(self) -> float:
        return (self.passed / self.total) if self.total else 0.0


def read_jsonl(name: str) -> list[dict]:
    """Load ``eval/datasets/<name>`` (one JSON object per non-blank line)."""
    path = _DATASETS / name
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


# ---------------------------------------------------------------------------
# Connecting to the real system (MCP tools)
# ---------------------------------------------------------------------------

_TOOLS: dict | None = None


def mcp_tools() -> dict:
    """Connect to the configured MCP servers once; return ``{tool_name: tool}``."""
    global _TOOLS
    if _TOOLS is None:
        from configs.settings import MCP_SERVERS
        from core.services.mcp_client import get_mcp_tools

        tools = asyncio.run(get_mcp_tools(MCP_SERVERS))
        _TOOLS = {t.name: t for t in tools}
    return _TOOLS


def call_tool(name: str, args: dict):
    """Invoke one MCP tool synchronously (raises KeyError if the server is down)."""
    tools = mcp_tools()
    if name not in tools:
        raise KeyError(f"MCP tool {name!r} not available — is the memory/task server running?")
    return asyncio.run(tools[name].ainvoke(args))


# ---------------------------------------------------------------------------
# Scoring / persistence
# ---------------------------------------------------------------------------


def summarize(eval_type: str, results: list[Result], note: str = "", *, persist: bool = True) -> Summary:
    """Print a per-case table, persist the run, and return the summary.

    Persisting writes to the observability store's ``eval_runs`` table via
    :func:`core.services.observability.record_eval_run`, so the dashboard can chart
    the score over time next to live traffic.
    """
    passed = sum(1 for r in results if r.passed)
    summary = Summary(eval_type, passed, len(results), note, results)

    print(f"\n=== {eval_type} eval ===")
    for r in results:
        mark = "✅" if r.passed else "❌"
        print(f"  {mark} {r.name}" + (f"  — {r.info}" if r.info else ""))
    print(f"  score: {passed}/{len(results)} = {summary.score * 100:.1f}%" + (f"   ({note})" if note else ""))

    if persist:
        try:
            from core.services.observability import record_eval_run

            record_eval_run(eval_type, summary.score, passed, len(results), note,
                            details={"cases": [{"name": r.name, "passed": r.passed} for r in results]})
        except Exception as exc:  # noqa: BLE001 — a store hiccup shouldn't fail the eval
            print(f"  (warning: could not persist eval run: {exc})")

    return summary
