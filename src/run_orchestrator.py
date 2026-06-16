"""Offline demo runner for the multi-agent daily planner.

Runs the full orchestrator pipeline — TodoAgent → contract → DailyPlannerAgent —
without needing Ollama or the MCP server, by using sample tasks and the
deterministic heuristic normalizer. This is enough to demonstrate the graded
behaviour, including the overload decision.

Examples::

    python src/run_orchestrator.py "plan my day"
    python src/run_orchestrator.py "what's on my list?"
    python src/run_orchestrator.py "plan my day" --day-end 12:00   # overloaded
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the package importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.orchestrator import Orchestrator  # noqa: E402
from agents.planner_agent import DailyPlannerAgent, Workday  # noqa: E402
from agents.todo_agent import TodoAgent, heuristic_normalize, sample_raw_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent daily planner demo")
    parser.add_argument("prompt", nargs="?", default="plan my day")
    parser.add_argument("--day-start", default="09:00", help="Workday start (HH:MM).")
    parser.add_argument(
        "--day-end",
        default="20:00",
        help="Workday end (HH:MM). Shorten it (e.g. 12:00) to force an overload.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show agent logs.")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    # Wire the agents. In production, swap the fetcher for an MCP-backed one and
    # the normalizer for `llm_normalize` — the orchestrator code is unchanged.
    todo_agent = TodoAgent(
        fetch_raw=sample_raw_tasks,
        normalize=heuristic_normalize,
    )
    workday = Workday(start=args.day_start, end=args.day_end)
    planner_agent = DailyPlannerAgent(workday=workday)
    orchestrator = Orchestrator(todo_agent, planner_agent)

    print(orchestrator.run(args.prompt))


if __name__ == "__main__":
    main()
