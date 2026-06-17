"""Visualize the multi-agent planner graph, and optionally trace a run.

Examples::

    # Print the Mermaid diagram and save planner_graph.mmd (+ .png if possible)
    python src/visualize_graph.py

    # Also run a prompt with the step/state observer (offline; no LLM/MCP)
    python src/visualize_graph.py "plan my day"
    python src/visualize_graph.py "plan my day" --day-end 12:00   # force an overload

The Mermaid text can be pasted into https://mermaid.live to see the diagram.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the package importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.graph_orchestrator import build_graph_orchestrator  # noqa: E402
from agents.planner_agent import Workday  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize / trace the planner graph")
    parser.add_argument("prompt", nargs="?", help="If given, run it with the step observer.")
    parser.add_argument("--day-start", default="09:00")
    parser.add_argument("--day-end", default="20:00")
    parser.add_argument("--out", default="planner_graph", help="Output file stem.")
    args = parser.parse_args()

    workday = Workday(start=args.day_start, end=args.day_end)
    # Offline build: no MCP tools, no LLM, observer on only when we run a prompt.
    go = build_graph_orchestrator(
        tools=[], model_name="dummy", workday=workday,
        use_llm=False, observe=bool(args.prompt),
    )

    # -- 1. Visualize --------------------------------------------------------
    print("=== Planner graph (Mermaid) — paste into https://mermaid.live ===\n")
    print(go.draw_mermaid())
    written = go.save_visualization(args.out)
    print("\nSaved: " + ", ".join(f"{k}={v}" for k, v in written.items()))

    try:
        print("\n=== ASCII ===\n")
        print(go.draw_ascii())
    except Exception as exc:  # noqa: BLE001 — needs the optional `grandalf` extra
        print(f"(ASCII skipped: {exc} — `pip install grandalf` to enable)")

    # -- 2. Trace a run (optional) ------------------------------------------
    if args.prompt:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        print(f"\n=== Tracing run: {args.prompt!r} ===\n")
        result = go.run(args.prompt, date="2026-06-17")
        print("\n=== RESULT ===\n" + result)


if __name__ == "__main__":
    main()
