"""Application configuration.

Values default to sensible local settings and can be overridden via environment
variables (a ``.env`` file at the project root is loaded automatically).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ``settings.py`` lives at ``src/configs/`` — ``src/`` is two levels up.
_SRC_DIR = Path(__file__).resolve().parent.parent

MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

APP_TITLE = "To-do Agent"

DATA_SOURCE = os.getenv("DATA_SOURCE", "data")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

MCP_SERVERS = {
    "task_mcp": os.getenv("TASK_MCP_URL", "http://localhost:8000/sse"),
    "memory_mcp": os.getenv("MEMORY_MCP_URL", "http://localhost:8002/sse"),
}

# Resolved relative to the package so it works regardless of the current
# working directory.
SKILL_SOURCES = os.getenv("SKILL_SOURCES", str(_SRC_DIR / "skills"))

# Persistent LangGraph checkpointer (SQLite). Stored at the project root so a
# pending human-in-the-loop approval can be resumed across restarts. Set
# ``CHECKPOINT_DB=""`` to disable persistence (falls back to an in-memory saver).
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", str(_SRC_DIR.parent / "agent-checkpoints.db"))

# When set (e.g. ``PLANNER_TRACE=1 streamlit run src/app.py``), the multi-agent
# planner logs each graph node and the state it produces, and the app raises the
# agent loggers to INFO so the trace appears in the terminal.
TRACE = bool(os.getenv("PLANNER_TRACE"))

# Persisted, listable multiturn chat history (SQLite), owned by the orchestrator.
# Survives restarts; separate from the LangGraph checkpointer (interrupt/resume).
HISTORY_DB = os.getenv("HISTORY_DB", str(_SRC_DIR.parent / "chat-history.db"))

# Observability: the shared structured-event + eval-run store (SQLite). Every
# component logs through core.services.observability into this one file; the
# dashboard reads it. Its own volume in the container setup.
EVENTS_DB = os.getenv("EVENTS_DB", str(_SRC_DIR.parent / "events.db"))

# LLM-enhanced guardrails: a screening model reviews user input and verifies that
# RAG answers are grounded, on top of (never instead of) the deterministic checks
# in guardrails.py. Any model failure falls back to the deterministic layer, so
# switching this off — or Ollama being down — never blocks the app.
GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL", "ministral-3:14b-cloud")
GUARDRAIL_USE_LLM = os.getenv("GUARDRAIL_USE_LLM", "1").lower() in ("1", "true", "yes")

# Debug: emit the FULL text of every retrieved chunk (and its parent context)
# into the RAG step trace — visible in the UI steps box and, with PLANNER_TRACE,
# the terminal. Off by default: it makes traces long.
RAG_DEBUG = bool(os.getenv("RAG_DEBUG"))

# Debug: when set (``AGENT_DEBUG=1``), agents persist fine-grained "debug" events
# (the classifier's raw decision, each RAG reformulation + judge verdict, the
# planner's chosen workday) into the events store, surfaced in the dashboard's
# Debug trace panel. Off by default; read live so it can be toggled per run.
AGENT_DEBUG = bool(os.getenv("AGENT_DEBUG"))
