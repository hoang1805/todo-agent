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
}

# Resolved relative to the package so it works regardless of the current
# working directory.
SKILL_SOURCES = os.getenv("SKILL_SOURCES", str(_SRC_DIR / "skills"))
