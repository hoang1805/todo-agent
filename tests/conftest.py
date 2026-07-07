"""Pytest config — put ``src/`` on the import path so tests import like the app.

Also point the persisted history at a throwaway DB (so tests never touch the
repo's ``chat-history.db``) and reset the singleton before each test.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ["HISTORY_DB"] = str(Path(tempfile.mkdtemp(prefix="todo-history-")) / "history.db")

# Observability writes to its own SQLite store; point it at a throwaway file so
# tests never touch the repo's events.db.
os.environ["EVENTS_DB"] = str(Path(tempfile.mkdtemp(prefix="todo-events-")) / "events.db")

# Tests exercise the deterministic guardrail layer; the LLM screen is opt-in and
# covered by dedicated tests that fake the model (never a live Ollama call).
os.environ["GUARDRAIL_USE_LLM"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest


@pytest.fixture(autouse=True)
def _fresh_history():
    """Each test starts with an empty history + events DB and fresh singletons."""
    from core.services.history import reset_history
    from core.services.observability import reset_observer

    for db in (os.environ["HISTORY_DB"], os.environ["EVENTS_DB"]):
        if os.path.exists(db):
            os.remove(db)
    reset_history()
    reset_observer()
    yield
    reset_history()
    reset_observer()
