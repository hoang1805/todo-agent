"""Persisted multiturn chat history — the orchestrator's canonical conversation log.

A real chat app survives a restart, so history lives in **SQLite**, not an
in-memory list. The orchestrator owns it (the one component that sees every turn,
whichever agent handles it).

Schema (designed here; reasoning below):

    sessions(id TEXT PRIMARY KEY, title TEXT, created_at TEXT)
    turns(id INTEGER PK AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, ts TEXT)

Two tables: one **session** row per conversation so they can be listed and resumed,
and an ordered **turn** log keyed by ``session_id`` (autoincrement ``id`` gives a
reliable order without depending on timestamp resolution). Reads use a **sliding
window** (:meth:`get_recent_turns`) so only the last few turns are injected per call.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone

from configs.settings import HISTORY_DB

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class History:
    def __init__(self, path: str = HISTORY_DB):
        self.path = path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS sessions"
                      "(id TEXT PRIMARY KEY, title TEXT, created_at TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS turns"
                      "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
                      "role TEXT, content TEXT, ts TEXT)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id)")

    def ensure_session(self, session_id: str, title: str | None = None) -> str:
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO sessions(id, title, created_at) VALUES (?, ?, ?)",
                (session_id, title or "New conversation", _now()),
            )
        return session_id

    def create_session(self, title: str | None = None) -> str:
        return self.ensure_session(str(uuid.uuid4()), title)

    def list_sessions(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        self.ensure_session(session_id)
        with self._conn() as c:
            c.execute(
                "INSERT INTO turns(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, role, content, _now()),
            )

    def get_recent_turns(self, session_id: str, limit: int = 6) -> list[dict]:
        """The last *limit* turns, oldest-first (the bounded window injected per call)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content, ts FROM turns WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_all_turns(self, session_id: str) -> list[dict]:
        """Every turn in order (used by the UI to replay a resumed conversation)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content, ts FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]


_history: History | None = None


def get_history() -> History:
    """Process-wide history singleton (lazy)."""
    global _history
    if _history is None:
        _history = History()
    return _history


def reset_history() -> None:
    """Drop the cached history (tests, after pointing HISTORY_DB elsewhere)."""
    global _history
    _history = None
