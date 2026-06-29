"""Checkpointer factory — persistent (SQLite) when available, in-memory otherwise.

A LangGraph **checkpointer** stores graph state keyed by ``thread_id``. That is
what makes two things work: human-in-the-loop **interrupt/resume** (the pending
approval is saved, then resumed with the user's decision) and multi-turn
**conversation threads**. The in-memory saver loses everything on restart; the
SQLite saver writes to disk, so a pending approval can be resumed even after the
process is restarted.

:func:`make_checkpointer_opener` returns a **zero-argument callable** that opens
an *async context manager* yielding a saver. The graph is compiled with that
saver inside the same coroutine that invokes it, so the async SQLite connection
lives and dies on one event loop — sidestepping the classic "connection bound to
a closed loop" error when each call spins its own loop via ``asyncio.run``.

* **SQLite path** — each call opens a fresh connection; the state lives on disk,
  so it is shared across calls *and* across process restarts.
* **In-memory path** — every call yields the **same** saver instance, so
  interrupt state still survives between ``start()`` and ``resume()`` within one
  process (a fresh instance per call would forget the pending interrupt).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncContextManager, Callable

from langgraph.checkpoint.memory import InMemorySaver

try:  # optional dependency — degrade gracefully if not installed
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except Exception:  # noqa: BLE001 — any import failure means "no sqlite saver"
    AsyncSqliteSaver = None  # type: ignore[assignment]


def sqlite_available() -> bool:
    """True if the persistent SQLite checkpointer can be used."""
    return AsyncSqliteSaver is not None


def make_checkpointer_opener(db_path: str | None = None) -> Callable[[], AsyncContextManager]:
    """Return a callable that opens a checkpointer async-context-manager.

    Args:
        db_path: Path to the SQLite file. When set and the sqlite saver is
            installed, state is persisted there; otherwise an in-memory saver is
            used (a single shared instance, so resume still works in-process).
    """
    if db_path and AsyncSqliteSaver is not None:

        @asynccontextmanager
        async def open_sqlite():
            async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
                await saver.setup()  # idempotent: CREATE TABLE IF NOT EXISTS
                yield saver

        return open_sqlite

    shared = InMemorySaver()

    @asynccontextmanager
    async def open_memory():
        yield shared

    return open_memory
