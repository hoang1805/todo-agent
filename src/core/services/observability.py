"""Shared observability — one logging interface, one persistent store.

Every reliability checkpoint in the app (routing, tool calls, guardrail checks,
the RAG loop, the planner's overload branch) records an **event** through the one
:func:`log_event` function, into one SQLite store. This is the week-4 ``guardrails.py``
pattern applied to logging: a single shared module called from many sites, instead
of each component inventing its own ad-hoc format.

Why SQLite (same reasoning as the chat-history store): the app is local and
self-hosted, events must survive a restart *and* a container teardown, and the
dashboard needs to *query* them (by time range, component, event type) — exactly
what a table with indexed columns gives, without a server to run.

Schema (designed here):

    events(id, ts, component, event_type, session_id, ok, latency_ms, details)
    eval_runs(id, ts, eval_type, score, passed, total, note, details)

Every event shares the same **shape** — a timestamp, the *component* that emitted
it, an *event_type*, the owning *session_id*, a success flag, an optional latency,
and a free-form JSON *details* blob for the fields specific to that event type.
The consistent top-level columns are what the dashboard groups/filters on; the
JSON keeps each call site free to attach what's relevant without a schema change.
``eval_runs`` is a second table (week-5 Part 2): each evaluation run's score,
persisted so the dashboard can chart quality over time.

All writes are **best-effort**: a logging failure must never break a real request,
so every public function swallows its own errors.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from configs.settings import EVENTS_DB

logger = logging.getLogger(__name__)

# The session (conversation) a deep call site belongs to, without threading it
# through every signature: the orchestrator sets it once per turn, and guardrails
# / tool wrappers / the RAG loop read it from here.
_session: contextvars.ContextVar[str | None] = contextvars.ContextVar("obs_session", default=None)


def set_session(session_id: str | None) -> None:
    """Bind the current session id for events logged on this call stack."""
    _session.set(session_id)


def current_session() -> str | None:
    return _session.get()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Observer:
    """Owns the events store: one writer, cheap indexed reads for the dashboard."""

    def __init__(self, path: str = EVENTS_DB):
        self.path = path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        # WAL: the dashboard reads while the app writes (separate containers on a
        # shared volume in the deployed setup) — WAL lets readers not block writers.
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init(self) -> None:
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS events("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, component TEXT, "
                "event_type TEXT, session_id TEXT, ok INTEGER, latency_ms REAL, details TEXT)"
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_events_component ON events(component, event_type)")
            c.execute(
                "CREATE TABLE IF NOT EXISTS eval_runs("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, eval_type TEXT, "
                "score REAL, passed INTEGER, total INTEGER, note TEXT, details TEXT)"
            )

    # -- writes -------------------------------------------------------------

    def log(
        self, component: str, event_type: str, *,
        session_id: str | None = None, ok: bool = True,
        latency_ms: float | None = None, details: dict | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO events(ts, component, event_type, session_id, ok, latency_ms, details) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_now(), component, event_type, session_id, 1 if ok else 0,
                 latency_ms, json.dumps(details or {}, default=str)),
            )

    def record_eval(
        self, eval_type: str, score: float, passed: int, total: int,
        note: str = "", details: dict | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO eval_runs(ts, eval_type, score, passed, total, note, details) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_now(), eval_type, score, passed, total, note,
                 json.dumps(details or {}, default=str)),
            )

    # -- reads (the dashboard only calls these; no logic lives in the UI) ----

    def recent_events(self, limit: int = 200) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, component, event_type, session_id, ok, latency_ms, details "
                "FROM events ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def counts_by_component(self, components: tuple[str, ...] | None = None) -> list[dict]:
        q = "SELECT component, COUNT(*) n FROM events"
        args: tuple = ()
        if components:
            q += f" WHERE component IN ({','.join('?' * len(components))})"
            args = components
        q += " GROUP BY component ORDER BY n DESC"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def requests_over_time(self, bucket: str = "minute") -> list[dict]:
        """Count ``orchestrator/request`` events per time bucket (for the trend)."""
        fmt = {"minute": "%Y-%m-%d %H:%M", "hour": "%Y-%m-%d %H:00", "day": "%Y-%m-%d"}.get(bucket, "%Y-%m-%d %H:%M")
        with self._conn() as c:
            rows = c.execute(
                "SELECT strftime(?, ts) bucket, COUNT(*) n FROM events "
                "WHERE component='orchestrator' AND event_type='request' GROUP BY bucket ORDER BY bucket",
                (fmt,),
            ).fetchall()
        return [dict(r) for r in rows]

    def guardrail_block_rate(self) -> dict:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) total, SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) blocked "
                "FROM events WHERE component='guardrail'"
            ).fetchone()
        total, blocked = (row["total"] or 0), (row["blocked"] or 0)
        return {"total": total, "blocked": blocked, "rate": (blocked / total) if total else 0.0}

    def error_rate(self) -> dict:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) total, SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) errors "
                "FROM events WHERE component='orchestrator' AND event_type='request'"
            ).fetchone()
        total, errors = (row["total"] or 0), (row["errors"] or 0)
        return {"total": total, "errors": errors, "rate": (errors / total) if total else 0.0}

    def avg_rag_iterations(self) -> float:
        with self._conn() as c:
            row = c.execute(
                "SELECT AVG(json_extract(details,'$.iterations')) x FROM events "
                "WHERE component='rag_agent' AND event_type='query'"
            ).fetchone()
        return float(row["x"]) if row and row["x"] is not None else 0.0

    def avg_latency_by_component(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT component, AVG(latency_ms) avg_ms, COUNT(latency_ms) n FROM events "
                "WHERE latency_ms IS NOT NULL GROUP BY component ORDER BY avg_ms DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def eval_history(self, eval_type: str | None = None) -> list[dict]:
        q = "SELECT ts, eval_type, score, passed, total, note FROM eval_runs"
        args: tuple = ()
        if eval_type:
            q += " WHERE eval_type=?"
            args = (eval_type,)
        q += " ORDER BY id"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def latest_eval_scores(self) -> list[dict]:
        """The most recent run of each eval type (for a scoreboard row)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT e.eval_type, e.score, e.passed, e.total, e.ts FROM eval_runs e "
                "JOIN (SELECT eval_type, MAX(id) mid FROM eval_runs GROUP BY eval_type) m "
                "ON e.id = m.mid ORDER BY e.eval_type"
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        try:
            d["details"] = json.loads(d.get("details") or "{}")
        except (ValueError, TypeError):
            d["details"] = {}
        d["ok"] = bool(d.get("ok"))
        return d


_observer: Observer | None = None


def get_observer() -> Observer:
    """Process-wide observer singleton (lazy)."""
    global _observer
    if _observer is None:
        _observer = Observer()
    return _observer


def reset_observer() -> None:
    """Drop the cached observer (tests, after pointing EVENTS_DB elsewhere)."""
    global _observer
    _observer = None


# ---------------------------------------------------------------------------
# The public logging interface — every call site uses these two.
# ---------------------------------------------------------------------------


def log_event(
    component: str, event_type: str, *,
    session_id: str | None = None, ok: bool = True,
    latency_ms: float | None = None, details: dict | None = None,
) -> None:
    """Record one event. Best-effort: never raises into the caller.

    *session_id* defaults to the current session bound by :func:`set_session`.
    """
    try:
        get_observer().log(
            component, event_type,
            session_id=session_id if session_id is not None else current_session(),
            ok=ok, latency_ms=latency_ms, details=details,
        )
    except Exception as exc:  # noqa: BLE001 — observability must not break the app
        logger.debug("log_event dropped (%s)", exc)


@contextmanager
def track(component: str, event_type: str, *, session_id: str | None = None, details: dict | None = None):
    """Time a block and log an event with its latency and success.

    ``ok`` is ``False`` if the block raises (the exception still propagates, so the
    caller's own error handling is unchanged); latency is recorded either way.
    Yields a mutable dict the block can update to enrich *details* before the write.
    """
    extra = dict(details or {})
    t0 = time.perf_counter()
    ok = True
    try:
        yield extra
    except Exception:
        ok = False
        raise
    finally:
        log_event(
            component, event_type, session_id=session_id, ok=ok,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2), details=extra,
        )


def timed_tool_call(tool, args: dict, invoke):
    """Run one MCP tool call through *invoke*, logging its latency and success.

    *invoke* is a callable ``(coroutine) -> result`` (the caller's own async
    runner, e.g. ``_run_coro``), so this stays decoupled from how the app drives
    coroutines. The tool's name and a compact arg-key list are recorded; the
    exception (if any) still propagates to the caller's existing error handling.
    """
    name = getattr(tool, "name", str(tool))
    with track("tool", name, details={"tool": name, "arg_keys": sorted(args.keys())}):
        return invoke(tool.ainvoke(args))


def record_eval_run(
    eval_type: str, score: float, passed: int, total: int,
    note: str = "", details: dict | None = None,
) -> None:
    """Persist an evaluation run's score (best-effort)."""
    try:
        get_observer().record_eval(eval_type, score, passed, total, note, details)
    except Exception as exc:  # noqa: BLE001
        logger.debug("record_eval_run dropped (%s)", exc)
