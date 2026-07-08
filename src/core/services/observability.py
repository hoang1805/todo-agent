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
import os
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

    def latest_eval_run(self, eval_type: str) -> dict | None:
        """The most recent run of one eval type, **with its per-case details**.

        Unlike :meth:`latest_eval_scores` (a scoreboard row), this parses the stored
        ``details`` blob so the dashboard can list each case's pass/fail and — for the
        generation eval — the per-criterion ✓/✗ breakdown the LLM judge produced.
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT ts, eval_type, score, passed, total, note, details FROM eval_runs "
                "WHERE eval_type=? ORDER BY id DESC LIMIT 1", (eval_type,),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["details"] = json.loads(d.get("details") or "{}")
        except (ValueError, TypeError):
            d["details"] = {}
        return d

    def latest_eval_scores(self) -> list[dict]:
        """The most recent run of each eval type (for a scoreboard row)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT e.eval_type, e.score, e.passed, e.total, e.ts FROM eval_runs e "
                "JOIN (SELECT eval_type, MAX(id) mid FROM eval_runs GROUP BY eval_type) m "
                "ON e.id = m.mid ORDER BY e.eval_type"
            ).fetchall()
        return [dict(r) for r in rows]

    # -- per-agent / per-session detail (the "all about the agents" view) -----

    def intent_breakdown(self) -> list[dict]:
        """How the orchestrator classified traffic: each intent → its agent + count.

        Reads the ``orchestrator/route`` events (the routing decision), so the panel
        shows *what the agents were actually asked to do*, not just how often they ran.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT json_extract(details,'$.intent') intent, "
                "json_extract(details,'$.agent') agent, COUNT(*) n FROM events "
                "WHERE component='orchestrator' AND event_type='route' "
                "GROUP BY intent, agent ORDER BY n DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def tool_usage(self) -> list[dict]:
        """Per MCP tool: call count, average latency, and failure count.

        Every tool call is logged under ``component='tool'`` with the tool name as
        the event type (via :func:`timed_tool_call`), so this is the tool mix.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT event_type tool, COUNT(*) n, AVG(latency_ms) avg_ms, "
                "SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) failures FROM events "
                "WHERE component='tool' GROUP BY event_type ORDER BY n DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def latency_by_operation(self) -> list[dict]:
        """Average/max latency per operation (component + event type) — finer than
        :meth:`avg_latency_by_component`. Surfaces e.g. how slow classification (an
        LLM call) is versus a plan versus each tool."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT component || '/' || event_type op, AVG(latency_ms) avg_ms, "
                "MAX(latency_ms) max_ms, COUNT(latency_ms) n FROM events "
                "WHERE latency_ms IS NOT NULL GROUP BY component, event_type ORDER BY avg_ms DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def planner_summary(self) -> dict:
        """DailyPlanner rollup: plans generated, how many overloaded (+ rate), and how
        many tasks got deferred in total. The overload branch is a first-class event."""
        with self._conn() as c:
            row = c.execute(
                "SELECT "
                "SUM(CASE WHEN event_type='plan' THEN 1 ELSE 0 END) plans, "
                "SUM(CASE WHEN event_type='overload' THEN 1 ELSE 0 END) overloads, "
                "SUM(CASE WHEN event_type='overload' "
                "    THEN json_extract(details,'$.deferred_count') ELSE 0 END) deferred, "
                "AVG(CASE WHEN event_type='overload' "
                "    THEN json_extract(details,'$.available_minutes') END) avg_available "
                "FROM events WHERE component='planner_agent'"
            ).fetchone()
        plans, overloads = (row["plans"] or 0), (row["overloads"] or 0)
        return {
            "plans": plans, "overloads": overloads,
            "rate": (overloads / plans) if plans else 0.0,
            "deferred": row["deferred"] or 0,
            "avg_available": row["avg_available"],
        }

    def recent_overloads(self, limit: int = 10) -> list[dict]:
        """The most recent overload events (with their deferred task titles)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, session_id, details FROM events "
                "WHERE component='planner_agent' AND event_type='overload' "
                "ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def session_summary(self, limit: int = 50) -> list[dict]:
        """One row per conversation: event count, request count, first/last timestamps.

        Feeds the session inspector — pick a session, see everything it did."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT session_id, COUNT(*) events, "
                "SUM(CASE WHEN component='orchestrator' AND event_type='request' THEN 1 ELSE 0 END) requests, "
                "SUM(CASE WHEN component='guardrail' AND ok=0 THEN 1 ELSE 0 END) blocks, "
                "MIN(ts) first_ts, MAX(ts) last_ts FROM events "
                "WHERE session_id IS NOT NULL GROUP BY session_id ORDER BY last_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def events_for_session(self, session_id: str, limit: int = 500) -> list[dict]:
        """The full ordered event timeline for one session (the drill-down)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, component, event_type, ok, latency_ms, details FROM events "
                "WHERE session_id=? ORDER BY id LIMIT ?", (session_id, limit),
            ).fetchall()
        return [self._row(r) for r in rows]

    def mcp_servers(self) -> list[dict]:
        """The latest registry event per MCP server (name, url, status, tools).

        Each server logs one ``mcp_server`` event when the app loads its tools (event
        type = the server name); we take the most recent per server so the panel
        reflects the current registry. The dashboard renders these — it never
        connects to the servers itself.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT e.event_type server, e.ok, e.ts, e.details FROM events e "
                "JOIN (SELECT event_type, MAX(id) mid FROM events "
                "      WHERE component='mcp_server' GROUP BY event_type) m "
                "ON e.id = m.mid ORDER BY e.event_type"
            ).fetchall()
        return [self._row(r) for r in rows]

    def debug_events(self, limit: int = 100) -> list[dict]:
        """Recent agent debug-trace events (only emitted when AGENT_DEBUG is on)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, component, event_type, session_id, ok, latency_ms, details "
                "FROM events WHERE event_type='debug' ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        return [self._row(r) for r in rows]

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


def debug_enabled() -> bool:
    """Whether agent debug tracing is on (``AGENT_DEBUG=1``).

    Read live from the environment so it can be toggled per run (CLI, tests) without
    a restart — the same "tag debug" switch, but persisted to the store instead of
    printed and lost.
    """
    return (os.getenv("AGENT_DEBUG") or "").lower() in ("1", "true", "yes")


def log_debug(component: str, message: str, *, session_id: str | None = None, **details) -> None:
    """Record a fine-grained agent debug event — but only when :func:`debug_enabled`.

    This is the shared "tag debug" trace: agents call it at decision points (the
    classifier's raw output, each RAG reformulation + judge verdict, the planner's
    chosen workday) and, when the flag is on, the detail lands in the events store
    where the dashboard's Debug trace panel shows it. A no-op when the flag is off,
    so leaving the calls in place costs nothing in normal operation.
    """
    if not debug_enabled():
        return
    log_event(component, "debug", session_id=session_id, ok=True,
              details={"message": message, **details})


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
