"""MCP client helpers.

Builds a :class:`MultiServerMCPClient` from the server map defined in
``configs.settings.MCP_SERVERS`` and exposes an async helper to fetch all
LangChain-compatible tools from every connected server.
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


def build_mcp_client(servers: dict[str, str]) -> MultiServerMCPClient:
    """Create a :class:`MultiServerMCPClient` from a ``{name: sse_url}`` map.

    Args:
        servers: Mapping of server name → SSE endpoint URL, e.g.::

            {"task_mcp": "http://localhost:8000/sse"}

    Returns:
        A configured ``MultiServerMCPClient`` (not yet connected — call
        ``await client.get_tools()`` to open sessions on demand).
    """
    connections = {
        name: {"url": url, "transport": "sse"}
        for name, url in servers.items()
    }
    return MultiServerMCPClient(connections)


async def get_mcp_tools(servers: dict[str, str]) -> list[BaseTool]:
    """Connect to the MCP servers and return a flat list of LangChain tools.

    Servers are loaded **independently** so one being down (e.g. the optional
    memory server) contributes no tools instead of failing the whole load — the
    task tools still come through.

    Args:
        servers: Mapping of server name → SSE endpoint URL.

    Returns:
        All tools exposed by the reachable servers, as ``BaseTool`` objects.
    """
    tools: list[BaseTool] = []
    for name, url in servers.items():
        try:
            client = build_mcp_client({name: url})
            server_tools = await client.get_tools()
            tools += server_tools
            _log_server_registry(name, url, server_tools, ok=True)
        except Exception as exc:  # noqa: BLE001 — skip an unreachable server
            logger.warning("MCP server %r unavailable (%s); skipping its tools.", name, exc)
            _log_server_registry(name, url, [], ok=False, error=str(exc))
    return tools


def _log_server_registry(
    name: str, url: str, server_tools: list[BaseTool], *, ok: bool, error: str = "",
) -> None:
    """Record which tools a server exposed, so the dashboard can show its registry.

    One event per server (``component='mcp_server'``, event type = the server name),
    carrying the tool count and each tool's name + description. The dashboard reads
    the *latest* such event per server — it never connects to the servers itself
    (it's a separate, read-only process). Best-effort: never breaks tool loading.
    """
    try:
        from core.services.observability import log_event

        details = {
            "url": url,
            "tool_count": len(server_tools),
            "tools": [
                {"name": t.name, "description": (getattr(t, "description", "") or "").strip()[:300]}
                for t in server_tools
            ],
        }
        if error:
            details["error"] = error[:200]
        log_event("mcp_server", name, ok=ok, details=details)
    except Exception:  # noqa: BLE001 — registry logging must never break the load
        pass