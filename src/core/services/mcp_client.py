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
            tools += await client.get_tools()
        except Exception as exc:  # noqa: BLE001 — skip an unreachable server
            logger.warning("MCP server %r unavailable (%s); skipping its tools.", name, exc)
    return tools