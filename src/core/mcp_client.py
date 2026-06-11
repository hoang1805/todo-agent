"""MCP client helpers.

Builds a :class:`MultiServerMCPClient` from the server map defined in
``configs.settings.MCP_SERVERS`` and exposes an async helper to fetch all
LangChain-compatible tools from every connected server.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


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
    """Connect to all MCP servers and return a flat list of LangChain tools.

    Each server connection is opened on demand (no persistent session).

    Args:
        servers: Mapping of server name → SSE endpoint URL.

    Returns:
        All tools exposed by all servers, as LangChain ``BaseTool`` objects.
    """
    client = build_mcp_client(servers)
    tools: list[BaseTool] = await client.get_tools()
    return tools