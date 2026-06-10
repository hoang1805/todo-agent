from langchain_mcp_adapters.client import MultiMCPClient

async def connect_to_mcp(servers: dict):
    async with MultiMCPClient() as mcp_hub:
        print("Connecting to MCP servers...")
        for server_name, sse_url in servers.items():
            try:
                print(f"Connecting to {server_name} at {sse_url}...")
                await mcp_hub.connect(server_name, sse_url)
                print(f"Connected to {server_name} at {sse_url}")
            except Exception as e:
                print(f"Failed to connect to {server_name} at {sse_url}: {e}")
        print("All MCP connections established.")
        yield mcp_hub