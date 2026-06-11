MODEL = "gemma4:31b-cloud"
TEMPERATURE = 0.2

APP_TITLE = "To-do Agent"

DATA_SOURCE = "data"

OLLAMA_HOST = "http://localhost:11434"

MCP_SERVERS = {
    "task_mcp": "http://localhost:8000/sse"
}

SKILL_SOURCES = "src/skills"