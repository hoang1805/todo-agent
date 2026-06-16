"""Ollama-backed agent using on-demand (progressive-disclosure) skills.

A single LangGraph tool-calling agent handles the whole request:

* Its system prompt lists every skill's **name + description** only.
* It is given a ``get_skill_detail`` tool (see :mod:`core.tools.skill_tools`) to load a
  skill's **full instructions** on demand, plus the live MCP tools.

The model therefore routes itself — it decides when a skill is relevant and
pulls its instructions via a tool call — instead of relying on a separate
"planner" LLM call. Conversation state persists across turns through a shared
LangGraph checkpointer.
"""

from __future__ import annotations

import asyncio
import queue
import re
import threading
from typing import TYPE_CHECKING, Generator

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from core.services.llm_client import create_ollama_model
from core.services.prompts import load_prompt
from core.services.skills import format_skill_roster
from core.tools.common_tools import make_common_tools
from core.tools.skill_tools import GET_SKILL_DETAIL, make_skill_tools

if TYPE_CHECKING:
    from models.skill import Skill


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class AgentChunk:
    """A single streamed chunk from the agent.

    ``type`` is one of:
        * ``"text"``      — assistant response text.
        * ``"reasoning"`` — chain-of-thought / thinking tokens.
        * ``"skill"``     — a skill was loaded (``content`` holds its name).
    """

    def __init__(self, type: str, content: str = "", reasoning: str = "") -> None:
        self.type = type
        self.content = content
        self.reasoning = reasoning


# ---------------------------------------------------------------------------
# Shared conversation memory
# ---------------------------------------------------------------------------
# A single checkpointer keyed by ``thread_id`` keeps the conversation thread
# state across turns natively.
_checkpointer = InMemorySaver()


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def _build_agent(
    skills: list[Skill],
    tools: list[BaseTool],
    model: str,
    temperature: float,
):
    """Create the single skill-aware tool-calling agent."""
    llm = create_ollama_model(model, temperature)
    system_prompt = load_prompt("agent_system").format(
        skills_roster=format_skill_roster(skills),
    )
    all_tools = [*make_common_tools(), *make_skill_tools(skills), *tools]
    return create_agent(
        llm,
        tools=all_tools,
        system_prompt=system_prompt,
        middleware=[
            SummarizationMiddleware(
                model=llm,
                trigger=("tokens", 4000),
                keep=("messages", 20),
            )
        ],
        checkpointer=_checkpointer,
    )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

_SKILL_NAME_RE = re.compile(r'"skill_name"\s*:\s*"([^"]+)"')


async def _astream_agent(agent, messages: list, thread_id: str):
    """Async generator: stream :class:`AgentChunk` objects from the agent.

    MCP tools are async-only, so we must use ``agent.astream()``. We surface
    three things: assistant text, reasoning (native ``reasoning_content`` and
    inline ``<think>`` blocks), and ``get_skill_detail`` calls (so the UI can
    show which skill the agent loaded).
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    in_think_block = False
    buffer = ""
    announced_skills: set[str] = set()

    async for msg, metadata in agent.astream(
        {"messages": messages},
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") not in ("agent", "model"):
            continue
        if not isinstance(msg, AIMessageChunk):
            continue

        # Surface skill loads: detect get_skill_detail tool calls as they stream.
        for tc in msg.tool_call_chunks or []:
            if tc.get("name") and tc["name"] != GET_SKILL_DETAIL:
                continue
            match = _SKILL_NAME_RE.search(tc.get("args") or "")
            if match and match.group(1) not in announced_skills:
                announced_skills.add(match.group(1))
                yield AgentChunk(type="skill", content=match.group(1))

        # Native reasoning field (ChatOllama reasoning=True).
        reasoning_content = msg.additional_kwargs.get("reasoning_content", "")
        if reasoning_content:
            yield AgentChunk(type="reasoning", reasoning=reasoning_content)

        if not msg.content:
            continue

        buffer += msg.content
        while True:
            if not in_think_block:
                idx = buffer.find("<think>")
                if idx != -1:
                    if idx > 0:
                        yield AgentChunk(type="text", content=buffer[:idx])
                    in_think_block = True
                    buffer = buffer[idx + len("<think>"):]
                else:
                    split_idx = len(buffer)
                    for i in range(1, min(8, len(buffer) + 1)):
                        if buffer[-i:].startswith("<"):
                            split_idx = len(buffer) - i
                            break
                    if split_idx > 0:
                        yield AgentChunk(type="text", content=buffer[:split_idx])
                        buffer = buffer[split_idx:]
                    break
            else:
                idx = buffer.find("</think>")
                if idx != -1:
                    if idx > 0:
                        yield AgentChunk(type="reasoning", reasoning=buffer[:idx])
                    in_think_block = False
                    buffer = buffer[idx + len("</think>"):]
                else:
                    split_idx = len(buffer)
                    for i in range(1, min(9, len(buffer) + 1)):
                        if buffer[-i:].startswith("<"):
                            split_idx = len(buffer) - i
                            break
                    if split_idx > 0:
                        yield AgentChunk(type="reasoning", reasoning=buffer[:split_idx])
                        buffer = buffer[split_idx:]
                    break

    if buffer:
        if in_think_block:
            yield AgentChunk(type="reasoning", reasoning=buffer)
        else:
            yield AgentChunk(type="text", content=buffer)


_SENTINEL = object()  # marks end of stream in the queue


def _stream_agent(
    agent, messages: list, thread_id: str
) -> Generator[AgentChunk, None, None]:
    """Sync bridge: run the async agent in a background thread, yield chunks.

    A thread + queue lets Streamlit receive chunks progressively while the
    async MCP tool calls run inside their own event loop.
    """
    chunk_queue: queue.Queue = queue.Queue()

    async def _producer() -> None:
        try:
            async for chunk in _astream_agent(agent, messages, thread_id):
                chunk_queue.put(chunk)
        except Exception as exc:  # noqa: BLE001 — forwarded to the consumer
            chunk_queue.put(exc)
        finally:
            chunk_queue.put(_SENTINEL)

    thread = threading.Thread(target=lambda: asyncio.run(_producer()), daemon=True)
    thread.start()

    while True:
        item = chunk_queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item

    thread.join()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_agent_with_skills(
    prompt: str,
    skills: list[Skill],
    tools: list[BaseTool],
    model: str,
    temperature: float,
    thread_id: str = "1",
) -> Generator[AgentChunk, None, None]:
    """Stream the skill-aware agent's response to *prompt*.

    The agent self-routes: it inspects the skill roster in its system prompt
    and calls ``get_skill_detail`` to load a skill's full instructions when one
    is relevant. Prior conversation is restored from the checkpointer via
    *thread_id*, so only the new message is passed in.

    Yields:
        :class:`AgentChunk` objects with ``type`` in
        ``{"skill", "reasoning", "text"}``.
    """
    agent = _build_agent(skills, tools, model, temperature)
    yield from _stream_agent(agent, [HumanMessage(content=prompt)], thread_id=thread_id)


def run_task_agent(
    question: str,
    skills: list[Skill],
    tools: list[BaseTool],
    model: str,
    temperature: float,
    thread_id: str = "1",
) -> str:
    """Non-streaming version of :func:`run_agent_with_skills`.

    Collects all text chunks and returns a single string.
    """
    parts: list[str] = []
    for chunk in run_agent_with_skills(
        question, skills, tools, model, temperature, thread_id
    ):
        if chunk.type == "text":
            parts.append(chunk.content)
    return "".join(parts) or "I encountered an issue. Please try again."


# ---------------------------------------------------------------------------
# Async MCP bootstrap (called once from app.py)
# ---------------------------------------------------------------------------


async def load_tools_async(servers: dict[str, str]) -> list[BaseTool]:
    """Async helper to fetch MCP tools — call once at startup."""
    from core.services.mcp_client import get_mcp_tools  # local import avoids circular dep

    try:
        return await get_mcp_tools(servers)
    except Exception as exc:  # noqa: BLE001 — MCP server may simply be down
        print(f"[MCP] Could not load tools: {exc}")
        return []


def load_tools(servers: dict[str, str]) -> list[BaseTool]:
    """Synchronous wrapper around :func:`load_tools_async` for Streamlit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an existing event loop (e.g. Streamlit's async context).
            import nest_asyncio  # type: ignore[import-untyped]

            nest_asyncio.apply()
        return loop.run_until_complete(load_tools_async(servers))
    except RuntimeError:
        return asyncio.run(load_tools_async(servers))
