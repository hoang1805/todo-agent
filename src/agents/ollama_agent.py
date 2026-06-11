"""Ollama-backed agent with skill-aware, two-phase execution.

Flow
----
1. **Plan** — :class:`~core.skill_planner.SkillPlanner` reads the user message
   and the skill roster (name + description only) and returns an ordered list
   of skill names.
2. **Execute** — for each chosen skill, a short-lived LangGraph agent is
   created with the skill's full ``detail`` injected as a system prompt, then
   run with the live MCP tools.  Chunks are yielded back to the UI via the
   same :class:`AgentChunk` dataclass used before.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import TYPE_CHECKING, Generator

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from core.llm_client import create_ollama_model
from core.prompts import load_prompt
from core.skill_planner import SkillPlanner

if TYPE_CHECKING:
    from core.models import Skill


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class AgentChunk:
    """A single streamed chunk from the agent."""

    def __init__(
        self,
        type: str,          # "text" | "reasoning" | "plan"
        content: str = "",
        reasoning: str = "",
    ) -> None:
        self.type = type
        self.content = content
        self.reasoning = reasoning


# ---------------------------------------------------------------------------
# Global Native Memory
# ---------------------------------------------------------------------------
# Share the same checkpointer across all short-lived skill agents so they
# can seamlessly pick up the conversation thread state natively.
_checkpointer = InMemorySaver()

# ---------------------------------------------------------------------------
# Low-level: single-skill agent
# ---------------------------------------------------------------------------


def _create_skill_agent(
    skill_detail: str,
    tools: list[BaseTool],
    model: str,
    temperature: float,
):
    """Build a LangGraph tool-calling agent scoped to one skill."""
    llm = create_ollama_model(model, temperature)
    system_prompt = skill_detail.strip() or load_prompt("fallback_system")
    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[SummarizationMiddleware(
            model=llm,
            trigger=("tokens", 4000),
            keep=("messages", 20),
        )],
        checkpointer=_checkpointer,
    )

async def _astream_agent(
    agent,
    messages: list,
    thread_id: str,
):
    """Async generator: stream chunks from a LangGraph agent via astream.

    MCP tools are async-only (StructuredTool with _arun only), so we MUST
    use ``agent.astream()`` — the sync ``agent.stream()`` would call the
    non-existent ``_run`` and raise ``NotImplementedError``.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    in_think_block = False
    buffer = ""

    async for msg, metadata in agent.astream(
        {"messages": messages},
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") not in ("agent", "model"):
            continue
        if not isinstance(msg, AIMessageChunk):
            continue

        # Native reasoning field (ChatOllama reasoning=True)
        reasoning_content = msg.additional_kwargs.get("reasoning_content", "")
        if reasoning_content:
            yield AgentChunk(type="reasoning", reasoning=reasoning_content)

        if msg.content:
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
                            yield AgentChunk(
                                type="reasoning", reasoning=buffer[:split_idx]
                            )
                            buffer = buffer[split_idx:]
                        break

    if buffer:
        if in_think_block:
            yield AgentChunk(type="reasoning", reasoning=buffer)
        else:
            yield AgentChunk(type="text", content=buffer)


_SENTINEL = object()  # marks end of stream in the queue


def _stream_agent(
    agent,
    messages: list,
    thread_id: str,
) -> Generator[AgentChunk, None, None]:
    """Sync bridge: runs the async agent in a background thread and yields chunks.

    Using a thread + queue lets Streamlit receive chunks progressively while
    the async MCP tool calls happen correctly inside their own event loop.
    """
    chunk_queue: queue.Queue = queue.Queue()

    async def _producer() -> None:
        try:
            async for chunk in _astream_agent(agent, messages, thread_id):
                chunk_queue.put(chunk)
        except Exception as exc:
            chunk_queue.put(exc)
        finally:
            chunk_queue.put(_SENTINEL)

    def _run_loop() -> None:
        asyncio.run(_producer())

    thread = threading.Thread(target=_run_loop, daemon=True)
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
    """Two-phase planner + executor pipeline.

    Phase 1 — Planning:
        :class:`~core.skill_planner.SkillPlanner` selects which skills to run.
        A ``"plan"`` chunk is yielded so the UI can display the routing decision.
        Recent ``history`` is summarised and appended to the planning prompt so
        the planner understands follow-up references like "the first task".

    Phase 2 — Execution:
        Each chosen skill runs as a short-lived agent with its full ``detail``
        as the system prompt.  **The full conversation history** is passed as the
        message list so the skill agent has context from earlier exchanges.
        Chunks are yielded in streaming order.
        If no skill matches, a fallback agent (no special system prompt) runs.

    Args:
        prompt: The user's current message.
        skills: All loaded :class:`~core.models.Skill` objects.
        tools: LangChain tools obtained from the MCP server(s).
        model: Ollama model name.
        temperature: Sampling temperature.
        thread_id: LangGraph checkpoint thread identifier.
        history: Prior conversation as LangChain ``BaseMessage`` objects
            (``HumanMessage`` / ``AIMessage``). Pass this so skill agents
            remember what was discussed in earlier turns.

    Yields:
        :class:`AgentChunk` objects with ``type`` in
        ``{"plan", "text", "reasoning"}``.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Fetch the raw conversation history natively from the LangGraph checkpointer
    state = _checkpointer.get_tuple(config)
    messages = state.checkpoint["channel_values"].get("messages", []) if state else []

    # ---- Phase 1: planning ------------------------------------------------
    # Give the planner a compact view of recent history so it can route
    # follow-up messages correctly (e.g. "show me the first task").
    planner = SkillPlanner(model=model, temperature=0.0)
    chosen_names: list[str] = planner.plan(prompt, skills, messages=messages)

    yield AgentChunk(
        type="plan",
        content=", ".join(chosen_names) if chosen_names else "(no specific skill)",
    )

    # ---- Phase 2: execution -----------------------------------------------
    skill_map = {s.name: s for s in skills}
    chosen_skills = [skill_map[n] for n in chosen_names if n in skill_map]

    # Fallback: no skill matched — run with the generic system prompt
    if not chosen_skills:
        agent = _create_skill_agent(load_prompt("fallback_system"), tools, model, temperature)
        yield from _stream_agent(agent, [HumanMessage(content=prompt)], thread_id=thread_id)
        return

    is_multi_skill = len(chosen_skills) > 1

    for i, skill in enumerate(chosen_skills):
        agent = _create_skill_agent(skill.detail, tools, model, temperature)
        
        # If this is the first skill in the chain, we pass the user's prompt.
        # If it's a subsequent skill, we pass an instruction to continue.
        # LangGraph's checkpointer natively handles everything else in the background!
        input_msg = prompt if i == 0 else f"Please continue fulfilling the user's request using your specialized '{skill.name}' skill."

        # If multi-skill, wrap skill execution in a pseudo-reasoning block header
        if is_multi_skill:
            yield AgentChunk(type="reasoning", reasoning=f"\n\n--- Running Skill: {skill.name} ---\n")

        for chunk in _stream_agent(agent, [HumanMessage(content=input_msg)], thread_id=thread_id):
            if is_multi_skill and chunk.type == "text":
                # Hide intermediate text output inside the thinking block
                yield AgentChunk(type="reasoning", reasoning=chunk.content)
            else:
                yield chunk

    # Final Synthesizer Step for Multi-Skill
    if is_multi_skill:
        # Fetch the very latest state containing all skill outputs
        final_state = _checkpointer.get_tuple(config)
        final_messages = final_state.checkpoint["channel_values"].get("messages", []) if final_state else []

        llm = create_ollama_model(model, temperature=temperature, with_thinking=False)
        synth_messages = final_messages + [SystemMessage(content=load_prompt("synthesizer_system"))]

        for chunk in llm.stream(synth_messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield AgentChunk(type="text", content=content)



# ---------------------------------------------------------------------------
# Legacy sync helper (kept for non-streaming "Normal" mode)
# ---------------------------------------------------------------------------


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
    from core.mcp_client import get_mcp_tools  # local import avoids circular dep
    try:
        return await get_mcp_tools(servers)
    except Exception as exc:
        print(f"[MCP] Could not load tools: {exc}")
        return []


def load_tools(servers: dict[str, str]) -> list[BaseTool]:
    """Synchronous wrapper around :func:`load_tools_async` for Streamlit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an existing event loop (e.g., Streamlit async context)
            import nest_asyncio  # type: ignore[import-untyped]
            nest_asyncio.apply()
        return loop.run_until_complete(load_tools_async(servers))
    except RuntimeError:
        return asyncio.run(load_tools_async(servers))