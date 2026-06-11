from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import streamlit as st
from langchain_core.tools import BaseTool

from agents.ollama_agent import AgentChunk, run_agent_with_skills, run_task_agent

if TYPE_CHECKING:
    from core.models import Skill



def render_main_view(
    model: str,
    temperature: float,
    mode: str,
    skills: list[Skill],
    tools: list[BaseTool],
) -> None:
    """Render the primary chat interface.

    Args:
        model: Ollama model name selected in the sidebar.
        temperature: Creativity slider value.
        mode: ``"Streaming"`` or ``"Normal"``.
        skills: All loaded :class:`~core.models.Skill` objects.
        tools: Live MCP tools (may be an empty list if the server is down).
    """
    st.markdown("<h1>✨ AI Task Assistant</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #94a3b8; font-size: 1.1rem;'>Manage your tasks with local AI</p>",
        unsafe_allow_html=True,
    )

    # Metrics row
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 2])
        col1.metric("Current Model", model)
        col2.metric("Creativity", f"{temperature:.1f}")
        col3.metric(
            "MCP Tools",
            len(tools),
            help="Number of tools loaded from the MCP server.",
        )

    # Skills badge row
    if skills:
        badge_row = "  ".join(
            f"`{s.name}`" for s in skills
        )
        st.markdown(
            f"<p style='color:#64748b; font-size:0.85rem; margin-top:0.25rem;'>"
            f"🧩 Active skills: {badge_row}</p>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("No skills loaded — check your `SKILL_SOURCES` path in settings.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []          # display dicts {role, content}
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask your agent (e.g., What should I do today?)"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("ai"):
            with st.spinner("Agent is thinking…", show_time=True):
                if mode == "Streaming":
                    response = _stream_response(
                        prompt, skills, tools, model, temperature,
                        st.session_state.thread_id,
                    )
                else:
                    response = run_task_agent(
                        prompt, skills, tools, model, temperature,
                        st.session_state.thread_id,
                    )
                    st.markdown(response)

        # Append this turn to the UI store
        st.session_state.messages.append({"role": "ai", "content": response})


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------


def _stream_response(
    prompt: str,
    skills: list[Skill],
    tools: list[BaseTool],
    model: str,
    temperature: float,
    thread_id: str,
) -> str:
    """Stream the two-phase agent output into the Streamlit UI.

    Returns the final concatenated text so it can be stored in chat history.
    """
    placeholder = st.empty()
    plan_text = ""
    thinking = ""
    response = ""

    for chunk in run_agent_with_skills(
        prompt, skills, tools, model, temperature, thread_id
    ):
        if chunk.type == "plan":
            plan_text = chunk.content
        elif chunk.type == "reasoning":
            thinking += chunk.reasoning
        elif chunk.type == "text":
            response += chunk.content

        # Build live display
        display = ""
        if plan_text:
            display += (
                f"<details><summary>🗺️ <strong>Skill routing</strong>: "
                f"<code>{plan_text}</code></summary></details>\n\n"
            )
        if thinking:
            display += f"🤔 **Thinking:**\n```text\n{thinking}\n```\n\n"
        if response:
            display += f"💡 **Response:**\n{response}"

        placeholder.markdown(display, unsafe_allow_html=True)

    return response
