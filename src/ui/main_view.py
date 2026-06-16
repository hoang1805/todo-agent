from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import streamlit as st
from langchain_core.tools import BaseTool

from agents.ollama_agent import AgentChunk, run_agent_with_skills, run_task_agent

PLANNER_MODE = "Planner (Multi-Agent)"

if TYPE_CHECKING:
    from models.skill import Skill



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
        skills: All loaded :class:`~models.skill.Skill` objects.
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

    # Skills badge row (skills are optional — the Planner mode and chat agent
    # both work without any).
    if skills:
        badge_row = "  ".join(
            f"`{s.name}`" for s in skills
        )
        st.markdown(
            f"<p style='color:#64748b; font-size:0.85rem; margin-top:0.25rem;'>"
            f"🧩 Active skills: {badge_row}</p>",
            unsafe_allow_html=True,
        )

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

    # Pending human-in-the-loop approval (Planner mode) — render until resolved.
    if mode == PLANNER_MODE and st.session_state.get("planner_pending"):
        _render_approval_panel(_get_planner(tools, model, temperature))

    # Handle new user input
    if prompt := st.chat_input("Ask your agent (e.g., What should I do today?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        if mode == PLANNER_MODE:
            # The planner may pause for approval, so route through session state
            # and rerun (the approval panel / result renders on the next pass).
            with st.spinner("Planning…", show_time=True):
                _start_planner_turn(_get_planner(tools, model, temperature), prompt)
            st.rerun()

        st.chat_message("user").markdown(prompt)
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
# Planner (multi-agent) helpers — with human-in-the-loop approval
# ---------------------------------------------------------------------------


def _get_planner(tools: list[BaseTool], model: str, temperature: float):
    """Return a cached GraphOrchestrator (so its checkpointer survives reruns).

    Rebuilt only when the model, temperature, or tool count changes — the
    checkpointer must persist across reruns for approval/resume to work.
    """
    import configs.settings as cfg
    from agents.graph_orchestrator import build_graph_orchestrator

    key = (model, round(temperature, 3), len(tools))
    if st.session_state.get("planner_key") != key:
        st.session_state.planner = build_graph_orchestrator(
            tools, model, temperature, checkpointer_db=cfg.CHECKPOINT_DB or None
        )
        st.session_state.planner_key = key
    return st.session_state.planner


def _consume_planner_result(result) -> None:
    """Turn a PlannerResult into chat output and/or a pending approval."""
    if result.status == "done":
        st.session_state.planner_pending = None
        st.session_state.messages.append(
            {"role": "ai", "content": f"```text\n{result.text}\n```"}
        )
        return

    intr = result.interrupt or {}
    st.session_state.planner_pending = {
        "thread_id": result.thread_id,
        "interrupt": intr,
    }
    st.session_state.messages.append(
        {
            "role": "ai",
            "content": (
                f"🔐 **Approval needed** to run `{intr.get('tool')}` "
                f"with arguments `{intr.get('args')}`."
            ),
        }
    )


def _start_planner_turn(planner, prompt: str) -> None:
    """Start a planner run for *prompt* (may pause for approval)."""
    from core.tools.common_tools import get_today_date

    result = planner.start(prompt, date=get_today_date.invoke({}))
    _consume_planner_result(result)


def _render_approval_panel(planner) -> None:
    """Render Approve/Reject controls for a pending mutating action."""
    pending = st.session_state.planner_pending
    intr = pending["interrupt"]

    st.warning(
        f"**{intr.get('message', 'Approve this action?')}**\n\n"
        f"Tool: `{intr.get('tool')}` — arguments: `{intr.get('args')}`"
    )
    approve, reject = st.columns(2)
    if approve.button("✅ Approve", use_container_width=True, key="planner_approve"):
        with st.spinner("Applying…"):
            result = planner.resume(pending["thread_id"], {"action": "accept"})
        _consume_planner_result(result)
        st.rerun()
    if reject.button("❌ Reject", use_container_width=True, key="planner_reject"):
        with st.spinner("Cancelling…"):
            result = planner.resume(
                pending["thread_id"], {"action": "reject", "reason": "user declined"}
            )
        _consume_planner_result(result)
        st.rerun()


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
    loaded_skills: list[str] = []
    thinking = ""
    response = ""

    for chunk in run_agent_with_skills(
        prompt, skills, tools, model, temperature, thread_id
    ):
        if chunk.type == "skill":
            if chunk.content not in loaded_skills:
                loaded_skills.append(chunk.content)
        elif chunk.type == "reasoning":
            thinking += chunk.reasoning
        elif chunk.type == "text":
            response += chunk.content

        # Build live display
        display = ""
        if loaded_skills:
            badges = ", ".join(f"<code>{name}</code>" for name in loaded_skills)
            display += (
                f"<details><summary>🧩 <strong>Skills loaded</strong>: "
                f"{badges}</summary></details>\n\n"
            )
        if thinking:
            display += f"🤔 **Thinking:**\n```text\n{thinking}\n```\n\n"
        if response:
            display += f"💡 **Response:**\n{response}"

        placeholder.markdown(display, unsafe_allow_html=True)

    return response
