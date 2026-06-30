import uuid

import streamlit as st

from configs.settings import MODEL, TEMPERATURE


def _render_conversations() -> None:
    """List persisted conversations; start a new one or resume an existing one.

    Resuming sets the session/thread id and replays the stored turns into the chat
    view — the conversation survives restarts because it lives in the history DB.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3>💬 Conversations</h3>", unsafe_allow_html=True)

    if st.sidebar.button("➕ New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pop("planner_pending", None)
        st.rerun()

    try:
        from core.services.history import get_history

        history = get_history()
        sessions = history.list_sessions()
    except Exception:  # noqa: BLE001 — history is optional; never break the sidebar
        return

    active = st.session_state.get("thread_id")
    for sess in sessions[:15]:
        label = (sess.get("title") or sess["id"])[:30]
        marker = "• " if sess["id"] == active else ""
        if st.sidebar.button(f"🗂️ {marker}{label}", key=f"sess_{sess['id']}", use_container_width=True):
            st.session_state.thread_id = sess["id"]
            st.session_state.messages = [
                {"role": "user" if t["role"] == "user" else "ai", "content": t["content"]}
                for t in history.get_all_turns(sess["id"])
            ]
            st.session_state.pop("planner_pending", None)
            st.rerun()


def render_sidebar():
    st.sidebar.markdown("<h2>⚙️ Agent Settings</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem;'>Configure your local AI models</p>", unsafe_allow_html=True)

    model_name = st.sidebar.text_input(
        "🧠 Ollama Model",
        value=MODEL,
        help="Make sure the model is pulled locally via 'ollama pull <model>'"
    )

    temperature = st.sidebar.slider(
        "🎨 Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=TEMPERATURE,
        step=0.1,
        help="Higher values make the output more random, lower values make it more focused."
    )

    mode = st.sidebar.selectbox(
        "Agent Mode",
        options=["Planner (Multi-Agent)", "Streaming", "Normal"],
        help=(
            "Planner: the multi-agent orchestrator (plan / summarize / multi-step). "
            "Streaming & Normal: the general chat assistant (with MCP tools)."
        ),
    )

    if st.sidebar.button("🧹 Clear Agent Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        # Drop all chat sessions; they're recreated fresh on the next render.
        for key in ("sessions", "active_sid"):
            st.session_state.pop(key, None)
        st.sidebar.success("✨ Memory cleared successfully!")

    _render_conversations()

    return {
        "model": model_name,
        "temperature": temperature,
        "mode": mode,
    }