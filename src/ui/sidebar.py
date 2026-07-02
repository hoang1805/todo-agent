import streamlit as st

from configs.settings import MODEL, TEMPERATURE


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

    return {
        "model": model_name,
        "temperature": temperature,
        "mode": mode,
    }