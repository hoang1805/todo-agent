from __future__ import annotations

import logging

import streamlit as st

import configs.settings as cfg
from core.services.skills import load_skills
from ui.main_view import render_main_view
from ui.sidebar import render_sidebar


def _configure_logging(trace: bool) -> None:
    """Send the agent loggers to the terminal — verbose (INFO) when tracing.

    Targets the ``agents`` logger only (so the planner's ``[trace]`` steps show)
    and stops propagation, keeping third-party libraries out of the output.
    Idempotent across Streamlit reruns.
    """
    agents_logger = logging.getLogger("agents")
    agents_logger.setLevel(logging.INFO if trace else logging.WARNING)
    agents_logger.propagate = False
    if not any(isinstance(h, logging.StreamHandler) for h in agents_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        agents_logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="🔌 Connecting to MCP server…")
def _load_tools(servers_frozen: tuple[tuple[str, str], ...]):
    """Load MCP tools once and cache them for the lifetime of the process.

    The *servers_frozen* argument is a tuple-of-tuples so it is hashable and
    can be used as a ``@st.cache_resource`` key.
    """
    from agents.ollama_agent import load_tools
    servers = dict(servers_frozen)
    tools = load_tools(servers)
    return tools


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _configure_logging(cfg.TRACE)

    st.set_page_config(
        page_title="Task Agent",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -- Top-level view switch: chat and the observability dashboard live in the
    #    SAME app (one port), toggled here — not a separate service. ----------
    page = st.sidebar.radio(
        "View", ["💬 Chat", "📊 Dashboard"],
        horizontal=True, label_visibility="collapsed",
    )
    st.sidebar.markdown("---")

    if page == "📊 Dashboard":
        from dashboard import render_dashboard

        render_dashboard()
        return

    # -- Skills (loaded once; cheap — just file reads) ----------------------
    skills, skill_errors = load_skills(cfg.SKILL_SOURCES)
    for error in skill_errors:
        st.warning(error)

    # -- MCP tools (cached; involves network connection to MCP server) -------
    servers_frozen = tuple(cfg.MCP_SERVERS.items())
    tools = _load_tools(servers_frozen)

    # -- Sidebar config -----------------------------------------------------
    config = render_sidebar()

    # -- Main chat UI -------------------------------------------------------
    render_main_view(
        model=config["model"],
        temperature=config["temperature"],
        mode=config["mode"],
        skills=skills,
        tools=tools,
    )


if __name__ == "__main__":
    main()
