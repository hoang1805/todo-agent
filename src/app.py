from __future__ import annotations

import streamlit as st

import configs.settings as cfg
from core.services.skills import load_skills
from ui.main_view import render_main_view
from ui.sidebar import render_sidebar


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
    st.set_page_config(
        page_title="Task Agent",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

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
