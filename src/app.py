from __future__ import annotations

import streamlit as st
from pathlib import Path

import configs.settings as cfg
from core.models import Skill
from ui.main_view import render_main_view
from ui.sidebar import render_sidebar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_skills(skill_dir: Path) -> list[Skill]:
    """Scan *skill_dir* for ``*/SKILL.md`` files and parse each into a Skill."""
    skills: list[Skill] = []
    if not skill_dir.is_dir():
        st.warning(f"Skill directory not found: {skill_dir}")
        return skills
    for skill_md in sorted(skill_dir.glob("*/SKILL.md")):
        try:
            skills.append(Skill.from_markdown(skill_md))
        except ValueError as exc:
            st.error(f"Could not parse {skill_md.name}: {exc}")
    return skills


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
    skill_dir = Path(cfg.SKILL_SOURCES)
    skills = _load_skills(skill_dir)

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
