"""Human-in-the-loop approval for mutating tools.

Reading tasks is safe to do autonomously; **creating, updating, or deleting** a
task changes the user's data, so those calls must be approved by a human first.

This is implemented with LangGraph's :func:`interrupt`: the wrapped tool pauses
the graph before it runs, surfacing the proposed action to the caller. The graph
is resumed with the user's decision (``accept`` / ``reject`` / ``edit``); only on
``accept`` (or ``edit``) does the real tool actually execute. Resuming requires
the graph to be compiled with a checkpointer and invoked with a ``thread_id``.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool as create_tool
from langgraph.types import interrupt

#: Substrings that mark a tool as mutating (and therefore approval-gated).
MUTATING_KEYWORDS = (
    "create",
    "add",
    "update",
    "delete",
    "remove",
    "complete",
    "edit",
    "set",
    "mark",
)


def is_mutating_tool(name: str) -> bool:
    """True if *name* looks like a tool that changes task data."""
    low = name.lower()
    return any(keyword in low for keyword in MUTATING_KEYWORDS)


def add_human_in_the_loop(target: BaseTool) -> BaseTool:
    """Wrap *target* so it requires human approval before it runs.

    The wrapper keeps the original tool's name, description, and argument schema
    so the model sees no difference — only execution is gated.
    """

    @create_tool(target.name, description=target.description, args_schema=target.args_schema)
    async def wrapper(**kwargs):
        decision = interrupt(
            {
                "type": "approval",
                "tool": target.name,
                "args": kwargs,
                "message": f"Approve running '{target.name}' with these arguments?",
            }
        )
        action = decision.get("action") if isinstance(decision, dict) else decision

        if action == "accept":
            return await target.ainvoke(kwargs)
        if action == "edit":
            edited = decision.get("args", kwargs) if isinstance(decision, dict) else kwargs
            return await target.ainvoke(edited)

        reason = decision.get("reason") if isinstance(decision, dict) else None
        return (
            f"The user declined to run '{target.name}'"
            f"{f' ({reason})' if reason else ''}. The action was NOT performed; "
            "do not retry it unless the user asks again."
        )

    return wrapper


def wrap_mutating_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """Return *tools* with the mutating ones gated by human approval."""
    return [add_human_in_the_loop(t) if is_mutating_tool(t.name) else t for t in tools]
