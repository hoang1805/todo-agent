from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING

import streamlit as st
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from agents.ollama_agent import AgentChunk, run_agent_with_skills, run_task_agent
from models.contract import TaskMutation

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

    # Render chat history (including the persisted step trace for planner turns).
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("steps"):
                _render_steps(st, message["steps"])
            st.markdown(message["content"])

    # Pending human-in-the-loop approval (Planner mode) — render until resolved.
    awaiting_approval = mode == PLANNER_MODE and bool(
        st.session_state.get("planner_pending")
    )
    if awaiting_approval:
        _render_approval_panel(_get_planner(tools, model, temperature), tools)

    # Handle new user input — locked while a review/approval is outstanding so the
    # user resolves the pending action before sending anything new.
    if prompt := st.chat_input(
        "Submit or reject the pending task above to continue…"
        if awaiting_approval
        else "Ask your agent (e.g., What should I do today?)",
        disabled=awaiting_approval,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})

        if mode == PLANNER_MODE:
            # The planner may pause for approval, so route through session state
            # and rerun (the approval panel / result renders on the next pass).
            # Echo the prompt now and show the live step box in the reply bubble;
            # both are re-rendered from history after the rerun.
            st.chat_message("user").markdown(prompt)
            with st.chat_message("assistant"):
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


# Friendly, user-facing label for each graph node, shown live as the planner works
# so the multi-agent steps are visible instead of hidden behind one spinner.
_PLANNER_STEP_LABELS = {
    "classify": "🧭 Understanding your request",
    "todo": "📥 Fetching & validating your tasks",
    "planner": "🗓️ Planning your day",
    "summary": "📋 Summarizing your tasks",
    "detail": "🔍 Looking up that task",
    "at_time": "🕖 Checking your plan at that time",
    "rag": "🧠 Searching your memory",
    "register_appointment": "📌 Registering the appointment",
    "extract_mutation": "✍️ Parsing your change",
    "confirm_plan": "⏳ Preparing the plan for your confirmation",
    "confirm_mutation": "⏳ Preparing the change for your approval",
    "execute_mutation": "💾 Applying the change",
    "apply_to_locked": "📝 Updating your locked plan",
    "show_locked": "📋 Showing your locked plan",
    "agent": "🤖 Deciding the next step",
    "tools": "🔧 Running a tool",
    "finalize": "✨ Wrapping up",
}


def _format_step(node: str, delta: dict) -> list[str]:
    """Format one completed node as trace lines — the node and its routing-relevant
    state delta (e.g. ``intent='recall'``), mirroring the terminal ``PLANNER_TRACE``
    output — plus any RAG loop sub-steps carried in ``delta['progress']``.
    """
    from agents.graph_orchestrator import _summarize_delta

    label = _PLANNER_STEP_LABELS.get(node, node)
    summary = _summarize_delta({k: v for k, v in (delta or {}).items() if k != "progress"})
    lines = [f"**▶ {label}** &nbsp;`{node}` → {summary}"]
    for sub in (delta or {}).get("progress", []):
        lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {sub}")
    return lines


def _render_steps(container, steps: list[str], *, expanded: bool = True) -> None:
    """Render an accumulated trace inside a (kept-open) expander on *container*."""
    with container.expander("🧭 Steps the agents took", expanded=expanded):
        for line in steps:
            st.markdown(line, unsafe_allow_html=True)


@contextmanager
def _planner_progress(label: str = "Working on your request…"):
    """Yield ``(on_step, steps)``.

    *on_step(node, delta)* writes each node's trace line into a live status box **and**
    accumulates it into *steps*, so the very same trace can be stored on the chat
    message and re-shown after the rerun — it no longer disappears. The box shows
    which node ran and where it routed (the delta), and stays expanded when done.
    """
    status = st.status(label, expanded=True)
    steps: list[str] = []

    def on_step(node: str, delta: dict) -> None:
        for line in _format_step(node, delta):
            steps.append(line)
            status.markdown(line, unsafe_allow_html=True)

    try:
        yield on_step, steps
    except Exception:
        status.update(label="⚠️ Something went wrong", state="error", expanded=True)
        raise
    else:
        status.update(label="✅ Done — steps below", state="complete", expanded=True)


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
            tools, model, temperature,
            checkpointer_db=cfg.CHECKPOINT_DB or None,
            observe=cfg.TRACE,  # log each node + state to the terminal when tracing
        )
        st.session_state.planner_key = key
    return st.session_state.planner


def _as_markdown(text: str) -> str:
    """Render preformatted text as Markdown without a code fence.

    Single newlines become hard line breaks (``  \\n``) so the plan keeps one
    item per line, while bold/emoji/etc. still render as Markdown.
    """
    return (text or "").replace("\n", "  \n")


def _consume_planner_result(result, steps: list[str] | None = None) -> None:
    """Turn a PlannerResult into chat output and/or a pending approval.

    *steps* (the trace collected during this run) is stored on the message so it
    re-renders in chat history after the rerun instead of vanishing.
    """
    # A resolved/new turn invalidates any field errors from a previous review.
    st.session_state.pop("planner_edit_errors", None)
    if result.status == "done":
        st.session_state.planner_pending = None
        st.session_state.messages.append(
            {"role": "ai", "content": _as_markdown(result.text), "steps": steps}
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
                f"📝 **Review needed** for `{intr.get('tool')}` — "
                "check the details in the table below, edit anything that's off, "
                "then submit."
            ),
            "steps": steps,
        }
    )


def _start_planner_turn(planner, prompt: str) -> None:
    """Start a planner run for *prompt* (may pause for approval).

    Uses the session's stable ``thread_id`` so the planner remembers the
    conversation across turns (the checkpointer keys state by thread_id) — a new
    id per turn would wipe its memory.
    """
    from core.tools.common_tools import get_today_date

    with _planner_progress() as (on_step, steps):
        result = planner.start(
            prompt,
            date=get_today_date.invoke({}),
            thread_id=st.session_state.thread_id,
            on_step=on_step,
        )
    _consume_planner_result(result, steps)


# Fields that describe the *operation* itself (which tool / what kind of write)
# rather than the task's data. These are hidden from the review table and never
# editable — the user reviews the task, not the plumbing.
HIDDEN_FIELDS = {"operation", "op", "action", "kind"}

# The task fields a CRUD mutation can carry, in the order to show them. `op` is
# hidden (plumbing); `task_id` only makes sense for update/delete and is added
# in front for those ops.
_MUTATION_FIELD_ORDER = (
    "title", "description", "priority", "status", "est_minutes", "category", "due",
)


def _is_mutation(args: dict) -> bool:
    """A CRUD-path interrupt carries an ``op`` field (a :class:`TaskMutation`)."""
    return isinstance(args, dict) and "op" in args


def _validate_mutation(args: dict) -> dict[str, list[str]]:
    """Return a ``{field: [messages]}`` map of validation errors (empty if valid).

    Mirrors the graph's ``TaskMutation.model_validate`` so the same edits that
    would be rejected server-side are caught here — letting us point at the bad
    field and keep the review open instead of silently dropping the change.
    """
    try:
        TaskMutation.model_validate(args)
        return {}
    except ValidationError as exc:
        errors: dict[str, list[str]] = {}
        for err in exc.errors():
            loc = err.get("loc") or ()
            field = str(loc[0]) if loc else "(form)"
            errors.setdefault(field, []).append(err.get("msg", "invalid value"))
        return errors


def _to_cell(value) -> str:
    """Render a tool-argument value as editable text for the review table."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _tool_schema(tool) -> dict:
    """Return the JSON-schema ``properties`` map for *tool* (empty if unknown).

    The MCP adapter stores each tool's ``inputSchema`` (a JSON-schema dict) as
    ``args_schema``; older/pydantic tools expose ``model_fields`` instead.
    """
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, dict):
        return schema.get("properties", {}) or {}
    if schema is not None and hasattr(schema, "model_fields"):
        return {name: {} for name in schema.model_fields}
    return {}


def _coerce_value(text: str, jtype: str | None, original):
    """Convert edited *text* back to the right type.

    Prefers the JSON-schema type (*jtype*); falls back to the *original* value's
    Python type when the schema is unknown. Empty text becomes ``None``.
    """
    text = "" if text is None else str(text)
    if text.strip() == "":
        return None

    jtype = jtype or {
        bool: "boolean",
        int: "integer",
        float: "number",
        list: "array",
        dict: "object",
    }.get(type(original))

    try:
        if jtype == "integer":
            return int(text)
        if jtype == "number":
            return float(text)
        if jtype == "boolean":
            return text.strip().lower() in ("true", "1", "yes", "y")
        if jtype in ("array", "object"):
            return json.loads(text)
    except (ValueError, TypeError):
        return text
    return text


def _rebuild_args(original_args: dict, edited_rows, properties: dict) -> dict:
    """Merge the user's edits back into the args, keeping hidden fields intact."""
    new_args = dict(original_args)  # preserves operation/op and any hidden fields
    for row in edited_rows:
        field = row.get("Field")
        if not field:
            continue
        jtype = (properties.get(field) or {}).get("type")
        value = _coerce_value(row.get("Value"), jtype, original_args.get(field))
        if value is None and field not in original_args:
            continue  # don't send empty optional fields the agent never set
        new_args[field] = value
    return new_args


def _render_plan_approval(planner, pending: dict) -> None:
    """Render the plan-confirmation panel: accept & lock, or re-plan with a hint."""
    intr = pending["interrupt"]
    st.info(intr.get("message", "Accept this plan for today?"))
    st.markdown(_as_markdown(intr.get("plan", "")))
    suggestion = st.text_input(
        "Suggest a change (optional) — e.g. \"I can work until 11pm\" or "
        "\"keep 3-4pm free for a call\"",
        key="plan_suggestion",
    )
    accept, reject = st.columns(2)
    if accept.button("✅ Accept & lock", use_container_width=True, key="plan_accept"):
        with _planner_progress("Locking the plan…") as (on_step, steps):
            result = planner.resume(pending["thread_id"], {"action": "accept"}, on_step=on_step)
        st.session_state.pop("plan_suggestion", None)
        _consume_planner_result(result, steps)
        st.rerun()
    if reject.button("✏️ Re-plan with my suggestion", use_container_width=True, key="plan_reject"):
        with _planner_progress("Re-planning…") as (on_step, steps):
            result = planner.resume(
                pending["thread_id"], {"action": "reject", "reason": suggestion}, on_step=on_step
            )
        st.session_state.pop("plan_suggestion", None)
        _consume_planner_result(result, steps)
        st.rerun()
    if st.button("🚫 Cancel", use_container_width=True, key="plan_cancel"):
        with _planner_progress("Cancelling…") as (on_step, steps):
            result = planner.resume(pending["thread_id"], {"action": "cancel"}, on_step=on_step)
        st.session_state.pop("plan_suggestion", None)
        _consume_planner_result(result, steps)
        st.rerun()


def _render_approval_panel(planner, tools: list[BaseTool]) -> None:
    """Render an editable review table for a pending mutating action.

    Shows **every** field of the task object (from the tool's schema), pre-filled
    with the agent's proposal and blank where it left a field empty, so the user
    can complete or correct any detail — due date, description, estimate,
    priority, category, … — before submitting. Operation/plumbing fields are
    hidden and the tool itself cannot be swapped.
    """
    pending = st.session_state.planner_pending
    intr = pending["interrupt"]
    # Plan confirmation uses a lighter accept/suggest panel, not the edit table.
    if intr.get("type") == "plan_approval":
        _render_plan_approval(planner, pending)
        return
    args = intr.get("args") or {}
    mutation = _is_mutation(args)

    # Decide which fields to show:
    #  • create — every task property (blank where the agent left it), so the user
    #    can complete a brand-new task.
    #  • update — only the fields actually being changed (the task's identity is in
    #    the confirmation message, not re-edited here).
    #  • delete — nothing to edit; just confirm.
    #  • non-CRUD tools — the live tool's JSON schema.
    if mutation:
        properties = {}
        op = args.get("op")
        if op == "create":
            field_names = list(_MUTATION_FIELD_ORDER)
        elif op == "delete":
            field_names = []
        else:  # update — just the changed fields
            field_names = [
                f for f in args if f not in ("op", "task_id")
            ]
    else:
        tool = next((t for t in tools if t.name == intr.get("tool")), None)
        properties = _tool_schema(tool)
        field_names = list(properties.keys())
        # Append any extra args present but not already listed.
        for name in args:
            if name not in field_names:
                field_names.append(name)
    visible = [f for f in field_names if f.lower() not in HIDDEN_FIELDS]

    st.warning(
        f"**{intr.get('message', 'Review this action before saving.')}**\n\n"
        f"Tool: `{intr.get('tool')}` — review and edit the details below, then submit."
    )

    # Show errors from a previous (rejected) submit and point at the bad fields.
    errors = st.session_state.get("planner_edit_errors") or {}
    if errors:
        lines = "\n".join(
            f"- **{field}**: {'; '.join(msgs)}" for field, msgs in errors.items()
        )
        st.error("Some fields need fixing before this can be saved:\n" + lines)

    if visible:
        rows = [{"Field": f, "Value": _to_cell(args.get(f))} for f in visible]
        edited_rows = st.data_editor(
            rows,
            key="planner_edit_table",
            use_container_width=True,
            hide_index=True,
            disabled=["Field"],
            column_config={
                "Field": st.column_config.TextColumn("Field", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="large"),
            },
        )
    else:
        # Nothing to edit (e.g. a delete) — confirm or reject only.
        edited_rows = []

    submit, reject = st.columns(2)
    if submit.button("✅ Submit", use_container_width=True, key="planner_submit"):
        new_args = _rebuild_args(args, edited_rows, properties)
        # Validate CRUD mutations up front: on failure, keep the review open with
        # the offending fields flagged so the user can correct and resubmit —
        # rather than the change being silently dropped downstream.
        validation_errors = _validate_mutation(new_args) if mutation else {}
        if validation_errors:
            st.session_state.planner_edit_errors = validation_errors
            st.rerun()
        st.session_state.pop("planner_edit_errors", None)
        with _planner_progress("Applying…") as (on_step, steps):
            result = planner.resume(
                pending["thread_id"], {"action": "edit", "args": new_args}, on_step=on_step
            )
        _consume_planner_result(result, steps)
        st.rerun()
    if reject.button("❌ Reject", use_container_width=True, key="planner_reject"):
        with _planner_progress("Cancelling…") as (on_step, steps):
            result = planner.resume(
                pending["thread_id"], {"action": "reject", "reason": "user declined"}, on_step=on_step
            )
        _consume_planner_result(result, steps)
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
