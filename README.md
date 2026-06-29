# To-do Agent

A local, **skill-aware** AI task-management assistant. It runs entirely against a
local [Ollama](https://ollama.com) model, manages your tasks through an
[MCP](https://modelcontextprotocol.io) server, and is driven by a
[Streamlit](https://streamlit.io) chat UI.

## How it works

The agent uses the **progressive-disclosure skill pattern** (the same idea behind
Anthropic's Agent Skills):

1. Each skill lives in `src/skills/<skill>/SKILL.md` with YAML front-matter
   (`name`, `description`) and a markdown body of full instructions.
2. At startup only every skill's **name + description** is injected into the agent's
   system prompt — keeping the context window small.
3. A single LangGraph tool-calling agent is given:
   - a `get_skill_detail` tool that returns a skill's **full instructions** on demand, and
   - the live **MCP tools** for reading/modifying tasks.
4. The model **routes itself**: when a request matches a skill it calls
   `get_skill_detail`, reads the instructions, then follows them using the MCP tools.

This replaces the older approach of a separate "planner" LLM call that pre-selected
skills — fewer model calls, simpler control flow, and the routing decision lives next
to the work that uses it.

Conversation state persists across turns via a shared LangGraph checkpointer keyed by
`thread_id`.

> The day-to-day task work (planning, summarizing, CRUD) now lives in the **multi-agent
> planner** below, which is the default mode. The skill-aware chat agent is kept as a
> general assistant; **no skills ship by default** — drop a `SKILL.md` into `src/skills/`
> to add one.

## Multi-agent planner

Alongside the skill-aware assistant there is a **multi-agent daily planner**, selectable
in the sidebar as the **Planner (Multi-Agent)** mode. It is a hub-and-spoke system: an
orchestrator classifies the request and routes it to specialist agents that communicate
**only through a schema-validated contract** — never free text.

```
user prompt
   │
   ▼
Orchestrator ── classify intent (plan | summary | create/update/delete | recall)
   │
   ▼
TodoAgent (data specialist)
   • fetch raw tasks (MCP tool, else sample data)
   • normalize → estimate duration, categorize, set priority
   • emit a Pydantic-validated TaskList   ◄── the handoff contract
   │   (malformed model output is caught and retried, never passed on)
   ▼
DailyPlannerAgent (reasoning specialist)
   • rank by priority → fit into the working hours, around meal breaks
   • DECISION: does it all fit?           ◄── what makes it an agent
       ├─ yes → schedule everything (lunch/dinner kept clear)
       └─ no  → defer the lowest-priority tasks, report what was dropped
   │
   ▼
Orchestrator → formatted, time-blocked day
```

**Design choices**

- **The contract is the linchpin** ([`models/contract.py`](src/models/contract.py)). `Task`/`TaskList`
  are bounded Pydantic models (`est_minutes` is range-checked, extra fields are forbidden),
  so a hallucinated task **cannot** flow downstream — it fails validation and the step retries.
- **The planner fits a real day, deterministically** ([`agents/planner_agent.py`](src/agents/planner_agent.py)).
  A `Workday` (start, end, meal breaks) defines the timeline; tasks are ranked by priority and laid out
  around lunch/dinner, and whatever doesn't fit is deferred. It's pure Python — no LLM call — so the
  graded "what do I cut?" behaviour stays reliable and unit-testable.
- **Writes go through a confirmed CRUD path.** A create/update/delete request takes a dedicated
  deterministic route in the graph: `TodoAgent` parses it into a **validated `TaskMutation`** (the
  write-side contract in [`models/contract.py`](src/models/contract.py) — an under-specified change,
  e.g. a delete with no task identified, is rejected and retried), the graph **pauses for human
  approval** (LangGraph `interrupt`), the change is applied via the MCP write tools, and the day is
  **re-planned** so you see the effect. Reads stay autonomous. (The generic agent⇄tools loop also
  gates its mutating tools, via [`agents/human_in_the_loop.py`](src/agents/human_in_the_loop.py).)
  Create/update can set any task field — title, description, priority, status,
  **estimated minutes**, **category** (work/home/health/…), and due date.
- **Plans are confirmed, then locked.** A generated plan pauses for the user to **Accept** (lock
  it for the day) or **Reject** with a suggestion (re-plan with it folded in — the confirm loop).
  Once locked, task changes patch the locked plan **in place** — marking a task done annotates it
  (`✅ ~~title~~`), no re-schedule; the plan is only regenerated on an explicit "replan".
  See `confirm_plan`/`show_locked`/`apply_to_locked` and `apply_mutation_to_plan`
  ([agents/planner_agent.py](src/agents/planner_agent.py)).
- **Approvals survive restarts.** The graph is compiled with a **persistent SQLite checkpointer**
  ([`core/services/checkpoint.py`](src/core/services/checkpoint.py)), so a pending approval — and the
  conversation thread — can be resumed even after the process restarts. It falls back to an in-memory
  saver when the SQLite checkpointer isn't installed.
- **Two orchestrator implementations, same agents.** A plain-Python one
  ([`agents/orchestrator.py`](src/agents/orchestrator.py)) — a `match` on intent, clearest to read —
  and a LangGraph one ([`agents/graph_orchestrator.py`](src/agents/graph_orchestrator.py)) where each
  step is a **node**, routing is **edges**, and the contract is a typed field on the shared **state**.
  Both register every agent in **one place** (`build_orchestrator` / `build_graph_orchestrator`), so
  adding a third agent is a single new branch/node. The Streamlit **Planner** mode uses the graph.
- **Complex / multi-step prompts loop — and the loop is an edge.** The graph's open-ended path is the
  classic ReAct cycle `agent → tools → agent → … → END`: a **tool-calling agent** decides which tool to
  call, a `ToolNode` runs it, and a **conditional edge** (`should_continue`) routes back to the agent
  while tool calls remain. The loop's memory is the message list in state, bounded by `recursion_limit`.
  The deterministic capabilities are themselves exposed as tools (`plan_my_day`, `summarize_tasks`)
  alongside **all** the local helper tools (date, **weather**) and the MCP CRUD tools, so a request that
  bundles several intents — *"create a task to call the bank, delete the gym task, re-plan my day, and
  what's the weather in Hanoi?"* — is parsed into a checklist and sequenced by the loop, one tool per
  iteration ([`prompts/complex_agent_system.md`](src/prompts/complex_agent_system.md) tells it to handle
  every part). A prompt that matches more than one intent family (or a non-task one like weather) is
  routed here automatically; single `plan`/`summary`/CRUD requests stay on the loop-free deterministic path.
- **Recall is an agentic RAG branch (the third agent).** A `recall` intent ("what do I usually
  defer when busy?", "what does my reference doc say…?") routes to a **`rag`** node backing the
  [`RAGAgent`](src/agents/rag_agent.py): a real **retrieve → judge → generate loop** (not one
  pass) that picks `retrieve_log` vs `retrieve_document`, validates each result against
  `DataChunk`, judges sufficiency (structured `RetrievalDecision`, with a heuristic fallback), and
  **reformulates + retries** up to `MAX_ITERATIONS` before answering grounded only in what it found.
  The data lives behind a separate **[memory MCP server](../mcps/memory-mcp)**; locking a plan
  auto-writes a `planning_log` summary there (the planner↔recall feedback loop). Adding this agent
  was **one** new branch in the graph (`route_by_intent` → `rag`) — the hub-and-spoke property.
- **The agent knows its tools.** The loop's system prompt is built from the live tool set (name +
  one-line description for every bound tool), so the model is told exactly what it can call.
- **Graceful degradation.** With the MCP server down it uses sample tasks; with Ollama down it uses
  a deterministic heuristic normalizer and keyword classification — the mode still works offline.

> **Tool reliability (deliverable #4).** Input validation, idempotency, and recoverable
> structured errors belong on the **MCP server** (a separate process/repo, not in this app).
> What this app controls — validating the data crossing the agent boundary, retrying on bad
> output, returning recoverable errors, and logging — lives in `TodoAgent` and the orchestrator
> runtime helpers.

**Try it offline** (no Ollama or MCP server needed):

```bash
python src/run_orchestrator.py "plan my day"
python src/run_orchestrator.py "plan my day" --day-end 12:00   # short day -> overload
python src/run_orchestrator.py "what's on my list?"
```

**Tests** (run without an LLM or MCP server):

```bash
pytest tests/ -q
```

**Visualize / observe the graph** (offline; writes `planner_graph.mmd` + `.png`):

```bash
python src/visualize_graph.py                  # draw the graph (Mermaid + PNG)
python src/visualize_graph.py "plan my day"    # also trace each node + state delta
```

The step observer is also available in any run by passing `observe=True` to
`build_graph_orchestrator(...)` or setting `PLANNER_TRACE=1` — each node logs
`[trace] ▶ <node>` and `[trace] ✓ <node> → <state delta>`.

To trace the **Streamlit app** in the Planner mode, launch it with the flag set —
the steps print in the terminal running Streamlit:

```bash
PLANNER_TRACE=1 streamlit run src/app.py     # or: python src/run_orchestrator.py "plan my day" --trace
```

## Project layout

This project and the [task MCP server](../mcps/task-mcp) share one skeleton —
`models/` (data), `core/services/` (logic + clients), `core/tools/` (tool
factories), `configs/`, and `utils/` — so the same mental map carries across
both. The agent app adds `agents/`, `prompts/`, `skills/`, and `ui/` on top.

```
src/
├── app.py                  # Streamlit entrypoint: load skills + MCP tools, render UI
├── run_orchestrator.py     # Offline CLI demo for the multi-agent planner
├── agents/
│   ├── ollama_agent.py     # The single skill-aware agent + streaming bridge
│   ├── todo_agent.py       # Multi-agent: data specialist (raw → validated TaskList)
│   ├── planner_agent.py    # Multi-agent: reasoning specialist (rank, time-block, defer)
│   ├── orchestrator.py     # Multi-agent: plain-Python classify + route + runtime wiring
│   ├── graph_orchestrator.py # Multi-agent: LangGraph StateGraph (nodes/edges + agent⇄tools loop)
│   └── human_in_the_loop.py  # Approval gate (interrupt) wrapping mutating tools
├── models/                 # Data models, no I/O
│   ├── contract.py         # Handoff contract (Task, TaskList, DayPlan) + write contract (TaskMutation)
│   └── skill.py            # Skill (SKILL.md parser)
├── core/
│   ├── services/           # Business logic + external clients
│   │   ├── checkpoint.py   #   Persistent (SQLite) LangGraph checkpointer, in-memory fallback
│   │   ├── llm_client.py   #   ChatOllama factory
│   │   ├── mcp_client.py   #   MCP server connection + tool loading
│   │   ├── prompts.py      #   Loads prompts from src/prompts/
│   │   └── skills.py       #   Skill loading + roster formatting
│   └── tools/              # LangChain tool factories
│       ├── common_tools.py #   Always-available native @tool helpers (date, weather, …)
│       └── skill_tools.py  #   get_skill_detail — the on-demand skill loader tool
├── prompts/
│   ├── agent_system.md            # Skill-agent system prompt ({skills_roster} slot)
│   ├── todo_agent_system.md       # TodoAgent normalization prompt
│   ├── crud_agent_system.md       # TodoAgent CRUD-request parsing prompt
│   ├── intent_classifier_system.md# Orchestrator intent prompt
│   └── complex_agent_system.md    # Tool-agent prompt for the graph's loop path
├── skills/                 # Optional chat-agent skills (none shipped by default)
├── ui/                     # Streamlit sidebar + main chat view
├── configs/settings.py     # Config (env-overridable)
└── utils/json_utils.py     # Generic JSON helpers

tests/                      # Unit tests (no LLM / MCP required)
```

## Setup

Requires Python 3.12+ and a running Ollama instance.

```bash
# 1. Install dependencies (uv recommended; pip also works)
uv sync                      # or: pip install -r requirements.txt

# 2. Pull the model you configured (see settings below)
ollama pull <your-model>

# 3. Start the task MCP server so it serves SSE at the configured URL
#    (default: http://localhost:8000/sse)

# 4. Run the app
streamlit run src/app.py
```

## Configuration

Defaults live in `src/configs/settings.py` and can be overridden with environment
variables (a `.env` file at the project root is loaded automatically):

| Variable             | Default                       | Purpose                          |
| -------------------- | ----------------------------- | -------------------------------- |
| `OLLAMA_MODEL`       | `gemma4:31b-cloud`            | Ollama model name                |
| `OLLAMA_TEMPERATURE` | `0.2`                         | Default sampling temperature     |
| `OLLAMA_HOST`        | `http://localhost:11434`      | Ollama server URL                |
| `TASK_MCP_URL`       | `http://localhost:8000/sse`   | Task MCP server SSE endpoint     |
| `SKILL_SOURCES`      | `src/skills`                  | Directory scanned for `SKILL.md` |
| `CHECKPOINT_DB`      | `agent-checkpoints.db`        | SQLite file for the persistent checkpointer (set empty to disable) |

## Adding a skill

Create `src/skills/<your-skill>/SKILL.md`:

```markdown
---
name: my_skill
description: >
  One or two sentences describing exactly when this skill should be used,
  including trigger phrases. This is what the agent sees when deciding to load it.
---

# My Skill

Full step-by-step instructions, MCP tool usage, and behavioral rules go here.
They are only loaded when the agent calls `get_skill_detail("my_skill")`.
```

No code changes are needed — skills are discovered automatically at startup.
