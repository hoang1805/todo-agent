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

## Multi-agent planner

Alongside the skill-aware assistant there is a **multi-agent daily planner**, selectable
in the sidebar as the **Planner (Multi-Agent)** mode. It is a hub-and-spoke system: an
orchestrator classifies the request and routes it to specialist agents that communicate
**only through a schema-validated contract** — never free text.

```
user prompt
   │
   ▼
Orchestrator ── classify intent (plan | summary | add)
   │
   ▼
TodoAgent (data specialist)
   • fetch raw tasks (MCP tool, else sample data)
   • normalize → estimate duration, categorize, set priority
   • emit a Pydantic-validated TaskList   ◄── the handoff contract
   │   (malformed model output is caught and retried, never passed on)
   ▼
DailyPlannerAgent (reasoning specialist)
   • rank → time-block across the available hours
   • DECISION: does it all fit?           ◄── what makes it an agent
       ├─ yes → schedule everything
       └─ no  → defer the lowest-priority tasks, report what was dropped
   │
   ▼
Orchestrator → formatted, time-blocked day
```

**Design choices**

- **The contract is the linchpin** ([`core/contract.py`](src/core/contract.py)). `Task`/`TaskList`
  are bounded Pydantic models (`est_minutes` is range-checked, extra fields are forbidden),
  so a hallucinated task **cannot** flow downstream — it fails validation and the step retries.
- **The overload decision is deterministic Python** ([`agents/planner_agent.py`](src/agents/planner_agent.py)),
  not an LLM call. That keeps the graded "what do I cut?" behaviour reliable and unit-testable
  without a model running.
- **The orchestrator is plain Python** ([`agents/orchestrator.py`](src/agents/orchestrator.py)) —
  intent classification falls back to keyword rules, and every agent is registered in **one place**
  (`build_orchestrator`), so adding a third agent is a single new branch.
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
python src/run_orchestrator.py "plan my day" --available-minutes 180   # force an overload
python src/run_orchestrator.py "what's on my list?"
```

**Tests** (run without an LLM or MCP server):

```bash
pytest tests/ -q
```

## Project layout

```
src/
├── app.py                  # Streamlit entrypoint: load skills + MCP tools, render UI
├── run_orchestrator.py     # Offline CLI demo for the multi-agent planner
├── agents/
│   ├── ollama_agent.py     # The single skill-aware agent + streaming bridge
│   ├── todo_agent.py       # Multi-agent: data specialist (raw → validated TaskList)
│   ├── planner_agent.py    # Multi-agent: reasoning specialist (rank, time-block, defer)
│   └── orchestrator.py     # Multi-agent: classify intent + route + runtime wiring
├── core/
│   ├── contract.py         # Pydantic handoff contract (Task, TaskList, DayPlan)
│   ├── common_tools.py     # Always-available native @tool helpers (get_today_date, …)
│   ├── llm_client.py       # ChatOllama factory
│   ├── mcp_client.py       # MCP server connection + tool loading
│   ├── models.py           # Skill (SKILL.md parser)
│   ├── prompts.py          # Loads prompts from src/prompts/
│   └── skills.py           # Skill loading, roster formatting, get_skill_detail tool
├── prompts/
│   ├── agent_system.md            # Skill-agent system prompt ({skills_roster} slot)
│   ├── todo_agent_system.md       # TodoAgent normalization prompt
│   └── intent_classifier_system.md# Orchestrator intent prompt
├── skills/
│   ├── daily-planner/SKILL.md
│   └── task-executor/SKILL.md
├── ui/                     # Streamlit sidebar + main chat view
└── configs/settings.py     # Config (env-overridable)

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
