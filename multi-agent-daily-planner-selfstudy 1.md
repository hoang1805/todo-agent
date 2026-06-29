# Self-Study: Building a Multi-Agent Daily Planner

**Format:** read this guide, then implement the project yourself. The code blocks are *hints and scaffolding*, not a finished solution — the parts that teach you the most are deliberately left as `TODO`s for you to write.

**Time:** one week. **Prerequisite:** last week's project (TodoAgent + the daily-planner skill + your MCP server).

---

## 0. Where you're starting from

Last week you built:

- a **TodoAgent** that reads tasks (from your task store via an MCP server) and summarizes the day,
- a **daily-planner skill** that turned a task list into a prioritized plan,
- an **MCP server** exposing task tools (`list_tasks`, `add_task`, `complete_task`).

This week the planner **graduates from a skill into a full agent**, and you make two agents work together. That single change — skill to agent — is the heart of the week, so it's worth being precise about what it means:

> **A skill is a procedure** the agent follows. **An agent has its own goal, its own loop, and its own tools, and it makes decisions.** The planner becomes an agent because it now has to *decide* things (what to cut when the day is overloaded), *use tools itself* (read tasks), and *loop* (re-plan after a change). If your planner just sorts tasks and prints them, you've built a sorter, not an agent.

---

## 1. This week's objectives

You are being assessed on two things:

1. **Improve prompts + tool reliability** — make the agents produce structured, predictable output, and make your tools fail safely and recoverably.
2. **Build agent logic + workflows** — give the Daily Planner Agent real decision-making, and coordinate it with the TodoAgent.

Keep these two in mind as you read the deliverables — every task below maps back to one or both.

---

## 2. The architecture: why an orchestrator

You have two reasonable ways to connect two agents.

**Pipeline (sequential):** `TodoAgent → DailyPlannerAgent`. The agents are wired directly to each other. Simple, easy to debug, fine for exactly two agents.

**Orchestrator (hub-and-spoke):** a thin coordinator receives the user's request and routes it to the right agent(s). The agents are wired only to the coordinator, **not to each other**.

**Use the orchestrator.** Here's the reason, and it's the whole point: when you add a third agent later (next week you'll add a memory/RAG service), the orchestrator only changes in **one place** — you register the new agent and describe what it's for. In a pipeline you'd have to cut into the chain and re-wire the neighbours every time. Hub-and-spoke means *N agents = N connections to the hub*, not a chain you keep re-threading.

You'll still *think* in pipeline terms for the data (Todo's output feeds the planner), but the control flow goes through the orchestrator.

### The two agents have different jobs

This is *why* you split into two agents at all:

- **TodoAgent = the data specialist.** Fetch, deduplicate, categorize, estimate durations, assign priority. Its output is clean structured data.
- **DailyPlannerAgent = the reasoning specialist.** Rank, time-block, and *decide* what to defer when things don't fit.

Each has a narrow system prompt and a small tool set, which is exactly what makes each one independently testable and reliable. That separation is the argument for multi-agent over one big do-everything agent. (The tradeoff: more moving parts and coordination cost. You split when responsibilities genuinely differ — which here they do.)

---

## 3. What you'll build (4 deliverables)

Build them in **this order** — building the orchestrator first leaves you coordinating nothing.

| # | Deliverable | Serves objective | Why it matters |
|---|-------------|------------------|----------------|
| 1 | **The handoff contract** | prompts + reliability | The linchpin. A validated data structure passed between agents. |
| 2 | **DailyPlannerAgent** | agent logic | The planner with a real overload *decision*. |
| 3 | **The Orchestrator** | agent logic | Routes the user's request; wires the agents together. |
| 4 | **Tool reliability hardening** | reliability | Validation, recoverable errors, idempotency, logging on your MCP tools. |

### 3.1 The handoff contract — *do this first*

Instead of TodoAgent handing the planner a paragraph of text, it hands a **schema-validated list of task objects**. This is small to state but it is where both objectives physically live: prompting for structured output (objective 1a) *and* validating that output (objective 1b) happen right here.

The rule: **never pass unvalidated data downstream.** If TodoAgent's LLM emits malformed JSON, you catch it at this boundary and retry that step — you do *not* let the planner choke on garbage.

### 3.2 DailyPlannerAgent

The planning loop: fetch the contract → rank tasks → time-block across available hours → **decide if it all fits** → if not, defer/cut the lowest-priority tasks and re-plan → return the schedule.

The diamond — *does it all fit?* — is the required, graded part. That branch is the difference between an agent and a script.

### 3.3 The Orchestrator

Receives the user prompt, classifies intent (`summary` / `plan` / `add task`), and routes. For "plan my day" it needs tasks first, so it runs TodoAgent, validates the contract, then runs the planner.

### 3.4 Tool reliability hardening

Applied to the MCP server you already have: validate inputs, return **recoverable** structured errors, make actions idempotent, and log every tool call.

---

## 4. Tooling

What I'd reach for, given you're self-hosting with Ollama:

- **Pydantic** — *the most important library this week.* Your handoff contract **is** a Pydantic model. You get validation, clear errors, and JSON schema generation for free.
- **Ollama structured output** — Ollama accepts a JSON schema via its `format` parameter and will constrain generation toward it. You generate that schema *from* your Pydantic model, so the model is steered toward valid output **and** you validate on the way in. Belt and suspenders.
- **LangGraph** — for the orchestrator. It models agents as **nodes** in a graph, routing as **edges**, and the data that flows as a shared **state** object. That maps one-to-one onto the orchestrator idea you're implementing, so the framework makes your architecture literal instead of hiding it.

**Allowed alternative:** you may build the orchestrator as **plain Python** — a function with a `match` on intent, agents as classes with a `run()` method, the contract as Pydantic. This is arguably clearer because nothing is hidden; you just lose the clean graph abstraction when agent #3 arrives. Either choice is acceptable. If LangGraph is slowing you down, drop it and go raw — **Pydantic is the only non-negotiable.**

**Avoid this week:** CrewAI, AutoGen, and similar high-level multi-agent frameworks. They abstract away the orchestration and contract logic that *is* this week's learning objective. Wrong tool for a learning week.

---

## 5. Data flow: user prompt → response

Trace this whole path before you write code. The two marked points are where the grading attention goes.

```
1. USER PROMPT
   "Plan my day"
        │
        ▼
2. ORCHESTRATOR  — classify intent
   decides: summary | plan | add task   → routes.
   For "plan", it needs tasks first.
        │
        ▼
3. TodoAgent  — the data specialist
   • calls MCP tool list_tasks → raw JSON from the task store
   • normalizes: dedupe, categorize, estimate est_minutes, set priority
   • emits the CONTRACT
        │
        ▼
   ┌──────────────────────────────────────────────┐
   │  HANDOFF CONTRACT  (Pydantic-validated)  ◄── checkpoint
   │  List[Task]                                   │
   │  Task = { id, title, priority,                │
   │           est_minutes, category, due }        │
   │  validation fails? → bounce back to TodoAgent │
   │  to fix. Do NOT pass garbage downstream.      │
   └──────────────────────────────────────────────┘
        │
        ▼
4. DailyPlannerAgent  — the reasoning specialist
   • rank (urgent / important)
   • time-block across available hours
   • DECISION: does it all fit?            ◄── this is what makes it an agent
        ├─ yes → build schedule
        └─ no  → defer/cut lowest-priority, then re-plan
        │
        ▼
5. ORCHESTRATOR  — collect result, format
        │
        ▼
6. RESPONSE — the time-blocked day
```

If you use LangGraph: steps 2/3/4/5 are **nodes**, the arrows are **edges**, and the contract is a typed field on the shared **state**.

---

## 6. Implementation hints

These are scaffolds. The `TODO`s are where your work goes — don't expect to copy-paste your way to a finished project.

### 6.1 The contract (Pydantic)

```python
# contract.py
from enum import Enum
from pydantic import BaseModel, Field

class Priority(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"

class Task(BaseModel):
    id: int
    title: str
    priority: Priority
    est_minutes: int = Field(gt=0, le=480)   # sane bounds = free validation
    category: str
    due: str | None = None                    # ISO date string, optional

class TaskList(BaseModel):
    tasks: list[Task]
```

That's your entire inter-agent contract. Because it's typed and bounded, a malformed task **cannot** silently pass through.

### 6.2 Ollama structured output

Steer the model toward valid JSON, then validate it:

```python
import ollama
from contract import TaskList

def normalize_tasks(raw_tasks: list[dict]) -> TaskList:
    schema = TaskList.model_json_schema()        # schema FROM your model
    resp = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": TODO_TODO_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Raw tasks:\n{raw_tasks}"},
        ],
        format=schema,                            # Ollama constrains output
    )
    # Validate on the way IN. If this raises, catch it and retry the step.
    return TaskList.model_validate_json(resp["message"]["content"])
```

> Note: you can also hit Ollama's OpenAI-compatible endpoint at `http://localhost:11434/v1` if you prefer that client. Same idea.

### 6.3 The Orchestrator

**LangGraph flavor (sketch):**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    user_input: str
    intent: str
    tasks: TaskList | None
    result: str | None

def classify(state):   # TODO: LLM or keyword classify → "summary"|"plan"|"add"
    ...
def run_todo(state):   # calls TodoAgent, validates contract, stores in state
    ...
def run_planner(state):# calls DailyPlannerAgent on state["tasks"]
    ...

g = StateGraph(State)
g.add_node("classify", classify)
g.add_node("todo", run_todo)
g.add_node("planner", run_planner)
# TODO: add edges. "plan" → todo → planner → END ; "summary" → todo → END
g.set_entry_point("classify")
app = g.compile()
```

**Plain-Python flavor (equally acceptable):**

```python
def orchestrate(user_input: str) -> str:
    intent = classify(user_input)               # TODO
    match intent:
        case "plan":
            tasks = todo_agent.run()            # returns validated TaskList
            schedule = planner_agent.run(tasks, available_minutes=480)
            return format_schedule(schedule)
        case "summary":
            ...
        case "add":
            ...
```

Notice both versions register each agent in exactly **one** place — that's the scaling property you're buying.

---

## 7. Definition of done

You're finished when all of these are true:

- [ ] TodoAgent returns a **Pydantic-validated** `TaskList`; malformed model output is caught and retried, never passed on.
- [ ] The two agents communicate **only** through the contract — no free-text handoff.
- [ ] DailyPlannerAgent produces a time-blocked schedule for a normal day.
- [ ] DailyPlannerAgent **makes a real decision when the day is overloaded** (defers/cuts lowest-priority, reports what it dropped) — and you can demo this with an overloaded task list.
- [ ] The orchestrator routes at least `plan` and `summary` correctly.
- [ ] Every MCP tool validates inputs, returns recoverable errors, is idempotent, and logs.
- [ ] You have unit tests for the MCP tools that run without the LLM.
- [ ] Adding a hypothetical new agent would require changes in **one** place (the orchestrator). Be ready to point to where.

---

## 8. Common pitfalls

- **Building a sorter, not an agent.** If the planner has no "does it fit?" decision, it isn't an agent. This is the most common miss.
- **Free-text handoff.** Passing a paragraph between agents instead of the validated contract defeats both objectives at once.
- **Orchestrator first.** Building the coordinator before the agents exist means you're wiring up nothing. Contract → planner → orchestrator.
- **Throwing instead of returning errors.** A tool that raises gives the LLM nothing to recover from. Return a structured error that says what's valid.
- **Over-engineering.** You don't need a framework to prove a point. Reach for LangGraph only if it's making things clearer, not harder.
