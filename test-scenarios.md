# Test Scenarios — Multi-Agent Daily Planner

Manual test plan covering every feature of the **Planner (Multi-Agent)** mode.
Type the prompts in the Streamlit chat; each row notes what it exercises and what
you should see.

## Prerequisites

1. **Ollama** running, with the chat model and `embeddinggemma` pulled.
2. **Task MCP** server on `:8000` and **Memory MCP** server on `:8002`
   (`cd ../todo-agent-mcps/memory-mcp && uv run python src/main.py`).
3. **Documents ingested** into the memory store (for the recall/document tests):
   `cd ../todo-agent-mcps/memory-mcp && python ingest_documents.py data/*.txt`
   then restart the memory server so it reloads.
4. Launch with the trace on so you can watch routing:
   `PLANNER_TRACE=1 streamlit run src/app.py` → pick **Planner (Multi-Agent)**.

> Tip: the live **🧭 Steps** box in each reply shows the node path + routing, so you
> can confirm the *intent* each prompt was classified as.

---

## A. Planning & the confirm loop

| # | Type this | Expect |
|---|-----------|--------|
| A1 | `plan my day` | A time-blocked plan, then a pause asking you to **Accept / Re-plan / Cancel**. Intent `plan` → `todo` → `planner` → `confirm_plan`. |
| A2 | (A1, then) **✅ Accept & lock** | Plan is locked (`🔒 Plan locked for today`) and a `planning_log` summary is written to memory. |
| A3 | (A1, then) **Re-plan** with *"I have lunch from 11h to 13h30"* | Re-plan blocks **11:00–13:30** for the lunch and drops the default 12:00 lunch. Tests appointment-from-suggestion. |
| A4 | (A1, then) **Re-plan** with *"actually I can work 8am to 10pm"* | Day window widens to 08:00–22:00 and the plan re-fits. Tests working-hours parsing. |
| A5 | (A1, then) **🚫 Cancel** | "Plan discarded — nothing was locked." No lock, chat input re-enabled. |
| A6 | (no lock) `plan my day` on a short day — re-plan with *"I can only work until 12pm"* | Day is **overloaded**; lowest-priority tasks listed under "deferred". |

## B. Appointments (LLM extraction, with regex fallback)

Use these as **Re-plan suggestions** (A1 → Re-plan) or stand-alone prompts.

| # | Type this | Expect |
|---|-----------|--------|
| B1 | `I have a dentist appointment at 3pm` | Registers **15:00–16:00** (single time → 1-hour block); day re-planned around it. |
| B2 | `lunch at noon for 2 hours` | **12:00–14:00** (worded time + duration — regex alone can't do this). |
| B3 | `dinner with family from 17:30 to 19:30` | **17:30–19:30**, replacing the default 18:00 dinner. |
| B4 | `team meeting from 14h to 15h` | **📌 team meeting** (not 🍽️ — a meeting isn't eating). |
| B5 | `I can work until 11pm` | **Not** an appointment — treated as working hours, not a blocked event. |

## C. "What's at this time?" lookup (`at_time`)

Lock a plan first (A1 → Accept).

| # | Type this | Expect |
|---|-----------|--------|
| C1 | `which task do I have at 7pm?` | The single block scheduled at 19:00 (or "Next up is …" if that slot is free). |
| C2 | `what's on at 12:30?` | The Lunch break. |
| C3 | `what do I have at 17:30?` (a free gap) | "Nothing is scheduled at 17:30. Next up is Dinner at 18:00." |
| C4 | `I have lunch at 7pm` | This is a **statement** → routed to `appointment`, **not** `at_time`. Confirms question-vs-statement disambiguation. |

## D. Summary & detail

| # | Type this | Expect |
|---|-----------|--------|
| D1 | `what's on my list?` | A summary of all tasks (priority, category, est. minutes). Intent `summary`. |
| D2 | `tell me about the report task` | Details of the one matching task. Intent `detail`. |

## E. CRUD with human-in-the-loop approval

| # | Type this | Expect |
|---|-----------|--------|
| E1 | `add a task to call the bank tomorrow` | An editable **review table** → **Submit** creates it, then the day re-plans (or patches the locked plan). |
| E2 | (with a locked plan) `mark the PR review done` | Patches the locked plan **in place**: `✅ ~~PR review~~`, no full re-schedule. |
| E3 | `delete the gym task` | Approval prompt → execute → re-plan. |
| E4 | (E1, then) **❌ Reject** | The change is **not** applied. |
| E5 | (E1) edit a field (e.g. priority/est_minutes) in the table, then Submit | The edited values are validated and saved; bad values keep the review open with the field flagged. |

## F. Recall — agentic RAG (the third agent)

Requires ingested documents (F1/F3/F4/F5) and ≥1 locked plan (F2).

| # | Type this | Expect |
|---|-----------|--------|
| F1 | `what does my reference doc say about deploying on Friday?` | Routes to `recall`; judge picks **documents**; grounded answer ("never deploy Friday"). |
| F2 | `what do I usually defer when I'm busy?` | Routes to `recall`; judge picks **logs** (`planning_log`); answer grounded in your accepted plans. |
| F3 | `tell me about the timing rules` (vague) | Weak first retrieval → a **reformulated** second query, visibly different in the **🧭 Steps** sub-steps. |
| F4 | `what does my doc say about the parking policy?` (absent) | Loops to `MAX_ITERATIONS` then answers honestly: "couldn't find anything relevant." |
| F5 | `how far ahead must I book international flights?` (bare factual) | Routed to `recall` (broadened intent) and answered from the expense-policy doc. |
| F6 | `hi there, how are you?` | Stays `unknown` (small talk) — **not** recall. |

## G. Multi-step / complex (the agent⇄tools loop)

| # | Type this | Expect |
|---|-----------|--------|
| G1 | `create a task to call the bank, delete the gym task, replan my day, and what's the weather in Hanoi?` | Routed to the **agent loop**, which sequences the tools one per step (CRUD steps still ask for approval). |
| G2 | `will it rain in Hanoi today?` | Uses the weather tool. Intent `weather`. |

## H. Live progress trace (UI)

| # | Do this | Expect |
|---|---------|--------|
| H1 | Run any planner turn | The **🧭 Steps** box fills live with each node + its routing delta (`classify → intent='plan'`, …) and **stays expanded** after it finishes. |
| H2 | Run a recall turn (F-series) | The RAG loop's **retrieve → judge → reformulate** sub-steps appear under the `rag` step. |

## I. Locked-plan behavior

| # | Type this | Expect |
|---|-----------|--------|
| I1 | (after locking) `plan my day` | **Shows** the locked plan as-is (no re-plan). Node `show_locked`. |
| I2 | `replan my day` | Regenerates the plan and re-enters the confirm loop. |

## J. Graceful degradation (optional)

| # | Do this | Expect |
|---|---------|--------|
| J1 | Stop the **memory** server, then run F1/F2 | Recall returns "couldn't find…"; planning/summary/CRUD still work. |
| J2 | Stop **Ollama**, run `python src/run_orchestrator.py "plan my day"` | Works offline via the heuristic normalizer + keyword classifier + deterministic planner. |

---

## When does the agent retrieve `planning_log`?

`planning_log` is one of the two memory sources (the other is `document`). Two facts
decide when it's read:

1. **It only contains data after you Accept & lock a plan.** Locking writes a one-line
   summary (`"On <date>: scheduled N task(s) … deferred M (…)"`) via the memory
   server's `remember` tool (see `confirm_plan` in
   [graph_orchestrator.py](src/agents/graph_orchestrator.py)). Reject/Cancel write
   nothing. With no locked plans, `retrieve_log` returns empty.

2. **It is read on a `recall` query.** Inside the RAGAgent loop
   ([rag_agent.py](src/agents/rag_agent.py)):
   - **Pass 1 always uses `source="both"`** → both `retrieve_log` *and*
     `retrieve_document` are called. So **every recall query touches `planning_log`
     on the first pass.**
   - The **judge** then sets `next_source` for any further pass — `logs`,
     `documents`, or `both` — based on the question
     ([rag_judge_system.md](src/prompts/rag_judge_system.md)).

So `planning_log` is the **decisive** source for questions about *your own planning
history / habits*, e.g.:

- "what do I usually defer when I'm busy?"
- "how many tasks did I schedule on 2026-06-23?"
- "what did my recent days look like?"
- "do I tend to drop reading tasks when overloaded?"

Whereas reference questions ("what does my doc say about X?") get retrieved on pass 1
too, but the judge keeps narrowing to `documents`. To test `planning_log` end to end:
**lock a few plans first** (A2), then ask F2 — and watch the **🧭 Steps** box show the
`rag` step retrieving from logs.
