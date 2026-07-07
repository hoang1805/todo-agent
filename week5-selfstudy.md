# Self-Study: Observability, Evaluation, and Deployment

## 0. Where you're starting from, and where this week takes it

Every reliability mechanism you've built so far checks **one request at a time, live**: a contract validates one payload, a guardrail checks one input, a unit test checks one function call. None of them can answer the question that actually matters once a system gets this complex: *across many realistic cases, is this actually good — and did my last change make it better or worse?*

Logging so far has also been scattered — week 2 said "log every tool call," week 3's RAG loop said "log every iteration," week 4's guardrails need pass/fail records. Each was bolted on locally, in its own format, with nowhere to actually look at them.

This week closes both gaps and ships the result: you make the system **observable** (Part 1), **measurably correct** (Part 2), and **deployable** (Part 3). That's a deliberate order — you can't usefully evaluate a system you can't see into, and you shouldn't ship a system you haven't measured.

---

## 1. This week's objectives

1. **Build a shared observability layer** — structured logging across every agent, tool, and guardrail, backed by persistent storage, surfaced through a dashboard.
2. **Evaluation** — a fixed set of test cases with known-good answers, run on demand, producing a score you can compare across changes over time.
3. **Containerize the whole system** — package every component as a deployable service, with Ollama correctly left outside the container boundary.

---

## 2. Observability: Logging, Storage, and a Dashboard

### 2.1 Why this is one layer, not scattered instrumentation

Every one of these already exists as a checkpoint in your code — the orchestrator's routing decision, every MCP tool call, every guardrail check, every RAGAgent loop iteration, DailyPlannerAgent's overload decision. What's missing isn't the checkpoints; it's a **single shared format** for what gets recorded at each one, and **one place** all of it ends up. Same pattern as `guardrails.py` in week 4: one shared module, called from many places, instead of each component inventing its own ad hoc logging.

### 2.2 What gets logged, and from where

At minimum, wire an event into each of these existing points:

- **Orchestrator** — every routing decision: which intent was classified, which agent(s) were invoked, the `session_id`.
- **Every MCP tool call** (task tools, memory tools) — inputs, success/failure, latency.
- **Guardrail checks** — input/output, pass/fail, reason.
- **RAGAgent's loop** — this generalizes the per-iteration trace log you already built in week 3 into something actually persisted and visible, rather than printed to a console and lost.
- **DailyPlannerAgent** — whether the overload branch triggered, and what got deferred.
- **Errors and exceptions**, and latency per LLM call.

### 2.3 Design decisions

**Storage is your choice.** You need something that persists events durably and lets you query them (by time range, by agent, by event type) — the same requirement you already solved once for chat history in week 4. Reuse whatever you chose there if it fits, or pick something else — justify the choice the same way you justified your chat-history schema.

**One shared logging interface, many call sites.** Design a single function or small module that every part of the system calls to record an event, with a consistent shape (timestamp, component, event type, details). Don't let each agent invent its own log format — that's the exact mistake this week is fixing.

**The dashboard reads, it doesn't compute.** The dashboard should be a thin layer that queries stored events and renders them — no business logic living in the dashboard itself. If a metric requires real computation, compute it when the event is logged or when it's queried, not by hardcoding it into a chart.

### 2.4 What you'll build

- A shared logging interface, called from the orchestrator, every MCP tool, both guardrail checks, RAGAgent's loop, and DailyPlannerAgent's overload branch.
- Persistent storage for logged events, in a schema you design.
- A dashboard with, at minimum: requests over time, a per-agent usage breakdown, the guardrail block rate, average RAGAgent iterations per query, and an error rate.
- A demo: generate enough real traffic (a handful of varied requests across all three agents, including at least one that trips a guardrail and one that requires multiple RAG iterations) to show every dashboard panel populated with real data, not placeholders.

### 2.5 Data flow

```
   ORCHESTRATOR          MCP TOOLS          GUARDRAILS          RAGAgent LOOP
  (routing decision)   (call + result)     (pass/fail)        (per iteration)
        │                    │                   │                    │
        └────────────────────┴──────────┬────────┴────────────────────┘
                                        ▼
                            SHARED LOGGING INTERFACE
                                        │
                                        ▼
                              PERSISTENT STORAGE
                             (your schema, your DB)
                                        │
                                        ▼
                                  DASHBOARD
                    (requests over time · per-agent split ·
                     guardrail block rate · avg RAG iterations ·
                     error rate)
```

---

## 3. Evaluation

### 3.1 What evaluation is, and how it's different from everything else you've built

Every mechanism in this project so far checks *one request, live, as it happens*. Evaluation is the opposite: a **fixed set of test cases with known-good answers**, run all at once, on demand, producing a score. It's the only tool in the project that can answer *"across many realistic cases, is this actually good — and did my last change help or hurt?"*

| | Contracts | Guardrails | Unit tests | **Evaluation** |
|---|---|---|---|---|
| Checks | shape of one payload | safety/policy of one input | exact function behavior | quality across many realistic cases |
| Runs | every request | every request | every code change | on demand, when you want to measure |
| Answers | "is this valid?" | "is this safe?" | "is this function correct?" | "is the system good, and did it improve?" |

### 3.2 Where eval code lives

Not in the runtime path — it never runs when a real user sends a message. It's its own folder, run manually like a test suite: a small dataset file per eval type, and a runner script that exercises part of the real system against that dataset and reports a score.

### 3.3 Suggested structure

A layout like this keeps each eval type self-contained and easy to run individually or all at once:

```
eval/
├── datasets/
│   ├── retrieval_cases.jsonl
│   ├── chunking_cases.jsonl
│   ├── loop_cases.jsonl
│   ├── generation_cases.jsonl
│   └── guardrail_cases.jsonl
├── run_retrieval_eval.py
├── run_chunking_eval.py
├── run_loop_eval.py
├── run_generation_eval.py
├── run_guardrail_eval.py
├── run_all.py          # runs all five, prints/persists a combined summary
└── report.py           # shared helper: score a run, persist it, format output
```

Treat this as a starting point, not a fixed requirement — the important properties to preserve are: one dataset file per eval type, one runner per eval type so any single type can be run in isolation, and a shared place (`report.py`, or whatever you call it) for the persist-and-score logic so five runners don't each reinvent it.

**Dataset format is your choice**, but JSON Lines (one JSON object per line, as sketched above) is a reasonable default — it's easy to append a new test case without touching existing ones, and easy to load line by line in a runner.

### 3.4 The five eval types

- **Retrieval eval.** Tests `retrieve_log`/`retrieve_document` directly — no LLM involved. Each case is a query plus what should come back. The runner reports a hit-rate: how often the expected chunk appeared in the top-k results. This is the exact tool that would have let you *measure* whether the week-3 embedding-model swap or the week-4 chunking-strategy change actually helped, instead of guessing.
- **Chunking eval.** Same mechanism as the retrieval eval, run twice: ingest the same sample document under two different chunking strategies (into separately tagged data so they don't mix), run the identical retrieval cases against both, and compare hit-rates side by side. This turns week 4's manual "eyeball the difference" comparison into a real measured number.
- **Agentic-loop eval.** Tests RAGAgent's *control flow*, not its final answer. A case is a query deliberately worded so the first retrieval should come back weak — the runner checks the agent's own trace log (from week 3 / Part 1's logging) to confirm it actually reformulated and retried, rather than quietly giving up after one pass, and confirms `MAX_ITERATIONS` terminates a query that can never be satisfied.
- **Generation (end-to-end) eval.** Needs an LLM to grade, because "is this a good answer" isn't a string match. A case is a question plus a rubric describing what a good answer looks like. The runner asks the real agent the question, then makes a **separate** LLM call — "LLM-as-judge" — with the question, the rubric, and the actual answer, asking for a pass/fail and a reason. This is what catches prompt regressions: change a system prompt later, re-run this file, find out immediately if quality dropped.
- **Guardrail eval.** Two datasets: adversarial inputs that should be blocked, and legitimate inputs that should be allowed. Both are required — testing only the adversarial set can't tell you if the guardrail became so strict it also blocks normal use.

### 3.5 Design decisions

**Datasets are small and fixed on purpose.** This isn't a benchmark suite — it's a regression check. A dozen or so well-chosen cases per type, hand-written by you based on what you know the system should and shouldn't do, is enough.

**Results should persist, not just print.** Now that Part 1's logging/storage exists, write each eval run's result into it — timestamp, eval type, score, and ideally a note on what changed since the last run. That's what turns "did this help?" from a one-off number you have to remember into a trend you can look back on.

### 3.6 What you'll build

- A dataset file and a runner script for each of the five eval types.
- Each runner produces a clear score (hit-rate, pass-rate, or pass/fail) and persists it to storage rather than only printing it.
- A demo showing at least one **before/after comparison** using the retrieval or chunking eval — pick a real change you've made to the project (an embedding model, a chunking strategy, a prompt) and show the eval score before and after.
- A dashboard panel (extending Part 1's dashboard) showing eval scores over time.

### 3.7 Data flow

```
   eval/datasets/*.jsonl  (your test cases, known-good answers)
              │
              ▼
   eval/run_*.py  ──── exercises the REAL system ────►  agents / tools / guardrails
              │                                                    │
              │◄───────────────── real output ─────────────────────┘
              ▼
      score it (hit-rate / pass-rate / LLM-as-judge)
              │
              ▼
      PERSISTENT STORAGE  (same store as Part 1's logging)
              │
              ▼
      DASHBOARD  — eval scores over time, before/after comparisons
```

---

## 4. Docker Deployment

### 4.1 What gets containerized, and what deliberately doesn't

Containerize the task MCP server, the memory MCP server, the orchestrator, and the dashboard — each as its own service, wired together with a compose file. This makes literal something that's been true in principle since week 1: an MCP server really is a separate process.

**Ollama stays on the host, not in a container.** Given the CPU-only, 16GB, two-models-loaded constraint from earlier weeks, putting Ollama inside a container buys nothing and risks breaking the RAM budgeting you already worked out. Your containers need to reach host Ollama over the network instead of `localhost` — this is the single most common point of confusion when containerizing an app that talks to Ollama, so expect to hit it and know it's expected, not a sign something is broken.

### 4.2 Design decisions

**Persistence must survive the container being destroyed, not just restarted.** This is a stronger test than week 4's "survives an app restart" requirement — tear the container down and recreate it, and your data (chat history, memory data, task data, logged events) should still be there. If it isn't, persistence was quietly relying on the process never fully dying, which isn't real persistence.

**Configuration via environment variables**, not hardcoded values — at minimum the Ollama host URL and the model names, extending the `config.yaml` instinct from week 1 into how containerized apps are actually configured.

**One service per component.** Each piece (task server, memory server, orchestrator, dashboard) is its own container, not one giant container running everything — this is what makes "the MCP server is a separate process" real rather than aspirational.

### 4.3 What you'll build

- A Dockerfile for each service.
- A compose file that starts all of them together, correctly networked to each other and to host Ollama.
- Persistent volumes for every store you've built (chat history, memory data, task data, logged events).
- A demo: tear down and recreate the containers, and show a previous conversation, previous memory data, and previous logged events are all still there afterward.
- A demo that the dashboard and at least one eval runner still work correctly when run against the containerized system — this is your integration test that the whole system, not just its packaging, actually works.

### 4.4 Data flow

```
                    HOST MACHINE
              ┌─────────────────────┐
              │   Ollama (native)   │◄────────────────┐
              └─────────────────────┘                 │
                                                      │ network call
                    DOCKER COMPOSE                    │ (not localhost)
   ┌────────────────────────────────────────────────┐ │
   │  ┌───────────┐  ┌───────────┐  ┌─────────────┐ │ │
   │  │ task MCP  │  │ memory MCP│  │orchestrator │─┼─┘
   │  │  server   │  │  server   │  │             │ │
   │  └─────┬─────┘  └─────┬─────┘  └──────┬──────┘ │
   │        │              │               │        │
   │        ▼              ▼               ▼        │
   │  ┌──────────────────────────────────────────┐  │
   │  │         persistent volumes               │  │
   │  │  (chat history · memory data · task data │  │
   │  │   · logged events)                       │  │
   │  └──────────────────────────────────────────┘  │
   │                        ▲                       │
   │                  ┌─────┴─────┐                 │
   │                  │ dashboard │                 │
   │                  └───────────┘                 │
   └────────────────────────────────────────────────┘
```

---

## 5. Definition of done

**Observability**
- [ ] A single shared logging interface is called from the orchestrator, every MCP tool, both guardrail checks, RAGAgent's loop, and DailyPlannerAgent's overload branch.
- [ ] Logged events persist in storage you designed and can justify.
- [ ] The dashboard shows requests over time, per-agent usage, guardrail block rate, average RAG iterations, and error rate — all populated with real generated traffic.

**Evaluation**
- [ ] All five eval types (retrieval, chunking, agentic-loop, generation, guardrail) have a dataset and a runner.
- [ ] Each runner produces a clear score and persists it.
- [ ] A real before/after comparison is demonstrated using at least one eval type against an actual change you made to the project.
- [ ] Eval scores appear on the dashboard over time.

**Deployment**
- [ ] Every component (task server, memory server, orchestrator, dashboard) runs in its own container via a single compose command.
- [ ] Ollama runs on the host and is reachable from every container that needs it.
- [ ] All persisted data survives a full container teardown and recreation.
- [ ] The dashboard and at least one eval runner work correctly against the fully containerized system.

---

