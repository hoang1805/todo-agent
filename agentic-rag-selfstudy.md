# Self-Study: Building an Agentic RAG Memory Agent

## 0. Where you're starting from

You have three things working: an Orchestrator that classifies a user's request and routes it, a TodoAgent that turns raw tasks into a validated contract, and a DailyPlannerAgent that turns that contract into a schedule (with a real decision when the day is overloaded).

This week you add a **third agent**: one that retrieves and reasons over stored data — its own planning logs, plus any documents you give it — to answer questions. This is also the first real test of the architecture you built last time — adding an agent should require a change in **one place** (the orchestrator), not a rewrite of what already works. If it doesn't, something about the design has gone wrong.

---

## 1. This week's objectives

1. **Understand the anatomy of a RAG system** — Ingestion, Retrieval, Generation — and implement all three yourself.
2. **Build a genuinely *agentic* retrieval loop.** Not "call a search function and stuff the result into a prompt" — an agent that decides *whether* to retrieve, *which* source to query, *whether* what came back is enough, and *iterates* until it has enough or gives up. That distinction is the actual content of this week, so don't skip §2.

---

## 2. Concept: the three parts of RAG, and what makes it "Agentic"

**Ingestion** (offline, before any query) — turning raw material into something searchable: chunk text → embed each chunk into a vector → store the vector + original text + metadata.

**Retrieval** (per query) — embed the query with the *same* embedding model used at ingestion, then similarity-search against the stored vectors for the top-k closest chunks.

**Generation** (per query) — assemble the retrieved chunks into a prompt, instruct the model to answer *only* from that context (the same grounding lesson as "don't invent tasks" from week 1, applied to this data), and call the LLM.

That's standard RAG: one retrieval pass, one generation pass, always. It's a fixed pipeline — the same three steps run every time, regardless of the question.

**Agentic RAG is different in one specific way: the agent controls the retrieval process itself.** Instead of always retrieving once and generating, the agent:

- **decides which source/tool to query** when more than one is available,
- **judges whether what it got back is actually sufficient** to answer,
- **iterates** — reformulates the query and retrieves again — if it isn't, up to some stopping point.

The test to apply to your own build: **if your agent always does exactly one retrieve-then-generate pass, it is standard RAG, even if an LLM is involved.** A single relevance-score threshold check ("if score > 0.5, answer; else say I don't know") is a good guardrail, but it is *not* agentic — it's still one pass. You need a loop with a real branch back to "retrieve again" for it to earn the name.

---

## 3. Design decisions for this project

### 3.1 What data you're working with

You need to decide the corpus, not just assume one. Build it from **two sources**:

- **Auto-logged planning summaries** — every time DailyPlannerAgent finishes building a schedule, it writes a short text summary into the data store ("On 2026-06-19, 6 tasks, deferred 2 low-priority reading tasks"). This is the main source — it creates a feedback loop between two agents, and questions like "what do I usually prioritize when busy?" become answerable.
- **Uploaded documents** — any text document you want the system to know about (notes, a reference doc, anything). This path is **not tied to any agent** — you (or a script) feed it in directly, independent of TodoAgent or DailyPlannerAgent.

Tag every chunk of data with a `source_type` (`"planning_log"` or `"document"`) when you write it. That tag is what makes tool/source selection possible later — without it, there's nothing for the agent to choose between.

### 3.2 The Memory MCP server — tools and where it plugs in

All of this data lives behind a dedicated MCP server, separate from your task server — same architecture as week 1: it's its own process, and anything that needs it connects through an MCP client, not by importing its code directly.

**Where it plugs in** — three different callers connect to it, for different reasons:

```
   DailyPlannerAgent          Document ingestion script          RAGAgent
          │                            │                             │
          │ remember(...)              │ remember(...)               │ retrieve_log(query)
          │ source_type=               │ source_type=                │ retrieve_document(query)
          │  "planning_log"            │  "document"                 │
          ▼                            ▼                             ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                  MEMORY MCP SERVER   (separate process)            │
   │      tools:  remember · retrieve_log · retrieve_document           │
   └────────────────────────────────────────────────────────────────────┘
```

**What tools it exposes:**

- `remember(text, source_type, metadata) -> id` — write a new chunk of data into the store. Called by DailyPlannerAgent (with `source_type="planning_log"`) and by your document ingestion script (with `source_type="document"`).
- `retrieve_log(query, k=3) -> list[DataChunk]` — search **only** planning-log data.
- `retrieve_document(query, k=3) -> list[DataChunk]` — search **only** document data.

Splitting retrieval into two tools instead of one is deliberate: tool selection only means something if there's more than one tool to select between. A question like "what do I usually prioritize when busy" clearly calls for `retrieve_log`; "what does my reference doc say about X" clearly calls for `retrieve_document` — and RAGAgent has to look at the question and decide, which is the tool-selection half of being agentic.

### 3.3 The sufficiency judgment is a decision, not a threshold

After retrieving, the agent has to decide: can I answer from this, or do I need to search again (and with what, and where)? Make that decision **structured LLM output**, the same discipline as your contracts from last week — not a numeric cutoff. See `RetrievalDecision` in §7.

### 3.4 The loop needs a hard stopping condition

Once retrieval can repeat, cost and latency stop being a fixed number and become a *distribution* — some queries finish in one pass, some take three. A hard `MAX_ITERATIONS` cap is not optional; an unbounded loop is a production risk, not just inefficiency.

---

## 4. What you'll build (5 deliverables)

Build in this order — each one depends on the last.

| # | Deliverable | Serves | Why it matters |
|---|---|---|---|
| 1 | **Contracts**: `DataChunk`, `RetrievalDecision` | ingestion/retrieval discipline | Same role as the handoff contract from last week — nothing ungoverned crosses a boundary. |
| 2 | **Memory MCP server**: `remember`, `retrieve_log`, `retrieve_document` | ingestion + retrieval | Hides the embedding model and vector store behind tools any caller can use. |
| 3 | **Two ingestion paths**: DailyPlannerAgent hook + standalone document script | ingestion | Builds the corpus — one automatic, one manual and agent-independent. |
| 4 | **RAGAgent**: retrieve → judge → generate loop | generation + agency | The agentic part. The required decision lives here. |
| 5 | **Orchestrator wiring**: new `recall` branch | integration | Proves the one-place-to-add-an-agent property still holds. |

---

## 5. Tooling

- **Embedding model suggestion `nomic-embed-text`/`all-minilm` or other model .** 
- **Hand-roll the vector store first** — a list of vectors plus a cosine-similarity loop in plain Python/numpy. It's a dozen lines and it makes "vector search" something you understand rather than something a library does for you. (Swapping in Chroma/FAISS later is fine — keep the same `retrieve_*` interface so nothing upstream changes.)
- **Ollama structured output for the judge step** — same pattern as last week's structured-output calls: generate a JSON schema from your Pydantic model, pass it via `format`, validate the response.

---

## 6. Data flow

**Ingestion workflow — building the data store:**

```
   DailyPlannerAgent                         a document you provide
   finishes a schedule                       (.txt file, etc.)
          │                                          │
          │ short text summary                       ▼
          │ (no chunking needed)              CHUNK into smaller pieces
          │                                          │
          └─────────────────┬────────────────────────┘
                             ▼
                     EMBED each piece
                  (same model used at retrieval!)
                             ▼
              remember(text, source_type, metadata)
                             ▼
                  ┌─────────────────────────┐
                  │        DATA STORE       │
                  │  vector + text +        │
                  │  source_type + metadata │
                  └─────────────────────────┘
```

**Orchestrator level — one new branch:**

```
USER PROMPT
    │
    ▼
ORCHESTRATOR — classify: summary | plan | add task | recall   ← new branch
    │                                              │
    ▼ (existing agents)                            ▼ (new)
TodoAgent / DailyPlannerAgent                   RAGAgent
                                                   │
                                              (see loop below)
                                                   │
                                                   ▼
                                                RESPONSE
```

**Inside RAGAgent — the agentic loop, with both contracts shown:**

```
            ┌───────────────┐
            │   RETRIEVE    │◄─────────────────────────┐
            │ choose source │                          │
            │ + query       │                          │
            └──────┬────────┘                          │
                   ▼                                   │
       ┌─────────────────────────────┐                 │
       │ CONTRACT: DataChunk[]       │                 │
       │ every tool result is checked│                 │
       │ against this shape (§7.1)   │                 │
       └──────────────┬──────────────┘                 │
                      ▼                                │
                ┌───────────────┐                      │
                │     JUDGE     │                      │
                │ enough to     │                      │
                │ answer?       │                      │
                └──────┬────────┘                      │
                       ▼                               │
       ┌───────────────────────────────┐               │
       │ CONTRACT: RetrievalDecision   │               │
       │ the judge's structured output │               │
       │ — validated before branching  │               │
       │ (§7.1)                        │               │
       └───────────────┬───────────────┘               │
                       │                               │
            sufficient?│  no, iterations left ─────────┘
                       │ yes — or out of iterations
                       ▼
                ┌───────────────┐
                │   GENERATE    │
                │ grounded      │
                │ answer        │
                └───────────────┘
```

## 7. Implementation hints

### 7.1 Contracts

```python
# contracts.py
from pydantic import BaseModel
from typing import Literal

class DataChunk(BaseModel):
    text: str
    score: float
    source_type: Literal["planning_log", "document"]
    metadata: dict

class RetrievalDecision(BaseModel):
    sufficient: bool
    reasoning: str
    next_source: Literal["logs", "documents", "both"] | None = None
    next_query: str | None = None     # reformulated, only if sufficient=False
```

### 7.2 Data store (hand-rolled)

```python
# store.py
import numpy as np
from embeddings import embed

class DataStore:
    def __init__(self, path="data/store.json"):
        self.path = path
        self.chunks = self._load()   # list of {text, vector, source_type, metadata}

    def add(self, text: str, source_type: str, metadata: dict) -> str:
        # TODO: embed(text), append to self.chunks, persist, return an id
        ...

    def search(self, query: str, k: int, source_type: str | None = None) -> list[dict]:
        # TODO: embed(query), cosine-similarity against self.chunks
        #       (filter by source_type if given), sort, return top k
        ...
```

### 7.3 Ingestion script — documents

```python
# ingest_documents.py — standalone script, NOT tied to any agent.
# Run it manually whenever you want to add a document to the data store.

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    # TODO: split `text` into ~chunk_size-character pieces
    #       (simplest approach: split on paragraphs, group until you hit chunk_size)
    ...

def ingest(file_path: str):
    text = open(file_path).read()
    for chunk in chunk_text(text):
        # TODO: call the Memory MCP client's `remember` tool with this chunk,
        #       source_type="document", metadata={"file": file_path}
        ...
```

### 7.4 RAGAgent — the loop

```python
# rag_agent.py
class RAGAgent:
    MAX_ITERATIONS = 3

    def run(self, query: str) -> str:
        collected: list[dict] = []     # accumulate — never discard between iterations
        current_query, source = query, "both"

        for i in range(self.MAX_ITERATIONS):
            # TODO: call retrieve_log / retrieve_document / both depending on `source`,
            #       extend `collected` with the results (validate each against DataChunk)
            decision = self._judge(query, collected)   # TODO: structured Ollama call
                                                          # returning RetrievalDecision
            log.info("rag_step", iter=i, query=current_query,
                     source=source, decision=decision.model_dump())

            if decision.sufficient:
                return self._generate(query, collected)   # TODO: grounded LLM answer

            current_query = decision.next_query or current_query
            source = decision.next_source or source

        # TODO: stopping condition — out of iterations.
        #       Answer from whatever was collected (with a caveat) if there's
        #       anything useful, otherwise say so honestly. Do not loop forever.
```

---

## 8. Definition of done

- [ ] `remember`, `retrieve_log`, and `retrieve_document` all work end-to-end through the Memory MCP server, with `embed()` shared between write and read paths.
- [ ] Every chunk of data is tagged with `source_type` (`planning_log` or `document`) at write time.
- [ ] DailyPlannerAgent automatically writes a `planning_log` entry after building a schedule.
- [ ] You can ingest at least one document independently of the agents (via the standalone script) and successfully retrieve from it.
- [ ] RAGAgent picks `retrieve_log` vs `retrieve_document` (or both) based on the question — demo one query that clearly hits each.
- [ ] A query whose first retrieval is weak triggers a **reformulated** second query — visibly different from the first in the trace log.
- [ ] `MAX_ITERATIONS` is enforced; a query that can never be satisfied still terminates and answers honestly instead of looping or crashing.
- [ ] Every tool result is validated against `DataChunk`, and every judge call against `RetrievalDecision`, before the loop branches on it.
- [ ] The trace log for any single run lets you reconstruct every query issued, source chosen, and judgment made.
- [ ] The orchestrator required exactly **one** new branch to add RAGAgent — point to the diff.

---
