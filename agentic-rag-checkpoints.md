# Agentic RAG — Definition of Done, checkpoint by checkpoint

A detailed companion to the "Definition of done" in
[agentic-rag-selfstudy.md](agentic-rag-selfstudy.md) §8. Each of the ten
checkpoints below is broken into four parts:

- **Requirement** — the checkbox, restated.
- **What it really means** — what the line is actually testing, and why it matters.
- **Done looks like** — the concrete acceptance bar.
- **How to verify** — something you can run or read to prove it.
- **Where it lives** — the file(s) that satisfy it in this project.

Two repos are involved:

- **`todo-agent`** (this repo) — the orchestrator, the agents, and `RAGAgent`.
- **`todo-agent-mcps/memory-mcp`** — the standalone Memory MCP server (store,
  embeddings, `remember` / `retrieve_log` / `retrieve_document` tools, and the
  document ingestion script).

---

## ✅ 1. `remember`, `retrieve_log`, `retrieve_document` work end-to-end, with `embed()` shared between write and read

**Requirement.** All three MCP tools work end-to-end through the Memory MCP
server, with the **same** `embed()` used on both the write and the read path.

**What it really means.** This is the spine of the whole feature. The subtle,
make-or-break part is *embedding consistency*: a vector written by `remember`
and a vector computed by `retrieve_*` must come from the **same model**, or
cosine similarity is meaningless and retrieval returns garbage (or nothing).
A classic failure here is the write path using a real embedding model while the
read path silently falls back to a hash/deterministic embedding (or vice versa)
— the vectors live in different spaces, every score is near-zero, and the agent
concludes "I couldn't find anything."

**Done looks like.**
- `remember(text, source_type, metadata)` writes a chunk and returns an id.
- `retrieve_log(query)` and `retrieve_document(query)` return ranked chunks.
- The exact same `embed()` function is imported by `store.add` and `store.search`
  — not two copies, not two models.

**How to verify.**
1. Start the server, `remember` one distinctive sentence, then `retrieve_*` with
   a paraphrase of it — it should come back with a healthy score (clearly above
   the noise floor of an unrelated chunk).
2. Confirm the same embedding backend is active on both paths (e.g. both using
   `embeddinggemma` via Ollama, or both using the fallback) — if one path used
   the model and the other the hash fallback, scores collapse toward zero.

**Where it lives.**
- Tools: `memory-mcp/src/core/tools/memory_tools.py`, exposed in `memory-mcp/src/server.py`.
- Store: `memory-mcp/src/core/services/store.py` (`add` / `search`).
- Embeddings: `memory-mcp/src/core/services/embeddings.py` — `embed()` imported by
  both `add` and `search`, which is what makes the spaces match.

---

## ✅ 2. Every chunk is tagged with `source_type` at write time

**Requirement.** Every chunk carries a `source_type` of `planning_log` or
`document`, set when it is written.

**What it really means.** The tag is not bookkeeping — it is the thing that makes
**source selection** (checkpoint 5) possible. If chunks weren't tagged, there'd
be nothing for `retrieve_log` vs `retrieve_document` to filter on, and the
"agentic" tool-choice half of the project would have no foundation. Tagging at
**write** time (not derived later) means the provenance can never be ambiguous.

**Done looks like.**
- `remember` requires/records a `source_type` on every chunk.
- The store filters by it, so `retrieve_log` only ever sees `planning_log`
  chunks and `retrieve_document` only ever sees `document` chunks.
- It's a constrained value, not free text (a `Literal` / enum), so a typo can't
  create a third invisible bucket.

**How to verify.**
- Inspect `data/store.json` — every entry has a `source_type` field, and it's one
  of the two allowed values.
- `retrieve_log` on a query that only matches a document returns nothing (the
  filter is real, not cosmetic).

**Where it lives.**
- Contract: `memory-mcp/src/models/contracts.py` (`SourceType` / `DataChunk`).
- Write + filter: `store.add(..., source_type=...)` and
  `store.search(..., source_type=...)` in `memory-mcp/src/core/services/store.py`.
- Mirror contract on the consumer side: [src/models/rag.py](src/models/rag.py)
  (`DataChunk.source_type`).

---

## ✅ 3. DailyPlannerAgent automatically writes a `planning_log` entry after building a schedule

**Requirement.** Every time a schedule is built, a `planning_log` entry is
written to memory — automatically, no manual step.

**What it really means.** This is the **feedback loop** between two agents: the
planner produces data that the RAG agent later reasons over. It's what turns
questions like "what do I usually defer when I'm busy?" from unanswerable into
answerable, because the history accumulates on its own. The word that matters is
*automatically* — if a human has to remember to log, the corpus rots.

**Done looks like.**
- Locking/accepting a plan triggers a `remember(..., source_type="planning_log")`
  with a short human-readable summary (date, task count, what got deferred).
- It fires as a side effect of the normal planning flow, not a separate command.
- It is best-effort: if the memory server is down, planning still succeeds (the
  log write degrades gracefully).

**How to verify.**
- Plan a day and **accept** it, then `retrieve_log("what did I schedule")` — the
  just-written summary comes back.
- Check `data/store.json` grows by one `planning_log` entry per accepted plan.

**Where it lives.**
- Summary line builder: [`_planning_log_line`](src/agents/graph_orchestrator.py#L177)
  in [graph_orchestrator.py](src/agents/graph_orchestrator.py).
- Write-on-accept: the `confirm_plan` node
  ([graph_orchestrator.py:493-507](src/agents/graph_orchestrator.py#L493-L507))
  calls the memory writer when a plan is locked.
- Writer wiring: [`make_memory_writer`](src/agents/rag_agent.py#L238) in
  [rag_agent.py](src/agents/rag_agent.py).

---

## ✅ 4. You can ingest a document independently of the agents and retrieve from it

**Requirement.** A standalone script can ingest at least one document, with no
agent involved, and the content is then retrievable.

**What it really means.** This proves the **document path is agent-independent**
(design §3.1): the corpus has two unrelated feeders, and uploading reference
material must not require the planner or todo agent to run. It also exercises the
one piece of real text processing on the ingestion side — **chunking** — since a
document is too big to embed as one vector.

**Done looks like.**
- A script you run by hand reads a `.txt`, chunks it, and calls
  `remember(chunk, source_type="document", metadata={"file": ...})` for each piece.
- Afterwards `retrieve_document(query)` returns those chunks.
- Nothing in the path imports or triggers TodoAgent / DailyPlannerAgent.

**How to verify.**
- Run the ingestion script on a sample doc, then `retrieve_document` with a
  question whose answer is in that doc — the relevant chunk comes back.
- Grep the script's imports: no agent modules.

**Where it lives.**
- `memory-mcp/ingest_documents.py` (chunk + `remember` loop).
- Sample corpus under `memory-mcp/data/` (e.g. the policy/handbook `.txt` files).

---

## ✅ 5. RAGAgent picks `retrieve_log` vs `retrieve_document` (or both) based on the question

**Requirement.** The agent chooses the source from the question — demo one query
that clearly hits logs and one that clearly hits documents.

**What it really means.** This is the **tool-selection** half of "agentic"
(§3.2). It only counts because there are two tools: a single merged "retrieve"
tool would make the choice vanish. The agent must read the *intent* of the
question ("what do **I** usually…" → my logs; "what does the **policy** say…" →
documents) and route accordingly, with `both` as the hedge when it's unsure.

**Done looks like.**
- A personal-history question retrieves from logs.
- A reference/policy question retrieves from documents.
- The choice is data-driven (judge's `next_source` and the loop's starting
  `source`), not hard-coded per query.

**How to verify (the two demo queries).**
- Logs: *"what do I usually defer when my day is overloaded?"* → trace shows
  `retrieve_log`.
- Documents: *"how far ahead must I book international flights?"* → trace shows
  `retrieve_document`.

**Where it lives.**
- Retriever over both tools: [`make_mcp_retriever`](src/agents/rag_agent.py#L134)
  (binds `retrieve_log` + `retrieve_document`).
- Source decision: the loop's `source` variable + `RetrievalDecision.next_source`
  ([src/models/rag.py:35](src/models/rag.py#L35)), driven by
  [`llm_judge`](src/agents/rag_agent.py#L201).

---

## ✅ 6. A weak first retrieval triggers a *reformulated* second query

**Requirement.** When the first retrieval is weak, the agent issues a second
query that is **visibly different** from the first in the trace.

**What it really means.** This is the **iteration** half of "agentic" and the
single clearest line between "real Agentic RAG" and "standard RAG with an LLM."
A reformulation that's byte-identical to the original proves nothing — the agent
must actually *rewrite* the query (different words, narrower/broader scope) based
on what the weak first pass told it, then search again.

**Done looks like.**
- The judge returns `sufficient=False` with a `next_query` that differs from the
  current query.
- The loop adopts `next_query` and retrieves again.
- Accumulated chunks are **kept** across iterations, never discarded.

**How to verify.**
- Ask something whose obvious phrasing retrieves poorly; in the trace, pass 1 and
  pass 2 show **different** query strings.
- Confirm pass 2's `collected` is a superset of pass 1's (accumulation, not reset).

**Where it lives.**
- Reformulation field: `RetrievalDecision.next_query`
  ([src/models/rag.py:36](src/models/rag.py#L36)).
- Loop adopting it + accumulating: [`RAGAgent.run`](src/agents/rag_agent.py#L43)
  (`current_query = decision.next_query or current_query`; `collected` is extended,
  not replaced).

---

## ✅ 7. `MAX_ITERATIONS` is enforced; an unsatisfiable query terminates honestly

**Requirement.** The loop has a hard cap; a query that can never be satisfied
still terminates and answers honestly instead of looping or crashing.

**What it really means.** Once retrieval can repeat, latency/cost become a
distribution, not a constant (§3.4). The cap is a **production safety property**,
not an optimization — without it, an adversarial or simply unanswerable question
spins forever. "Honestly" matters too: on exhaustion the agent must either answer
from what little it has *with a caveat*, or say it doesn't know — never fabricate.

**Done looks like.**
- A constant `MAX_ITERATIONS` bounds the loop.
- On exhaustion there's an explicit stopping branch: generate-with-caveat if
  anything useful was collected, otherwise an honest "couldn't find it."
- No infinite loop, no unhandled exception.

**How to verify.**
- Ask something the corpus can't possibly answer; the trace shows exactly
  `MAX_ITERATIONS` passes, then a graceful "I don't have that" — not a hang.

**Where it lives.**
- Cap: [`RAGAgent.MAX_ITERATIONS = 3`](src/agents/rag_agent.py#L35) and the
  bounded `for i in range(self.MAX_ITERATIONS)` loop.
- Honest stop: the post-loop branch
  ([rag_agent.py:73-80](src/agents/rag_agent.py#L73-L80)) — generate from
  whatever was collected, else say so.

---

## ✅ 8. Every tool result validated against `DataChunk`, every judge call against `RetrievalDecision`, before branching

**Requirement.** Each tool result is validated against `DataChunk`, and each
judge output against `RetrievalDecision`, *before* the loop branches on it.

**What it really means.** Same discipline as the week-1 handoff contracts:
**nothing ungoverned crosses a boundary.** Two boundaries matter here — the MCP
tool result (untyped JSON coming back over the wire) and the judge's LLM output
(free-form unless constrained). Validating *before* branching means a malformed
chunk or a judge reply missing `sufficient` is caught at the edge, not deep
inside a decision. In this project there's an extra wrinkle: MCP returns results
wrapped in text-content blocks, so the raw payload must be **unwrapped** before it
can be validated.

**Done looks like.**
- Every retrieved item is coerced/unwrapped and parsed into a `DataChunk`;
  malformed items are dropped or surfaced, never silently trusted.
- The judge call uses **structured output** (Pydantic → JSON schema → Ollama
  `format=`) and the reply is parsed into `RetrievalDecision` before any
  `if decision.sufficient` check.

**How to verify.**
- Feed a deliberately malformed tool payload — it does not crash the loop and does
  not get treated as a valid chunk.
- Inspect the judge call: it passes a JSON schema and validates the response into
  `RetrievalDecision` before branching.

**Where it lives.**
- Unwrap + validate chunks:
  [`_unwrap_chunk`](src/agents/rag_agent.py#L90) /
  [`_coerce_chunks`](src/agents/rag_agent.py#L115) → `DataChunk` in
  [rag_agent.py](src/agents/rag_agent.py).
- Structured judge: [`llm_judge`](src/agents/rag_agent.py#L201) returning a
  validated [`RetrievalDecision`](src/models/rag.py#L26).

---

## ✅ 9. The trace log lets you reconstruct every query, source, and judgment

**Requirement.** For any single run, the trace log is enough to reconstruct every
query issued, every source chosen, and every judgment made.

**What it really means.** Observability is a *requirement*, not a nicety — an
agentic loop is only debuggable if you can replay its decisions after the fact.
A good test: hand the trace of one run to someone who didn't watch it, and they
can say exactly how many passes ran, what each query was, which source it hit, and
why the agent stopped. (This project also surfaces these steps live in the UI.)

**Done looks like.**
- Per pass, the log records: iteration index, the query used, the source chosen,
  and the judge's decision (`sufficient` + reasoning + any reformulation).
- The final stop reason (sufficient vs. exhausted) is recoverable.

**How to verify.**
- Run one recall query with tracing on (`PLANNER_TRACE`) and read back the log:
  every pass's query/source/decision is there, in order.

**Where it lives.**
- Per-step trace: the `on_step` callback + `logger.info("[rag] …")` lines inside
  [`RAGAgent.run`](src/agents/rag_agent.py#L43).
- Graph-level tracing: the `PLANNER_TRACE` observer in
  [graph_orchestrator.py](src/agents/graph_orchestrator.py#L823-L826), which also
  drives the live progress UI.

---

## ✅ 10. The orchestrator needed exactly **one** new branch to add RAGAgent

**Requirement.** Adding the RAG agent required exactly one new branch in the
orchestrator — point to the diff.

**What it really means.** This is the architectural payoff and the real test of
last week's design: a new capability should plug in at a **single seam**
(classify a new intent → route to one new node), not force edits scattered across
the existing agents. If adding RAGAgent had touched the todo or planner paths,
the hub-and-spoke design would have failed its own promise.

**Done looks like.**
- One new intent (`recall`) in the classifier.
- One new route entry mapping that intent to one new node (`rag` → the RAG
  handler), which runs the agent and finalizes.
- The existing plan/summary/CRUD paths are untouched.

**How to verify.**
- `git diff` for the RAG integration commit: the orchestrator change is a single
  added branch (intent + route + node), not edits across existing handlers.

**Where it lives.**
- Intent: `recall` in [intent_classifier_system.md](src/prompts/intent_classifier_system.md).
- Route entry: `"rag": "rag"` in the shared `intent_routes` map
  ([graph_orchestrator.py:829-833](src/agents/graph_orchestrator.py#L829-L833)).
- The node: [`run_rag`](src/agents/graph_orchestrator.py#L619) → `rag` node →
  `finalize`.

---

## At-a-glance map

| # | Checkpoint | Primary location |
|---|---|---|
| 1 | Tools work; shared `embed()` | `memory-mcp` server + `core/services/embeddings.py` |
| 2 | `source_type` at write | `memory-mcp` store + contracts |
| 3 | Auto `planning_log` | `confirm_plan` + `_planning_log_line` (this repo) |
| 4 | Standalone doc ingest | `memory-mcp/ingest_documents.py` |
| 5 | Source selection | `make_mcp_retriever` + judge `next_source` |
| 6 | Reformulated retry | `RAGAgent.run` + `next_query` |
| 7 | `MAX_ITERATIONS` + honest stop | `rag_agent.py` loop |
| 8 | Contract validation at both boundaries | `_coerce_chunks` + `llm_judge` |
| 9 | Reconstructable trace | `on_step` / `[rag]` logs / `PLANNER_TRACE` |
| 10 | One orchestrator branch | `recall` intent + `rag` route/node |
