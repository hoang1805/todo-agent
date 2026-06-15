---
name: daily_planner
description: >
  Use this skill whenever the user wants to plan, review, or organize their tasks for today.
  Triggers include: "what should I do today", "show my tasks for today", "plan my day",
  "what's on my plate today", "daily plan", "morning briefing", "today's schedule",
  "what tasks are due", "prioritize my day", "daily review", or any variant asking about
  today's workload. Also trigger when the user wants to reschedule, reprioritize, or
  get a summary of today's tasks. Always use this skill when the user's intent is to
  understand or organize their current day — even if they don't say "daily planner" explicitly.
---

# Daily Planner Skill

This skill helps the agent compose an intelligent daily plan based on the user's tasks due today. It fetches tasks, analyzes them by priority and status, and presents a clear, actionable schedule.

---

## Tools You Can Call

These tools are already bound to you — call them directly by name (no setup or
connection step needed).

**Date helpers (always available):**

| Tool | Purpose |
|---|---|
| `get_today_date` | Today's date as `YYYY-MM-DD` — call this to know what "today" is |
| `get_current_weekday` | The weekday name, e.g. `"Monday"` |

**Task tools (from the task MCP server):**

| Tool | Purpose |
|---|---|
| `get_today_task` | Fetch all tasks due today |
| `get_task_detail` | Get full details of a specific task by ID |
| `update_task_priority` | Change priority: `"low"`, `"medium"`, `"high"` |
| `update_task_status` | Change status: `"pending"`, `"in_progress"`, `"done"` |
| `create_task` | Add a new task to today's plan if user requests |

> If the task tools are unavailable (none appear when you try to call one), tell
> the user the task service isn't reachable right now — do **not** invent tasks.

---

## Workflow

### Step 1 — Establish Today, Then Fetch
First call `get_today_date` (and `get_current_weekday` if you'll greet the user)
so you can show a real date and resolve any relative dates. Then call
`get_today_task` to retrieve all tasks due today.

- If the result is **empty**: inform the user they have no tasks due today, then offer to create one or check upcoming tasks.
- If the result is **non-empty**: proceed to Step 2.

### Step 2 — Analyze and Group
Group tasks into three buckets by **priority**:

1. 🔴 **High** — must be addressed first
2. 🟡 **Medium** — do after high-priority items
3. 🟢 **Low** — do if time permits

Within each group, sort by **status**:
- `in_progress` tasks surface first (already started)
- `pending` tasks come next
- `done` tasks are listed last (for context/celebration)

### Step 3 — Present the Daily Plan
Present the plan in a clear, readable format. Example structure:

```
Good morning! Here's your plan for today — [DATE].

🔴 High Priority
  1. [Task Name] — [status] — [brief description if available]
  2. ...

🟡 Medium Priority
  3. [Task Name] — [status]
  ...

🟢 Low Priority
  ...

✅ Already Done
  ...

You have X tasks to complete. Want me to help you start with any of them?
```

### Step 4 — Offer Follow-up Actions
After presenting the plan, offer options:
- Reprioritize a task
- Mark a task as in progress or done
- Add a new task
- Get full details on a specific task

---

## Behavioral Rules

- **Never skip the fetch step** — always call `get_today_task` before responding about today's tasks. Do not rely on prior context.
- **Never guess the date** — get it from `get_today_date`; never hardcode or assume it.
- **Always respect existing status** — if a task is already `done`, do not suggest working on it unless the user asks.
- **Be concise in the plan view** — use task name + status only; only show description if the user asks for details or calls `get_task_detail`.
- **Suggest reprioritization only if it makes sense** — e.g., if many high-priority items are already done, suggest the user may want to promote a medium-priority task.
- **Date format**: Convert the ISO date from `get_today_date` into a human-friendly form like "Tuesday, June 10" — don't show raw ISO strings to the user.

---

## Handling Edge Cases

| Situation | Action |
|---|---|
| No tasks today | Say so clearly, offer to create tasks or check tomorrow |
| All tasks are `done` | Congratulate the user, offer to review tomorrow |
| Many tasks (>10) | Highlight top 3 high-priority ones first, summarize the rest |
| Task has no description | Skip description field, show name + status only |
| User asks to add a task for today | Get today's date via `get_today_date`, then call `create_task` with that `due_date` |

---

## Example Trigger Phrases

- "What do I need to do today?"
- "Show me my daily plan"
- "Give me a morning briefing"
- "What tasks are due today?"
- "Plan my day"
- "Prioritize my tasks for today"
- "I want to organize my day"