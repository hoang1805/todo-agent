---
name: task_execution
description: >
  Use this skill whenever the user wants to act on a specific task — including creating,
  updating, deleting, or changing the status or priority of tasks. Triggers include:
  "create a task", "add a task", "mark task as done", "mark as in progress", "update this task",
  "change priority", "delete task", "remove task", "I finished X", "start working on X",
  "set X to high priority", "reschedule task", "edit task", "complete task", or any phrase
  implying a direct action on one or more tasks. Also trigger when the user references a
  task by name or ID and wants to do something with it. Always use this skill for any
  create/read/update/delete (CRUD) operation on tasks — even if phrased casually
  like "just mark that done" or "add one more thing".
---

# Task Execution Skill

This skill handles all direct operations on tasks: creating new ones, reading details, updating fields, changing status or priority, and deleting tasks. It is the action layer of the task management system.

---

## MCP Tools Available

Connect to the task MCP server (HTTP) and use these tools:

| Tool | Signature | Purpose |
|---|---|---|
| `create_task` | `name, description?, status?, priority?, due_date?` | Create a new task |
| `get_task_detail` | `task_id` | Fetch full details of a task |
| `update_task_by_id` | `task_id, name?, description?, due_date?` | Update name/description/due date |
| `update_task_status` | `task_id, status` | Set status: `pending`, `in_progress`, `done` |
| `update_task_priority` | `task_id, priority` | Set priority: `low`, `medium`, `high` |
| `delete_task_by_id` | `task_id` | Permanently delete a task |
| `get_today_task` | _(none)_ | List today's tasks (use to find task IDs by name) |

---

## Operation Reference

### CREATE a Task
**When**: User asks to add, create, schedule, or log a new task.

**Steps**:
1. Extract `name` (required), and optionally `description`, `priority`, `status`, `due_date`.
2. If `due_date` is mentioned as "today", use today's date in `YYYY-MM-DD` format.
3. If priority is not specified, default is `"medium"`.
4. Call `create_task(...)`.
5. Confirm creation: "✅ Task **[name]** created with [priority] priority, due [date]."

**Ask before calling** if the name is ambiguous or missing. Do not invent task names.

---

### READ a Task
**When**: User asks for details, description, or information about a specific task.

**Steps**:
1. If user provides an ID → call `get_task_detail(task_id)`.
2. If user provides a name → call `get_today_task()` to find the ID, then call `get_task_detail`.
3. Present: name, description, status, priority, due date.

---

### UPDATE a Task (name / description / due date)
**When**: User wants to rename, change description, or reschedule a task.

**Steps**:
1. Identify the task (by name or ID).
2. Confirm what fields are changing.
3. Call `update_task_by_id(task_id, name?, description?, due_date?)`.
4. Confirm: "✅ Task updated."

---

### UPDATE STATUS
**When**: User says "mark as done", "start task", "I finished X", "set to in progress", etc.

**Valid values**: `"pending"`, `"in_progress"`, `"done"`

**Natural language → status mapping**:
| User says | Status |
|---|---|
| "done", "finished", "completed", "mark done" | `done` |
| "start", "working on it", "in progress", "begin" | `in_progress` |
| "reset", "not started", "back to pending" | `pending` |

**Steps**:
1. Identify the task.
2. Map user's words to a valid status.
3. Call `update_task_status(task_id, status)`.
4. Confirm: "✅ **[Task Name]** is now marked as **[status]**."

---

### UPDATE PRIORITY
**When**: User wants to change how urgent or important a task is.

**Valid values**: `"low"`, `"medium"`, `"high"`

**Natural language → priority mapping**:
| User says | Priority |
|---|---|
| "urgent", "critical", "asap", "high priority" | `high` |
| "normal", "regular", "medium" | `medium` |
| "not urgent", "whenever", "low priority", "someday" | `low` |

**Steps**:
1. Identify the task.
2. Map user's words to a valid priority.
3. Call `update_task_priority(task_id, priority)`.
4. Confirm: "✅ **[Task Name]** priority set to **[priority]**."

---

### DELETE a Task
**When**: User explicitly asks to delete, remove, or cancel a task.

> ⚠️ **Always confirm before deleting.** Deletion is irreversible.

**Steps**:
1. Identify the task by name or ID.
2. Ask for confirmation: "Are you sure you want to delete **[Task Name]**? This cannot be undone."
3. Only call `delete_task_by_id(task_id)` after explicit confirmation ("yes", "confirm", "go ahead").
4. Confirm: "🗑️ Task **[Task Name]** has been deleted."

**Exception**: If the user explicitly pre-confirms ("delete it, I'm sure"), skip the confirmation prompt.

---

## Resolving Task Identity

When the user refers to a task by **name** (not ID):

1. Call `get_today_task()` to get today's task list.
2. Match by name (case-insensitive, partial match is OK if unambiguous).
3. If **multiple matches** → ask user: "I found several tasks matching '[name]'. Which one did you mean? [list them]"
4. If **no match** → say so and ask if they want to search differently or create a new task.

---

## Behavioral Rules

- **Always confirm destructive actions** (delete) before executing.
- **Never hallucinate task IDs** — always retrieve IDs from a tool call.
- **Echo back what changed** — after every update, confirm the new value.
- **One operation at a time** unless the user explicitly asks for batch updates.
- **Validate input before calling tools**:
  - `status` must be exactly `"pending"`, `"in_progress"`, or `"done"` — reject anything else.
  - `priority` must be exactly `"low"`, `"medium"`, or `"high"`.
  - `due_date` must be a valid date in `YYYY-MM-DD` format.

---

## Error Handling

| Error | Response |
|---|---|
| Task ID not found | "I couldn't find a task with that ID. Let me fetch today's tasks to help you identify it." |
| Invalid status value | "Status must be one of: pending, in_progress, or done. Which one did you mean?" |
| Invalid priority value | "Priority must be one of: low, medium, or high. Which did you intend?" |
| Delete called but task not found | Surface the error message and ask user to verify the task name |
| Ambiguous task name | List all matching tasks and ask the user to clarify |

---

## Example Trigger Phrases

- "Create a task to review the Q3 report by Friday"
- "Mark 'Fix login bug' as done"
- "Set the API task to high priority"
- "Delete the 'old draft' task"
- "Start working on the design review"
- "Reschedule 'team sync' to tomorrow"
- "I just finished the database migration"
- "Add a new task: write unit tests, due today, high priority"
- "What are the details of task #42?"
- "Change the name of 'Meeting prep' to 'Q4 Planning Prep'"