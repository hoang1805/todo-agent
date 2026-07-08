# The app image — the chat orchestrator (app.py). The observability dashboard is
# a view *inside* this same app (a "📊 Dashboard" tab), so it's one service on one
# port, not a separate container/port.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY eval/ ./eval/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HISTORY_DB=/data/chat-history.db \
    EVENTS_DB=/data/events.db \
    CHECKPOINT_DB=/data/agent-checkpoints.db \
    AGENT_DEBUG=1

# One app, one port — chat + dashboard both served here.
# NOTE: exec-form CMD can't do `VAR=value cmd` (that only works in a shell) — set
# env vars via ENV above (or `environment:` in compose), never as an argv element.
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
