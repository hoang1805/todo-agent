# The app image — runs BOTH the chat orchestrator (app.py) and the observability
# dashboard (dashboard.py). Same code + deps; compose picks the entrypoint per
# service, so "one service per component" holds without a second image to build.
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
    CHECKPOINT_DB=/data/agent-checkpoints.db

# Default command is the chat app; the dashboard service overrides it in compose.
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
