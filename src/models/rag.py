"""RAG contracts — the typed shapes the retrieval loop validates before branching.

Same discipline as the task/plan contracts: every tool result is checked against
:class:`DataChunk` and every judge output against :class:`RetrievalDecision`, so
nothing ungoverned crosses the agent's boundary.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["planning_log", "document"]


class DataChunk(BaseModel):
    """One retrieved chunk: the text, its similarity score, and where it came from."""

    text: str
    score: float
    source_type: SourceType
    metadata: dict = Field(default_factory=dict)


class RetrievalDecision(BaseModel):
    """The judge's structured verdict after a retrieval pass (not a numeric cutoff).

    ``sufficient`` stops the loop and triggers generation; otherwise
    ``next_source``/``next_query`` steer a reformulated retry.
    """

    sufficient: bool
    reasoning: str
    next_source: Literal["logs", "documents", "both"] | None = None
    next_query: str | None = None  # reformulated query, only when sufficient is False
