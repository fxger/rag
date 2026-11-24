"""Pydantic schemas for the knowledge-base retrieval endpoint.

This API is deliberately *retrieval only*: no LLM call, no answer synthesis.
The calling agent (``cs-agent``) already owns the conversation and its own
generation step; what it lacks is原文 chunks.  Returning ``answer`` from here
would mean paying for a second LLM round-trip and then throwing the answer
away.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

PLATFORM_TENANT = "__platform__"


class KbRetrieveRequest(BaseModel):
    """One retrieval request, scoped to one tenant."""

    query: str = Field(..., min_length=1, max_length=2000, description="检索问句")
    tenant_id: str = Field(
        default=PLATFORM_TENANT,
        max_length=64,
        description="租户键，如 store_1000001；平台共享库用 __platform__",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    deadline_ms: int = Field(
        default=1500,
        ge=100,
        le=10000,
        description="服务端预算（毫秒）。超时返回 degraded=true 的空结果，不阻塞调用方",
    )
    rerank: bool = Field(
        default=False,
        description="是否精排。默认关闭：cross-encoder 在本机 CPU 上耗时数百毫秒到数秒",
    )


class KbChunk(BaseModel):
    """One retrieved chunk — the full text, not a 200-char preview."""

    chunk_id: str = ""
    document_id: str = ""
    content: str = ""
    score: float = 0.0
    tenant_id: str = PLATFORM_TENANT
    metadata: dict[str, Any] = Field(default_factory=dict)


class KbRetrieveResponse(BaseModel):
    """Retrieval result envelope.

    ``degraded=true`` is a valid, expected outcome (timeout / Milvus down /
    embedding failure).  Callers must treat it as "no knowledge available"
    and fall back to their own data source — never as a hard error.
    """

    tenant_id: str = PLATFORM_TENANT
    chunks: list[KbChunk] = Field(default_factory=list)
    degraded: bool = False
    degraded_reason: str | None = None
    latency_ms: int = 0
    mode: str = "vector"
