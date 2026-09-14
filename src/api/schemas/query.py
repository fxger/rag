"""Query and RAG response schemas."""
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema for RAG query request."""
    query: str = Field(..., min_length=1, max_length=2000, description="The search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    use_reranking: bool = Field(default=True, description="Whether to use reranking")
    filter_metadata: Optional[dict[str, Any]] = Field(default=None, description="Metadata filters")


class SourceDocument(BaseModel):
    """Source document reference in query response."""
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Schema for RAG query response."""
    answer: str
    sources: list[SourceDocument]
    latency_ms: float
    query_id: Optional[UUID] = None


# ── 流式响应 Schema ─────────────────────────────────────────────

class StreamMetadata(BaseModel):
    """流式响应元数据 (搜索完成后立即发送)"""
    sources: list[SourceDocument]
    latency_search_ms: int
    total_sources: int


class StreamChunk(BaseModel):
    """流式响应文本块"""
    content: str


class StreamDone(BaseModel):
    """流式响应完成事件"""
    status: str = "completed"
    latency_total_ms: int


class StreamError(BaseModel):
    """流式响应错误事件"""
    error: str


# ── 其他 Schema ─────────────────────────────────────────────────

class QueryHistoryItem(BaseModel):
    """Schema for query history entry."""
    id: UUID
    query: str
    answer: str
    sources: list[dict[str, Any]]
    latency_ms: float
    created_at: datetime
    feedback_score: Optional[int] = None

    class Config:
        from_attributes = True


class FeedbackRequest(BaseModel):
    """Schema for submitting query feedback."""
    score: int = Field(..., ge=1, le=5, description="Feedback score from 1 to 5")
    comment: Optional[str] = Field(None, max_length=1000)
