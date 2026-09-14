"""Document management schemas."""
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    id: UUID
    filename: str
    status: str
    message: str
    upload_time: datetime


class DocumentResponse(BaseModel):
    """Full document details."""
    id: UUID
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    metadata: dict[str, Any] = {}
    uploaded_by: UUID
    upload_time: datetime
    processed_time: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Paginated document list response."""
    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DocumentStatusResponse(BaseModel):
    """Document processing status."""
    id: UUID
    filename: str
    status: str
    progress: Optional[float] = Field(None, description="Processing progress 0-100")
    chunks_processed: int = 0
    total_chunks: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentMetadataUpdate(BaseModel):
    """Schema for updating document metadata."""
    metadata: dict[str, Any]
