"""Document API endpoints — simplified for demo."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, UploadFile, File

from src.api.middleware.auth import get_current_user
from src.api.schemas.common import ResponseModel, PaginatedResponse
from src.api.schemas.document import DocumentResponse, DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["Documents"])

# In-memory document store
_documents: list[dict] = []


@router.post("/upload", response_model=ResponseModel[DocumentUploadResponse])
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
) -> Any:
    """Upload a document (demo mode)."""
    doc = {
        "id": str(uuid.uuid4()),
        "title": file.filename,
        "file_type": file.filename.split(".")[-1] if "." in file.filename else "txt",
        "file_size": file.size or 0,
        "status": "completed",
        "owner_id": current_user["id"],
        "chunk_count": 10,
    }
    _documents.append(doc)
    
    return ResponseModel(
        data=DocumentUploadResponse(
            id=doc["id"],
            title=doc["title"],
            status=doc["status"],
        )
    )


@router.get("/", response_model=PaginatedResponse[DocumentResponse])
async def list_documents(
    current_user: dict = Depends(get_current_user),
    page: int = 1,
    page_size: int = 10,
) -> Any:
    """List documents for current user."""
    user_docs = [d for d in _documents if d["owner_id"] == current_user["id"]]
    start = (page - 1) * page_size
    end = start + page_size
    
    return PaginatedResponse(
        items=[DocumentResponse(**d) for d in user_docs[start:end]],
        total=len(user_docs),
        page=page,
        page_size=page_size,
        total_pages=(len(user_docs) + page_size - 1) // page_size,
    )


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
) -> Any:
    """Get document details."""
    doc = next((d for d in _documents if d["id"] == document_id), None)
    if not doc:
        return {"success": False, "message": "Document not found"}
    return ResponseModel(data=DocumentResponse(**doc))


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
) -> Any:
    """Delete a document."""
    global _documents
    before = len(_documents)
    _documents = [d for d in _documents if d["id"] != document_id]
    deleted = before > len(_documents)
    return ResponseModel(message="Deleted" if deleted else "Not found")