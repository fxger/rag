"""Document API endpoints — upload, list, inspect, delete.

The pipeline is real: an upload is written to MinIO, a database record is
created, and a background task parses → chunks → embeds → indexes the
content into Milvus so it becomes retrievable.

This module previously was a stub that appended records to a process-local
list, hardcoded ``chunk_count: 10`` and reported every upload as
``completed`` — no parsing, no chunking, no vectors.  Nothing uploaded
through the API was ever searchable.
"""
import logging
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from src.api.middleware.auth import get_current_user
from src.api.metrics import (
    DOCUMENTS_FAILED_TOTAL,
    DOCUMENTS_PROCESSED_TOTAL,
    DOCUMENTS_UPLOADED_TOTAL,
    DOCUMENT_PROCESSING_DURATION,
)
from src.api.schemas.common import PaginatedResponse, ResponseModel
from src.api.schemas.document import DocumentResponse, DocumentUploadResponse
from src.services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

SUPPORTED_EXTENSIONS = frozenset({"pdf", "docx", "xlsx", "md", "txt"})

_service: DocumentService | None = None


def get_document_service() -> DocumentService:
    """Lazily build the shared service (construction opens no connections)."""
    global _service
    if _service is None:
        _service = DocumentService()
    return _service


def _to_response(row: dict[str, Any]) -> DocumentResponse:
    """Map a serialized document row onto the API schema.

    The service serializes ORM ``title`` as ``file_name``; the schema calls
    it ``filename`` / ``original_filename``.  Mapping here keeps the public
    contract stable without pushing ORM naming into the API.
    """
    created = row.get("created_at")
    owner = row.get("owner_id")
    return DocumentResponse(
        id=uuid.UUID(str(row["id"])),
        filename=row.get("file_name") or "",
        original_filename=row.get("file_name") or "",
        file_type=row.get("file_type") or "",
        file_size=row.get("file_size") or 0,
        status=row.get("status") or "unknown",
        chunk_count=row.get("chunk_count") or 0,
        metadata={},
        uploaded_by=uuid.UUID(str(owner)) if owner else uuid.UUID(int=0),
        upload_time=(
            datetime.fromisoformat(created) if created else datetime.now(timezone.utc)
        ),
        processed_time=None,
        error_message=row.get("error_message"),
    )


async def _process_in_background(document_id: str) -> None:
    """Run parse → chunk → embed → index outside the request lifecycle."""
    started = time.time()
    try:
        result = await get_document_service().process_document(document_id)
        elapsed = time.time() - started
        DOCUMENT_PROCESSING_DURATION.observe(elapsed)

        if result.get("status") == "completed":
            DOCUMENTS_PROCESSED_TOTAL.inc()
            logger.info(
                "Document %s indexed: %d chunk(s) in %.2fs",
                document_id,
                result.get("indexed_chunks", 0),
                elapsed,
            )
        else:
            DOCUMENTS_FAILED_TOTAL.inc()
            logger.error(
                "Document %s processing failed: %s", document_id, result.get("error")
            )
    except Exception as e:  # noqa: BLE001 — background tasks must not raise
        DOCUMENT_PROCESSING_DURATION.observe(time.time() - started)
        DOCUMENTS_FAILED_TOTAL.inc()
        logger.exception("Background processing crashed for %s: %s", document_id, e)


@router.post("/upload", response_model=ResponseModel[DocumentUploadResponse])
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    tenant_id: str = Form(
        "__platform__",
        max_length=64,
        description="租户键（如 store_1000001）；不传则进平台共享库",
    ),
    current_user: dict = Depends(get_current_user),
) -> Any:
    """Upload a document and kick off processing in the background."""
    filename = file.filename or "untitled"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"不支持的文件类型 '.{ext}'；"
                f"支持：{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # The service works from a local path, so spool the upload to disk first.
    temp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    local_path = temp_dir / f"{uuid.uuid4()}_{filename}"

    try:
        with local_path.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                fh.write(chunk)

        created = await get_document_service().upload_document(
            file_path=str(local_path),
            owner_id=current_user["id"],
            file_name=filename,
            content_type=file.content_type,
            # Persisted as document_metadata rows and copied into every chunk's
            # Milvus payload, which is what the tenant filter reads.
            metadata={"tenant_id": tenant_id},
        )
    finally:
        local_path.unlink(missing_ok=True)

    document_id = created["document_id"]
    DOCUMENTS_UPLOADED_TOTAL.inc()
    background.add_task(_process_in_background, document_id)

    return ResponseModel(
        data=DocumentUploadResponse(
            id=uuid.UUID(document_id),
            filename=filename,
            status="uploaded",
            message="已受理，正在后台解析、分块、向量化并写入 Milvus",
            upload_time=datetime.now(timezone.utc),
        )
    )


@router.get("/", response_model=PaginatedResponse[DocumentResponse])
async def list_documents(
    current_user: dict = Depends(get_current_user),
    page: int = 1,
    page_size: int = 10,
) -> Any:
    """List documents, newest first."""
    service = get_document_service()
    page = max(1, page)
    page_size = max(1, min(page_size, 100))

    rows = await service.list_documents(limit=page_size, offset=(page - 1) * page_size)
    total = await service.count_documents()

    return PaginatedResponse(
        items=[_to_response(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/{document_id}", response_model=ResponseModel[DocumentResponse])
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
) -> Any:
    """Get a single document's details, including its chunk count."""
    row = await get_document_service().get_document(document_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return ResponseModel(data=_to_response(row))


@router.get("/{document_id}/status")
async def get_document_status(
    document_id: str,
    current_user: dict = Depends(get_current_user),
) -> Any:
    """Poll processing status (useful right after an upload)."""
    row = await get_document_service().get_processing_status(document_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return ResponseModel(data=row)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
) -> Any:
    """Delete a document record and its MinIO object (chunks cascade)."""
    deleted = await get_document_service().delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return ResponseModel(message="Deleted")
