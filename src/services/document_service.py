"""Document service — business logic for document management."""
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, func, delete

from src.config.settings import get_settings
from src.config.minio import upload_file, download_file
from src.config.database import async_session_factory
from src.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentStatus,
    FileType,
)
from src.core.document.processor import DocumentProcessor

logger = logging.getLogger(__name__)

# File extension → FileType mapping
_EXT_MAP: dict[str, FileType] = {
    "pdf": FileType.PDF,
    "docx": FileType.DOCX,
    "xlsx": FileType.XLSX,
    "md": FileType.MARKDOWN,
    "txt": FileType.TEXT,
}


class DocumentService:
    """Service layer for document CRUD and processing orchestration.
    
    Handles uploading to MinIO, creating database records, triggering
    the processing pipeline, and querying document status.
    """

    def __init__(self):
        settings = get_settings()
        self._bucket = settings.get("MINIO_BUCKET", "rag-documents")
        self._processor = DocumentProcessor()

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    async def upload_document(
        self,
        file_path: str,
        file_name: str | None = None,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Upload a file to MinIO and create a database record.
        
        Args:
            file_path: Local path to the file to upload.
            file_name: Override filename (defaults to basename of *file_path*).
            content_type: MIME type override.
            metadata: Optional user-defined metadata tags.
            
        Returns:
            Dictionary with ``document_id``, ``file_name``, ``status``.
            
        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ValueError: If the file type is not supported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        fname = file_name or path.name
        ext = path.suffix.lstrip(".").lower()

        if ext not in _EXT_MAP:
            supported = ", ".join(sorted(_EXT_MAP))
            raise ValueError(f"Unsupported file type '.{ext}'. Supported: {supported}")

        file_type = _EXT_MAP[ext]
        document_id = str(uuid.uuid4())
        storage_path = f"{document_id}/{fname}"

        # Upload to MinIO
        upload_file(str(path), storage_path, content_type=content_type)

        # Create DB record
        doc = Document(
            id=document_id,
            file_name=fname,
            file_type=file_type,
            storage_path=storage_path,
            status=DocumentStatus.UPLOADED,
            file_size=path.stat().st_size,
            content_type=content_type,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        async with async_session_factory() as session:
            async with session.begin():
                session.add(doc)

                # Persist user-supplied metadata
                if metadata:
                    for key, value in metadata.items():
                        meta = DocumentMetadata(
                            id=str(uuid.uuid4()),
                            document_id=document_id,
                            key=key,
                            value=str(value),
                            source="user",
                        )
                        session.add(meta)

        logger.info("Document uploaded: %s (%s)", document_id, fname)

        return {
            "document_id": document_id,
            "file_name": fname,
            "file_type": file_type.value if hasattr(file_type, "value") else str(file_type),
            "status": "uploaded",
            "storage_path": storage_path,
        }

    # ------------------------------------------------------------------
    # Processing trigger
    # ------------------------------------------------------------------

    async def process_document(self, document_id: str) -> dict[str, Any]:
        """Trigger the processing pipeline for an uploaded document.
        
        Args:
            document_id: UUID of the document to process.
            
        Returns:
            Processing result dictionary.
        """
        return await self._processor.process(document_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Retrieve a single document record.
        
        Args:
            document_id: UUID of the document.
            
        Returns:
            Document data dictionary, or ``None`` if not found.
        """
        async with async_session_factory() as session:
            stmt = select(Document).where(Document.id == document_id)
            result = await session.execute(stmt)
            doc = result.scalar_one_or_none()

        if doc is None:
            return None

        return self._serialize_document(doc)

    async def list_documents(
        self,
        status: DocumentStatus | None = None,
        file_type: FileType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List documents with optional filters.
        
        Args:
            status: Filter by processing status.
            file_type: Filter by file type.
            limit: Maximum number of results.
            offset: Pagination offset.
            
        Returns:
            List of document data dictionaries.
        """
        async with async_session_factory() as session:
            stmt = select(Document).order_by(Document.created_at.desc())

            if status is not None:
                stmt = stmt.where(Document.status == status)
            if file_type is not None:
                stmt = stmt.where(Document.file_type == file_type)

            stmt = stmt.limit(limit).offset(offset)
            result = await session.execute(stmt)
            docs = result.scalars().all()

        return [self._serialize_document(d) for d in docs]

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all associated data.
        
        Removes chunks, metadata, and the MinIO object.
        
        Args:
            document_id: UUID of the document to delete.
            
        Returns:
            ``True`` if the document existed and was deleted.
        """
        async with async_session_factory() as session:
            async with session.begin():
                # Delete chunks
                await session.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
                )
                # Delete metadata
                await session.execute(
                    delete(DocumentMetadata).where(DocumentMetadata.document_id == document_id)
                )
                # Delete document record
                result = await session.execute(
                    delete(Document).where(Document.id == document_id)
                )

        # Best-effort MinIO cleanup (ignore errors)
        try:
            doc = await self.get_document(document_id)
            if doc and doc.get("storage_path"):
                from src.config.minio import delete_file  # lazy import
                delete_file(doc["storage_path"])
        except Exception as exc:
            logger.warning("MinIO cleanup failed for %s: %s", document_id, exc)

        logger.info("Document deleted: %s", document_id)
        return result.rowcount > 0

    async def get_processing_status(self, document_id: str) -> dict[str, Any] | None:
        """Get the processing status and progress of a document.
        
        Args:
            document_id: UUID of the document.
            
        Returns:
            Dictionary with ``status``, ``chunk_count``, ``error_message``,
            or ``None`` if the document does not exist.
        """
        async with async_session_factory() as session:
            stmt = select(
                Document.status,
                Document.chunk_count,
                Document.error_message,
            ).where(Document.id == document_id)
            result = await session.execute(stmt)
            row = result.one_or_none()

        if row is None:
            return None

        return {
            "document_id": document_id,
            "status": row.status.value if hasattr(row.status, "value") else str(row.status),
            "chunk_count": row.chunk_count or 0,
            "error_message": row.error_message,
        }

    # ------------------------------------------------------------------
    # Serialization helper
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_document(doc: Document) -> dict[str, Any]:
        """Convert a ``Document`` ORM instance to a plain dictionary."""
        return {
            "id": doc.id,
            "file_name": doc.file_name,
            "file_type": doc.file_type.value if hasattr(doc.file_type, "value") else str(doc.file_type),
            "storage_path": doc.storage_path,
            "status": doc.status.value if hasattr(doc.status, "value") else str(doc.status),
            "file_size": doc.file_size,
            "content_type": doc.content_type,
            "chunk_count": doc.chunk_count or 0,
            "error_message": doc.error_message,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        }

