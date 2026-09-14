"""Document service — business logic for document management."""
import asyncio
import hashlib
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
from src.core.embedding.embedder import EmbeddingGenerator
from src.config.milvus import insert_vectors

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
        # Settings is a pydantic model, not a dict — the previous
        # settings.get(...) raised AttributeError, so this service could
        # never be constructed (which is why the API layer kept its own
        # in-memory stub).
        self._bucket = settings.MINIO_BUCKET
        self._processor = DocumentProcessor()
        self._embedder = EmbeddingGenerator()

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    async def upload_document(
        self,
        file_path: str,
        owner_id: str,
        file_name: str | None = None,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Upload a file to MinIO and create a database record.

        Note on naming: this service used to write ``file_name`` /
        ``storage_path`` / ``content_type`` onto ``Document``, none of which
        exist on the model.  The ORM (which mirrors the real database) is the
        source of truth, so the service follows it: ``title`` / ``file_path``,
        plus the non-nullable ``content_hash`` and ``owner_id`` that were
        previously never supplied.

        Args:
            file_path: Local path to the file to upload.
            owner_id: UUID of the uploading user (``Document.owner_id`` is a
                non-nullable foreign key).
            file_name: Override filename (defaults to basename of *file_path*).
            content_type: MIME type override (passed to MinIO only).
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
        object_path = f"{document_id}/{fname}"

        # MinIO client is synchronous — keep it off the event loop.
        await asyncio.to_thread(
            lambda: upload_file(str(path), object_path, content_type=content_type)
        )

        # Create DB record — field names follow the Document model.
        doc = Document(
            id=uuid.UUID(document_id),
            title=fname,
            content_hash=self._content_hash(path),
            file_path=object_path,
            file_type=file_type,
            # DocumentStatus has no UPLOADED member (only pending /
            # processing / completed / failed) — referencing it raised
            # AttributeError on every upload.
            status=DocumentStatus.PENDING,
            file_size=path.stat().st_size,
            owner_id=uuid.UUID(str(owner_id)),
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
            "storage_path": object_path,
        }

    @staticmethod
    def _content_hash(path: Path) -> str:
        """SHA-256 of the file contents.

        ``Document.content_hash`` is non-nullable and unique; the service
        never populated it, so inserts would have failed outright.
        """
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    # ------------------------------------------------------------------
    # Processing trigger
    # ------------------------------------------------------------------

    async def process_document(self, document_id: str) -> dict[str, Any]:
        """Run the whole pipeline: parse → chunk → embed → index in Milvus.

        ``DocumentProcessor.process`` stops once the chunks are persisted to
        PostgreSQL; its docstring explicitly leaves embedding and Milvus
        insertion to the caller.  This method fills that gap — without it the
        chunks exist in the database but remain invisible to retrieval.

        Args:
            document_id: UUID of the document to process.

        Returns:
            Processing result dictionary, including ``indexed_chunks`` when
            the run succeeds.
        """
        result = await self._processor.process(document_id)
        if result.get("status") != "completed":
            return result

        result["indexed_chunks"] = await self._index_chunks(document_id)
        return result

    async def _index_chunks(self, document_id: str) -> int:
        """Embed a document's persisted chunks and insert them into Milvus.

        Returns:
            Number of vectors written.
        """
        async with async_session_factory() as session:
            stmt = (
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
            )
            rows = (await session.execute(stmt)).scalars().all()

        if not rows:
            logger.warning("Document %s produced no chunks to index", document_id)
            return 0

        embeddings = await self._embedder.generate([row.content for row in rows])

        documents = [
            {
                "document_id": str(document_id),
                "chunk_id": str(row.id),
                "content": row.content,
                "embedding": embedding,
                "metadata": {
                    **(row.metadata_json or {}),
                    "chunk_index": row.chunk_index,
                    "token_count": row.token_count,
                },
            }
            for row, embedding in zip(rows, embeddings)
        ]

        # The pymilvus client is synchronous — keep it off the event loop.
        inserted = await asyncio.to_thread(insert_vectors, documents)
        logger.info(
            "Indexed %d/%d chunks for document %s",
            inserted,
            len(documents),
            document_id,
        )
        return inserted

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

    async def count_documents(
        self,
        status: DocumentStatus | None = None,
        file_type: FileType | None = None,
    ) -> int:
        """Count documents (optionally filtered).

        ``list_documents`` returns a single page without a total, which the
        API layer needs to build pagination metadata.
        """
        async with async_session_factory() as session:
            stmt = select(func.count()).select_from(Document)
            if status is not None:
                stmt = stmt.where(Document.status == status)
            if file_type is not None:
                stmt = stmt.where(Document.file_type == file_type)
            return (await session.execute(stmt)).scalar_one()

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all associated data.
        
        Removes chunks, metadata, and the MinIO object.
        
        Args:
            document_id: UUID of the document to delete.
            
        Returns:
            ``True`` if the document existed and was deleted.
        """
        # Look the row up BEFORE deleting.  The previous order deleted first
        # and queried afterwards, so the MinIO cleanup below could never run.
        existing = await self.get_document(document_id)
        if existing is None:
            return False

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

        # Best-effort object cleanup (errors are non-fatal).
        object_path = existing.get("file_path")
        if object_path:
            try:
                from src.config.minio import delete_file  # lazy import

                await asyncio.to_thread(delete_file, object_path)
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
            # The ORM calls these title / file_path; kept as file_name /
            # file_path here to match what the API layer consumes.
            "file_name": doc.title,
            "file_type": doc.file_type.value if hasattr(doc.file_type, "value") else str(doc.file_type),
            "file_path": doc.file_path,
            "status": doc.status.value if hasattr(doc.status, "value") else str(doc.status),
            "file_size": doc.file_size,
            "chunk_count": doc.chunk_count or 0,
            "error_message": doc.error_message,
            "owner_id": str(doc.owner_id) if doc.owner_id else None,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        }

