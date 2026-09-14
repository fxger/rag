"""Document processing pipeline — parse, clean, chunk, persist."""
import asyncio
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update

from src.config.settings import get_settings
from src.config.minio import download_file
from src.config.database import async_session_factory
from src.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentStatus,
    FileType,
    MetadataSource,
    MetadataValueType,
)
from src.core.document.parsers import get_parser
from src.core.document.chunker import SmartChunker, Chunk
from src.core.document.cleaner import ContentCleaner

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Full document processing pipeline.
    
    Workflow:
    
    1. Download the source file from MinIO to a local temp directory.
    2. Select the appropriate parser based on file type.
    3. Parse the document to extract text and metadata.
    4. Clean the extracted text.
    5. Chunk the cleaned text into embedding-ready segments.
    6. Persist chunks and metadata to the database.
    7. Update the document status to ``COMPLETED`` or ``FAILED``.
    
    The caller is responsible for generating embeddings from the
    returned chunks and inserting them into Milvus.
    
    Args:
        chunk_size: Target token count per chunk (passed to ``SmartChunker``).
        chunk_overlap: Token overlap between consecutive chunks.
        temp_dir: Local directory for temporary file downloads.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        temp_dir: str | None = None,
    ):
        settings = get_settings()
        # Settings is a pydantic model, not a dict: settings.get(...) raised
        # AttributeError, so this class could never be constructed.  The old
        # "/tmp/..." default was also wrong on Windows, where it resolves
        # against the current drive root.
        self._temp_dir = Path(
            temp_dir
            or settings.TEMP_DIR
            or str(Path(tempfile.gettempdir()) / "rag_uploads")
        )
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        self._chunker = SmartChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._cleaner = ContentCleaner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, document_id: str) -> dict[str, Any]:
        """Run the full processing pipeline for a single document.
        
        Args:
            document_id: UUID of the ``Document`` record in the database.
            
        Returns:
            Dictionary with ``status``, ``chunk_count`` and ``error``
            (if any).
        """
        local_path: Path | None = None

        try:
            # 1. Load document record
            async with async_session_factory() as session:
                stmt = select(Document).where(Document.id == document_id)
                result = await session.execute(stmt)
                doc: Document | None = result.scalar_one_or_none()

            if doc is None:
                raise ValueError(f"Document not found: {document_id}")

            # Mark as processing
            await self._update_status(document_id, DocumentStatus.PROCESSING)

            # 2. Download from MinIO (the ORM names these title / file_path;
            #    the synchronous client is pushed off the event loop)
            local_path = self._temp_dir / f"{document_id}_{doc.title}"
            await asyncio.to_thread(download_file, doc.file_path, str(local_path))

            # 3. Parse
            # NB: the parentheses matter — without them this parses as
            # `(file_ext or value) if hasattr(...) else str(...)`, which
            # discards the extension whenever file_type has no `.value`.
            file_ext = Path(doc.title).suffix.lstrip(".").lower()
            file_type = file_ext or (
                doc.file_type.value
                if hasattr(doc.file_type, "value")
                else str(doc.file_type)
            )

            parser = get_parser(file_type)
            parsed = await parser.parse(str(local_path))

            # 4. Clean
            cleaned_text = self._cleaner.clean(parsed.content)

            # 5. Chunk
            chunks: list[Chunk] = self._chunker.chunk(
                content=cleaned_text,
                file_type=file_type,
                metadata={"document_id": document_id, "file_type": file_type},
            )

            # 6. Persist chunks and metadata
            await self._persist_chunks(document_id, chunks, parsed.metadata)

            # 7. Mark completed
            await self._update_status(
                document_id,
                DocumentStatus.COMPLETED,
                chunk_count=len(chunks),
            )

            logger.info(
                "Document %s processed successfully: %d chunks", document_id, len(chunks),
            )

            return {
                "status": "completed",
                "chunk_count": len(chunks),
                "page_count": parsed.page_count,
            }

        except Exception as exc:
            logger.error("Processing failed for document %s: %s", document_id, exc, exc_info=True)
            await self._update_status(
                document_id, DocumentStatus.FAILED, error=str(exc),
            )
            return {"status": "failed", "error": str(exc)}

        finally:
            # Clean up temp file
            if local_path and local_path.exists():
                try:
                    local_path.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    async def _persist_chunks(
        self,
        document_id: str,
        chunks: list[Chunk],
        parsed_metadata: dict[str, Any],
    ) -> None:
        """Write chunks and metadata rows to the database."""
        async with async_session_factory() as session:
            async with session.begin():
                doc_uuid = uuid.UUID(str(document_id))

                # Insert metadata
                for key, value in parsed_metadata.items():
                    session.add(
                        DocumentMetadata(
                            id=uuid.uuid4(),
                            document_id=doc_uuid,
                            key=key,
                            value=str(value),
                            value_type=self._infer_value_type(value),
                            source=MetadataSource.PARSER,
                        )
                    )

                # Insert chunks.
                # start_offset / end_offset are deliberately not written: the
                # chunker produces them, but neither the DocumentChunk model
                # nor the document_chunks table has those columns.  Passing
                # them raised
                #   TypeError: 'start_offset' is an invalid keyword argument
                # and failed every processing run.
                for chunk in chunks:
                    session.add(
                        DocumentChunk(
                            id=uuid.uuid4(),
                            document_id=doc_uuid,
                            chunk_index=chunk.index,
                            content=chunk.content,
                            token_count=chunk.token_count,
                            metadata_json=chunk.metadata,
                        )
                    )

    async def _update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> None:
        """Update the document status and optional processing stats."""
        # func.now() rather than datetime.now(timezone.utc): these columns are
        # `timestamp WITHOUT time zone`, and asyncpg rejects an aware datetime
        # ("can't subtract offset-naive and offset-aware datetimes").  Letting
        # the server stamp it also avoids clock skew between app and database.
        values: dict[str, Any] = {
            "status": status,
            "updated_at": func.now(),
        }
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        if error is not None:
            values["error_message"] = error

        async with async_session_factory() as session:
            async with session.begin():
                stmt = (
                    update(Document)
                    .where(Document.id == document_id)
                    .values(**values)
                )
                await session.execute(stmt)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_value_type(value: Any) -> MetadataValueType:
        """Infer the metadata value type from a Python value.

        ``MetadataValueType`` defines STRING / NUMBER / BOOLEAN / JSON — there
        is no INTEGER or FLOAT member.  This used to return those two names,
        so every processing run died with AttributeError at the metadata
        insert.  Both collapse into NUMBER; the value is persisted as TEXT
        anyway (see ``value=str(value)``), so the int/float distinction would
        not survive.
        """
        if isinstance(value, bool):
            # Must precede the int check — bool is a subclass of int.
            return MetadataValueType.BOOLEAN
        if isinstance(value, (int, float)):
            return MetadataValueType.NUMBER
        return MetadataValueType.STRING
