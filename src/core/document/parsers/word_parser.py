"""Word document parser using python-docx."""
import logging
from datetime import datetime
from typing import Any
from pathlib import Path

from docx import Document as DocxDocument

from src.core.document.parser import BaseDocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class WordParser(BaseDocumentParser):
    """Extract text and metadata from .docx files.
    
    Uses ``python-docx`` to read paragraphs and core properties.
    """

    async def parse(self, file_path: str) -> ParsedDocument:
        """Parse a Word document and return its content and metadata.
        
        Args:
            file_path: Absolute or relative path to the .docx file.
            
        Returns:
            ``ParsedDocument`` with extracted text, metadata and page count.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Word file not found: {file_path}")

        try:
            doc = DocxDocument(str(path))
        except Exception as exc:
            logger.error("Failed to open Word document '%s': %s", file_path, exc)
            raise RuntimeError(f"Word parsing failed: {exc}") from exc

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        raw_text = "\n\n".join(paragraphs)

        core = doc.core_properties
        created = core.created
        modified = core.modified

        metadata: dict[str, Any] = {
            "title": core.title or "",
            "author": core.author or "",
            "subject": core.subject or "",
            "created": created.isoformat() if isinstance(created, datetime) else str(created or ""),
            "modified": modified.isoformat() if isinstance(modified, datetime) else str(modified or ""),
            "revision": core.revision,
            "category": core.category or "",
            "keywords": core.keywords or "",
            "paragraph_count": len(paragraphs),
        }

        # python-docx does not expose page count; estimate from paragraph count
        # (rough heuristic: ~3 paragraphs per page for typical documents)
        estimated_pages = max(1, len(paragraphs) // 3) if paragraphs else 0

        return ParsedDocument(
            content=raw_text,
            metadata=metadata,
            page_count=estimated_pages,
            raw_text=raw_text,
        )

    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata only, without parsing full content.
        
        Args:
            file_path: Path to the .docx file.
            
        Returns:
            Dictionary of core document properties.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Word file not found: {file_path}")

        try:
            doc = DocxDocument(str(path))
            core = doc.core_properties
            created = core.created
            modified = core.modified
            return {
                "title": core.title or "",
                "author": core.author or "",
                "subject": core.subject or "",
                "created": created.isoformat() if isinstance(created, datetime) else str(created or ""),
                "modified": modified.isoformat() if isinstance(modified, datetime) else str(modified or ""),
            }
        except Exception as exc:
            logger.error("Failed to extract Word metadata '%s': %s", file_path, exc)
            raise RuntimeError(f"Word metadata extraction failed: {exc}") from exc
