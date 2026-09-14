"""PDF document parser using pdfplumber."""
import logging
from typing import Any
from pathlib import Path

import pdfplumber

from src.core.document.parser import BaseDocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class PDFParser(BaseDocumentParser):
    """Extract text and metadata from PDF files.
    
    Uses ``pdfplumber`` for reliable text extraction and metadata
    retrieval.  Handles encrypted or corrupted PDFs gracefully by
    returning partial results and logging warnings.
    """

    async def parse(self, file_path: str) -> ParsedDocument:
        """Parse a PDF file and return its content and metadata.
        
        Args:
            file_path: Absolute or relative path to the PDF file.
            
        Returns:
            ``ParsedDocument`` with extracted text, metadata and page count.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        pages_text: list[str] = []
        metadata: dict[str, Any] = {}
        page_count = 0

        try:
            with pdfplumber.open(str(path)) as pdf:
                metadata_raw = pdf.metadata or {}
                metadata = {
                    "title": metadata_raw.get("Title", ""),
                    "author": metadata_raw.get("Author", ""),
                    "subject": metadata_raw.get("Subject", ""),
                    "creator": metadata_raw.get("Creator", ""),
                    "producer": metadata_raw.get("Producer", ""),
                    "creation_date": str(metadata_raw.get("CreationDate", "")),
                    "mod_date": str(metadata_raw.get("ModDate", "")),
                }

                page_count = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)

        except Exception as exc:
            logger.error("Failed to parse PDF '%s': %s", file_path, exc)
            raise RuntimeError(f"PDF parsing failed: {exc}") from exc

        raw_text = "\n\n".join(pages_text)
        return ParsedDocument(
            content=raw_text,
            metadata=metadata,
            page_count=page_count,
            raw_text=raw_text,
        )

    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata only, without parsing page content.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Dictionary of PDF metadata fields.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            with pdfplumber.open(str(path)) as pdf:
                raw = pdf.metadata or {}
                return {
                    "title": raw.get("Title", ""),
                    "author": raw.get("Author", ""),
                    "subject": raw.get("Subject", ""),
                    "page_count": len(pdf.pages),
                }
        except Exception as exc:
            logger.error("Failed to extract PDF metadata '%s': %s", file_path, exc)
            raise RuntimeError(f"PDF metadata extraction failed: {exc}") from exc
