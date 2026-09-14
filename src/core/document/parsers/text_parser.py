"""Plain text document parser with encoding detection fallback."""
import logging
from typing import Any
from pathlib import Path

from src.core.document.parser import BaseDocumentParser, ParsedDocument

logger = logging.getLogger(__name__)

# Ordered list of encodings to try when reading text files
_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "gbk", "big5", "shift_jis"]


class TextParser(BaseDocumentParser):
    """Extract text from plain .txt files.
    
    Attempts UTF-8 first, then falls back through common encodings
    until one succeeds.  Uses ``errors="replace"`` as a last resort
    so that no file is ever completely unreadable.
    """

    async def parse(self, file_path: str) -> ParsedDocument:
        """Parse a text file and return its content and metadata.
        
        Args:
            file_path: Path to the .txt file.
            
        Returns:
            ``ParsedDocument`` with extracted text and metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        raw_text, encoding_used = self._read_with_fallback(path)

        # Basic line/word statistics
        lines = raw_text.splitlines()
        words = raw_text.split()

        metadata: dict[str, Any] = {
            "encoding": encoding_used,
            "line_count": len(lines),
            "word_count": len(words),
            "char_count": len(raw_text),
        }

        return ParsedDocument(
            content=raw_text,
            metadata=metadata,
            page_count=max(1, len(lines) // 50),  # ~50 lines per page heuristic
            raw_text=raw_text,
        )

    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata only, without returning full content.
        
        Args:
            file_path: Path to the .txt file.
            
        Returns:
            Dictionary of file statistics.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        raw_text, encoding_used = self._read_with_fallback(path)
        lines = raw_text.splitlines()
        return {
            "encoding": encoding_used,
            "line_count": len(lines),
            "char_count": len(raw_text),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_with_fallback(path: Path) -> tuple[str, str]:
        """Read a text file, trying multiple encodings in order.
        
        Returns:
            Tuple of (text content, encoding that succeeded).
        """
        for enc in _ENCODINGS:
            try:
                text = path.read_text(encoding=enc)
                logger.debug("Read '%s' with encoding %s", path, enc)
                return text, enc
            except (UnicodeDecodeError, LookupError):
                continue

        # Last resort: read as bytes and decode with replacement
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8", errors="replace")
        logger.warning("Fell back to UTF-8 with replacement for '%s'", path)
        return text, "utf-8-replace"
