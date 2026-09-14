"""Abstract base class for document parsers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ParsedDocument:
    """Result of parsing a document.
    
    Attributes:
        content: Cleaned, readable text content.
        metadata: Key-value metadata extracted from the document.
        page_count: Number of pages (0 for non-paginated formats).
        raw_text: Unprocessed original text before any cleaning.
    """
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_count: int = 0
    raw_text: str = ""

class BaseDocumentParser(ABC):
    """Abstract base for all document parsers.
    
    Every concrete parser must implement ``parse`` and
    ``extract_metadata``.  The ``parse`` method returns a
    ``ParsedDocument`` while ``extract_metadata`` is a lighter
    operation that avoids full content extraction.
    """

    @abstractmethod
    async def parse(self, file_path: str) -> ParsedDocument:
        """Parse a document and return extracted content + metadata."""
        ...

    @abstractmethod
    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from the document without full parsing."""
        ...
