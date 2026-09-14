"""Markdown document parser using markdown + BeautifulSoup."""
import logging
import re
from typing import Any
from pathlib import Path

import markdown
from bs4 import BeautifulSoup

from src.core.document.parser import BaseDocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class MarkdownParser(BaseDocumentParser):
    """Extract text and metadata from Markdown files.
    
    Converts Markdown to HTML via the ``markdown`` library, then
    strips tags with ``BeautifulSoup`` to produce clean text.
    """

    async def parse(self, file_path: str) -> ParsedDocument:
        """Parse a Markdown file and return its content and metadata.
        
        Args:
            file_path: Path to the .md file.
            
        Returns:
            ``ParsedDocument`` with extracted text and metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = path.read_text(encoding="utf-8", errors="replace")

        html_content = markdown.markdown(
            raw_text,
            extensions=["extra", "codehilite", "toc", "meta"],
        )

        soup = BeautifulSoup(html_content, "html.parser")
        clean_text = soup.get_text(separator="\n")

        # Extract YAML front-matter if present (simple heuristic)
        metadata = self._extract_front_matter(raw_text)

        # Count headings as pseudo-pages
        heading_count = len(soup.find_all(re.compile(r"^h[1-3]$")))

        return ParsedDocument(
            content=clean_text,
            metadata=metadata,
            page_count=max(1, heading_count),
            raw_text=raw_text,
        )

    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata only, without full parsing.
        
        Args:
            file_path: Path to the .md file.
            
        Returns:
            Dictionary of extracted metadata fields.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = path.read_text(encoding="utf-8", errors="replace")

        return self._extract_front_matter(raw_text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_front_matter(text: str) -> dict[str, Any]:
        """Parse optional YAML front-matter delimited by ``---``.
        
        Falls back to extracting the first ``#`` heading as title.
        """
        metadata: dict[str, Any] = {}

        # Simple front-matter detection
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                front_matter = parts[1].strip()
                for line in front_matter.splitlines():
                    if ":" in line:
                        key, _, value = line.partition(":")
                        metadata[key.strip().lower()] = value.strip()

        # Fallback: extract first H1 heading as title
        if "title" not in metadata:
            m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            if m:
                metadata["title"] = m.group(1).strip()

        return metadata
