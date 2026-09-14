"""Intelligent document chunking with type-aware strategies."""
import re
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with positional metadata.
    
    Attributes:
        content: The text content of this chunk.
        index: Zero-based index of the chunk in the document.
        start_offset: Character offset where this chunk begins in the
            original document.
        end_offset: Character offset where this chunk ends.
        token_count: Estimated token count for this chunk.
        metadata: Optional metadata carried forward from the document
            or injected by the chunking strategy.
    """
    content: str
    index: int
    start_offset: int
    end_offset: int
    token_count: int
    metadata: dict[str, Any] | None = None


class SmartChunker:
    """Split documents into chunks using type-aware strategies.
    
    For Markdown files the chunker first splits by headings, then
    applies a sliding-window strategy within each section.  For all
    other file types a standard sliding-window approach with
    sentence-boundary detection is used.
    
    Args:
        chunk_size: Target maximum token count per chunk.
        chunk_overlap: Number of tokens to overlap between
            consecutive chunks for context continuity.
    """

    # Sentence-ending patterns for multiple languages.
    # CJK punctuation is self-delimiting (no trailing space required),
    # while Latin punctuation must be followed by whitespace or the end
    # of the string so that abbreviations like "U.S.A." are not split.
    _SENTENCE_END = re.compile(r"[。！？]|[.!?](?=\s|$)")

    # Markdown heading pattern
    _HEADING = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)

    # Token pattern.  Each CJK ideograph is its own token; runs of
    # non-CJK, non-whitespace characters form a single token.  Chinese
    # text has no inter-word spaces, so whitespace-only tokenization
    # would collapse an entire document into one token.
    _TOKEN = re.compile(
        r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
        r"|[^\s\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
    )

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        content: str,
        file_type: str = "txt",
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split *content* into chunks.
        
        Args:
            content: The full document text.
            file_type: File extension hint (e.g. ``"md"``, ``"pdf"``).
            metadata: Optional metadata to attach to every chunk.
            
        Returns:
            Ordered list of ``Chunk`` objects.
        """
        if not content or not content.strip():
            return []

        if file_type.lower() == "md":
            return self._markdown_chunks(content, metadata)
        return self._sliding_window_chunks(content, metadata)

    # ------------------------------------------------------------------
    # Strategy: Markdown-aware
    # ------------------------------------------------------------------

    def _markdown_chunks(
        self, text: str, metadata: dict[str, Any] | None,
    ) -> list[Chunk]:
        """Split Markdown by headings first, then chunk each section."""
        # Find all heading positions
        heading_positions = [m.start() for m in self._HEADING.finditer(text)]

        if not heading_positions:
            # No headings — fall back to sliding window
            return self._sliding_window_chunks(text, metadata)

        sections: list[str] = []
        # Content before the first heading (if any)
        if heading_positions[0] > 0:
            sections.append(text[: heading_positions[0]])

        for i, start in enumerate(heading_positions):
            end = heading_positions[i + 1] if i + 1 < len(heading_positions) else len(text)
            sections.append(text[start:end])

        chunks: list[Chunk] = []
        global_offset = 0

        for section in sections:
            section = section.strip()
            if not section:
                global_offset += len(section) + 1
                continue

            section_chunks = self._sliding_window_chunks(section, metadata)
            # Re-index and adjust offsets relative to full document
            for ch in section_chunks:
                ch.index = len(chunks)
                ch.start_offset += global_offset
                ch.end_offset += global_offset
                chunks.append(ch)

            global_offset += len(section) + 1

        return chunks

    # ------------------------------------------------------------------
    # Strategy: Sliding window with sentence boundaries
    # ------------------------------------------------------------------

    def _sliding_window_chunks(
        self, text: str, metadata: dict[str, Any] | None,
    ) -> list[Chunk]:
        """Standard sliding-window chunking with sentence-boundary detection.
        
        The algorithm walks through the text in steps of
        ``chunk_size - chunk_overlap`` tokens.  At each step it
        snaps the end boundary to the nearest sentence end to avoid
        splitting mid-sentence.
        """
        # Tokens are tracked as character spans rather than strings so
        # that the original text can be sliced back verbatim.  Joining
        # tokens would inject spaces between CJK characters and destroy
        # the source text.
        spans = self._token_spans(text)
        total_tokens = len(spans)

        if total_tokens == 0:
            return []

        chunks: list[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        pos = 0  # token index

        while pos < total_tokens:
            end_pos = min(pos + self.chunk_size, total_tokens)

            # Try to snap to a sentence boundary within the last 20 % of
            # the window.
            snap_window = max(1, self.chunk_size // 5)
            snap_start = max(pos, end_pos - snap_window)

            if snap_start < end_pos:
                snap_text = text[spans[snap_start][0]:spans[end_pos - 1][1]]
                match = None
                for m in self._SENTENCE_END.finditer(snap_text):
                    match = m

                if match:
                    # Character position of the sentence end, then the
                    # first token boundary at or past it.
                    cut_char = spans[snap_start][0] + match.end()
                    new_end = snap_start
                    for i in range(snap_start, end_pos):
                        if spans[i][1] >= cut_char:
                            new_end = i + 1
                            break
                    if new_end > pos:
                        end_pos = new_end

            raw_text = text[spans[pos][0]:spans[end_pos - 1][1]]
            chunk_text = raw_text.strip()
            if chunk_text:
                # Keep offsets exact after trimming surrounding whitespace.
                leading = len(raw_text) - len(raw_text.lstrip())
                start_char = spans[pos][0] + leading
                end_char = start_char + len(chunk_text)

                chunk = Chunk(
                    content=chunk_text,
                    index=len(chunks),
                    start_offset=start_char,
                    end_offset=end_char,
                    token_count=self._estimate_tokens(chunk_text),
                    metadata=metadata,
                )
                chunks.append(chunk)

            pos += step

        return chunks

    # ------------------------------------------------------------------
    # Token estimation helpers
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Rough token-count estimate.
        
        English: ~4 chars per token.  CJK characters: ~1 char ≈ 1 token.
        Mixed text uses a weighted average.
        """
        if not text:
            return 0

        cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        non_cjk = len(text) - cjk_chars

        # 4 chars per token for Latin text, 1 token per CJK char
        return (non_cjk // 4) + cjk_chars

    # ------------------------------------------------------------------
    # Tokenization (CJK-aware, no external dependency)
    # ------------------------------------------------------------------

    @classmethod
    def _token_spans(cls, text: str) -> list[tuple[int, int]]:
        """Return ``(start, end)`` character spans for every token.

        Each CJK ideograph is one token; a run of non-CJK,
        non-whitespace characters forms one token.  The spans index
        into *text*, so the original string can be sliced back
        losslessly — unlike re-joining token strings, which would
        inject spaces between CJK characters.
        """
        return [(m.start(), m.end()) for m in cls._TOKEN.finditer(text)]

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """Split text into CJK-aware tokens.

        Each CJK ideograph becomes its own token; runs of Latin
        characters, digits and punctuation are grouped by whitespace.
        """
        return [text[start:end] for start, end in cls._token_spans(text)]
