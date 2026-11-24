"""Intelligent document chunking with type-aware strategies."""
import logging
import re
from dataclasses import dataclass
from typing import Any

from src.core.tokenization import (
    SENTENCE_END_CHARS,
    count_chars,
    estimate_tokens,
    split_tokens,
    token_spans,
    tokens_for_counts,
)

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

    For Markdown files the chunker first splits by headings, then applies
    a sliding-window strategy within each section.  For all other file
    types a standard sliding-window approach with sentence-boundary
    detection is used.

    Both the window boundary *and* the reported ``token_count`` come from
    :mod:`src.core.tokenization`.  Keeping them on one measure is what
    makes ``chunk_size`` an actual constraint — previously the window was
    measured in whitespace tokens while the reported count used a
    character heuristic, so a "512-token" chunk could report 764.

    Args:
        chunk_size: Maximum estimated tokens per chunk.
        chunk_overlap: Estimated tokens of overlap between consecutive
            chunks, for context continuity.
    """

    # Markdown heading pattern
    _HEADING = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)

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
            # Re-index and adjust offsets relative to the full document
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
        """Sliding-window chunking bounded by an estimated token budget.

        Tokens are tracked as character spans so the original text can be
        sliced back verbatim — re-joining token strings would inject
        spaces between CJK characters.  Windows advance by
        ``chunk_size - chunk_overlap`` *estimated tokens*, so the overlap
        means the same thing for Chinese and English input.
        """
        spans = token_spans(text)
        if not spans:
            return []

        # Count each token's characters once, so budgets can be accumulated
        # in a single pass instead of re-measuring substrings.
        weights = [count_chars(text[s:e]) for s, e in spans]
        total = len(spans)
        step = max(1, self.chunk_size - self.chunk_overlap)

        chunks: list[Chunk] = []
        start = 0

        while start < total:
            end = self._window_end(text, spans, weights, start)

            raw_text = text[spans[start][0]:spans[end - 1][1]]
            chunk_text = raw_text.strip()
            if chunk_text:
                # Keep offsets exact after trimming surrounding whitespace.
                leading = len(raw_text) - len(raw_text.lstrip())
                start_char = spans[start][0] + leading
                chunks.append(Chunk(
                    content=chunk_text,
                    index=len(chunks),
                    start_offset=start_char,
                    end_offset=start_char + len(chunk_text),
                    token_count=estimate_tokens(chunk_text),
                    metadata=metadata,
                ))

            # Advance by `step` estimated tokens (at least one token, so a
            # pathologically short window can never stall the loop).
            cjk = other = 0
            nxt = start
            while nxt < end:
                c, o = weights[nxt]
                if nxt > start and tokens_for_counts(cjk + c, other + o) > step:
                    break
                cjk += c
                other += o
                nxt += 1
            start = max(nxt, start + 1)

        return chunks

    def _window_end(
        self,
        text: str,
        spans: list[tuple[int, int]],
        weights: list[tuple[int, int]],
        start: int,
    ) -> int:
        """Exclusive end index of the window beginning at token *start*.

        The window grows while the accumulated estimate stays within
        ``chunk_size`` (always accepting at least one token).  It is then
        pulled back to the last sentence-ending token found within the
        final fifth of the window, unless that would empty the window.
        """
        total = len(spans)
        cjk = other = 0
        end = start

        while end < total:
            c, o = weights[end]
            if end > start and tokens_for_counts(cjk + c, other + o) > self.chunk_size:
                break
            cjk += c
            other += o
            end += 1

        if end - start <= 1:
            return end

        tail_start = max(start, end - max(1, (end - start) // 5))
        cut = 0
        for i in range(tail_start, end):
            token = text[spans[i][0]:spans[i][1]]
            if token and token[-1] in SENTENCE_END_CHARS:
                cut = i + 1

        return cut if cut > start else end

    # ------------------------------------------------------------------
    # Token helpers
    #
    # Thin wrappers kept for backwards compatibility.  The implementation
    # lives in src.core.tokenization — do not reintroduce a local copy of
    # the counting rules here, that is exactly how the two measures drifted
    # apart in the first place.
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate the token count of *text*."""
        return estimate_tokens(text)

    @staticmethod
    def _token_spans(text: str) -> list[tuple[int, int]]:
        """Return ``(start, end)`` character spans for every token."""
        return token_spans(text)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split *text* into CJK-aware tokens."""
        return split_tokens(text)
