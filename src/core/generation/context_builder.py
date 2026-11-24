"""Context assembly from retrieval results with source citation support.

Builds structured context strings for LLM consumption, respecting token budgets
and including proper source attribution for grounded generation.
"""
import logging
from typing import Optional

from src.core.retrieval.models import SearchResult
from src.core.tokenization import estimate_tokens

logger = logging.getLogger(__name__)

# Citation formatting templates
CITATION_HEADER = "## Source [{source_id}]: {title}"
CITATION_METADATA = "Document: {document} | Section: {section} | Relevance: {score:.3f}"
CONTEXT_SEPARATOR = "\n---\n"


class ContextBuilder:
    """Assembles context strings from search results with source citations.
    
    Optimizes context window usage by prioritizing high-relevance results
    and truncating to fit within token budgets.
    """

    def build(
        self,
        query: str,
        results: list[SearchResult],
        max_tokens: int = 4000,
    ) -> str:
        """Assemble context string from search results.
        
        Formats search results into a structured context block suitable for
        RAG prompt injection. Each result includes source attribution.
        
        Args:
            query: Original user query for context.
            results: List of search results sorted by relevance.
            max_tokens: Maximum token budget for context (default: 4000).
            
        Returns:
            Formatted context string with source citations.
        """
        if not results:
            logger.warning("No search results to build context from")
            return "No relevant documents found."

        context_parts: list[str] = []
        current_tokens = 0
        
        for idx, result in enumerate(results, start=1):
            # Build source block
            source_block = self._format_source_block(idx, result)
            block_tokens = estimate_tokens(source_block)
            
            # Check token budget
            if current_tokens + block_tokens > max_tokens:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 50:  # Minimum useful content
                    truncated = self._truncate_to_tokens(source_block, remaining_tokens)
                    context_parts.append(truncated)
                logger.info(
                    "Context truncated at result %d/%d (token limit reached)",
                    idx - 1,
                    len(results),
                )
                break
            
            context_parts.append(source_block)
            current_tokens += block_tokens

        context = CONTEXT_SEPARATOR.join(context_parts)
        logger.info(
            "Built context: %d sources, ~%d tokens",
            len(context_parts),
            current_tokens,
        )
        return context

    def build_with_sources(
        self,
        query: str,
        results: list[SearchResult],
        max_tokens: int = 4000,
    ) -> tuple[str, list[dict]]:
        """Build context and return structured source references.
        
        Args:
            query: Original user query.
            results: List of search results sorted by relevance.
            max_tokens: Maximum token budget for context.
            
        Returns:
            Tuple of (context_string, source_references_list).
            Each source reference contains: id, title, document, section, score.
        """
        if not results:
            return "No relevant documents found.", []

        context_parts: list[str] = []
        sources: list[dict] = []
        current_tokens = 0

        for idx, result in enumerate(results, start=1):
            source_block = self._format_source_block(idx, result)
            block_tokens = estimate_tokens(source_block)

            if current_tokens + block_tokens > max_tokens:
                remaining = max_tokens - current_tokens
                if remaining > 50:
                    truncated = self._truncate_to_tokens(source_block, remaining)
                    context_parts.append(truncated)
                    sources.append(self._extract_source_ref(idx, result))
                break

            context_parts.append(source_block)
            sources.append(self._extract_source_ref(idx, result))
            current_tokens += block_tokens

        context = CONTEXT_SEPARATOR.join(context_parts)
        logger.info(
            "Built context with sources: %d included of %d available",
            len(sources),
            len(results),
        )
        return context, sources

    def _format_source_block(self, source_id: int, result: SearchResult) -> str:
        """Format a single search result as a context block.
        
        Args:
            source_id: Sequential source identifier for citation.
            result: Search result to format.
            
        Returns:
            Formatted source block string.
        """
        header = CITATION_HEADER.format(
            source_id=source_id,
            title=getattr(result, "title", "Untitled"),
        )
        
        metadata = CITATION_METADATA.format(
            document=getattr(result, "document", "Unknown"),
            section=getattr(result, "section", "N/A"),
            score=getattr(result, "score", 0.0),
        )

        content = getattr(result, "content", getattr(result, "text", ""))
        
        return f"{header}\n{metadata}\n\n{content}"

    def _extract_source_ref(self, source_id: int, result: SearchResult) -> dict:
        """Extract source reference metadata from search result.
        
        Args:
            source_id: Sequential source identifier.
            result: Search result to extract from.
            
        Returns:
            Dictionary with source reference fields.
        """
        return {
            "id": source_id,
            "title": getattr(result, "title", "Untitled"),
            "document": getattr(result, "document", "Unknown"),
            "section": getattr(result, "section", "N/A"),
            "score": round(getattr(result, "score", 0.0), 4),
            "chunk_id": getattr(result, "chunk_id", None),
            "metadata": getattr(result, "metadata", {}),
        }

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate the token count of *text*.

        Delegates to :mod:`src.core.tokenization` so the context budget is
        measured the same way as chunking.  The previous ``len(text) // 4``
        heuristic under-counted Chinese by roughly 4x, which let a
        4000-token budget silently accept ~16000 tokens of CJK text.
        """
        return estimate_tokens(text)

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate *text* so it fits within *max_tokens*.

        The character budget is located by binary search over the estimated
        token count — a fixed ``max_tokens * 4`` character budget only holds
        for English.  The cut is then pulled back to the last sentence
        boundary in the retained half, so Chinese text (whose sentences end
        in ``。！？``) is no longer hard-sliced mid-sentence.

        Args:
            text: Text to truncate.
            max_tokens: Maximum tokens allowed.

        Returns:
            Truncated text with an ellipsis marker if truncation occurred.
        """
        if estimate_tokens(text) <= max_tokens:
            return text

        # Largest character prefix that still fits the token budget.
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if estimate_tokens(text[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1

        if low <= 0:
            return "[...]"

        truncated = text[:low]

        # Prefer to end on a sentence boundary, keeping at least half.
        for punct in ("\n\n", "。", "！", "？", "；", ". ", "! ", "? "):
            pos = truncated.rfind(punct)
            if pos > low * 0.5:
                return truncated[: pos + len(punct)].rstrip() + " [...]"

        return truncated.rstrip() + " [...]"
