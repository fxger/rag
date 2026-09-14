"""Context assembly from retrieval results with source citation support.

Builds structured context strings for LLM consumption, respecting token budgets
and including proper source attribution for grounded generation.
"""
import logging
from typing import Optional

from src.core.retrieval.models import SearchResult

logger = logging.getLogger(__name__)

# Approximate characters per token (conservative estimate for English text)
CHARS_PER_TOKEN = 4

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
            block_tokens = self._estimate_tokens(source_block)
            
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
            block_tokens = self._estimate_tokens(source_block)

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
        """Estimate token count using character-based heuristic.
        
        Args:
            text: Text to estimate tokens for.
            
        Returns:
            Approximate token count.
        """
        return len(text) // CHARS_PER_TOKEN

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit.
        
        Preserves complete sentences where possible by truncating at
        sentence boundaries within the token budget.
        
        Args:
            text: Text to truncate.
            max_tokens: Maximum tokens allowed.
            
        Returns:
            Truncated text with ellipsis indicator if truncated.
        """
        max_chars = max_tokens * CHARS_PER_TOKEN
        
        if len(text) <= max_chars:
            return text

        # Try to truncate at sentence boundary
        truncated = text[:max_chars]
        
        # Find last sentence-ending punctuation
        for punct in [". ", "! ", "? ", "\n\n"]:
            last_punct = truncated.rfind(punct)
            if last_punct > max_chars * 0.5:  # Keep at least 50% of content
                return truncated[: last_punct + len(punct)].rstrip() + " [...]"

        # Fallback: truncate at word boundary
        last_space = truncated.rfind(" ")
        if last_space > 0:
            return truncated[:last_space].rstrip() + " [...]"

        return truncated.rstrip() + " [...]"
