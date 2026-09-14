"""Data models for retrieval results and query analysis."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """A single retrieval result.
    
    Attributes:
        chunk_id: Unique identifier for the document chunk.
        document_id: Parent document identifier.
        content: Text content of the chunk.
        score: Relevance score (higher is better).
        rank: Position in the result list.
        metadata: Additional metadata about the chunk.
        source_type: Origin of the result (vector, sparse, or metadata).
    """
    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    source_type: str = "vector"  # vector | sparse | metadata


@dataclass
class QueryAnalysis:
    """Analysis of a user query.
    
    Attributes:
        original_query: The original user query string.
        query_type: Classified type (factual, conceptual, comparative, navigational).
        keywords: Extracted keywords from the query.
        entities: Extracted named entities with types.
        features: Query features for dynamic weight calculation.
        expanded_queries: Generated query variations for better retrieval.
    """
    original_query: str
    query_type: str  # factual | conceptual | comparative | navigational
    keywords: list[str]
    entities: list[dict[str, str]]
    features: dict[str, Any] = field(default_factory=dict)
    expanded_queries: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Complete retrieval result with metadata.
    
    Attributes:
        results: List of search results.
        query_analysis: Analysis of the original query.
        weights: Dynamic weights used for fusion.
        total_candidates: Total number of candidates evaluated.
        retrieval_time_ms: Time taken for retrieval in milliseconds.
        reranked: Whether reranking was applied.
    """
    results: list[SearchResult]
    query_analysis: QueryAnalysis
    weights: dict[str, float] = field(default_factory=lambda: {"dense": 0.6, "sparse": 0.4})
    total_candidates: int = 0
    retrieval_time_ms: int = 0
    reranked: bool = False
