"""Core module for Enterprise RAG System.

Provides singleton management for heavy resources like models and connections.
"""

from .singleton import (
    SingletonManager,
    get_reranker,
    get_vector_retriever,
    get_sparse_retriever,
    get_query_analyzer,
    get_hybrid_retriever,
    get_embedder,
)

__all__ = [
    "SingletonManager",
    "get_reranker",
    "get_vector_retriever",
    "get_sparse_retriever",
    "get_query_analyzer",
    "get_hybrid_retriever",
    "get_embedder",
]
