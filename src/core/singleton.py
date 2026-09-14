"""Singleton manager for model instances and heavy resources.

Provides thread-safe singleton pattern for:
- Cross-Encoder reranker model
- Milvus collection connections
- Redis connections
- Query analyzer (stateless, but expensive to create)

Usage:
    reranker = SingletonManager.get_reranker()
    vector_retriever = SingletonManager.get_vector_retriever()
"""

import threading
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SingletonManager:
    """Thread-safe singleton manager for heavy resources.
    
    All instances are lazily initialized on first access and
    shared across the application.
    """
    
    _lock = threading.Lock()
    _instances: dict[str, Any] = {}
    
    @classmethod
    def get_reranker(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Any:
        """Get or create singleton Reranker instance.
        
        Args:
            model_name: Cross-encoder model name.
            
        Returns:
            Singleton Reranker instance.
        """
        key = f"reranker:{model_name}"
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    from src.core.retrieval.reranker import Reranker
                    cls._instances[key] = Reranker(model_name=model_name)
                    logger.info(f"Created singleton Reranker: {model_name}")
        
        return cls._instances[key]
    
    @classmethod
    def get_vector_retriever(cls, collection_name: str = "enterprise_rag") -> Any:
        """Get or create singleton VectorRetriever instance.
        
        Args:
            collection_name: Milvus collection name.
            
        Returns:
            Singleton VectorRetriever instance.
        """
        key = f"vector_retriever:{collection_name}"
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    from src.core.retrieval.vector_retriever import VectorRetriever
                    cls._instances[key] = VectorRetriever(collection_name=collection_name)
                    logger.info(f"Created singleton VectorRetriever: {collection_name}")
        
        return cls._instances[key]
    
    @classmethod
    def get_sparse_retriever(cls) -> Any:
        """Get or create singleton SparseRetriever instance.
        
        Returns:
            Singleton SparseRetriever instance.
        """
        key = "sparse_retriever"
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    from src.core.retrieval.sparse_retriever import SparseRetriever
                    cls._instances[key] = SparseRetriever()
                    logger.info("Created singleton SparseRetriever")
        
        return cls._instances[key]
    
    @classmethod
    def get_query_analyzer(cls) -> Any:
        """Get or create singleton QueryAnalyzer instance.
        
        Returns:
            Singleton QueryAnalyzer instance.
        """
        key = "query_analyzer"
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    from src.core.retrieval.query_analyzer import QueryAnalyzer
                    cls._instances[key] = QueryAnalyzer()
                    logger.info("Created singleton QueryAnalyzer")
        
        return cls._instances[key]
    
    @classmethod
    def get_hybrid_retriever(
        cls,
        collection_name: str = "enterprise_rag",
        enable_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fusion_method: str = "adaptive",
    ) -> Any:
        """Get or create singleton HybridRetriever instance.
        
        Args:
            collection_name: Milvus collection name.
            enable_reranker: Whether to enable reranking.
            reranker_model: Cross-encoder model name.
            fusion_method: Fusion method (adaptive/rrf/weighted).
            
        Returns:
            Singleton HybridRetriever instance.
        """
        key = f"hybrid_retriever:{collection_name}"
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    from src.core.retrieval.hybrid_retriever import HybridRetriever
                    cls._instances[key] = HybridRetriever(
                        collection_name=collection_name,
                        enable_reranker=enable_reranker,
                        reranker_model=reranker_model,
                        fusion_method=fusion_method,
                    )
                    logger.info(f"Created singleton HybridRetriever: {collection_name}")
        
        return cls._instances[key]
    
    @classmethod
    def get_embedder(cls) -> Any:
        """Get or create singleton Embedder instance.
        
        Returns:
            Singleton Embedder instance.
        """
        key = "embedder"
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    from src.core.embedding.embedder import QwenEmbedder
                    cls._instances[key] = QwenEmbedder()
                    logger.info("Created singleton Embedder")
        
        return cls._instances[key]
    
    @classmethod
    def clear(cls, pattern: Optional[str] = None) -> int:
        """Clear singleton instances.
        
        Args:
            pattern: Optional pattern to match keys. If None, clear all.
            
        Returns:
            Number of instances cleared.
        """
        with cls._lock:
            if pattern:
                keys_to_remove = [k for k in cls._instances if pattern in k]
            else:
                keys_to_remove = list(cls._instances.keys())
            
            for key in keys_to_remove:
                instance = cls._instances.pop(key)
                # Try to close/cleanup if method exists
                if hasattr(instance, 'close'):
                    try:
                        instance.close()
                    except Exception:
                        pass
                logger.info(f"Cleared singleton: {key}")
            
            return len(keys_to_remove)
    
    @classmethod
    def get_stats(cls) -> dict:
        """Get statistics about singleton instances.
        
        Returns:
            Dictionary with instance counts and keys.
        """
        return {
            "count": len(cls._instances),
            "instances": list(cls._instances.keys()),
        }


# Convenience functions
def get_reranker(**kwargs) -> Any:
    """Get singleton reranker."""
    return SingletonManager.get_reranker(**kwargs)

def get_vector_retriever(**kwargs) -> Any:
    """Get singleton vector retriever."""
    return SingletonManager.get_vector_retriever(**kwargs)

def get_sparse_retriever() -> Any:
    """Get singleton sparse retriever."""
    return SingletonManager.get_sparse_retriever()

def get_query_analyzer() -> Any:
    """Get singleton query analyzer."""
    return SingletonManager.get_query_analyzer()

def get_hybrid_retriever(**kwargs) -> Any:
    """Get singleton hybrid retriever."""
    return SingletonManager.get_hybrid_retriever(**kwargs)

def get_embedder() -> Any:
    """Get singleton embedder."""
    return SingletonManager.get_embedder()
