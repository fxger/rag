"""Cross-encoder reranking with graceful fallback."""
import hashlib
import threading
import logging
import time
from typing import Any

from .models import SearchResult

logger = logging.getLogger(__name__)

# Global model cache to avoid repeated loading attempts
_model_cache = {}
_model_load_attempted = set()


class Reranker:
    _load_lock = threading.Lock()
    _semaphore = threading.Semaphore(4)
    """Cross-encoder reranker with caching and graceful fallback.
    
    Uses a cross-encoder model to rerank retrieval results. Includes:
    - Lazy model loading with global cache
    - Result caching (LRU)
    - Batch inference
    - Graceful fallback to original scores
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32,
        cache_enabled: bool = True,
        cache_maxsize: int = 256,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.model = None
        self._model_loaded = False
        self._load_lock = threading.Lock()
        self._semaphore = threading.Semaphore(4)
        
        if cache_enabled:
            from cachetools import LRUCache
            self._cache = LRUCache(maxsize=cache_maxsize)
        else:
            self._cache = None

    def _ensure_model_loaded(self) -> bool:
        """Lazy-load the cross-encoder model with global caching."""
        if self._model_loaded:
            return self.model is not None
        
        with self._load_lock:
            if self._model_loaded:
                return self.model is not None
            self._model_loaded = True
        
        # Check global cache
        if self.model_name in _model_cache:
            self.model = _model_cache[self.model_name]
            logger.info(f"Using cached cross-encoder model: {self.model_name}")
            return True
        
        # Skip if previous load attempt failed
        if self.model_name in _model_load_attempted:
            return False
        
        _model_load_attempted.add(self.model_name)
        
        try:
            from sentence_transformers import CrossEncoder
            
            start_time = time.time()
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )
            elapsed = time.time() - start_time
            
            # Cache globally
            _model_cache[self.model_name] = self.model
            logger.info(f"Loaded cross-encoder model: {self.model_name} ({elapsed:.2f}s)")
            return True
            
        except ImportError:
            logger.warning("sentence-transformers not installed")
            return False
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder model: {e}")
            self.model = None
            return False

    def _get_cache_key(self, query: str, doc_ids: list[str]) -> str:
        """Generate cache key."""
        content = f"{query}:{','.join(sorted(doc_ids))}"
        return hashlib.md5(content.encode()).hexdigest()

    async def rerank(
        self,
        query: str,
        documents: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """Rerank documents based on query-document relevance."""
        if not documents:
            return []
        
        # Check cache first
        doc_ids = [doc.chunk_id for doc in documents]
        if self.cache_enabled and self._cache is not None:
            cache_key = self._get_cache_key(query, doc_ids)
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Rerank cache hit")
                return cached_result[:top_n] if top_n else cached_result
        
        # Try to use model
        if self._ensure_model_loaded() and self.model is not None:
            try:
                result = await self._rerank_with_model(query, documents, top_n)
                # Cache result
                if self.cache_enabled and self._cache is not None:
                    self._cache[cache_key] = result
                return result
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using fallback")
        
        # Fallback to original scores
        return self._fallback_rerank(documents, top_n)

    async def _rerank_with_model(
        self,
        query: str,
        documents: list[SearchResult],
        top_n: int | None,
    ) -> list[SearchResult]:
        """Rerank using the cross-encoder model."""
        start_time = time.time()
        
        # Acquire semaphore for concurrency control
        if not self._semaphore.acquire(timeout=30.0):
            logger.warning("Semaphore timeout during reranking")
            return self._fallback_rerank(documents, top_n)
        
        try:
            pairs = [(query, doc.content[:self.max_length]) for doc in documents]
            
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        finally:
            self._semaphore.release()
        
        import numpy as np
        normalized_scores = 1.0 / (1.0 + np.exp(-np.array(scores)))
        
        reranked_results = []
        for doc, new_score in zip(documents, normalized_scores):
            metadata = doc.metadata.copy()
            metadata["original_score"] = doc.score
            metadata["rerank_score"] = float(new_score)
            metadata["reranked"] = True
            
            reranked_results.append(SearchResult(
                chunk_id=doc.chunk_id,
                document_id=doc.document_id,
                content=doc.content,
                score=float(new_score),
                rank=0,
                metadata=metadata,
                source_type=doc.source_type,
            ))
        
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(reranked_results, start=1):
            result.rank = rank
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Reranking: {len(documents)} docs in {elapsed_ms}ms")
        
        return reranked_results[:top_n] if top_n else reranked_results

    def _fallback_rerank(
        self, 
        documents: list[SearchResult], 
        top_n: int | None
    ) -> list[SearchResult]:
        """Fallback using original scores."""
        results = documents.copy()
        results.sort(key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(results, start=1):
            result.rank = rank
        return results[:top_n] if top_n else results

    def is_available(self) -> bool:
        """Check if reranker model is available."""
        return self._ensure_model_loaded()

    def clear_cache(self) -> None:
        """Clear the reranking cache."""
        if self._cache is not None:
            self._cache.clear()

    @property
    def cache_stats(self) -> dict:
        """Get cache statistics."""
        if self._cache is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "size": len(self._cache),
            "maxsize": self._cache.maxsize,
        }
