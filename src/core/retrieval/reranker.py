"""Cross-encoder reranking with graceful fallback."""
import asyncio
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
# Guards _model_cache / _model_load_attempted. Model loading is a
# process-wide concern, so a single module-level lock serialises it
# across every Reranker instance.
_model_cache_lock = threading.Lock()


class Reranker:
    """Cross-encoder reranker with caching and graceful fallback.

    Uses a cross-encoder model to rerank retrieval results. Includes:
    - Lazy model loading with global cache
    - Result caching (LRU)
    - Batch inference
    - Graceful fallback to original scores

    Concurrency:
    - Model loading is serialised by a module-level lock so the model is
      constructed exactly once, even under concurrent first requests.
    - ``_semaphore`` caps concurrent cross-encoder inference; CPU-bound
      work is dispatched to a worker thread, never run on the event loop.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32,
        cache_enabled: bool = True,
        cache_maxsize: int = 256,
        max_concurrent: int = 4,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.model = None
        self._model_loaded = False
        self._load_lock = threading.Lock()

        # asyncio.Semaphore, not threading.Semaphore: acquiring a blocking
        # semaphore inside a coroutine freezes the entire event loop, so a
        # saturated reranker would stall every other request too.
        self._semaphore = asyncio.Semaphore(max_concurrent)

        self.max_concurrent = max_concurrent
        # Counters exposed through ``stats`` for Prometheus scraping.  The
        # reranker is a process-wide singleton and every mutation happens on
        # the event loop, so plain ints are sufficient.
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._inference_count = 0
        self._active = 0

        if cache_enabled:
            from cachetools import LRUCache
            self._cache = LRUCache(maxsize=cache_maxsize)
        else:
            self._cache = None

    def _ensure_model_loaded(self) -> bool:
        """Lazy-load the cross-encoder model with global caching.

        The whole load path runs under ``_model_cache_lock``, and
        ``_model_loaded`` is only set once ``self.model`` is genuinely
        ready.  Setting the flag before the model exists (the previous
        behaviour) let a concurrent caller observe "loaded" while
        ``self.model`` was still ``None``, silently skipping reranking
        with no log line to explain it.
        """
        # Fast path: only trust the flag when the model is actually present.
        if self._model_loaded and self.model is not None:
            return True

        with _model_cache_lock:
            # Re-check under the lock, again requiring a real model.
            if self._model_loaded and self.model is not None:
                return True

            if self.model_name in _model_cache:
                self.model = _model_cache[self.model_name]
                self._model_loaded = True
                logger.info("Using cached cross-encoder model: %s", self.model_name)
                return True

            # A previous attempt already failed in this process — don't retry.
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

                # Cache globally so other instances / callers reuse it.
                _model_cache[self.model_name] = self.model
                self._model_loaded = True
                logger.info(
                    "Loaded cross-encoder model: %s (%.2fs)", self.model_name, elapsed
                )
                return True

            except ImportError:
                logger.warning("sentence-transformers not installed")
                return False
            except Exception as e:
                logger.warning("Failed to load cross-encoder model: %s", e)
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

        self._total_requests += 1
        
        # Check cache first
        doc_ids = [doc.chunk_id for doc in documents]
        if self.cache_enabled and self._cache is not None:
            cache_key = self._get_cache_key(query, doc_ids)
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                self._cache_hits += 1
                logger.debug("Rerank cache hit")
                return cached_result[:top_n] if top_n else cached_result
            self._cache_misses += 1
        
        # Load the model off the event loop: the first load reads weights
        # from disk and can take seconds, which would freeze every other
        # request if executed inline.
        model_ready = await asyncio.to_thread(self._ensure_model_loaded)
        if model_ready and self.model is not None:
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
        timeout: float = 30.0,
    ) -> list[SearchResult]:
        """Rerank using the cross-encoder model.

        The semaphore is acquired asynchronously and the CPU-bound
        inference runs in a worker thread, so a saturated reranker
        degrades this one request instead of freezing the event loop.
        """
        start_time = time.time()

        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Semaphore timeout during reranking, falling back")
            return self._fallback_rerank(documents, top_n)

        self._inference_count += 1
        self._active += 1
        try:
            pairs = [(query, doc.content[:self.max_length]) for doc in documents]

            scores = await asyncio.to_thread(
                self.model.predict,
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        finally:
            self._active -= 1
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

    @property
    def stats(self) -> dict:
        """Runtime statistics consumed by the Prometheus collector.

        ``src.api.metrics.update_concurrent_metrics`` reads these keys —
        treat the names as a contract and keep both sides in sync.
        """
        return {
            "model_name": self.model_name,
            "model_loaded": self._model_loaded,
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._cache) if self._cache is not None else 0,
            "inference_count": self._inference_count,
            "active": self._active,
            "waiting": 0,
            "max_concurrent": self.max_concurrent,
        }

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
