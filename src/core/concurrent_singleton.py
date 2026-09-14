"""Thread-safe singleton manager with concurrency control.

Provides:
- Double-checked locking for initialization
- Semaphore-based concurrency control for model inference
- Connection pooling for Milvus/Redis
- Request queuing with backpressure
"""

import asyncio
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelSemaphore:
    """Semaphore-based concurrency limiter for model inference.
    
    Limits concurrent model predictions to prevent resource exhaustion.
    """
    
    def __init__(self, max_concurrent: int = 4):
        """Initialize semaphore.
        
        Args:
            max_concurrent: Maximum concurrent model inferences.
        """
        self._semaphore = threading.Semaphore(max_concurrent)
        self._waiting = 0
        self._active = 0
        self._lock = threading.Lock()
    
    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire semaphore for model inference.
        
        Args:
            timeout: Maximum wait time in seconds.
            
        Returns:
            True if acquired, False if timeout.
        """
        with self._lock:
            self._waiting += 1
        
        try:
            result = self._semaphore.acquire(timeout=timeout)
            with self._lock:
                self._waiting -= 1
                if result:
                    self._active += 1
            return result
        except Exception:
            with self._lock:
                self._waiting -= 1
            raise
    
    def release(self) -> None:
        """Release semaphore after model inference."""
        with self._lock:
            self._active -= 1
        self._semaphore.release()
    
    @property
    def stats(self) -> dict:
        """Get semaphore statistics."""
        with self._lock:
            return {
                "active": self._active,
                "waiting": self._waiting,
                "max_concurrent": self._semaphore._value + self._active,
            }


class ThreadSafeReranker:
    """Thread-safe reranker with concurrency control.
    
    Wraps Cross-Encoder model with:
    - Thread-safe predict using semaphore
    - Result caching with TTL
    - Graceful fallback
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        max_concurrent: int = 4,
        cache_maxsize: int = 256,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self._model_loaded = False
        self._load_lock = threading.Lock()
        
        # Concurrency control
        self._semaphore = ModelSemaphore(max_concurrent)
        
        # Cache
        self._cache_maxsize = cache_maxsize
        self._cache: dict[str, tuple[float, list]] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 300  # 5 minutes
        
        # Statistics
        self._total_requests = 0
        self._cache_hits = 0
        self._inference_count = 0
        self._stats_lock = threading.Lock()
    
    def _ensure_model_loaded(self) -> bool:
        """Double-checked locking for model loading."""
        if self._model_loaded:
            return self.model is not None
        
        with self._load_lock:
            if self._model_loaded:
                return self.model is not None
            
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name, max_length=self.max_length)
                self._model_loaded = True
                logger.info(f"Loaded model: {self.model_name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                return False
    
    def _get_cache_key(self, query: str, doc_ids: list[str]) -> str:
        """Generate cache key."""
        import hashlib
        content = f"{query}:{','.join(sorted(doc_ids))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[list]:
        """Get result from cache if valid."""
        with self._cache_lock:
            if key in self._cache:
                timestamp, result = self._cache[key]
                if time.time() - timestamp < self._cache_ttl:
                    return result
                del self._cache[key]
        return None
    
    def _set_cache(self, key: str, result: list) -> None:
        """Set result in cache with eviction."""
        with self._cache_lock:
            # Evict oldest if full
            if len(self._cache) >= self._cache_maxsize:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), result)
    
    def predict(
        self,
        pairs: list[tuple[str, str]],
        timeout: float = 30.0,
    ) -> list[float] | None:
        """Thread-safe prediction with concurrency control.

        Args:
            pairs: List of (query, document) pairs.
            timeout: Maximum wait time for semaphore.

        Returns:
            Relevance scores, or ``None`` when the model is unavailable or
            the semaphore times out.  Callers must treat ``None`` as "keep
            the original ranking" — an earlier version returned
            ``[0.0] * len(pairs)`` here, which the log line described as
            falling back to the original scores while actually scrambling
            them.
        """
        with self._stats_lock:
            self._total_requests += 1
        
        # Check cache
        cache_key = self._get_cache_key(
            pairs[0][0] if pairs else "",
            [p[1][:50] for p in pairs]
        )
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            with self._stats_lock:
                self._cache_hits += 1
            return cached
        
        # Fallback if model not available
        if not self._ensure_model_loaded():
            return None
        
        # Acquire semaphore for concurrency control
        if not self._semaphore.acquire(timeout=timeout):
            logger.warning("Semaphore timeout, falling back to original scores")
            return None
        
        try:
            with self._stats_lock:
                self._inference_count += 1
            
            import numpy as np
            scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
            normalized = (1.0 / (1.0 + np.exp(-np.array(scores)))).tolist()
            
            # Cache result
            self._set_cache(cache_key, normalized)
            
            return normalized
        finally:
            self._semaphore.release()
    
    @property
    def stats(self) -> dict:
        """Get reranker statistics."""
        with self._stats_lock:
            hit_rate = self._cache_hits / max(self._total_requests, 1)
            return {
                "model": self.model_name,
                "loaded": self._model_loaded,
                "total_requests": self._total_requests,
                "cache_hits": self._cache_hits,
                "hit_rate": f"{hit_rate:.2%}",
                "inference_count": self._inference_count,
                "semaphore": self._semaphore.stats,
                "cache_size": len(self._cache),
            }


class ConnectionPool:
    """Generic connection pool for Milvus/Redis.
    
    Provides thread-safe connection management with:
    - Pool of reusable connections
    - Health checking
    - Automatic reconnection
    """
    
    def __init__(
        self,
        factory: callable,
        min_size: int = 2,
        max_size: int = 10,
        health_check: callable = None,
    ):
        """Initialize connection pool.
        
        Args:
            factory: Callable that creates a new connection.
            min_size: Minimum pool size.
            max_size: Maximum pool size.
            health_check: Callable to check connection health.
        """
        self._factory = factory
        self._min_size = min_size
        self._max_size = max_size
        self._health_check = health_check
        
        self._pool: deque = deque()
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_size)
        self._created = 0
        self._active = 0
        
        # Initialize minimum connections
        for _ in range(min_size):
            conn = factory()
            self._pool.append(conn)
            self._created += 1
    
    def acquire(self, timeout: float = 30.0) -> Any:
        """Acquire a connection from the pool.
        
        Args:
            timeout: Maximum wait time.
            
        Returns:
            Connection object.
        """
        if not self._semaphore.acquire(timeout=timeout):
            raise TimeoutError("Connection pool exhausted")
        
        with self._lock:
            if self._pool:
                conn = self._pool.popleft()
                self._active += 1
                return conn
            # Create new connection if under limit
            if self._created < self._max_size:
                conn = self._factory()
                self._created += 1
                self._active += 1
                return conn
        
        # Unreachable in practice: the semaphore guarantees at most
        # ``max_size`` checkouts at a time.  Release the permit before
        # raising so a logic error cannot leak a slot permanently.
        self._semaphore.release()
        raise RuntimeError("Connection pool error")
    
    def release(self, conn: Any) -> None:
        """Release a connection back to the pool.
        
        Args:
            conn: Connection to release.
        """
        # Check health before returning to pool
        if self._health_check and not self._health_check(conn):
            with self._lock:
                self._created -= 1
                self._active = max(0, self._active - 1)
            self._semaphore.release()
            return
        
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(conn)
            else:
                self._created -= 1
            self._active = max(0, self._active - 1)
        self._semaphore.release()
    
    @property
    def stats(self) -> dict:
        """Get pool statistics.

        ``available`` is the number of idle connections held in the
        pool, ``active`` is the number currently checked out, and
        ``created`` is the number of live connections.  ``pool_size``
        is retained as an alias of ``available`` so existing metric
        readers keep working.
        """
        with self._lock:
            available = len(self._pool)
            return {
                "available": available,
                "pool_size": available,
                "active": self._active,
                "created": self._created,
                "max_size": self._max_size,
            }


class SingletonManager:
    """Thread-safe singleton manager with concurrency control.
    
    Features:
    - Double-checked locking for initialization
    - Connection pooling for external services
    - Concurrency limiting for model inference
    - Statistics and monitoring
    """
    
    _lock = threading.Lock()
    _instances: dict[str, Any] = {}
    _pools: dict[str, ConnectionPool] = {}
    
    @classmethod
    def get_reranker(
        cls,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_concurrent: int = 4,
    ) -> ThreadSafeReranker:
        """Get or create singleton ThreadSafeReranker."""
        key = f"reranker:{model_name}"
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    cls._instances[key] = ThreadSafeReranker(
                        model_name=model_name,
                        max_concurrent=max_concurrent,
                    )
                    logger.info(f"Created singleton ThreadSafeReranker: {model_name}")
        
        return cls._instances[key]
    
    @classmethod
    def get_milvus_pool(
        cls,
        collection_name: str = "enterprise_rag",
        min_size: int = 2,
        max_size: int = 5,
    ) -> ConnectionPool:
        """Get or create Milvus connection pool."""
        key = f"milvus_pool:{collection_name}"
        
        if key not in cls._pools:
            with cls._lock:
                if key not in cls._pools:
                    from pymilvus import connections, Collection
                    
                    def factory():
                        from src.config.settings import get_settings

                        settings = get_settings()
                        # pymilvus registers connections process-wide by
                        # alias, and Collection() attaches to "default".  This
                        # used to connect under f"pool_{thread_id}", so the
                        # pooled connection was never the one retrieval used.
                        connections.connect(
                            alias="default",
                            host=settings.MILVUS_HOST,
                            port=str(settings.MILVUS_PORT),
                        )
                        collection = Collection(collection_name)
                        collection.load()
                        return collection

                    def health_check(conn):
                        try:
                            # Cheap probe. num_entities runs a statistic on the
                            # collection and is too heavy for a release path.
                            return connections.has_connection("default")
                        except Exception:
                            return False
                    
                    cls._pools[key] = ConnectionPool(
                        factory=factory,
                        min_size=min_size,
                        max_size=max_size,
                        health_check=health_check,
                    )
                    logger.info(f"Created Milvus pool: {collection_name}")
        
        return cls._pools[key]
    
    @classmethod
    def get_redis_pool(
        cls,
        min_size: int = 5,
        max_size: int = 20,
    ) -> ConnectionPool:
        """Get or create Redis connection pool."""
        key = "redis_pool"
        
        if key not in cls._pools:
            with cls._lock:
                if key not in cls._pools:
                    import redis
                    from urllib.parse import urlparse
                    from src.config.settings import get_settings
                    
                    settings = get_settings()
                    parsed = urlparse(settings.REDIS_URL)
                    
                    def factory():
                        return redis.Redis(
                            host=parsed.hostname,
                            port=parsed.port,
                            db=int(parsed.path.lstrip("/") or "0"),
                            password=settings.REDIS_PASSWORD,
                            decode_responses=True,
                        )
                    
                    def health_check(conn):
                        try:
                            conn.ping()
                            return True
                        except Exception:
                            return False
                    
                    cls._pools[key] = ConnectionPool(
                        factory=factory,
                        min_size=min_size,
                        max_size=max_size,
                        health_check=health_check,
                    )
                    logger.info("Created Redis pool")
        
        return cls._pools[key]
    
    @classmethod
    def get_stats(cls) -> dict:
        """Get comprehensive statistics."""
        stats = {
            "instances": list(cls._instances.keys()),
            "pools": {},
        }
        
        for key, pool in cls._pools.items():
            stats["pools"][key] = pool.stats
        
        # Get reranker stats
        for key, instance in cls._instances.items():
            if hasattr(instance, 'stats'):
                stats[key] = instance.stats
        
        return stats
    
    @classmethod
    def clear(cls, pattern: Optional[str] = None) -> int:
        """Clear all singletons and pools."""
        with cls._lock:
            count = 0
            
            if pattern:
                keys = [k for k in cls._instances if pattern in k]
                pool_keys = [k for k in cls._pools if pattern in k]
            else:
                keys = list(cls._instances.keys())
                pool_keys = list(cls._pools.keys())
            
            for key in keys:
                del cls._instances[key]
                count += 1
            
            for key in pool_keys:
                del cls._pools[key]
                count += 1
            
            return count


# Convenience functions
def get_thread_safe_reranker(**kwargs) -> ThreadSafeReranker:
    """Get thread-safe reranker singleton."""
    return SingletonManager.get_reranker(**kwargs)

def get_milvus_pool(**kwargs) -> ConnectionPool:
    """Get Milvus connection pool."""
    return SingletonManager.get_milvus_pool(**kwargs)

def get_redis_pool(**kwargs) -> ConnectionPool:
    """Get Redis connection pool."""
    return SingletonManager.get_redis_pool(**kwargs)

