"""
Cache anti-avalanche strategies.

Prevents cache stampedes and thundering herds via:
- Random TTL offsets to prevent simultaneous expiration
- Mutex locks for single-flight cache rebuilding
"""

import asyncio
import logging
import random
from typing import Any, Callable, Optional

from src.config.redis import get_cache_redis

logger = logging.getLogger(__name__)


class CacheAntiAvalanche:
    """Protects against cache stampedes and thundering herd problems."""

    def __init__(self, prefix: str = "rag:lock:", lock_ttl: int = 30):
        self.prefix = prefix
        self.lock_ttl = lock_ttl

    def get_random_ttl(self, base_ttl: int, jitter_pct: float = 0.1) -> int:
        """
        Add random jitter to TTL to prevent synchronized expiration.

        Args:
            base_ttl: Base time-to-live in seconds.
            jitter_pct: Jitter as fraction of base (default 10%).

        Returns:
            Randomized TTL.
        """
        jitter = int(base_ttl * jitter_pct)
        return base_ttl + random.randint(-jitter, jitter)

    async def get_with_lock(
        self,
        key: str,
        fetch_fn: Callable[..., Any],
        ttl: int = 300,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Get value with single-flight protection.

        If the key is missing, attempts to acquire a Redis lock.
        Only the lock holder calls fetch_fn; others wait and read the cache.

        Args:
            key: Cache key.
            fetch_fn: Async callable to generate the value on cache miss.
            ttl: Cache TTL in seconds.
        """
        from src.core.cache.multi_level_cache import MultiLevelCache

        cache = MultiLevelCache(prefix="rag:data:")

        # Try cache first
        value = await cache.get(key)
        if value is not None:
            return value

        # Try to acquire lock
        lock_key = f"{self.prefix}{key}"
        lock_acquired = False

        try:
            redis = await get_cache_redis()
            lock_acquired = await redis.set(lock_key, "1", nx=True, ex=self.lock_ttl)

            if lock_acquired:
                # We are the builder — fetch and cache
                logger.debug("Lock acquired, building cache for: %s", key)
                value = await fetch_fn(*args, **kwargs)
                randomized_ttl = self.get_random_ttl(ttl)
                await cache.set(key, value, ttl=randomized_ttl)
                return value
            else:
                # Someone else is building — poll until ready
                logger.debug("Waiting for cache build: %s", key)
                for _ in range(self.lock_ttl * 2):
                    await asyncio.sleep(0.5)
                    value = await cache.get(key)
                    if value is not None:
                        return value
                # Timeout — fallback to direct fetch
                logger.warning("Lock wait timeout for %s, fetching directly", key)
                return await fetch_fn(*args, **kwargs)

        finally:
            if lock_acquired:
                try:
                    redis = await get_cache_redis()
                    await redis.delete(lock_key)
                except Exception:
                    pass