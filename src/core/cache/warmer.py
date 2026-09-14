"""
Cache warming utilities.

Provides batch and scheduled cache warming to prevent cold-start latency.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from src.config.redis import get_cache_redis

logger = logging.getLogger(__name__)


class CacheWarmer:
    """Proactively warms cache with frequently accessed data."""

    def __init__(self, prefix: str = "rag:warm:", default_ttl: int = 600):
        self.prefix = prefix
        self.default_ttl = default_ttl
        self._running = False

    async def warm_up(
        self,
        keys_and_fetchers: list[tuple[str, Callable[..., Any]]],
    ) -> int:
        """
        Batch-warm cache entries.

        Args:
            keys_and_fetchers: List of (cache_key, async_fetcher) tuples.

        Returns:
            Number of entries successfully warmed.
        """
        from src.core.cache.multi_level_cache import MultiLevelCache

        cache = MultiLevelCache(prefix="rag:data:")
        warmed = 0

        for key, fetcher in keys_and_fetchers:
            try:
                value = await fetcher()
                if value is not None:
                    await cache.set(key, value, ttl=self.default_ttl)
                    warmed += 1
            except Exception as e:
                logger.warning("Cache warm failed for key %s: %s", key, e)

        logger.info("Cache warmed: %d/%d entries", warmed, len(keys_and_fetchers))
        return warmed

    async def schedule_warm_up(
        self,
        pattern: str,
        fetch_fn: Callable[..., Any],
        interval: int = 300,
    ) -> None:
        """
        Periodically refresh cache entries matching a pattern.

        Runs until self._running is set to False.

        Args:
            pattern: Redis key pattern to match.
            fetch_fn: Async callable returning {key: value} dict.
            interval: Seconds between warm-up cycles.
        """
        self._running = True
        logger.info("Scheduled cache warm-up every %ds for pattern: %s", interval, pattern)

        while self._running:
            try:
                data = await fetch_fn()
                if data:
                    from src.core.cache.multi_level_cache import MultiLevelCache
                    cache = MultiLevelCache(prefix="rag:data:")
                    for key, value in data.items():
                        await cache.set(key, value, ttl=self.default_ttl)
                    logger.debug("Scheduled warm-up refreshed %d entries", len(data))
            except Exception as e:
                logger.error("Scheduled warm-up error: %s", e)

            await asyncio.sleep(interval)

    def stop(self) -> None:
        """Stop scheduled warming."""
        self._running = False