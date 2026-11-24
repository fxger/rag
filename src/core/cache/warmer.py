"""
Cache warming utilities.

Provides batch and scheduled cache warming to prevent cold-start latency.
"""

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Callable
from typing import Any

from src.config.redis import get_cache_redis

logger = logging.getLogger(__name__)

QUERY_LOG_KEY = "rag:query_log"  # 与 RAGService._log_query 写入的键保持一致


class CacheWarmer:
    """Proactively warms cache with frequently accessed data."""

    def __init__(
        self,
        prefix: str = "rag:warm:",
        default_ttl: int = 600,
        cache: Any | None = None,
    ):
        self.prefix = prefix
        self.default_ttl = default_ttl
        # 注入调用方的 MultiLevelCache：预热写入的键必须落在与读取方同一个
        # keyspace（同一实例、同一 prefix），否则预热白做。
        self.cache = cache
        self._running = False

    def _resolve_cache(self) -> Any:
        if self.cache is not None:
            return self.cache
        from src.core.cache.multi_level_cache import MultiLevelCache

        return MultiLevelCache(prefix="rag:data:")

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
        cache = self._resolve_cache()
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
                    cache = self._resolve_cache()
                    for key, value in data.items():
                        await cache.set(key, value, ttl=self.default_ttl)
                    logger.debug("Scheduled warm-up refreshed %d entries", len(data))
            except Exception as e:
                logger.error("Scheduled warm-up error: %s", e)

            await asyncio.sleep(interval)

    def stop(self) -> None:
        """Stop scheduled warming."""
        self._running = False


async def top_queries_from_log(limit: int = 10) -> list[str]:
    """Read the query log written by ``RAGService._log_query`` and return the
    most frequent queries, hottest first.

    ``rag:query_log`` is a Redis list, most-recent-first, capped at 1000
    entries.  Fresh deploy (empty log) or Redis down → empty list, which the
    callers treat as "nothing to warm", not an error.
    """
    try:
        redis = await get_cache_redis()
        raw = await redis.lrange(QUERY_LOG_KEY, 0, 999)
    except Exception as e:
        logger.warning("top_queries_from_log failed: %s", e)
        return []

    counts: Counter[str] = Counter()
    for item in raw:
        try:
            entry = json.loads(item)
            query = (entry.get("query") or "").strip()
            if query:
                counts[query] += 1
        except Exception as e:
            # 单条日志损坏只跳过该条，不影响整体预热
            logger.debug("Skipping malformed query log entry: %s", e)
            continue

    return [q for q, _ in counts.most_common(limit)]
