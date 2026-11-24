"""
Multi-level caching: L1 (in-process) + L2 (Redis).

L1 uses cachetools.TTLCache for sub-millisecond hot data access.
L2 uses Redis for distributed cross-instance caching.
"""

import json
import logging
from typing import Any

from cachetools import TTLCache

from src.config.redis import get_cache_redis

logger = logging.getLogger(__name__)


class MultiLevelCache:
    """
    Two-tier cache: L1 (process-local TTLCache) → L2 (Redis).
    Reads cascade L1 → L2; writes populate both levels.
    """

    def __init__(
        self,
        l1_maxsize: int = 1000,
        l1_ttl: int = 60,
        l2_ttl: int = 300,
        prefix: str = "rag:cache:",
    ):
        self.l1: TTLCache[str, Any] = TTLCache(maxsize=l1_maxsize, ttl=l1_ttl)
        self.l2_ttl = l2_ttl
        self.prefix = prefix
        self._hits = {"l1": 0, "l2": 0, "miss": 0}

    # ── Read ─────────────────────────────────────────────────────────

    async def get_with_source(self, key: str) -> tuple[str, Any | None]:
        """Retrieve value and report which tier served it: ``"l1" | "l2" | "miss"``."""
        # L1 hit
        if key in self.l1:
            self._hits["l1"] += 1
            logger.debug("L1 cache hit: %s", key)
            return "l1", self.l1[key]

        # L2 hit
        try:
            redis = await get_cache_redis()
            raw = await redis.get(f"{self.prefix}{key}")
            if raw is not None:
                value = json.loads(raw)
                self.l1[key] = value  # promote to L1
                self._hits["l2"] += 1
                logger.debug("L2 cache hit: %s", key)
                return "l2", value
        except Exception as e:
            logger.warning("L2 cache read error: %s", e)

        self._hits["miss"] += 1
        return "miss", None

    async def get(self, key: str) -> Any | None:
        """Retrieve value from L1, then L2 if missing."""
        _, value = await self.get_with_source(key)
        return value

    # ── Write ────────────────────────────────────────────────────────

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Write to both L1 and L2."""
        effective_ttl = ttl or self.l2_ttl

        # L1
        self.l1[key] = value

        # L2
        try:
            redis = await get_cache_redis()
            await redis.set(
                f"{self.prefix}{key}",
                json.dumps(value, default=str),
                ex=effective_ttl,
            )
        except Exception as e:
            logger.warning("L2 cache write error: %s", e)

    # ── Invalidation ─────────────────────────────────────────────────

    async def delete(self, key: str) -> None:
        """Remove from both L1 and L2."""
        self.l1.pop(key, None)
        try:
            redis = await get_cache_redis()
            await redis.delete(f"{self.prefix}{key}")
        except Exception as e:
            logger.warning("L2 cache delete error: %s", e)

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all L2 keys matching a pattern; clear entire L1."""
        count = 0
        self.l1.clear()
        try:
            redis = await get_cache_redis()
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor=cursor, match=f"{self.prefix}{pattern}", count=100)
                if keys:
                    count += await redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning("L2 cache clear error: %s", e)
        return count

    # ── Metrics ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        total = sum(self._hits.values())
        return {
            "l1_hits": self._hits["l1"],
            "l2_hits": self._hits["l2"],
            "misses": self._hits["miss"],
            "hit_rate": (self._hits["l1"] + self._hits["l2"]) / max(total, 1),
            "l1_size": len(self.l1),
        }
