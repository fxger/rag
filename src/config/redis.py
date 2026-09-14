"""
Redis client factory with connection pooling.

Provides async Redis client for caching, rate limiting, and session storage.
Uses VM1 (192.168.86.132) Redis instance.
"""

from typing import Optional

from redis.asyncio import ConnectionPool, Redis
from redis.asyncio.client import PubSub

from src.config.settings import get_settings

# Connection pools (initialized lazily)
_main_pool: Optional[ConnectionPool] = None
_cache_pool: Optional[ConnectionPool] = None


def get_redis_pool() -> ConnectionPool:
    """Get or create the main Redis connection pool."""
    global _main_pool
    if _main_pool is None:
        settings = get_settings()
        _main_pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=True,
            password=settings.REDIS_PASSWORD,
        )
    return _main_pool


def get_cache_pool() -> ConnectionPool:
    """Get or create the cache-dedicated Redis connection pool."""
    global _cache_pool
    if _cache_pool is None:
        settings = get_settings()
        cache_url = settings.REDIS_URL.replace(
            f"/{settings.REDIS_URL.split('/')[-1]}",
            f"/{settings.REDIS_CACHE_DB}",
        )
        _cache_pool = ConnectionPool.from_url(
            cache_url,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=True,
            password=settings.REDIS_PASSWORD,
        )
    return _cache_pool


async def get_redis() -> Redis:
    """FastAPI dependency that yields a Redis client."""
    return Redis(connection_pool=get_redis_pool())


async def get_cache_redis() -> Redis:
    """Get a Redis client connected to the cache database."""
    return Redis(connection_pool=get_cache_pool())


async def close_redis() -> None:
    """Close all Redis connection pools."""
    global _main_pool, _cache_pool
    if _main_pool:
        await _main_pool.disconnect()
        _main_pool = None
    if _cache_pool:
        await _cache_pool.disconnect()
        _cache_pool = None