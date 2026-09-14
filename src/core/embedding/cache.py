"""Redis-backed embedding cache with binary-safe serialization."""
import json
import logging
from typing import Optional

from src.config.redis import get_cache_redis

logger = logging.getLogger(__name__)

# Cache key prefix for embedding namespace
EMBEDDING_CACHE_PREFIX = "emb:"


def _make_key(text_hash: str) -> str:
    """Generate Redis key for embedding cache entry."""
    return f"{EMBEDDING_CACHE_PREFIX}{text_hash}"


async def get(text_hash: str) -> Optional[list[float]]:
    """Retrieve a cached embedding by text hash.
    
    Args:
        text_hash: SHA-256 hash of the source text.
        
    Returns:
        Cached embedding vector, or None if not found/expired.
    """
    redis = await get_cache_redis()
    key = _make_key(text_hash)
    
    try:
        data = await redis.get(key)
        if data is None:
            return None
        
        # Decode bytes to string if needed
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        
        embedding = json.loads(data)
        logger.debug("Cache hit for embedding %s", text_hash[:16])
        return embedding
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Corrupted cache entry for %s: %s", text_hash[:16], e)
        return None
    except Exception as e:
        logger.error("Redis get failed for %s: %s", text_hash[:16], e)
        return None


async def set(text_hash: str, embedding: list[float], ttl: int = 86400) -> bool:
    """Store an embedding in cache with TTL.
    
    Args:
        text_hash: SHA-256 hash of the source text.
        embedding: Embedding vector to cache.
        ttl: Time-to-live in seconds (default: 24 hours).
        
    Returns:
        True if stored successfully, False otherwise.
    """
    redis = await get_cache_redis()
    key = _make_key(text_hash)
    
    try:
        serialized = json.dumps(embedding, separators=(",", ":"))
        await redis.setex(key, ttl, serialized)
        logger.debug("Cached embedding %s (ttl=%ds)", text_hash[:16], ttl)
        return True
    except Exception as e:
        logger.error("Redis set failed for %s: %s", text_hash[:16], e)
        return False


async def get_batch(text_hashes: list[str]) -> dict[str, list[float]]:
    """Retrieve multiple cached embeddings in a single pipeline.
    
    Args:
        text_hashes: List of SHA-256 hashes to look up.
        
    Returns:
        Dictionary mapping hash -> embedding for cache hits only.
    """
    if not text_hashes:
        return {}
    
    redis = await get_cache_redis()
    keys = [_make_key(h) for h in text_hashes]
    
    try:
        # Use pipeline for batch retrieval
        pipeline = redis.pipeline()
        for key in keys:
            pipeline.get(key)
        results = await pipeline.execute()
        
        cached: dict[str, list[float]] = {}
        for text_hash, raw_data in zip(text_hashes, results):
            if raw_data is None:
                continue
            
            try:
                if isinstance(raw_data, bytes):
                    raw_data = raw_data.decode("utf-8")
                embedding = json.loads(raw_data)
                cached[text_hash] = embedding
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Corrupted cache entry for %s: %s", text_hash[:16], e)
                continue
        
        hit_rate = len(cached) / len(text_hashes) * 100 if text_hashes else 0
        logger.info("Embedding cache batch: %d/%d hits (%.1f%%)", len(cached), len(text_hashes), hit_rate)
        return cached
    except Exception as e:
        logger.error("Redis batch get failed: %s", e)
        return {}


async def invalidate(text_hash: str) -> bool:
    """Remove a specific embedding from cache.
    
    Args:
        text_hash: SHA-256 hash to invalidate.
        
    Returns:
        True if key existed and was deleted, False otherwise.
    """
    redis = await get_cache_redis()
    key = _make_key(text_hash)
    
    try:
        deleted = await redis.delete(key)
        return bool(deleted)
    except Exception as e:
        logger.error("Redis delete failed for %s: %s", text_hash[:16], e)
        return False
