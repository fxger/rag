"""Tests for multi-level cache."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestMultiLevelCache:

    @pytest.mark.asyncio
    async def test_l1_hit(self):
        from src.core.cache.multi_level_cache import MultiLevelCache
        cache = MultiLevelCache(l1_maxsize=10, l1_ttl=60)
        cache.l1["key1"] = "value1"
        result = await cache.get("key1")
        assert result == "value1"
        assert cache._hits["l1"] == 1

    @pytest.mark.asyncio
    async def test_l2_hit_promotes_to_l1(self):
        from src.core.cache.multi_level_cache import MultiLevelCache
        cache = MultiLevelCache(l1_maxsize=10, l1_ttl=60)
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value='"cached_value"')
        with patch("src.core.cache.multi_level_cache.get_cache_redis", return_value=mock_redis):
            result = await cache.get("key1")
            assert result == "cached_value"
            assert cache.l1["key1"] == "cached_value"
            assert cache._hits["l2"] == 1

    @pytest.mark.asyncio
    async def test_miss(self):
        from src.core.cache.multi_level_cache import MultiLevelCache
        cache = MultiLevelCache()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        with patch("src.core.cache.multi_level_cache.get_cache_redis", return_value=mock_redis):
            result = await cache.get("missing")
            assert result is None
            assert cache._hits["miss"] == 1

    def test_stats(self):
        from src.core.cache.multi_level_cache import MultiLevelCache
        cache = MultiLevelCache()
        cache._hits = {"l1": 10, "l2": 5, "miss": 3}
        stats = cache.stats
        assert stats["l1_hits"] == 10
        assert stats["hit_rate"] == 15 / 18