"""Tests for rate limiter."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import time


class TestRateLimiter:

    def test_parse_limit(self):
        from src.api.middleware.rate_limiter import RateLimiter
        limiter = RateLimiter()
        count, seconds = limiter._parse_limit("100/min")
        assert count == 100
        assert seconds == 60

        count, seconds = limiter._parse_limit("10/hour")
        assert count == 10
        assert seconds == 3600

    @pytest.mark.asyncio
    async def test_check_rate_limit_allows_under_limit(self):
        from src.api.middleware.rate_limiter import RateLimiter
        limiter = RateLimiter()
        mock_redis = AsyncMock()
        pipe = AsyncMock()
        pipe.execute = AsyncMock(return_value=[None, 0, None, None])
        mock_redis.pipeline = MagicMock(return_value=pipe)

        with patch("src.api.middleware.rate_limiter.get_redis", return_value=mock_redis):
            result = await limiter.check_rate_limit("test_key", "100/min")
            assert result is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_blocks_over_limit(self):
        from src.api.middleware.rate_limiter import RateLimiter
        limiter = RateLimiter()
        mock_redis = AsyncMock()
        pipe = AsyncMock()
        pipe.execute = AsyncMock(return_value=[None, 100, None, None])
        mock_redis.pipeline = MagicMock(return_value=pipe)

        with patch("src.api.middleware.rate_limiter.get_redis", return_value=mock_redis):
            result = await limiter.check_rate_limit("test_key", "100/min")
            assert result is False