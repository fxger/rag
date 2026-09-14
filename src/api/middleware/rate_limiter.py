"""
Redis-based sliding window rate limiter.

Supports per-user and global rate limiting with configurable limits.
"""

import logging
import time
from typing import Optional

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.config.redis import get_redis

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding window rate limiter using Redis sorted sets.

    Limits are expressed as "N/period" (e.g., "100/min", "10/hour").
    """

    def __init__(self):
        self._limits: dict[str, int] = {}

    @staticmethod
    def _parse_limit(limit_str: str) -> tuple[int, int]:
        """Parse 'N/period' into (count, seconds)."""
        count, period = limit_str.split("/")
        period_map = {
            "sec": 1, "second": 1, "seconds": 1,
            "min": 60, "minute": 60, "minutes": 60,
            "hour": 3600, "hours": 3600,
            "day": 86400, "days": 86400,
        }
        return int(count), period_map.get(period, 60)

    async def check_rate_limit(self, key: str, limit: str) -> bool:
        """
        Check if a request is allowed under the given rate limit.

        Uses Redis sorted set sliding window algorithm.

        Args:
            key: Unique identifier (e.g., user_id, "global").
            limit: Rate limit string like "100/min".

        Returns:
            True if allowed, False if rate limited.
        """
        try:
            redis = await get_redis()
            max_requests, window_seconds = self._parse_limit(limit)
            now = time.time()
            window_start = now - window_seconds

            pipe = redis.pipeline()
            # Remove expired entries
            pipe.zremrangebyscore(f"rl:{key}", 0, window_start)
            # Count current window
            pipe.zcard(f"rl:{key}")
            # Add current request
            pipe.zadd(f"rl:{key}", {str(now): now})
            # Set expiry
            pipe.expire(f"rl:{key}", window_seconds)

            results = await pipe.execute()
            current_count = results[1]

            if current_count >= max_requests:
                logger.warning("Rate limit exceeded for key=%s: %d/%s", key, current_count, limit)
                return False

            return True

        except Exception as e:
            logger.error("Rate limit check failed: %s", e)
            return True  # Fail open

    async def user_limit(self, user_id: str, limit: str = "100/min") -> bool:
        """Check per-user rate limit."""
        return await self.check_rate_limit(f"user:{user_id}", limit)

    async def global_limit(self, limit: str = "10000/min") -> bool:
        """Check global rate limit."""
        return await self.check_rate_limit("global", limit)

    async def get_remaining(self, key: str, limit: str) -> int:
        """Get remaining allowed requests in current window."""
        try:
            redis = await get_redis()
            max_requests, window_seconds = self._parse_limit(limit)
            now = time.time()
            window_start = now - window_seconds
            current = await redis.zcount(f"rl:{key}", window_start, now)
            return max(0, max_requests - current)
        except Exception:
            return 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces rate limits on every request."""

    def __init__(self, app, user_limit: str = "100/min", global_limit: str = "10000/min"):
        super().__init__(app)
        self.limiter = RateLimiter()
        self.user_limit_str = user_limit
        self.global_limit_str = global_limit

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Global check
        if not await self.limiter.global_limit(self.global_limit_str):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Global rate limit exceeded. Please try again later.",
            )

        # Per-user check (extract from JWT if available)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            if not await self.limiter.user_limit(str(user_id), self.user_limit_str):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="User rate limit exceeded. Please try again later.",
                )

        response = await call_next(request)
        return response