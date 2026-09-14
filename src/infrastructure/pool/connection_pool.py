"""
Connection pool manager for database, Redis, and other services.

Provides centralized pool management with health monitoring and metrics.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PoolMetrics:
    """Metrics for a connection pool."""
    name: str
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time_ms: float = 0.0
    last_health_check: float = 0.0
    is_healthy: bool = True


class ConnectionPoolManager:
    """
    Centralized connection pool manager with health monitoring.

    Tracks pool metrics for PostgreSQL, Redis, and other external services.
    """

    def __init__(self):
        self._pools: dict[str, Any] = {}
        self._metrics: dict[str, PoolMetrics] = {}
        self._response_times: dict[str, list[float]] = {}

    def register_pool(self, name: str, pool: Any) -> None:
        """Register an external connection pool for monitoring."""
        self._pools[name] = pool
        self._metrics[name] = PoolMetrics(name=name)
        self._response_times[name] = []
        logger.info("Registered connection pool: %s", name)

    def record_request(self, pool_name: str, duration_ms: float, success: bool = True) -> None:
        """Record a request metric for a pool."""
        if pool_name not in self._metrics:
            return

        metrics = self._metrics[pool_name]
        metrics.total_requests += 1

        if not success:
            metrics.total_errors += 1

        times = self._response_times[pool_name]
        times.append(duration_ms)
        if len(times) > 100:
            times.pop(0)
        metrics.avg_response_time_ms = sum(times) / len(times)

    async def health_check(self, pool_name: str) -> bool:
        """
        Check if a pool's underlying service is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        if pool_name not in self._metrics:
            return False

        metrics = self._metrics[pool_name]
        start = time.monotonic()

        try:
            pool = self._pools.get(pool_name)

            if pool_name == "postgresql" and pool:
                # Async check via pool
                async with pool.connect() as conn:
                    await conn.execute(
                        __import__("sqlalchemy").text("SELECT 1")
                    )
            elif pool_name == "redis" and pool:
                await pool.ping()
            elif pool_name == "milvus":
                from src.config.milvus import get_milvus_connection
                from pymilvus import utility as milvus_util
                get_milvus_connection()
                milvus_util.has_collection("__health_check__")

            elapsed = (time.monotonic() - start) * 1000
            metrics.is_healthy = True
            metrics.last_health_check = time.time()
            logger.debug("Pool '%s' healthy (%.1fms)", pool_name, elapsed)
            return True

        except Exception as e:
            metrics.is_healthy = False
            metrics.last_health_check = time.time()
            logger.warning("Pool '%s' unhealthy: %s", pool_name, e)
            return False

    async def check_all(self) -> dict[str, bool]:
        """Check health of all registered pools."""
        results = {}
        for name in self._pools:
            results[name] = await self.health_check(name)
        return results

    @property
    def all_metrics(self) -> dict[str, dict]:
        """Return metrics for all pools."""
        return {
            name: {
                "active": m.active_connections,
                "idle": m.idle_connections,
                "total_requests": m.total_requests,
                "total_errors": m.total_errors,
                "avg_response_ms": round(m.avg_response_time_ms, 2),
                "healthy": m.is_healthy,
            }
            for name, m in self._metrics.items()
        }


# Singleton
pool_manager = ConnectionPoolManager()