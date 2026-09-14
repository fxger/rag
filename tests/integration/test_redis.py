"""Redis integration tests (requires Redis on VM1)."""
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_redis_connection():
    """Test Redis connectivity to VM1."""
    from src.config.redis import get_redis
    redis = await get_redis()
    result = await redis.ping()
    assert result is True


@pytest.mark.asyncio
async def test_redis_set_get():
    """Test Redis set/get operations."""
    from src.config.redis import get_redis
    redis = await get_redis()
    test_key = "rag:test:integration"
    await redis.set(test_key, "test_value", ex=60)
    value = await redis.get(test_key)
    assert value == "test_value"
    await redis.delete(test_key)