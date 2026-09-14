"""Tests for embedding generation and caching."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestEmbeddingGenerator:

    @pytest.mark.asyncio
    async def test_generate_single(self):
        from src.core.embedding.embedder import EmbeddingGenerator
        gen = EmbeddingGenerator.__new__(EmbeddingGenerator)
        gen.model = "text-embedding-3-large"
        gen.dimension = 3072
        gen.client = MagicMock()
        gen.client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 3072)])
        )
        result = await gen.generate_single("test text")
        assert len(result) == 3072

    def test_text_hash_deterministic(self):
        from src.core.embedding.embedder import EmbeddingGenerator
        h1 = EmbeddingGenerator.text_hash("hello world")
        h2 = EmbeddingGenerator.text_hash("hello world")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256

    def test_text_hash_unique(self):
        from src.core.embedding.embedder import EmbeddingGenerator
        h1 = EmbeddingGenerator.text_hash("text one")
        h2 = EmbeddingGenerator.text_hash("text two")
        assert h1 != h2


class TestEmbeddingCache:

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        from src.core.embedding.cache import EmbeddingCache
        cache = EmbeddingCache.__new__(EmbeddingCache)
        cache.prefix = "test:emb:"
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        with patch("src.core.embedding.cache.get_cache_redis", return_value=mock_redis):
            result = await cache.get("nonexistent")
            assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        from src.core.embedding.cache import EmbeddingCache
        cache = EmbeddingCache.__new__(EmbeddingCache)
        cache.prefix = "test:emb:"
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value="[0.1, 0.2, 0.3]")
        with patch("src.core.embedding.cache.get_cache_redis", return_value=mock_redis):
            await cache.set("key1", [0.1, 0.2, 0.3])
            result = await cache.get("key1")
            assert result is not None