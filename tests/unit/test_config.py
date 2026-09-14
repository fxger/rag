"""Tests for configuration management."""
import os
import pytest
from unittest.mock import patch


class TestSettings:
    """Test Pydantic settings loading and validation."""

    def test_default_settings(self):
        """Settings load with defaults from .env.example."""
        from src.config.settings import Settings
        settings = Settings(
            DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/test",
            OPENAI_API_KEY="test-key",
        )
        assert settings.APP_NAME == "Enterprise RAG System"
        assert settings.CHUNK_SIZE == 512
        assert settings.MILVUS_PORT == 19530

    def test_database_url_validation(self):
        """Reject non-asyncpg database URLs."""
        from src.config.settings import Settings
        with pytest.raises(ValueError, match="asyncpg"):
            Settings(DATABASE_URL="postgresql://user:pass@localhost/db", OPENAI_API_KEY="k")

    def test_database_url_accepts_asyncpg(self):
        """Accept valid asyncpg URLs."""
        from src.config.settings import Settings
        s = Settings(
            DATABASE_URL="postgresql+asyncpg://u:p@192.168.86.134:5432/postgres",
            OPENAI_API_KEY="k",
        )
        assert s.DATABASE_URL.startswith("postgresql+asyncpg://")

    def test_vm1_defaults(self):
        """Default Redis/Milvus/MinIO point to VM1."""
        from src.config.settings import Settings
        s = Settings(
            DATABASE_URL="postgresql+asyncpg://u:p@localhost:5432/db",
            OPENAI_API_KEY="k",
        )
        assert "192.168.86.132" in s.REDIS_URL
        assert s.MILVUS_HOST == "192.168.86.132"
        assert "192.168.86.132" in s.MINIO_ENDPOINT

    def test_vm2_defaults(self):
        """Default PostgreSQL/Kafka point to VM2."""
        from src.config.settings import Settings
        s = Settings(
            DATABASE_URL="postgresql+asyncpg://u:p@192.168.86.134:5432/db",
            OPENAI_API_KEY="k",
        )
        assert "192.168.86.134" in s.DATABASE_URL
        assert "192.168.86.134" in s.KAFKA_BOOTSTRAP_SERVERS