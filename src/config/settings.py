"""
Application settings and configuration management.

Uses Pydantic BaseSettings for type-safe environment variable loading.
All VM connection endpoints are configured here.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────
    APP_NAME: str = "Enterprise RAG System"
    APP_VERSION: str = "0.2.1"
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")

    # ── PostgreSQL (VM2: 192.168.86.134) ─────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@192.168.86.134:5432/postgres",
        description="Async PostgreSQL connection string",
    )
    DATABASE_POOL_SIZE: int = Field(default=20)
    DATABASE_MAX_OVERFLOW: int = Field(default=10)
    DATABASE_POOL_TIMEOUT: int = Field(default=30)
    DATABASE_ECHO: bool = Field(default=False)

    # ── Redis (VM1: 192.168.86.132) ─────────────────────────────────
    REDIS_URL: str = Field(
        default="redis://192.168.86.132:6379/0",
        description="Redis connection string",
    )
    REDIS_CACHE_DB: int = Field(default=1, description="DB for embedding cache")
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_MAX_CONNECTIONS: int = Field(default=50)

    # ── Milvus (VM1: 192.168.86.132) ────────────────────────────────
    MILVUS_HOST: str = Field(default="192.168.86.132")
    MILVUS_PORT: int = Field(default=19530)
    MILVUS_COLLECTION: str = Field(default="enterprise_rag")
    MILVUS_DIMENSION: int = Field(default=1024, description="Qwen embedding dim")

    # ── MinIO (VM1: 192.168.86.132) ─────────────────────────────────
    MINIO_ENDPOINT: str = Field(default="192.168.86.132:9000")
    MINIO_ACCESS_KEY: str = Field(default="minioadmin")
    MINIO_SECRET_KEY: str = Field(default="minioadmin123")
    MINIO_BUCKET: str = Field(default="rag-documents")
    MINIO_SECURE: bool = Field(default=False)

    # ── OpenAI / DeepSeek ───────────────────────────────────────────
    OPENAI_API_KEY: str = Field(default="your-api-key")
    OPENAI_BASE_URL: Optional[str] = Field(default=None)
    EMBEDDING_MODEL: str = Field(default="qwen-embedding")
    LLM_MODEL: str = Field(default="deepseek-chat")
    EMBEDDING_DIMENSION: int = Field(default=1024)
    LLM_MAX_TOKENS: int = Field(default=4096)
    LLM_TEMPERATURE: float = Field(default=0.7)

    # ── Kafka (VM2: 192.168.86.134) ─────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = Field(
        default="192.168.86.134:9092",
        description="Kafka broker addresses",
    )
    KAFKA_DOCUMENT_TOPIC: str = Field(default="rag.document.processing")
    KAFKA_ENABLED: bool = Field(default=False, description="Enable Kafka in v1")

    # ── JWT Authentication ───────────────────────────────────────────
    JWT_SECRET_KEY: str = Field(
        default="change-me-in-production-use-a-real-secret",
        description="Secret key for JWT token signing",
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)

    # ── RAG Settings ─────────────────────────────────────────────────
    CHUNK_SIZE: int = Field(default=512, description="Default chunk size in tokens")
    CHUNK_OVERLAP: int = Field(default=50, description="Overlap between chunks")
    DEFAULT_TOP_K: int = Field(default=10, description="Default retrieval top_k")
    RERANK_TOP_N: int = Field(default=5, description="Top N after reranking")

    # ── Reranker Settings ────────────────────────────────────────────
    RERANKER_ENABLED: bool = Field(default=True, description="Enable cross-encoder reranking")
    RERANKER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model name"
    )
    RERANKER_MAX_LENGTH: int = Field(default=512, description="Max sequence length")
    RERANKER_BATCH_SIZE: int = Field(default=32, description="Batch size for inference")
    RERANKER_CACHE_ENABLED: bool = Field(default=True, description="Enable reranking cache")
    RERANKER_CACHE_MAXSIZE: int = Field(default=256, description="Max cache entries")

    # ── Hybrid Retrieval Settings ────────────────────────────────────
    RETRIEVAL_FUSION_METHOD: str = Field(
        default="adaptive",
        description="Fusion method: adaptive/rrf/weighted"
    )
    RETRIEVAL_SPARSE_BOOST_THRESHOLD: float = Field(
        default=2.0,
        description="Sparse score threshold for weight boost"
    )
    RETRIEVAL_BASE_WEIGHTS: dict = Field(
        default={
            "factual": {"dense": 0.6, "sparse": 0.4},
            "conceptual": {"dense": 0.8, "sparse": 0.2},
            "comparative": {"dense": 0.5, "sparse": 0.3},
            "navigational": {"dense": 0.4, "sparse": 0.4},
        },
        description="Base weights by query type"
    )

    # ── Rate Limiting ────────────────────────────────────────────────
    RATE_LIMIT_USER_PER_MINUTE: int = Field(default=100)
    RATE_LIMIT_GLOBAL_PER_MINUTE: int = Field(default=10000)

    # ── CORS ─────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = Field(default=["*"])

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use the asyncpg driver: "
                "postgresql+asyncpg://user:pass@host:port/db"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton of application settings."""
    return Settings()
