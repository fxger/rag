"""
Production environment configuration.
"""

from src.config.settings import Settings


class ProdSettings(Settings):
    """Production-specific settings override."""

    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 50
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30

    # Stricter JWT expiry
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Production rate limits
    RATE_LIMIT_USER_PER_MINUTE: int = 100
    RATE_LIMIT_GLOBAL_PER_MINUTE: int = 10000

    # Production RAG settings
    CHUNK_SIZE: int = 512
    DEFAULT_TOP_K: int = 10
    RERANK_TOP_N: int = 5


def get_prod_settings() -> ProdSettings:
    return ProdSettings()