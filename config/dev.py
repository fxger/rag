"""
Development environment configuration.
"""

from src.config.settings import Settings


class DevSettings(Settings):
    """Development-specific settings override."""

    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DATABASE_ECHO: bool = True  # Log all SQL queries
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 5

    # Shorter token expiry for dev
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Lower rate limits for testing
    RATE_LIMIT_USER_PER_MINUTE: int = 1000
    RATE_LIMIT_GLOBAL_PER_MINUTE: int = 100000


def get_dev_settings() -> DevSettings:
    return DevSettings()