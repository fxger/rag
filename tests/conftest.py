"""
Shared test fixtures for the RAG system test suite.
"""

import asyncio
import uuid
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create a single event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client for FastAPI app."""
    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def db_session():
    """Mock async database session for unit tests."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.execute = AsyncMock()
    session.get = AsyncMock()
    session.add = AsyncMock()
    session.begin = AsyncMock()
    session.begin.return_value.__aenter__ = AsyncMock(return_value=session)
    session.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    yield session


@pytest.fixture
def mock_redis() -> MagicMock:
    """Mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.ping = AsyncMock(return_value=True)
    redis.scan = AsyncMock(return_value=(0, []))
    redis.pipeline = MagicMock()
    pipe = AsyncMock()
    pipe.execute = AsyncMock(return_value=[None, 0, None, None])
    redis.pipeline.return_value = pipe
    return redis


@pytest.fixture
def mock_milvus() -> MagicMock:
    """Mock Milvus collection."""
    collection = MagicMock()
    collection.search = MagicMock(return_value=[])
    collection.insert = MagicMock(return_value=[0])
    collection.load = MagicMock()
    collection.count = MagicMock(return_value=0)
    return collection


@pytest.fixture
def mock_openai() -> MagicMock:
    """Mock OpenAI client."""
    mock = MagicMock()
    mock.embeddings = MagicMock()
    mock.embeddings.create = AsyncMock()
    mock.chat = MagicMock()
    mock.chat.completions = MagicMock()
    mock.chat.completions.create = AsyncMock()
    return mock


@pytest.fixture
def sample_user_data() -> dict:
    """Sample user registration data."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPass123!",
        "role": "user",
        "department": "Engineering",
    }


@pytest.fixture
def sample_query_data() -> dict:
    """Sample query request data."""
    return {
        "query": "What is the company's revenue growth strategy?",
        "top_k": 10,
        "use_reranking": True,
    }


@pytest.fixture
def mock_user() -> MagicMock:
    """Mock authenticated user object."""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.username = "testuser"
    user.email = "test@example.com"
    user.role = "user"
    user.is_active = True
    return user


@pytest.fixture
def mock_admin_user() -> MagicMock:
    """Mock admin user object."""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.username = "admin"
    user.email = "admin@example.com"
    user.role = "admin"
    user.is_active = True
    return user