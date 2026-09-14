"""API tests for query endpoints."""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport


class TestQueryAPI:

    @pytest.mark.asyncio
    async def test_query_requires_auth(self):
        """Test query endpoint requires authentication."""
        from src.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/query/", json={"query": "test"})
            assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_query_history_requires_auth(self):
        """Test query history requires authentication."""
        from src.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/query/history")
            assert response.status_code in [401, 403]