"""API tests for document endpoints."""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport


class TestDocumentAPI:

    @pytest.mark.asyncio
    async def test_document_list_requires_auth(self):
        """Test document listing requires authentication."""
        from src.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/documents/")
            assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_upload_requires_auth(self):
        """Test document upload requires authentication."""
        from src.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/documents/upload")
            assert response.status_code in [401, 403]