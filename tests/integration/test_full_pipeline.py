"""Full pipeline integration test."""
import pytest
import uuid
from unittest.mock import AsyncMock, patch, MagicMock

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_end_to_end_rag_query():
    """Test complete RAG pipeline: embed → retrieve → generate."""
    from src.services.rag_service import RAGService
    from src.core.retrieval.models import SearchResult

    service = RAGService.__new__(RAGService)
    service.embedder = AsyncMock()
    service.retriever = AsyncMock()
    service.context_builder = MagicMock()
    service.generator = AsyncMock()
    service.cache = AsyncMock()

    # Mock pipeline steps
    service.embedder.generate_single = AsyncMock(return_value=[0.1] * 3072)
    service.retriever.retrieve = AsyncMock(return_value=MagicMock(
        results=[
            SearchResult(chunk_id="1", document_id="d1", content="relevant info", score=0.95)
        ],
        retrieval_time_ms=50,
    ))
    service.context_builder.build = MagicMock(return_value="Context: relevant info")
    service.generator.generate_with_sources = AsyncMock(return_value=MagicMock(
        answer="The answer is based on the context.",
        sources=[{"chunk_id": "1", "score": 0.95}],
    ))
    service.cache.get = AsyncMock(return_value=None)
    service.cache.set = AsyncMock()

    with patch.object(service, "__init__", lambda self: None):
        result = await service.query("What is relevant?", user_id=str(uuid.uuid4()))
        assert result is not None
        service.retriever.retrieve.assert_called_once()