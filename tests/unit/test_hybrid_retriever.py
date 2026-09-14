"""Tests for hybrid retriever."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestHybridRetriever:

    @pytest.mark.asyncio
    async def test_retrieve_calls_fusion(self):
        from src.core.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.dense_retriever = AsyncMock()
        retriever.sparse_retriever = AsyncMock()
        retriever.fusion = MagicMock()
        retriever.reranker = AsyncMock()
        retriever.analyzer = MagicMock()
        retriever.embedder = AsyncMock()
        retriever.context_builder = MagicMock()

        from src.core.retrieval.models import SearchResult
        retriever.dense_retriever.search = AsyncMock(return_value=[
            SearchResult(chunk_id="1", document_id="d1", content="test", score=0.9, source_type="vector")
        ])
        retriever.sparse_retriever.search = AsyncMock(return_value=[
            SearchResult(chunk_id="2", document_id="d1", content="test2", score=0.8, source_type="sparse")
        ])
        retriever.fusion.weighted_fusion = MagicMock(return_value=[
            SearchResult(chunk_id="1", document_id="d1", content="test", score=0.9)
        ])
        retriever.reranker.rerank = AsyncMock(return_value=[
            SearchResult(chunk_id="1", document_id="d1", content="test", score=0.95)
        ])
        retriever.analyzer.analyze = MagicMock(return_value=MagicMock(query_type="factual", keywords=["test"]))
        retriever.embedder.generate_single = AsyncMock(return_value=[0.1] * 3072)

        result = await retriever.retrieve("test query", top_k=5)
        assert result is not None
        retriever.fusion.weighted_fusion.assert_called_once()