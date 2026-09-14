"""Tests for reranker."""
import pytest
from unittest.mock import MagicMock, patch


class TestReranker:

    @pytest.mark.asyncio
    async def test_rerank_returns_sorted_results(self):
        from src.core.retrieval.reranker import Reranker
        from src.core.retrieval.models import SearchResult

        reranker = Reranker.__new__(Reranker)
        reranker.model = None  # Force fallback
        reranker.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        docs = [
            SearchResult(chunk_id="1", document_id="d1", content="a", score=0.5),
            SearchResult(chunk_id="2", document_id="d1", content="b", score=0.9),
            SearchResult(chunk_id="3", document_id="d1", content="c", score=0.7),
        ]
        result = await reranker.rerank("query", docs, top_n=2)
        assert len(result) == 2
        # Without model, should preserve original order
        assert result[0].score >= result[1].score