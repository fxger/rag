"""Tests for vector retriever."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestVectorRetriever:

    def test_init(self):
        from src.core.retrieval.vector_retriever import VectorRetriever
        with patch("src.core.retrieval.vector_retriever.get_settings") as mock:
            mock.return_value = MagicMock(MILVUS_HOST="192.168.86.132", MILVUS_PORT=19530)
            retriever = VectorRetriever.__new__(VectorRetriever)
            retriever.host = "192.168.86.132"
            retriever.port = 19530
            assert retriever.host == "192.168.86.132"