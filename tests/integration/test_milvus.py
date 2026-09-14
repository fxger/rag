"""Milvus integration tests (requires Milvus on VM1)."""
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_milvus_connection():
    """Test Milvus connectivity to VM1."""
    from src.config.milvus import get_milvus_connection
    get_milvus_connection()
    from pymilvus import utility
    # Just verify connection works by listing collections
    collections = utility.list_collections()
    assert isinstance(collections, list)


@pytest.mark.asyncio
async def test_milvus_collection_schema():
    """Test enterprise RAG collection has correct schema."""
    from src.config.milvus import get_or_create_collection
    collection = get_or_create_collection()
    schema = collection.schema
    field_names = [f.name for f in schema.fields]
    assert "document_id" in field_names
    assert "chunk_id" in field_names
    assert "embedding" in field_names
    assert "content" in field_names