"""
Milvus vector database connection and collection management.
Uses MilvusClient (pymilvus 2.x+ recommended API).
"""

import json
import logging
from typing import Optional

from pymilvus import (
    MilvusClient,
    DataType,
    CollectionSchema,
    FieldSchema,
)

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """Get or create MilvusClient singleton."""
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    _client = MilvusClient(uri=uri)
    logger.info("Connected to Milvus at %s", uri)
    return _client


def get_or_create_collection() -> str:
    """
    Get or create the enterprise_rag collection.
    Returns the collection name.
    """
    settings = get_settings()
    client = get_milvus_client()
    collection_name = settings.MILVUS_COLLECTION

    # Check if collection exists (list_collections returns list of strings)
    try:
        collections = client.list_collections()
        # pymilvus 2.x returns list of strings
        if collection_name in collections:
            logger.info("Using existing Milvus collection: %s", collection_name)
            return collection_name
    except Exception as e:
        logger.warning("Error listing collections: %s", e)

    # Create collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
        # Tenant the chunk belongs to.  Scalar filtering (`tenant_id == "…"`)
        # is cheap and exact — unlike the old approach of stuffing the tenant
        # into the `metadata` JSON *string*, which Milvus can only `like`
        # on a prefix, so a tenant filter either missed rows or scanned
        # everything.  "__platform__" is the shared pool every store may read.
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.MILVUS_DIMENSION,
        ),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=8192),
    ]

    schema = CollectionSchema(fields, description="Enterprise RAG document embeddings")

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )

    # Create index for fast ANN search
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 1024},
    )
    client.create_index(collection_name, index_params)

    logger.info("Created Milvus collection: %s", collection_name)
    return collection_name


def get_collection_fields(collection_name: Optional[str] = None) -> set[str]:
    """Return the set of field names of a collection.

    Collections created before the tenant field existed do not have
    ``tenant_id``; callers must branch on this instead of assuming, otherwise
    every search raises "field tenant_id not exist".
    """
    client = get_milvus_client()
    col_name = collection_name or get_settings().MILVUS_COLLECTION
    try:
        desc = client.describe_collection(collection_name=col_name)
        return {f["name"] for f in desc.get("fields", [])}
    except Exception as e:
        logger.warning("Failed to describe collection %s: %s", col_name, e)
        return set()


def load_collection(collection_name: Optional[str] = None) -> bool:
    """Load collection into memory for faster search."""
    client = get_milvus_client()
    col_name = collection_name or get_settings().MILVUS_COLLECTION
    
    try:
        client.load_collection(col_name)
        logger.info("Loaded Milvus collection: %s", col_name)
        return True
    except Exception as e:
        logger.warning("Failed to load collection %s: %s", col_name, e)
        return False


def search_vectors(
    query_embedding: list[float],
    top_k: int = 10,
    collection_name: Optional[str] = None,
    expr: Optional[str] = None,
) -> list[dict]:
    """
    Search for similar vectors in Milvus.

    Args:
        query_embedding: Query vector.
        top_k: Number of hits.
        collection_name: Override the configured collection.
        expr: Optional Milvus boolean expression, e.g. ``tenant_id == "x"``.
    """
    settings = get_settings()
    client = get_milvus_client()
    col_name = collection_name or settings.MILVUS_COLLECTION

    try:
        results = client.search(
            collection_name=col_name,
            data=[query_embedding],
            limit=top_k,
            expr=expr,
            output_fields=["document_id", "chunk_id", "content", "metadata"],
        )

        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append({
                    "id": hit.id,
                    "document_id": hit.entity.get("document_id", ""),
                    "chunk_id": hit.entity.get("chunk_id", ""),
                    "content": hit.entity.get("content", ""),
                    "score": hit.score,
                    "metadata": json.loads(hit.entity.get("metadata", "{}") or "{}"),
                })

        return search_results

    except Exception as e:
        logger.error("Milvus search error: %s", e)
        return []


def insert_vectors(
    documents: list[dict],
    collection_name: Optional[str] = None,
) -> int:
    """
    Insert vectors into Milvus.
    """
    settings = get_settings()
    client = get_milvus_client()
    col_name = collection_name or settings.MILVUS_COLLECTION

    try:
        has_tenant_field = "tenant_id" in get_collection_fields(col_name)
        data = []
        for doc in documents:
            row = {
                "document_id": doc["document_id"],
                "chunk_id": doc["chunk_id"],
                "content": doc["content"],
                "embedding": doc["embedding"],
                "metadata": json.dumps(doc.get("metadata", {})),
            }
            if has_tenant_field:
                # Legacy collections have no tenant field; only add the key
                # when the schema accepts it, otherwise the insert rejects the
                # whole batch with "field tenant_id not exist".
                row["tenant_id"] = doc.get("tenant_id") or "__platform__"
            data.append(row)

        result = client.insert(collection_name=col_name, data=data)
        logger.info("Inserted %d vectors into Milvus", len(data))
        return len(data)

    except Exception as e:
        logger.error("Milvus insert error: %s", e)
        return 0


def delete_vectors_by_document(
    document_id: str,
    collection_name: Optional[str] = None,
) -> int:
    """Delete every vector belonging to a document.

    Needed because deleting a document removes its PostgreSQL rows and its
    MinIO object, but nothing used to touch Milvus — so the vectors stayed
    behind and retrieval could still return chunks of a deleted document.

    Args:
        document_id: UUID of the document whose vectors should go.
        collection_name: Override the configured collection.

    Returns:
        Number of entities Milvus reports as deleted, or 0 on failure.
    """
    settings = get_settings()
    client = get_milvus_client()
    col_name = collection_name or settings.MILVUS_COLLECTION

    try:
        result = client.delete(
            collection_name=col_name,
            filter=f'document_id == "{document_id}"',
        )
        deleted = 0
        if isinstance(result, dict):
            deleted = int(result.get("delete_count", 0) or 0)
        logger.info("Deleted %d vectors for document %s", deleted, document_id)
        return deleted

    except Exception as e:
        logger.error("Milvus delete error for document %s: %s", document_id, e)
        return 0


def close_milvus() -> None:
    """Close Milvus connection."""
    global _client
    _client = None
    logger.info("Milvus connection released")
