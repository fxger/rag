"""Milvus vector search implementation for dense retrieval."""
import logging
import time
from typing import Any

from pymilvus import Collection, connections, utility

from src.config.settings import get_settings
from src.config.milvus import get_or_create_collection
from .models import SearchResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Dense vector retriever using Milvus.
    
    Performs similarity search using cosine similarity on document embeddings
    stored in Milvus vector database.
    
    Attributes:
        collection: Milvus collection instance.
        collection_name: Name of the Milvus collection.
        search_params: Parameters for Milvus search.
    """

    def __init__(self, collection_name: str = "document_chunks"):
        """Initialize the vector retriever.
        
        Args:
            collection_name: Name of the Milvus collection to search.
        """
        self.settings = get_settings()
        self.collection_name = collection_name
        self.collection: Collection | None = None
        self.search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16},
        }
        self._connect()
        self._load_collection()

    def _connect(self) -> None:
        """Establish connection to Milvus server."""
        try:
            milvus_host = getattr(self.settings, "MILVUS_HOST", "192.168.86.132")
            milvus_port = getattr(self.settings, "MILVUS_PORT", "19530")
            
            # Disconnect existing connection if any
            if connections.has_connection("default"):
                connections.disconnect("default")
            
            connections.connect(
                alias="default",
                host=milvus_host,
                port=milvus_port,
            )
            logger.info(f"Connected to Milvus at {milvus_host}:{milvus_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _load_collection(self) -> None:
        """Load the Milvus collection into memory."""
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                logger.info(f"Loaded collection: {self.collection_name}")
            else:
                logger.warning(f"Collection {self.collection_name} not found")
                self.collection = None
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            self.collection = None

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector embedding.
            top_k: Maximum number of results to return.
            filters: Optional filters to apply (e.g., document_id, metadata filters).
            
        Returns:
            List of SearchResult objects sorted by relevance score.
            
        Raises:
            RuntimeError: If collection is not loaded or search fails.
        """
        if self.collection is None:
            logger.warning("Collection not loaded, attempting to reload")
            self._load_collection()
            if self.collection is None:
                raise RuntimeError(f"Collection {self.collection_name} not available")

        start_time = time.time()
        
        try:
            # Build filter expression if filters provided
            expr = self._build_filter_expression(filters)
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=self.search_params,
                limit=top_k,
                expr=expr,
                output_fields=["chunk_id", "document_id", "content", "metadata"],
            )
            
            # Convert to SearchResult objects
            search_results = []
            for hits in results:
                for rank, hit in enumerate(hits):
                    entity = hit.entity
                    metadata = entity.get("metadata", {})
                    if isinstance(metadata, str):
                        import json
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    
                    search_result = SearchResult(
                        chunk_id=entity.get("chunk_id", ""),
                        document_id=entity.get("document_id", ""),
                        content=entity.get("content", ""),
                        score=hit.score,
                        rank=rank + 1,
                        metadata=metadata,
                        source_type="vector",
                    )
                    search_results.append(search_result)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Vector search completed: {len(search_results)} results in {elapsed_ms}ms"
            )
            
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    def _build_filter_expression(self, filters: dict[str, Any] | None) -> str | None:
        """Build Milvus filter expression from filter dictionary.
        
        Args:
            filters: Dictionary of field-value pairs to filter by.
            
        Returns:
            Milvus filter expression string or None.
        """
        if not filters:
            return None
        
        expressions = []
        for field, value in filters.items():
            if isinstance(value, str):
                expressions.append(f'{field} == "{value}"')
            elif isinstance(value, (int, float)):
                expressions.append(f'{field} == {value}')
            elif isinstance(value, list):
                if all(isinstance(v, str) for v in value):
                    values_str = ", ".join(f'"{v}"' for v in value)
                else:
                    values_str = ", ".join(str(v) for v in value)
                expressions.append(f'{field} in [{values_str}]')
        
        return " and ".join(expressions) if expressions else None

    def close(self) -> None:
        """Close the Milvus connection."""
        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
                logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus: {e}")
