"""Initialize BM25 index from existing Milvus data."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import asyncio
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MILVUS_URI = "http://192.168.86.132:19530"
COLLECTION_NAME = "enterprise_rag"


async def initialize_bm25_index():
    """Initialize BM25 index from Milvus data."""
    logger.info("Starting BM25 index initialization...")
    
    try:
        # Import pymilvus
        import pymilvus
        from pymilvus import connections, Collection
        
        # Connect to Milvus
        connections.connect("default", uri=MILVUS_URI)
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        # Get all documents
        logger.info("Fetching documents from Milvus...")
        results = collection.query(
            expr="",
            output_fields=["chunk_id", "document_id", "content", "metadata"],
            limit=10000,
        )
        
        if not results:
            logger.warning("No documents found in Milvus")
            return
        
        logger.info(f"Found {len(results)} documents in Milvus")
        
        # Initialize hybrid retriever
        retriever = HybridRetriever(
            collection_name=COLLECTION_NAME,
            enable_reranker=False,
        )
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        await retriever.build_sparse_index(results)
        
        logger.info("BM25 index initialization completed successfully!")
        
        # Test search
        logger.info("Testing hybrid search...")
        test_query = "公司营收"
        result = await retriever.retrieve(
            query=test_query,
            top_k=3,
            fusion_method="rrf",
        )
        
        logger.info(f"Test query: {test_query}")
        logger.info(f"Found {len(result.results)} results")
        for i, r in enumerate(result.results, 1):
            logger.info(f"  {i}. Score: {r.score:.3f} | Source: {r.source_type} | Content: {r.content[:50]}...")
        
        retriever.close()
        
    except Exception as e:
        logger.error(f"Failed to initialize BM25 index: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(initialize_bm25_index())
