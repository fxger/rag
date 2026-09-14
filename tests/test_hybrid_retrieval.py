"""Test hybrid retrieval functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import asyncio
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.sparse_retriever import SparseRetriever
from src.core.retrieval.vector_retriever import VectorRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MILVUS_URI = "http://192.168.86.132:19530"
COLLECTION_NAME = "enterprise_rag"


async def test_bm25_retrieval():
    """Test BM25 sparse retrieval."""
    logger.info("=== Testing BM25 Sparse Retrieval ===")
    
    try:
        # Sample documents for testing
        documents = [
            {"chunk_id": "1", "document_id": "doc1", "content": "公司2025年营收为5.2亿元，同比增长15%。"},
            {"chunk_id": "2", "document_id": "doc1", "content": "员工总数达到500人，分布在5个城市。"},
            {"chunk_id": "3", "document_id": "doc2", "content": "技术架构采用微服务设计，使用Kubernetes部署。"},
            {"chunk_id": "4", "document_id": "doc2", "content": "数据库使用PostgreSQL，缓存使用Redis。"},
            {"chunk_id": "5", "document_id": "doc3", "content": "HR政策规定每年享有15天年假。"},
        ]
        
        # Initialize sparse retriever
        sparse_retriever = SparseRetriever()
        await sparse_retriever.build_index(documents)
        
        # Test queries
        test_queries = [
            "公司营收",
            "员工数量",
            "技术架构",
            "年假政策",
        ]
        
        for query in test_queries:
            results = await sparse_retriever.search(query, top_k=2)
            logger.info(f"\nQuery: {query}")
            for r in results:
                logger.info(f"  Score: {r.score:.3f} | Content: {r.content[:30]}...")
        
        sparse_retriever.clear_index()
        logger.info("\n✓ BM25 retrieval test completed")
        
    except Exception as e:
        logger.error(f"BM25 test failed: {e}")
        raise


async def test_hybrid_retrieval():
    """Test hybrid retrieval with Milvus."""
    logger.info("\n=== Testing Hybrid Retrieval ===")
    
    try:
        # Initialize hybrid retriever
        retriever = HybridRetriever(
            collection_name=COLLECTION_NAME,
            enable_reranker=False,
        )
        
        # Connect to Milvus and get sample data
        import pymilvus
        from pymilvus import connections, Collection
        
        connections.connect("default", uri=MILVUS_URI)
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        # Get sample documents
        results = collection.query(
            expr="",
            output_fields=["chunk_id", "document_id", "content", "metadata"],
            limit=100,
        )
        
        if not results:
            logger.warning("No documents found in Milvus, using sample data")
            results = [
                {"chunk_id": "1", "document_id": "doc1", "content": "公司2025年营收为5.2亿元"},
                {"chunk_id": "2", "document_id": "doc1", "content": "员工总数达到500人"},
                {"chunk_id": "3", "document_id": "doc2", "content": "技术架构采用微服务设计"},
            ]
        
        # Build BM25 index
        logger.info(f"Building BM25 index with {len(results)} documents...")
        await retriever.build_sparse_index(results)
        
        # Test hybrid search
        test_queries = [
            "公司财务状况",
            "员工管理",
            "技术架构",
        ]
        
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            
            # Get embedding for query
            from src.api.v1.query import get_qwen_embedding_sync
            embedding = get_qwen_embedding_sync(query)
            
            # Hybrid search
            result = await retriever.retrieve(
                query=query,
                top_k=3,
                embedding=embedding,
                fusion_method="rrf",
            )
            
            logger.info(f"  Results: {len(result.results)} (candidates: {result.total_candidates})")
            logger.info(f"  Time: {result.retrieval_time_ms}ms")
            
            for i, r in enumerate(result.results, 1):
                logger.info(f"  {i}. Score: {r.score:.3f} | Source: {r.source_type} | Content: {r.content[:30]}...")
        
        retriever.close()
        logger.info("\n✓ Hybrid retrieval test completed")
        
    except Exception as e:
        logger.error(f"Hybrid test failed: {e}")
        raise


async def main():
    """Run all tests."""
    logger.info("Starting retrieval tests...")
    
    # Test BM25
    await test_bm25_retrieval()
    
    # Test hybrid
    await test_hybrid_retrieval()
    
    logger.info("\n=== All tests completed ===")


if __name__ == "__main__":
    asyncio.run(main())
