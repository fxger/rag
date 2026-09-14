"""Hybrid retriever orchestrating the complete retrieval pipeline."""
import asyncio
import logging
import time
from typing import Any

from .models import SearchResult, QueryAnalysis, RetrievalResult
from .fusion import ResultFusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Main hybrid retriever orchestrating the retrieval pipeline.
    
    Uses singleton pattern for heavy resources (retrievers, analyzer, reranker).
    
    Pipeline: analyze → parallel search → dynamic weights → fuse → rerank
    
    Attributes:
        dense_retriever: Vector-based dense retriever (singleton).
        sparse_retriever: BM25-based sparse retriever (singleton).
        fusion: Result fusion strategies.
        reranker: Cross-encoder reranker (singleton).
        analyzer: Query analyzer (singleton).
    """

    def __init__(
        self,
        collection_name: str = "document_chunks",
        enable_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fusion_method: str = "adaptive",
        sparse_boost_threshold: float = 2.0,
    ):
        """Initialize the hybrid retriever.
        
        Args:
            collection_name: Name of the Milvus collection for vector search.
            enable_reranker: Whether to enable cross-encoder reranking.
            reranker_model: Cross-encoder model name.
            fusion_method: Fusion method ('adaptive', 'rrf', or 'weighted').
            sparse_boost_threshold: Threshold for sparse quality boost.
        """
        # Use singleton instances for heavy resources
        from ..singleton import (
            get_vector_retriever,
            get_sparse_retriever,
            get_query_analyzer,
        )
        
        self.dense_retriever = get_vector_retriever(collection_name=collection_name)
        self.sparse_retriever = get_sparse_retriever()
        self.analyzer = get_query_analyzer()
        self.fusion = ResultFusion()
        
        # Reranker config (lazy-loaded via singleton)
        self.enable_reranker = enable_reranker
        self.reranker_model = reranker_model
        self._reranker = None
        
        # Fusion config
        self.fusion_method = fusion_method
        self.sparse_boost_threshold = sparse_boost_threshold
        
        logger.info(
            f"Initialized HybridRetriever "
            f"(reranker={enable_reranker}, fusion={fusion_method}, "
            f"collection={collection_name})"
        )

    @property
    def reranker(self):
        """Lazy-load reranker via singleton."""
        if self._reranker is None and self.enable_reranker:
            from ..singleton import get_reranker
            self._reranker = get_reranker(model_name=self.reranker_model)
        return self._reranker

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        embedding: list[float] | None = None,
        filters: dict[str, Any] | None = None,
        fusion_method: str | None = None,
        rerank: bool = True,
        rerank_top_n: int | None = None,
    ) -> RetrievalResult:
        """Execute the full hybrid retrieval pipeline.
        
        Pipeline: analyze → parallel search → dynamic weights → fuse → rerank
        
        Args:
            query: User query string.
            top_k: Final number of results to return.
            embedding: Pre-computed query embedding (optional).
            filters: Optional filters for vector search.
            fusion_method: Override fusion method.
            rerank: Whether to apply reranking.
            rerank_top_n: Number of results after reranking.
            
        Returns:
            RetrievalResult with ranked results, query analysis, and weights.
        """
        start_time = time.time()
        
        if fusion_method is None:
            fusion_method = self.fusion_method
        if rerank_top_n is None:
            rerank_top_n = min(5, top_k)
        
        # Step 1: Analyze the query
        query_analysis = self.analyzer.analyze(query)
        logger.info(f"Query type: {query_analysis.query_type}")
        
        # Step 2: Execute searches in parallel
        dense_results: list[SearchResult] = []
        sparse_results: list[SearchResult] = []
        total_candidates = 0
        
        search_tasks = []
        
        if embedding is not None:
            search_tasks.append(
                self._dense_search(embedding, top_k * 2, filters)
            )
        
        search_tasks.append(
            self._sparse_search(query_analysis.original_query, top_k * 2)
        )
        
        if search_tasks:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    logger.error(f"Search task {i} failed: {result}")
                elif isinstance(result, list):
                    if embedding is not None and i == 0:
                        dense_results = result
                    else:
                        sparse_results = result
                    total_candidates += len(result)
        
        # Step 3: Calculate dynamic weights
        sparse_scores = [r.score for r in sparse_results] if sparse_results else None
        
        if fusion_method == "adaptive":
            weights = self.analyzer.get_dynamic_weights(
                query_analysis,
                sparse_scores=sparse_scores,
                sparse_boost_threshold=self.sparse_boost_threshold,
            )
        else:
            weights = self.analyzer.get_retrieval_weights(query_analysis.query_type)
        
        logger.info(f"Fusion weights: {weights}")
        
        # Step 4: Fuse results
        if dense_results and sparse_results:
            if fusion_method in ("adaptive", "weighted"):
                fused_results = self.fusion.weighted_fusion(
                    [dense_results, sparse_results],
                    [weights["dense"], weights["sparse"]],
                )
            else:  # rrf
                fused_results = self.fusion.rrf_fusion(
                    [dense_results, sparse_results],
                    k=60,
                )
        elif dense_results:
            fused_results = self.fusion.normalize_scores(dense_results)
        elif sparse_results:
            fused_results = self.fusion.normalize_scores(sparse_results)
        else:
            fused_results = []
        
        # Step 5: Rerank if enabled
        reranked = False
        if rerank and self.reranker and fused_results:
            rerank_start = time.time()
            final_results = await self.reranker.rerank(
                query,
                fused_results,
                top_n=rerank_top_n,
            )
            reranked = True
            # Imported lazily: metrics are a cross-cutting concern, and a
            # module-level import would make the retrieval layer depend on
            # the API layer from the moment it is imported.
            # TODO: move the metric definitions to src/core/observability so
            # the dependency points in the right direction.
            from src.api.metrics import RAG_RERANK_DURATION
            RAG_RERANK_DURATION.observe(time.time() - rerank_start)
        else:
            final_results = fused_results[:top_k]
            for rank, result in enumerate(final_results, start=1):
                result.rank = rank
        
        retrieval_time_ms = int((time.time() - start_time) * 1000)
        
        retrieval_result = RetrievalResult(
            results=final_results,
            query_analysis=query_analysis,
            weights=weights,
            total_candidates=total_candidates,
            retrieval_time_ms=retrieval_time_ms,
            reranked=reranked,
        )
        
        logger.info(
            f"Retrieval completed: {len(final_results)} results "
            f"from {total_candidates} candidates in {retrieval_time_ms}ms "
            f"(reranked={reranked}, weights={weights})"
        )
        
        return retrieval_result

    async def _dense_search(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Execute dense vector search."""
        try:
            return await self.dense_retriever.search(
                query_embedding=embedding,
                top_k=top_k,
                filters=filters,
            )
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    async def _sparse_search(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Execute sparse BM25 search."""
        try:
            return await self.sparse_retriever.search(
                query=query,
                top_k=top_k,
            )
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []

    async def build_sparse_index(self, documents: list[dict[str, Any]]) -> None:
        """Build the sparse BM25 index."""
        await self.sparse_retriever.build_index(documents)
        logger.info(f"Built sparse index with {len(documents)} documents")

    def close(self) -> None:
        """Close all connections and release resources."""
        try:
            self.dense_retriever.close()
            logger.info("Closed HybridRetriever connections")
        except Exception as e:
            logger.warning(f"Error closing HybridRetriever: {e}")

    @property
    def stats(self) -> dict:
        """Get retriever statistics."""
        return {
            "reranker": {
                "enabled": self.enable_reranker,
                "available": self.reranker.is_available() if self.reranker else False,
                "model": self.reranker_model,
                "cache": self.reranker.cache_stats if self.reranker else None,
            },
            "fusion": {
                "method": self.fusion_method,
                "sparse_boost_threshold": self.sparse_boost_threshold,
            },
        }
