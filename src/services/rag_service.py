"""End-to-end RAG service orchestrating retrieval, context building, and generation.

Main entry point for the RAG pipeline, coordinating embedding generation,
hybrid retrieval, context assembly, and answer generation with logging.
"""
import logging
import time
from typing import AsyncGenerator, Optional

from src.core.embedding.embedder import EmbeddingGenerator
from src.core.embedding import cache as embedding_cache
from src.core.generation.context_builder import ContextBuilder
from src.core.generation.generator import AnswerGenerator, GenerationResult
from src.core.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service — orchestrates the full retrieval-augmented generation pipeline.
    
    Coordinates:
    1. Query embedding generation (with cache)
    2. Hybrid retrieval (vector + keyword search)
    3. Context assembly with source citations
    4. LLM answer generation (sync or streaming)
    5. Query logging for analytics
    """

    def __init__(self):
        # Lazy imports to avoid circular dependencies
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()
        self.context_builder = ContextBuilder()
        self.embedder = EmbeddingGenerator()

    async def query(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
    ) -> dict:
        """Execute full RAG pipeline: embed → retrieve → build context → generate.
        
        Args:
            query: User question to answer.
            user_id: User identifier for logging and personalization.
            top_k: Number of search results to retrieve (default: 10).
            
        Returns:
            Dictionary containing:
                - answer: Generated answer text
                - sources: List of cited source references
                - model: LLM model used
                - tokens_used: Total tokens consumed
                - retrieval_time_ms: Retrieval latency
                - generation_time_ms: Generation latency
                - total_time_ms: End-to-end latency
                
        Raises:
            ValueError: If query is empty.
            Exception: On pipeline failures.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        total_start = time.monotonic()
        timings = {}

        # Step 1: Generate query embedding (with cache check)
        embed_start = time.monotonic()
        query_hash = EmbeddingGenerator.text_hash(query)
        cached_embedding = await embedding_cache.get(query_hash)

        if cached_embedding:
            query_embedding = cached_embedding
            logger.info("Using cached embedding for query")
        else:
            query_embedding = await self.embedder.generate_single(query)
            await embedding_cache.set(query_hash, query_embedding)
            logger.info("Generated and cached new embedding")

        timings["embedding_ms"] = int((time.monotonic() - embed_start) * 1000)

        # Step 2: Hybrid retrieval
        #
        # ``retrieve()`` returns a RetrievalResult wrapper, but the context
        # builder expects a plain list of SearchResult.  Unwrap here so the
        # success and failure paths hand downstream the same type.
        # Previously the success path passed the wrapper (which has no
        # __len__) and the error path passed [], so the TypeError raised by
        # len() was swallowed by the except clause below and the whole
        # pipeline quietly degraded to "No relevant documents found."
        retrieval_start = time.monotonic()
        try:
            retrieval = await self.retriever.retrieve(
                query=query,
                embedding=query_embedding,
                top_k=top_k,
            )
            results = retrieval.results
            timings["retrieval_ms"] = int((time.monotonic() - retrieval_start) * 1000)
            logger.info("Retrieved %d results", len(results))
        except Exception as e:
            logger.error("Retrieval failed: %s", e)
            timings["retrieval_ms"] = int((time.monotonic() - retrieval_start) * 1000)
            results = []

        # Step 3: Build context with sources
        context, sources = self.context_builder.build_with_sources(
            query=query,
            results=results,
            max_tokens=4000,
        )

        # Step 4: Generate answer
        generation_start = time.monotonic()
        try:
            generation_result = await self.generator.generate_with_sources(
                query=query,
                context=context,
                sources=sources,
            )
            timings["generation_ms"] = int((time.monotonic() - generation_start) * 1000)
        except Exception as e:
            logger.error("Generation failed: %s", e)
            timings["generation_ms"] = int((time.monotonic() - generation_start) * 1000)
            generation_result = GenerationResult(
                answer="I apologize, but I encountered an error generating an answer. Please try again.",
                sources=sources,
                model=self.generator.model,
            )

        # Step 5: Log query for analytics
        total_ms = int((time.monotonic() - total_start) * 1000)
        await self._log_query(
            user_id=user_id,
            query=query,
            answer=generation_result.answer,
            sources_count=len(generation_result.sources),
            tokens_used=generation_result.tokens_used,
            timings=timings,
            total_ms=total_ms,
        )

        return {
            "answer": generation_result.answer,
            "sources": generation_result.sources,
            "model": generation_result.model,
            "tokens_used": generation_result.tokens_used,
            "retrieval_time_ms": timings.get("retrieval_ms", 0),
            "generation_time_ms": timings.get("generation_ms", 0),
            "total_time_ms": total_ms,
        }

    async def query_stream(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
    ) -> AsyncGenerator[str, None]:
        """Streaming RAG query — yields answer tokens as generated.
        
        Performs full retrieval pipeline but streams generation tokens
        for real-time response display.
        
        Args:
            query: User question to answer.
            user_id: User identifier for logging.
            top_k: Number of search results to retrieve.
            
        Yields:
            Answer text tokens as they are generated.
            
        Raises:
            ValueError: If query is empty.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        total_start = time.monotonic()

        # Step 1: Generate query embedding
        query_hash = EmbeddingGenerator.text_hash(query)
        cached_embedding = await embedding_cache.get(query_hash)

        if cached_embedding:
            query_embedding = cached_embedding
        else:
            query_embedding = await self.embedder.generate_single(query)
            await embedding_cache.set(query_hash, query_embedding)

        # Step 2: Hybrid retrieval
        # Same unwrapping as the non-streaming path: the context builder
        # enumerates the results, and a RetrievalResult is not iterable.
        try:
            retrieval = await self.retriever.retrieve(
                query=query,
                embedding=query_embedding,
                top_k=top_k,
            )
            results = retrieval.results
        except Exception as e:
            logger.error("Retrieval failed during streaming query: %s", e)
            results = []

        # Step 3: Build context
        context, sources = self.context_builder.build_with_sources(
            query=query,
            results=results,
            max_tokens=4000,
        )

        # Step 4: Stream generation
        answer_parts: list[str] = []
        try:
            async for token in self.generator.generate_stream(
                query=query,
                context=context,
            ):
                answer_parts.append(token)
                yield token
        except Exception as e:
            logger.error("Streaming generation failed: %s", e)
            yield "\n\n[Error: Answer generation interrupted]"
            return

        # Step 5: Log after streaming completes
        total_ms = int((time.monotonic() - total_start) * 1000)
        full_answer = "".join(answer_parts)

        await self._log_query(
            user_id=user_id,
            query=query,
            answer=full_answer,
            sources_count=len(sources),
            tokens_used=0,  # Streaming doesn't report usage
            timings={"total_ms": total_ms},
            total_ms=total_ms,
        )

    async def _log_query(
        self,
        user_id: str,
        query: str,
        answer: str,
        sources_count: int,
        tokens_used: int,
        timings: dict,
        total_ms: int,
    ) -> None:
        """Log query for analytics and debugging.
        
        Args:
            user_id: User identifier.
            query: User query.
            answer: Generated answer.
            sources_count: Number of sources cited.
            tokens_used: Token usage.
            timings: Performance timing breakdown.
            total_ms: Total pipeline latency.
        """
        try:
            from src.config.redis import get_cache_redis
            import json

            redis = await get_cache_redis()
            log_entry = {
                "user_id": user_id,
                "query": query[:500],  # Truncate for storage
                "answer_length": len(answer),
                "sources_cited": sources_count,
                "tokens_used": tokens_used,
                "timings": timings,
                "total_ms": total_ms,
            }

            # Append to recent queries list (keep last 1000)
            await redis.lpush("rag:query_log", json.dumps(log_entry))
            await redis.ltrim("rag:query_log", 0, 999)

            logger.info(
                "Query logged: user=%s, sources=%d, tokens=%d, time=%dms",
                user_id,
                sources_count,
                tokens_used,
                total_ms,
            )
        except Exception as e:
            # Non-critical: don't fail the request if logging fails
            logger.warning("Query logging failed (non-critical): %s", e)
