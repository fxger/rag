"""End-to-end RAG service — the single orchestration layer of the query path.

接入说明（2026-10-19）
----------------------
此前本文件与 ``src/api/v1/query.py`` 是**两套并行实现**：路由里自建了一整条流水线
（嵌入、检索、缓存、LLM 调用），本文件的 RAGService 从未被任何调用方触达。
两套实现并存的代价是——链路上的每一处改动都要改两遍，而只改一遍的那次不会生效。

本次把生产链路收敛进 RAGService，路由层退化为「协议转换 + SSE 事件组装」：

  检索器    SingletonManager 单例（原 RAGService 自己 new 一个 HybridRetriever，
            等于绕开了全局复用与连接池）
  检索缓存  SearchCache，复用 MultiLevelCache 的读写，叠加 Prometheus 上报
  防雪崩    缓存未命中时经 CacheAntiAvalanche 做 single-flight + TTL 抖动
  查询重写  QueryRewriter，由 settings.ENABLE_QUERY_REWRITE 控制，默认关闭
  上下文    ContextBuilder（token 预算 + [Source N] 编号）；给前端的 SourceDocument
            仍走原结构，避免改动 SSE 契约
  生成      AnswerGenerator，外层包三态熔断 + 指标，system prompt 注入中文 RAG 提示词
  降级      混合检索失败时退回纯向量检索（Milvus 连接池）
"""
import asyncio
import hashlib
import json
import logging
import threading
import time
from typing import Any, AsyncGenerator, Optional

from src.api.middleware.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)
from src.api.metrics import (
    AI_GENERATION_TOKENS,
    AI_GENERATION_TOTAL,
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    CACHE_SIZE,
    RAG_GENERATION_DURATION,
    RAG_QUERY_DURATION,
    RAG_QUERY_TOTAL,
    RAG_RETRIEVAL_DURATION,
    RAG_RETRIEVAL_RESULTS,
)
from src.api.schemas.query import SourceDocument
from src.config.settings import get_settings
from src.core.cache.anti_avalanche import CacheAntiAvalanche
from src.core.cache.multi_level_cache import MultiLevelCache
from src.core.embedding.embedder import EmbeddingGenerator, IntelligentEmbeddingCache
from src.core.generation.context_builder import ContextBuilder
from src.core.generation.generator import AnswerGenerator, GenerationResult
from src.core.retrieval.models import SearchResult
from src.core.singleton import get_hybrid_retriever

logger = logging.getLogger(__name__)

_settings = get_settings()

# 三态熔断器：连续 5 次失败 → OPEN；30s 后 → HALF_OPEN；半开下连续 2 次成功 → CLOSED。
#
# CircuitBreaker 是纯状态机（``await breaker.call(fn, ...)``），不是 HTTP 中间件，
# 所以挂在调用点而不是 app.add_middleware：下游抖动时该熔断的是这一条依赖，
# 而不是把所有用户一起拒掉。
_llm_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    success_threshold=2,
    name="deepseek",
)


def get_llm_breaker() -> CircuitBreaker:
    """Return the process-wide LLM circuit breaker (single source of truth).

    ``src/api/metrics.py`` reads its state from here.  It used to reach into
    ``query._llm_breaker``, which broke once the breaker moved into this
    module — the metric silently fell back to zero.
    """
    return _llm_breaker


# AnswerGenerator 默认的 RAG_SYSTEM_PROMPT 是英文，而本项目语料与用户查询都是中文。
# 这里沿用路由层此前实际在用的中文提示词，改为注入给 generator，行为保持一致。
RAG_SYSTEM_PROMPT_ZH = """你是一个企业知识库助手。请根据以下检索到的文档片段回答用户的问题。

要求：
1. 基于检索到的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、简洁、专业
4. 引用具体的文档内容作为依据"""


# ── 检索结果缓存 ─────────────────────────────────────────────────
class SearchCache:
    """搜索结果 L1+L2 双层缓存。

    L1/L2 的读写交给 ``MultiLevelCache``，这里只叠加两件事：把原始 query 归一化成
    定长 key（原始 query 可能很长且含任意字符，不适合直接当 Redis key），以及
    把命中情况报到 Prometheus。

    此前这里与 MultiLevelCache 是两份几乎相同的实现，改一处不会同步另一处。
    """

    def __init__(self, l1_maxsize: int = 200, l1_ttl: int = 60, l2_ttl: int = 300):
        self.cache = MultiLevelCache(
            l1_maxsize=l1_maxsize,
            l1_ttl=l1_ttl,
            l2_ttl=l2_ttl,
            prefix="rag:search:",
        )
        self._hits = {"l1": 0, "l2": 0, "miss": 0}

    @property
    def l2_ttl(self) -> int:
        """Default TTL handed to the anti-avalanche rebuild path."""
        return self.cache.l2_ttl

    def key_for(self, query: str, top_k: int, rerank: bool = False) -> str:
        """Build the cache key.

        ``rerank`` 必须在 key 里：非流式端点精排、流式端点不精排，两个端点共用
        同一个检索器实例，若 key 不区分就会把未精排的结果当成精排结果返回。
        """
        return hashlib.sha256(f"{query}:{top_k}:{int(rerank)}".encode()).hexdigest()

    async def get(self, query: str, top_k: int, rerank: bool = False) -> list[dict] | None:
        tier, value = await self.cache.get_with_source(self.key_for(query, top_k, rerank))
        self._hits[tier] += 1

        if value is None:
            CACHE_MISSES_TOTAL.labels(cache_type="search").inc()
            return None

        CACHE_HITS_TOTAL.labels(cache_type="search").inc()
        CACHE_SIZE.labels(cache_type="search").set(len(self.cache.l1))
        return value

    async def set(
        self,
        query: str,
        top_k: int,
        results: list[dict],
        rerank: bool = False,
        ttl: int | None = None,
    ) -> None:
        await self.cache.set(self.key_for(query, top_k, rerank), results, ttl=ttl)
        CACHE_SIZE.labels(cache_type="search").set(len(self.cache.l1))

    @property
    def stats(self) -> dict:
        total = sum(self._hits.values())
        return {
            "l1_hits": self._hits["l1"],
            "l2_hits": self._hits["l2"],
            "misses": self._hits["miss"],
            "hit_rate": (self._hits["l1"] + self._hits["l2"]) / max(total, 1),
        }


# 进程级共享：缓存命中率只有跨请求累积才有意义，不能每个请求新建一份。
_search_cache = SearchCache()
_embedding_cache = IntelligentEmbeddingCache()


class RAGService:
    """Main RAG service — orchestrates the full retrieval-augmented generation pipeline.

    Coordinates:
    1. Query embedding generation (with cache)
    2. Optional query rewriting / HyDE (off by default)
    3. Hybrid retrieval (vector + keyword) with cache and anti-avalanche
    4. Context assembly with source citations
    5. LLM answer generation (sync or streaming), guarded by a circuit breaker
    6. Query logging for analytics

    Every collaborator is injectable so tests can substitute fakes; the defaults
    are the production singletons.
    """

    def __init__(
        self,
        retriever: Any = None,
        generator: Any = None,
        context_builder: Any = None,
        embedder: Any = None,
        embedding_cache: Any = None,
        search_cache: Any = None,
        anti_avalanche: Any = None,
        rewriter: Any = None,
    ):
        self.settings = get_settings()
        self.retriever = retriever if retriever is not None else self._default_retriever()
        self.generator = generator if generator is not None else AnswerGenerator()
        self.context_builder = context_builder if context_builder is not None else ContextBuilder()
        self.embedder = embedder if embedder is not None else EmbeddingGenerator()
        self.embedding_cache = embedding_cache if embedding_cache is not None else _embedding_cache
        self.search_cache = search_cache if search_cache is not None else _search_cache
        self.anti_avalanche = anti_avalanche if anti_avalanche is not None else CacheAntiAvalanche(
            lock_ttl=self.settings.CACHE_LOCK_TTL,
            jitter_pct=self.settings.CACHE_TTL_JITTER_PCT,
        )
        self.rewriter = rewriter

        self.collection_name = self.settings.MILVUS_COLLECTION
        self._query_log: list[dict] = []

    def _default_retriever(self) -> Any:
        """Build the shared hybrid retriever.

        ``enable_reranker=True`` only permits the reranker to be lazily loaded;
        whether it actually runs is decided per call via ``retrieve(rerank=...)``.

        ``reranker_model`` comes from settings — both ``Reranker`` and
        ``HybridRetriever`` hard-code an English cross-encoder otherwise, and
        that model measurably *hurts* on this project's Chinese corpus.
        """
        return get_hybrid_retriever(
            collection_name=self.settings.MILVUS_COLLECTION,
            enable_reranker=True,
            reranker_model=self.settings.RERANKER_MODEL,
            fusion_method=self.settings.RETRIEVAL_FUSION_METHOD,
        )

    # ── 1. 查询重写（可选） ──────────────────────────────────────
    @property
    def _rewrite_enabled(self) -> bool:
        return bool(self.settings.ENABLE_QUERY_REWRITE)

    def _rewriter_instance(self) -> Any:
        if self.rewriter is not None:
            return self.rewriter
        # 懒加载：只有开关打开时才构造，避免为一条不走的分支建 OpenAI client。
        from src.core.generation.query_rewriter import QueryRewriter

        self.rewriter = QueryRewriter()
        return self.rewriter

    def _should_rewrite(self, query: str) -> bool:
        """Rewrite only short queries by default.

        长查询本身信息量足够，BM25 侧也已按词匹配；短查询才是词汇鸿沟最明显、
        重写收益最大的地方。max_chars<=0 表示不限制。
        """
        if not self._rewrite_enabled:
            return False
        limit = self.settings.QUERY_REWRITE_MAX_CHARS
        return limit <= 0 or len(query.strip()) <= limit

    async def expand_query(self, query: str) -> list[str]:
        """Return the query variants to retrieve with.

        Always returns at least ``[query]``.  When rewriting is disabled this is
        a no-op wrapper, so callers need no branching.
        """
        if not self._should_rewrite(query):
            return [query]

        try:
            rewriter = self._rewriter_instance()
            mode = (self.settings.QUERY_REWRITE_MODE or "rewrite").lower()

            if mode == "hyde":
                hypothetical = await rewriter.hyde(query)
                return [query, hypothetical] if hypothetical and hypothetical != query else [query]

            variants = await rewriter.rewrite(query, self.settings.QUERY_REWRITE_VARIANTS)
            return variants or [query]
        except Exception as e:
            logger.warning("Query rewrite failed, using original query: %s", e)
            return [query]

    # ── 2. Embedding ────────────────────────────────────────────
    async def embed_query(self, text: str) -> list[float]:
        """Get the query vector, hitting the multi-level cache first."""
        cached = await self.embedding_cache.get(text)
        if cached is not None:
            CACHE_HITS_TOTAL.labels(cache_type="embedding").inc()
            return cached

        CACHE_MISSES_TOTAL.labels(cache_type="embedding").inc()

        try:
            embedding = await self.embedder.generate_single(text)
        except Exception as e:
            logger.error("Qwen embedding failed: %s", e)
            return [0.0] * self.settings.EMBEDDING_DIMENSION

        await self.embedding_cache.set(text, embedding)
        CACHE_SIZE.labels(cache_type="embedding").set(len(self.embedding_cache.l1))
        return embedding

    # ── 3. 检索 ─────────────────────────────────────────────────
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[dict]:
        """Hybrid retrieval (dense + BM25) with cache and single-flight rebuild.

        Args:
            rerank: Whether to run cross-encoder reranking on the fused results.
                The non-streaming endpoint enables it; the streaming one does
                not (time-to-first-token matters more there).  Part of the
                cache key.
        """
        cached = await self.search_cache.get(query, top_k, rerank)
        if cached is not None:
            logger.debug("Search cache hit for query: %s", query[:50])
            RAG_RETRIEVAL_RESULTS.observe(len(cached))
            return cached

        async def _build() -> list[dict]:
            return await self._retrieve_uncached(query, top_k, rerank)

        if self.settings.ENABLE_CACHE_ANTI_AVALANCHE:
            # 热点 key 在缓存过期瞬间会有大量并发同时 miss；只有抢到锁的那个请求
            # 去查 Milvus，其余等待并读缓存。
            key = self.search_cache.key_for(query, top_k, rerank)
            try:
                results = await self.anti_avalanche.get_with_lock(
                    key,
                    _build,
                    ttl=self.search_cache.l2_ttl,
                    cache=self.search_cache.cache,
                )
            except Exception as e:
                logger.warning("Anti-avalanche path failed, retrieving directly: %s", e)
                results = await _build()
        else:
            results = await _build()

        RAG_RETRIEVAL_RESULTS.observe(len(results))
        return results

    async def _retrieve_uncached(self, query: str, top_k: int, rerank: bool) -> list[dict]:
        """Run one hybrid retrieval and cache the result."""
        retrieval_start = time.time()

        try:
            queries = await self.expand_query(query)
            embedding = await self.embed_query(query)

            merged: list[dict] = []
            for variant in queries:
                result = await self.retriever.retrieve(
                    query=variant,
                    top_k=top_k,
                    embedding=embedding,
                    fusion_method=self.settings.RETRIEVAL_FUSION_METHOD,
                    rerank=rerank,
                )
                merged.extend(self._to_dicts(result.results))

            # 多变体检索会把同一 chunk 重复召回；按 chunk_id 去重，保留最高分。
            search_results = self._dedupe(merged)[:top_k]

            RAG_RETRIEVAL_DURATION.observe(time.time() - retrieval_start)
            await self.search_cache.set(query, top_k, search_results, rerank)

            logger.info(
                "Hybrid search completed: %d results (variants: %d, rerank: %s) in %dms",
                len(search_results),
                len(queries),
                rerank,
                result.retrieval_time_ms,
            )
            return search_results

        except Exception as e:
            logger.error("Hybrid search failed: %s", e)
            # 降级到纯向量搜索
            return await self.vector_search_fallback(query, top_k)

    @staticmethod
    def _to_dicts(results: list[SearchResult]) -> list[dict]:
        return [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "content": r.content,
                "score": r.score,
                "rank": r.rank,
                "metadata": r.metadata,
                "source_type": r.source_type,
            }
            for r in results
        ]

    @staticmethod
    def _dedupe(results: list[dict]) -> list[dict]:
        """Collapse repeated chunks across query variants, keeping the best score."""
        best: dict[str, dict] = {}
        for r in results:
            key = r.get("chunk_id") or r.get("content", "")[:64]
            existing = best.get(key)
            if existing is None or r.get("score", 0.0) > existing.get("score", 0.0):
                best[key] = r

        ordered = sorted(best.values(), key=lambda r: r.get("score", 0.0), reverse=True)
        for i, r in enumerate(ordered, 1):
            r["rank"] = i
        return ordered

    # ── 3b. 降级：纯向量检索 ────────────────────────────────────
    class _MilvusSession:
        """Borrow a Milvus collection handle from the pool; return it on exit.

        Connections come from ``SingletonManager.get_milvus_pool`` (reuse plus
        health probing), replacing the previous pattern of calling
        ``connections.connect()`` on every request.
        """

        _pool = None
        _lock = threading.Lock()

        @classmethod
        def _get_pool(cls, collection_name: str):
            if cls._pool is None:
                with cls._lock:
                    if cls._pool is None:
                        from src.core.concurrent_singleton import SingletonManager

                        cls._pool = SingletonManager.get_milvus_pool(
                            collection_name=collection_name,
                            min_size=1,
                            max_size=8,
                        )
            return cls._pool

        def __init__(self, collection_name: str):
            self.collection_name = collection_name
            self._conn = None

        def __enter__(self):
            self._pool = self._get_pool(self.collection_name)
            self._conn = self._pool.acquire()
            return self._conn

        def __exit__(self, *exc_info) -> bool:
            self._pool.release(self._conn)
            return False

    def _vector_search_sync(self, embedding: list[float], top_k: int) -> list[dict]:
        """Run a Milvus vector search (called from a worker thread)."""
        with self._MilvusSession(self.collection_name) as collection:
            results = collection.search(
                data=[embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k,
                output_fields=["chunk_id", "document_id", "content", "metadata"],
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "chunk_id": hit.entity.get("chunk_id", ""),
                        "document_id": hit.entity.get("document_id", ""),
                        "content": hit.entity.get("content", ""),
                        "score": float(hit.score),
                        "rank": len(search_results) + 1,
                        "metadata": hit.entity.get("metadata", {}),
                        "source_type": "dense",
                    })

            return search_results

    async def vector_search_fallback(self, query: str, top_k: int) -> list[dict]:
        """向量搜索降级方案（Milvus 调用在线程池中执行，避免阻塞事件循环）。"""
        try:
            embedding = await self.embed_query(query)
            return await asyncio.to_thread(self._vector_search_sync, embedding, top_k)
        except Exception as e:
            logger.error("Vector search fallback failed: %s", e)
            return []

    # ── 4. 上下文 ───────────────────────────────────────────────
    def build_context(self, search_results: list[dict]) -> tuple[str, list[SourceDocument]]:
        """Build the LLM context and the source list returned to the client.

        Two outputs on purpose:
          · context  —— 交给 ContextBuilder，带 token 预算与 [Source N] 编号，
                        这一份是给模型看的；
          · sources  —— 沿用原有的 SourceDocument 结构，SSE 契约不变，
                        这一份是给前端看的。
        """
        if not search_results:
            return "", []

        as_results = [SearchResult(**r) for r in search_results]
        context, _ = self.context_builder.build_with_sources(
            query="",
            results=as_results,
            max_tokens=self.settings.LLM_MAX_TOKENS,
        )

        sources: list[SourceDocument] = []
        for r in search_results:
            content = r.get("content", "")
            sources.append(SourceDocument(
                chunk_id=r.get("chunk_id", ""),
                document_id=r.get("document_id", ""),
                content=content[:200] + "..." if len(content) > 200 else content,
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            ))

        return context, sources

    # ── 5. 生成 ─────────────────────────────────────────────────
    async def generate(self, query: str, context: str, sources: list[Any]) -> GenerationResult:
        """Generate an answer, guarded by the circuit breaker."""
        generation_start = time.time()

        async def _call() -> GenerationResult:
            return await self.generator.generate_with_sources(
                query=query,
                context=context,
                sources=[s.model_dump() for s in sources],
                system_prompt=RAG_SYSTEM_PROMPT_ZH,
            )

        try:
            result = await _llm_breaker.call(_call)
            AI_GENERATION_TOTAL.labels(status="success").inc()
            AI_GENERATION_TOKENS.inc(result.tokens_used or 0)
            RAG_GENERATION_DURATION.observe(time.time() - generation_start)
            return result

        except CircuitBreakerError as e:
            AI_GENERATION_TOTAL.labels(status="circuit_open").inc()
            logger.warning("DeepSeek circuit is OPEN, rejecting call: %s", e)
            return GenerationResult(
                answer="AI 服务暂时不可用（熔断保护中），请稍后重试。",
                sources=[s.model_dump() for s in sources],
                model=getattr(self.generator, "model", ""),
            )

        except Exception as e:
            AI_GENERATION_TOTAL.labels(status="error").inc()
            logger.error("Answer generation failed: %s", e)
            return GenerationResult(
                answer=f"AI 生成失败: {e}",
                sources=[s.model_dump() for s in sources],
                model=getattr(self.generator, "model", ""),
            )

    async def stream_generate(self, query: str, context: str) -> AsyncGenerator[str, None]:
        """Stream answer tokens, guarded by the circuit breaker.

        A streaming body is consumed lazily and cannot be wrapped by
        ``breaker.call()``, so the outcome is reported directly.
        """
        if _llm_breaker.state == CircuitState.OPEN:
            logger.warning("DeepSeek circuit is OPEN, rejecting stream call")
            AI_GENERATION_TOTAL.labels(status="circuit_open").inc()
            yield "[AI 服务暂时不可用（熔断保护中），请稍后重试。]"
            return

        generation_start = time.time()

        try:
            async for token in self.generator.generate_stream(
                query=query,
                context=context,
                system_prompt=RAG_SYSTEM_PROMPT_ZH,
            ):
                yield token

            _llm_breaker.record_success()
            usage = getattr(self.generator, "last_usage", {}) or {}
            AI_GENERATION_TOTAL.labels(status="success").inc()
            AI_GENERATION_TOKENS.inc(usage.get("total_tokens", 0))
            RAG_GENERATION_DURATION.observe(time.time() - generation_start)

        except Exception as e:
            _llm_breaker.record_failure()
            AI_GENERATION_TOTAL.labels(status="error").inc()
            logger.error("Streaming generation failed: %s", e)
            yield f"\n\n[AI 生成错误: {e}]"

    # ── 6. 对外主入口 ───────────────────────────────────────────
    async def query(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        rerank: bool | None = None,
    ) -> dict:
        """Execute the full non-streaming pipeline.

        Args:
            rerank: Override for reranking.  Defaults to ``settings.ENABLE_RERANK``.

        Returns:
            Dict with answer / sources / model / tokens_used / timings.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        total_start = time.monotonic()
        use_rerank = self.settings.ENABLE_RERANK if rerank is None else rerank

        retrieval_start = time.monotonic()
        search_results = await self.retrieve(query, top_k, rerank=use_rerank)
        retrieval_ms = int((time.monotonic() - retrieval_start) * 1000)

        context, sources = self.build_context(search_results)

        generation_start = time.monotonic()
        result = await self.generate(query, context, sources)
        generation_ms = int((time.monotonic() - generation_start) * 1000)

        total_ms = int((time.monotonic() - total_start) * 1000)
        await self._log_query(
            user_id=user_id,
            query=query,
            answer=result.answer,
            sources_count=len(sources),
            tokens_used=result.tokens_used,
            timings={
                "embedding_ms": 0,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
            },
            total_ms=total_ms,
            search_mode="hybrid+rerank" if use_rerank else "hybrid",
        )

        return {
            "answer": result.answer,
            "sources": sources,
            "model": result.model,
            "tokens_used": result.tokens_used,
            "retrieval_time_ms": retrieval_ms,
            "generation_time_ms": generation_ms,
            "total_time_ms": total_ms,
        }

    async def prepare_stream(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[SourceDocument], str]:
        """Retrieve and build the context for a streaming request.

        Split out from ``query_stream`` so the route can emit the ``metadata``
        SSE event (which carries the sources) *before* the first token arrives —
        that ordering is what lets the frontend render citations immediately.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        # 流式不做精排：优先保证首字节延迟。
        search_results = await self.retrieve(query, top_k, rerank=False)
        context, sources = self.build_context(search_results)
        return sources, context

    async def query_stream(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> AsyncGenerator[str, None]:
        """Streaming RAG query — yields answer tokens as generated."""
        total_start = time.monotonic()
        sources, context = await self.prepare_stream(query, top_k)

        answer_parts: list[str] = []
        async for token in self.stream_generate(query, context):
            answer_parts.append(token)
            yield token

        total_ms = int((time.monotonic() - total_start) * 1000)
        await self._log_query(
            user_id=user_id,
            query=query,
            answer="".join(answer_parts),
            sources_count=len(sources),
            tokens_used=0,  # 流式的 usage 由 generator.last_usage 单独上报
            timings={"total_ms": total_ms},
            total_ms=total_ms,
            search_mode="hybrid",
        )

    async def log_stream_query(
        self,
        user_id: str,
        query: str,
        answer: str,
        sources_count: int,
        total_ms: int,
    ) -> None:
        """Log a completed streaming query.

        路由层在 SSE ``done`` 事件之后调用：只有在答案完整收齐后才有东西可记。
        token 数取自 generator 最后一次流式的 usage。
        """
        usage = getattr(self.generator, "last_usage", {}) or {}
        await self._log_query(
            user_id=user_id,
            query=query,
            answer=answer,
            sources_count=sources_count,
            tokens_used=usage.get("total_tokens", 0),
            timings={"total_ms": total_ms},
            total_ms=total_ms,
            search_mode="hybrid",
        )

    # ── 运维接口 ────────────────────────────────────────────────
    async def build_sparse_index(self) -> int:
        """(Re)build the BM25 sparse index from every chunk in Milvus."""
        def _fetch_all() -> list[dict]:
            with self._MilvusSession(self.collection_name) as collection:
                return collection.query(
                    expr="",
                    output_fields=["chunk_id", "document_id", "content", "metadata"],
                    limit=10000,
                )

        results = await asyncio.to_thread(_fetch_all)
        if results:
            await self.retriever.build_sparse_index(results)
        return len(results)

    async def warm_hot_queries(self, limit: int | None = None) -> int:
        """Pre-warm the search cache with the most frequent recent queries.

        Cold start is the worst case for latency: the first request after a
        deploy pays for embedding + Milvus + BM25 with nothing cached.  Warming
        shifts that cost off the user-facing path.

        Reads the query log written by ``_log_query``; with an empty log (fresh
        deploy) this is a no-op, not an error.
        """
        from src.core.cache.warmer import CacheWarmer, top_queries_from_log

        queries = await top_queries_from_log(
            limit=limit or self.settings.CACHE_WARM_HOT_QUERIES,
        )
        if not queries:
            logger.info("Cache warm-up skipped: no query history yet")
            return 0

        warmer = CacheWarmer(
            default_ttl=self.settings.CACHE_WARM_TTL,
            cache=self.search_cache.cache,
        )
        top_k = self.settings.DEFAULT_TOP_K

        pairs = [
            (self.search_cache.key_for(q, top_k, False), self._make_warm_fetcher(q, top_k))
            for q in queries
        ]
        return await warmer.warm_up(pairs)

    def _make_warm_fetcher(self, query: str, top_k: int):
        """Build a zero-arg fetcher for the warmer."""
        async def _fetch() -> list[dict]:
            return await self._retrieve_uncached(query, top_k, rerank=False)

        return _fetch

    async def build_warm_payloads(self, limit: int | None = None) -> dict[str, list[dict]]:
        """Return ``{cache_key: results}`` for the periodic warmer.

        ``CacheWarmer.schedule_warm_up`` expects a fetcher returning a mapping,
        so this is the bridge between "which queries are hot" and "what the
        cache expects to be handed".
        """
        from src.core.cache.warmer import top_queries_from_log

        queries = await top_queries_from_log(
            limit=limit or self.settings.CACHE_WARM_HOT_QUERIES,
        )
        top_k = self.settings.DEFAULT_TOP_K

        payload: dict[str, list[dict]] = {}
        for q in queries:
            try:
                payload[self.search_cache.key_for(q, top_k, False)] = (
                    await self._retrieve_uncached(q, top_k, rerank=False)
                )
            except Exception as e:
                logger.warning("Warm-up fetch failed for %r: %s", q, e)
        return payload

    def pipeline_features(self) -> list[str]:
        """Describe the active pipeline (drives GET /search-mode)."""
        return [
            "Dense retrieval: Semantic similarity via embeddings",
            "Sparse retrieval: Keyword matching via BM25",
            f"Adaptive fusion: {self.settings.RETRIEVAL_FUSION_METHOD}",
            "Caching: L1 (in-memory) + L2 (Redis)",
            "Reranker: Cross-Encoder (lazy-loaded with cache)",
            f"Anti-avalanche: {'on' if self.settings.ENABLE_CACHE_ANTI_AVALANCHE else 'off'}",
            f"Cache warmer: {'on' if self.settings.ENABLE_CACHE_WARMER else 'off'}",
            f"Query rewrite: {'on' if self.settings.ENABLE_QUERY_REWRITE else 'off'}",
        ]

    def history(self, user_id: str, limit: int = 10) -> list[dict]:
        """Recent queries for a user (in-process; resets on restart)."""
        user_queries = [q for q in self._query_log if q.get("user_id") == user_id]
        return user_queries[-limit:]

    @property
    def cache_stats(self) -> dict:
        return {"search_cache": self.search_cache.stats}

    # ── 日志 ────────────────────────────────────────────────────
    async def _log_query(
        self,
        user_id: str,
        query: str,
        answer: str,
        sources_count: int,
        tokens_used: int,
        timings: dict,
        total_ms: int,
        search_mode: str,
    ) -> None:
        """Log query for analytics and debugging."""
        self._query_log.append({
            "user_id": user_id,
            "query": query,
            "latency_ms": total_ms,
            "search_mode": search_mode,
        })
        # 无界增长会吃内存，只保留最近的记录。
        if len(self._query_log) > 1000:
            del self._query_log[:-500]

        try:
            from src.config.redis import get_cache_redis

            redis = await get_cache_redis()
            log_entry = {
                "user_id": user_id,
                "query": query[:500],  # Truncate for storage
                "answer_length": len(answer),
                "sources_cited": sources_count,
                "tokens_used": tokens_used,
                "timings": timings,
                "total_ms": total_ms,
                "search_mode": search_mode,
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


# 进程级单例：让 history / 缓存命中率跨请求累积。
_service: Optional[RAGService] = None
_service_lock = threading.Lock()


def get_rag_service() -> RAGService:
    """Return the process-wide RAGService."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = RAGService()
    return _service


__all__ = [
    "RAGService",
    "SearchCache",
    "get_llm_breaker",
    "get_rag_service",
]
