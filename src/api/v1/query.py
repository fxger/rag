"""Query API endpoints — Hybrid Retrieval (Dense + BM25) with SSE streaming."""

import asyncio
import json
import threading
import time
import hashlib
import logging
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.middleware.auth import get_current_user
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
from src.api.schemas.common import ResponseModel
from src.api.schemas.query import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    StreamMetadata,
    StreamChunk,
    StreamDone,
    StreamError,
)
from src.config.settings import get_settings
from src.core.embedding.embedder import IntelligentEmbeddingCache
from src.core.singleton import get_hybrid_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])

_query_log: list[dict] = []

_settings = get_settings()

_llm_base = (_settings.OPENAI_BASE_URL or "https://api.deepseek.com").rstrip("/")
DEEPSEEK_URL = f"{_llm_base}/chat/completions"
QWEN_EMBED_URL = _settings.QWEN_EMBED_URL
QWEN_EMBED_MODEL = _settings.QWEN_EMBED_MODEL
COLLECTION_NAME = _settings.MILVUS_COLLECTION
# Milvus 连接统一由 SingletonManager.get_milvus_pool() 管理（见 _MilvusSession）。
# 这里刻意不再保留 URI 常量，避免又出现一条绕过连接池的建连路径。

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

# 全局混合检索器实例


def get_hybrid_retriever_instance():
    """获取全局混合检索器实例（单例模式）。

    ``enable_reranker=True`` only permits the reranker to be lazily loaded;
    whether it actually runs is decided per call via ``retrieve(rerank=...)``.
    The two endpoints deliberately differ: the non-streaming one reranks
    (the caller already waits for a full answer, so tens of extra
    milliseconds are acceptable), while the streaming one does not
    (time-to-first-token is what matters there).

    Because the two share a single retriever instance, the rerank flag is
    part of the search cache key — otherwise one endpoint would serve the
    other's results.
    """
    return get_hybrid_retriever(
        collection_name=COLLECTION_NAME,
        enable_reranker=True,
        fusion_method="adaptive",
    )


# ── 搜索结果缓存 ─────────────────────────────────────────────────
class SearchCache:
    """搜索结果 L1+L2 双层缓存"""
    
    def __init__(self, l1_maxsize: int = 200, l1_ttl: int = 60, l2_ttl: int = 300):
        from cachetools import TTLCache
        self.l1 = TTLCache(maxsize=l1_maxsize, ttl=l1_ttl)
        self.l2_ttl = l2_ttl
        self.prefix = "rag:search:"
        self._hits = {"l1": 0, "l2": 0, "miss": 0}
    
    def _make_key(self, query: str, top_k: int, rerank: bool = False) -> str:
        # ``rerank`` is deliberately part of the key.  The non-streaming and
        # streaming endpoints run different pipelines on purpose, and sharing
        # entries between them would serve un-reranked results as though they
        # had been reranked.
        return hashlib.sha256(f"{query}:{top_k}:{int(rerank)}".encode()).hexdigest()

    async def get(self, query: str, top_k: int, rerank: bool = False) -> list[dict] | None:
        key = self._make_key(query, top_k, rerank)

        # L1
        if key in self.l1:
            self._hits["l1"] += 1
            CACHE_HITS_TOTAL.labels(cache_type="search").inc()
            CACHE_SIZE.labels(cache_type="search").set(len(self.l1))
            return self.l1[key]

        # L2
        try:
            from src.config.redis import get_cache_redis
            redis = await get_cache_redis()
            raw = await redis.get(f"{self.prefix}{key}")
            if raw is not None:
                value = json.loads(raw)
                self.l1[key] = value
                self._hits["l2"] += 1
                CACHE_HITS_TOTAL.labels(cache_type="search").inc()
                CACHE_SIZE.labels(cache_type="search").set(len(self.l1))
                return value
        except Exception as e:
            logger.warning("Search cache L2 read error: %s", e)

        self._hits["miss"] += 1
        CACHE_MISSES_TOTAL.labels(cache_type="search").inc()
        return None

    async def set(self, query: str, top_k: int, results: list[dict], rerank: bool = False):
        key = self._make_key(query, top_k, rerank)

        self.l1[key] = results
        CACHE_SIZE.labels(cache_type="search").set(len(self.l1))

        try:
            from src.config.redis import get_cache_redis
            redis = await get_cache_redis()
            await redis.set(
                f"{self.prefix}{key}",
                json.dumps(results, default=str),
                ex=self.l2_ttl,
            )
        except Exception as e:
            logger.warning("Search cache L2 write error: %s", e)
    
    @property
    def stats(self) -> dict:
        total = sum(self._hits.values())
        return {
            "l1_hits": self._hits["l1"],
            "l2_hits": self._hits["l2"],
            "misses": self._hits["miss"],
            "hit_rate": (self._hits["l1"] + self._hits["l2"]) / max(total, 1),
        }


# 全局缓存实例
_search_cache = SearchCache()


# ── Embedding 生成 (多级缓存) ────────────────────────────────────
# 缓存实现来自 src/core/embedding/embedder.py::IntelligentEmbeddingCache：
# 文本标准化后做 L1 进程内 TTL + L2 Redis，并对近似重复的查询按
# SequenceMatcher(0.85) 做模糊匹配复用。
# 这里此前另写了一套 Redis 精确 hash 缓存（还硬编码了 Redis 地址），
# 与 embedder 里的实现重复；现统一走后者。
_embedding_cache = IntelligentEmbeddingCache()


def _request_embedding_sync(text: str) -> list[float]:
    """向 Qwen Embedding 服务发起同步请求（不带缓存，供线程池调用）。"""
    response = httpx.post(
        QWEN_EMBED_URL,
        json={"input": text, "model": QWEN_EMBED_MODEL},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


async def get_qwen_embedding(text: str) -> list[float]:
    """获取查询向量，优先命中多级缓存。

    未命中时在线程池中发起请求 —— 同步 httpx 调用会阻塞事件循环最长 30s。
    """
    cached = await _embedding_cache.get(text)
    if cached is not None:
        CACHE_HITS_TOTAL.labels(cache_type="embedding").inc()
        return cached

    CACHE_MISSES_TOTAL.labels(cache_type="embedding").inc()

    try:
        embedding = await asyncio.to_thread(_request_embedding_sync, text)
    except Exception as e:
        logger.error("Qwen embedding failed: %s", e)
        return [0.0] * 1024

    await _embedding_cache.set(text, embedding)
    CACHE_SIZE.labels(cache_type="embedding").set(len(_embedding_cache.l1))
    return embedding


# ── 混合检索 (带缓存) ──────────────────────────────────────────
async def hybrid_search_cached(
    query: str,
    top_k: int = 5,
    rerank: bool = False,
) -> list[dict]:
    """混合检索：Dense + BM25，带缓存。

    Args:
        rerank: 是否对融合结果做 Cross-Encoder 精排。非流式端点开启，
            流式端点关闭（首字节延迟敏感）。该标志参与缓存 key。
    """

    # 检查缓存
    cached = await _search_cache.get(query, top_k, rerank)
    if cached is not None:
        logger.debug("Search cache hit for query: %s", query[:50])
        RAG_RETRIEVAL_RESULTS.observe(len(cached))
        return cached

    # 缓存未命中，执行混合检索
    retrieval_start = time.time()
    try:
        retriever = get_hybrid_retriever_instance()

        embedding = await get_qwen_embedding(query)

        # 执行混合检索
        result = await retriever.retrieve(
            query=query,
            top_k=top_k,
            embedding=embedding,
            fusion_method="adaptive",
            rerank=rerank,
        )

        RAG_RETRIEVAL_DURATION.observe(time.time() - retrieval_start)
        RAG_RETRIEVAL_RESULTS.observe(len(result.results))

        # 转换为字典格式
        search_results = []
        for r in result.results:
            search_results.append({
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "content": r.content,
                "score": r.score,
                "rank": r.rank,
                "metadata": r.metadata,
                "source_type": r.source_type,
            })

        # 写入缓存
        await _search_cache.set(query, top_k, search_results, rerank)

        logger.info(
            "Hybrid search completed: %d results (dense: %d, sparse: %d, rerank: %s) in %dms",
            len(search_results),
            len([r for r in search_results if "dense" in r.get("source_type", "")]),
            len([r for r in search_results if "sparse" in r.get("source_type", "")]),
            rerank,
            result.retrieval_time_ms,
        )

        return search_results

    except Exception as e:
        logger.error("Hybrid search failed: %s", e)
        # 降级到纯向量搜索
        return await vector_search_fallback(query, top_k)


class _MilvusSession:
    """Borrow a Milvus collection handle from the pool; return it on exit.

    Connections are managed by ``SingletonManager.get_milvus_pool`` (reuse
    plus health probing), replacing the previous pattern of calling
    ``connections.connect()`` on every single request.
    """

    _pool = None
    _lock = threading.Lock()

    @classmethod
    def _get_pool(cls):
        if cls._pool is None:
            with cls._lock:
                if cls._pool is None:
                    from src.core.concurrent_singleton import SingletonManager

                    cls._pool = SingletonManager.get_milvus_pool(
                        collection_name=COLLECTION_NAME,
                        min_size=1,
                        max_size=8,
                    )
        return cls._pool

    def __enter__(self):
        self._pool = self._get_pool()
        self._conn = self._pool.acquire()
        return self._conn

    def __exit__(self, *exc_info) -> bool:
        self._pool.release(self._conn)
        return False


def _vector_search_sync(embedding: list[float], top_k: int) -> list[dict]:
    """Run a Milvus vector search (called from a worker thread)."""
    with _MilvusSession() as collection:
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


async def vector_search_fallback(query: str, top_k: int) -> list[dict]:
    """向量搜索降级方案（Milvus 调用在线程池中执行，避免阻塞事件循环）。"""
    try:
        embedding = await get_qwen_embedding(query)
        return await asyncio.to_thread(_vector_search_sync, embedding, top_k)
    except Exception as e:
        logger.error("Vector search fallback failed: %s", e)
        return []


# ── 构建搜索上下文 ──────────────────────────────────────────────
def build_search_context(search_results: list[dict]) -> tuple[str, list[SourceDocument]]:
    """构建 LLM 上下文和源文档列表"""
    if not search_results:
        return "", []
    
    context_parts = []
    sources = []
    
    for i, result in enumerate(search_results, 1):
        content = result.get("content", "")
        score = result.get("score", 0.0)
        source_type = result.get("source_type", "unknown")
        metadata = result.get("metadata", {})
        
        context_parts.append(f"[文档片段 {i}] (相关度: {score:.3f}, 来源: {source_type})\n{content}")
        
        sources.append(SourceDocument(
            chunk_id=result.get("chunk_id", ""),
            document_id=result.get("document_id", ""),
            content=content[:200] + "..." if len(content) > 200 else content,
            score=score,
            metadata=metadata,
        ))
    
    context = "\n\n".join(context_parts)
    return context, sources


# ── DeepSeek AI 生成 ────────────────────────────────────────────
async def call_deepseek(query: str, context: str) -> str:
    """调用 DeepSeek Chat 生成回答"""
    settings = get_settings()
    api_key = settings.OPENAI_API_KEY
    
    system_prompt = """你是一个企业知识库助手。请根据以下检索到的文档片段回答用户的问题。

要求：
1. 基于检索到的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、简洁、专业
4. 引用具体的文档内容作为依据

检索到的文档片段：
{context}"""
    
    messages = [
        {"role": "system", "content": system_prompt.format(context=context)},
        {"role": "user", "content": query},
    ]
    
    generation_start = time.time()

    def _request() -> dict:
        response = httpx.post(
            DEEPSEEK_URL,
            json={
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
            },
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()

    try:
        # The synchronous httpx call must not run on the event loop (it can
        # block for up to 60s), and breaker.call() needs an awaitable — so
        # it goes through a worker thread.
        result = await _llm_breaker.call(asyncio.to_thread, _request)

        usage = result.get("usage") or {}
        AI_GENERATION_TOTAL.labels(status="success").inc()
        AI_GENERATION_TOKENS.inc(usage.get("total_tokens", 0))
        RAG_GENERATION_DURATION.observe(time.time() - generation_start)

        return result.get("choices", [{}])[0].get("message", {}).get("content", "")

    except CircuitBreakerError as e:
        AI_GENERATION_TOTAL.labels(status="circuit_open").inc()
        logger.warning("DeepSeek circuit is OPEN, rejecting call: %s", e)
        return "AI 服务暂时不可用（熔断保护中），请稍后重试。"

    except Exception as e:
        AI_GENERATION_TOTAL.labels(status="error").inc()
        logger.error("DeepSeek API failed: %s", e)
        return f"AI 生成失败: {str(e)}"


# ── 流式 DeepSeek 调用 ─────────────────────────────────────────
async def stream_deepseek(query: str, context: str):
    """流式调用 DeepSeek Chat"""
    settings = get_settings()
    api_key = settings.OPENAI_API_KEY
    
    system_prompt = """你是一个企业知识库助手。请根据以下检索到的文档片段回答用户的问题。

要求：
1. 基于检索到的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、简洁、专业
4. 引用具体的文档内容作为依据

检索到的文档片段：
{context}"""
    
    messages = [
        {"role": "system", "content": system_prompt.format(context=context)},
        {"role": "user", "content": query},
    ]
    
    if _llm_breaker.state == CircuitState.OPEN:
        logger.warning("DeepSeek circuit is OPEN, rejecting stream call")
        AI_GENERATION_TOTAL.labels(status="circuit_open").inc()
        yield "[AI 服务暂时不可用（熔断保护中），请稍后重试。]"
        return

    generation_start = time.time()
    token_total = 0

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                DEEPSEEK_URL,
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                timeout=60.0,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                response.raise_for_status()
                # Reaching here means the upstream accepted the request.
                # A streaming body is consumed lazily and cannot be wrapped
                # by breaker.call(), so the outcome is reported directly.
                _llm_breaker.record_success()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            usage = data.get("usage")
                            if usage:
                                token_total = usage.get("total_tokens", token_total)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        AI_GENERATION_TOTAL.labels(status="success").inc()
        AI_GENERATION_TOKENS.inc(token_total)
        RAG_GENERATION_DURATION.observe(time.time() - generation_start)

    except Exception as e:
        _llm_breaker.record_failure()
        AI_GENERATION_TOTAL.labels(status="error").inc()
        yield f"\n\n[AI 生成错误: {str(e)}]"


# ── API 端点 ────────────────────────────────────────────────────

@router.post("/", response_model=ResponseModel[QueryResponse])
async def query_rag(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
) -> Any:
    """Execute a RAG query (非流式): Hybrid Retrieval → (可选精排) → DeepSeek AI."""
    start_time = time.time()
    status = "success"

    try:
        # Step 1: 混合检索 (带缓存)
        # 精排由 settings.ENABLE_RERANK 控制，默认**关闭**：实测英文
        # Cross-Encoder 在本项目中文语料上会拉低效果（HitRate@5 0.95→0.80，
        # 耗时 21.6s→92.4s）。详见 retrieval_experiment.json。
        search_results = await hybrid_search_cached(
            request.query, top_k=request.top_k, rerank=_settings.ENABLE_RERANK
        )

        # Step 2: 构建上下文
        context, sources = build_search_context(search_results)

        # Step 3: DeepSeek AI 生成
        answer = await call_deepseek(request.query, context)
    except Exception as e:
        status = "error"
        logger.exception("RAG query failed: %s", e)
        raise
    finally:
        latency_ms = int((time.time() - start_time) * 1000)
        RAG_QUERY_TOTAL.labels(status=status).inc()
        RAG_QUERY_DURATION.observe(latency_ms / 1000)

    _query_log.append({
        "user_id": current_user["id"],
        "query": request.query,
        "latency_ms": latency_ms,
        "search_mode": "hybrid+rerank",
    })

    # 记录缓存统计
    logger.info(
        "Query completed: %dms | Search cache: %s",
        latency_ms,
        _search_cache.stats,
    )

    return ResponseModel(
        data=QueryResponse(
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
        )
    )


@router.post("/stream")
async def query_rag_stream(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """Execute a RAG query with SSE streaming response (Hybrid Retrieval)."""
    start_time = time.time()

    async def event_generator():
        """SSE 事件生成器"""
        try:
            # Step 1: 混合检索 (带缓存；流式不做精排，优先保证首字节延迟)
            search_start = time.time()
            search_results = await hybrid_search_cached(
                request.query, top_k=request.top_k, rerank=False
            )
            search_latency = int((time.time() - search_start) * 1000)

            # 构建上下文和源文档
            context, sources = build_search_context(search_results)

            # Step 2: 发送元数据事件
            metadata = StreamMetadata(
                sources=sources,
                latency_search_ms=search_latency,
                total_sources=len(sources),
            )
            yield f"event: metadata\ndata: {metadata.model_dump_json()}\n\n"

            # Step 3: 流式调用 DeepSeek
            full_answer = ""
            async for chunk in stream_deepseek(request.query, context):
                full_answer += chunk
                chunk_event = StreamChunk(content=chunk)
                yield f"event: chunk\ndata: {chunk_event.model_dump_json()}\n\n"

            # Step 4: 完成事件
            total_latency = int((time.time() - start_time) * 1000)
            done_event = StreamDone(latency_total_ms=total_latency)
            yield f"event: done\ndata: {done_event.model_dump_json()}\n\n"

            # 记录日志
            _query_log.append({
                "user_id": current_user["id"],
                "query": request.query,
                "latency_ms": total_latency,
                "search_mode": "hybrid",
            })

            logger.info(
                "Stream query completed: %dms (search: %dms) | Cache: %s",
                total_latency,
                search_latency,
                _search_cache.stats,
            )
            RAG_QUERY_TOTAL.labels(status="success").inc()

        except Exception as e:
            RAG_QUERY_TOTAL.labels(status="error").inc()
            error_event = StreamError(error=str(e))
            yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"
        finally:
            RAG_QUERY_DURATION.observe(time.time() - start_time)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/history")
async def query_history(
    current_user: dict = Depends(get_current_user),
    limit: int = 10,
) -> Any:
    """Get query history for current user."""
    user_queries = [q for q in _query_log if q["user_id"] == current_user["id"]]
    return ResponseModel(data=user_queries[-limit:])


@router.get("/cache-stats")
async def cache_stats(
    current_user: dict = Depends(get_current_user),
) -> Any:
    """获取缓存统计信息"""
    return ResponseModel(
        data={
            "search_cache": _search_cache.stats,
        }
    )


@router.post("/build-index")
async def build_bm25_index(
    current_user: dict = Depends(get_current_user),
) -> Any:
    """手动构建 BM25 索引"""
    def _fetch_all() -> list[dict]:
        with _MilvusSession() as collection:
            return collection.query(
                expr="",
                output_fields=["chunk_id", "document_id", "content", "metadata"],
                limit=10000,
            )

    try:
        # 从 Milvus 获取所有文档块（连接经连接池复用；调用移出事件循环）
        results = await asyncio.to_thread(_fetch_all)

        if not results:
            return ResponseModel(message="No documents found to index")

        # 构建 BM25 索引
        retriever = get_hybrid_retriever_instance()
        await retriever.build_sparse_index(results)

        return ResponseModel(
            message=f"BM25 index built successfully with {len(results)} documents"
        )

    except Exception as e:
        logger.error("Failed to build BM25 index: %s", e)
        return ResponseModel(message=f"Failed to build index: {str(e)}")


@router.get("/search-mode")
async def get_search_mode(
    current_user: dict = Depends(get_current_user),
) -> Any:
    """获取当前检索模式"""
    return ResponseModel(
        data={
            "mode": "hybrid",
            "description": "Dense (Qwen Embedding + Milvus) + Sparse (BM25) with RRF fusion",
            "features": [
                "Dense retrieval: Semantic similarity via embeddings",
                "Sparse retrieval: Keyword matching via BM25",
                "Adaptive fusion: Dynamic weight adjustment based on query features",
                "Caching: L1 (in-memory) + L2 (Redis)", "Reranker: Cross-Encoder (lazy-loaded with cache)",
            ],
        }
    )
