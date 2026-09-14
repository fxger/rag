"""Query API endpoints — Hybrid Retrieval (Dense + BM25) with SSE streaming."""

import json
import time
import hashlib
import logging
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.middleware.auth import get_current_user
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
from src.core.singleton import get_hybrid_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])

_query_log: list[dict] = []

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
QWEN_EMBED_URL = "http://192.168.86.132:8000/v1/embeddings"
QWEN_EMBED_MODEL = "qwen-embedding"
MILVUS_URI = "http://192.168.86.132:19530"
COLLECTION_NAME = "enterprise_rag"

# 全局混合检索器实例


def get_hybrid_retriever_instance():
    """获取全局混合检索器实例（单例模式）"""
    return get_hybrid_retriever(
        collection_name=COLLECTION_NAME,
        enable_reranker=False,
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
    
    def _make_key(self, query: str, top_k: int) -> str:
        return hashlib.sha256(f"{query}:{top_k}".encode()).hexdigest()
    
    async def get(self, query: str, top_k: int) -> list[dict] | None:
        key = self._make_key(query, top_k)
        
        # L1
        if key in self.l1:
            self._hits["l1"] += 1
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
                return value
        except Exception as e:
            logger.warning("Search cache L2 read error: %s", e)
        
        self._hits["miss"] += 1
        return None
    
    async def set(self, query: str, top_k: int, results: list[dict]):
        key = self._make_key(query, top_k)
        
        self.l1[key] = results
        
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


# ── Embedding 生成 (带缓存) ──────────────────────────────────────
def get_qwen_embedding_sync(text: str) -> list[float]:
    """同步获取 Qwen Embedding (带缓存检查)"""
    import hashlib
    
    # 计算缓存 key
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cache_key = f"rag:emb:{text_hash}"
    
    # 尝试从 Redis L2 缓存获取
    try:
        import redis
        r = redis.Redis(host="192.168.86.132", port=6379, db=1)
        cached = r.get(cache_key)
        if cached:
            import json
            logger.debug("Embedding cache hit (sync)")
            return json.loads(cached)
    except Exception as e:
        logger.debug("Redis cache check failed: %s", e)
    
    # 缓存未命中，调用 API
    try:
        response = httpx.post(
            QWEN_EMBED_URL,
            json={"input": text, "model": QWEN_EMBED_MODEL},
            timeout=30.0,
        )
        result = response.json()
        embedding = result["data"][0]["embedding"]
        
        # 写入缓存
        try:
            import json
            r.set(cache_key, json.dumps(embedding, separators=(",", ":")), ex=3600)
        except:
            pass
        
        return embedding
    except Exception as e:
        logger.error("Qwen embedding failed: %s", e)
        return [0.0] * 1024


# ── 混合检索 (带缓存) ──────────────────────────────────────────
async def hybrid_search_cached(query: str, top_k: int = 5) -> list[dict]:
    """混合检索：Dense + BM25，带缓存"""
    
    # 检查缓存
    cached = await _search_cache.get(query, top_k)
    if cached is not None:
        logger.debug("Search cache hit for query: %s", query[:50])
        return cached
    
    # 缓存未命中，执行混合检索
    try:
        retriever = get_hybrid_retriever_instance()
        
        # 获取查询的 embedding
        embedding = get_qwen_embedding_sync(query)
        
        # 执行混合检索
        result = await retriever.retrieve(
            query=query,
            top_k=top_k,
            embedding=embedding,
            fusion_method="adaptive",  # 使用 RRF 融合
            rerank=False,  # 暂时禁用重排序
        )
        
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
        await _search_cache.set(query, top_k, search_results)
        
        logger.info(
            "Hybrid search completed: %d results (dense: %d, sparse: %d) in %dms",
            len(search_results),
            len([r for r in search_results if "dense" in r.get("source_type", "")]),
            len([r for r in search_results if "sparse" in r.get("source_type", "")]),
            result.retrieval_time_ms,
        )
        
        return search_results
        
    except Exception as e:
        logger.error("Hybrid search failed: %s", e)
        # 降级到纯向量搜索
        return await vector_search_fallback(query, top_k)


async def vector_search_fallback(query: str, top_k: int) -> list[dict]:
    """向量搜索降级方案"""
    try:
        embedding = get_qwen_embedding_sync(query)
        
        import pymilvus
        from pymilvus import connections, Collection
        
        connections.connect("default", uri=MILVUS_URI)
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
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
    
    try:
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
        
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
    except Exception as e:
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
                },
                timeout=60.0,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield f"\n\n[AI 生成错误: {str(e)}]"


# ── API 端点 ────────────────────────────────────────────────────

@router.post("/", response_model=ResponseModel[QueryResponse])
async def query_rag(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
) -> Any:
    """Execute a RAG query (非流式): Hybrid Retrieval → DeepSeek AI."""
    start_time = time.time()

    # Step 1: 混合检索 (带缓存)
    search_results = await hybrid_search_cached(request.query, top_k=request.top_k)

    # Step 2: 构建上下文
    context, sources = build_search_context(search_results)

    # Step 3: DeepSeek AI 生成
    answer = await call_deepseek(request.query, context)

    latency_ms = int((time.time() - start_time) * 1000)

    _query_log.append({
        "user_id": current_user["id"],
        "query": request.query,
        "latency_ms": latency_ms,
        "search_mode": "hybrid",
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
            # Step 1: 混合检索 (带缓存)
            search_start = time.time()
            search_results = await hybrid_search_cached(request.query, top_k=request.top_k)
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

        except Exception as e:
            error_event = StreamError(error=str(e))
            yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"

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
    try:
        # 从 Milvus 获取所有文档块
        import pymilvus
        from pymilvus import connections, Collection
        
        connections.connect("default", uri=MILVUS_URI)
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        # 获取所有数据
        results = collection.query(
            expr="",
            output_fields=["chunk_id", "document_id", "content", "metadata"],
            limit=10000,
        )
        
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
