"""Query API endpoints — Qwen Embedding + Milvus + DeepSeek AI (with SSE streaming)."""

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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])

_query_log: list[dict] = []

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
QWEN_EMBED_URL = "http://192.168.86.132:8000/v1/embeddings"
QWEN_EMBED_MODEL = "qwen-embedding"
MILVUS_URI = "http://192.168.86.132:19530"
COLLECTION_NAME = "enterprise_rag"


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


# ── Milvus 搜索 (使用单例) ──────────────────────────────────────
def vector_search_sync(query: str, top_k: int = 5) -> list[dict]:
    """同步 Milvus 搜索 (使用单例客户端)"""
    try:
        from src.config.milvus import get_milvus_client, get_or_create_collection
        
        # 获取 Embedding
        embedding = get_qwen_embedding_sync(query)
        
        # 使用单例客户端
        client = get_milvus_client()
        collection_name = get_or_create_collection()
        
        results = client.search(
            collection_name=collection_name,
            data=[embedding],
            limit=top_k,
            output_fields=["document_id", "chunk_id", "content", "metadata"],
        )
        
        search_results = []
        for hits in results:
            for hit in hits:
                content = hit.entity.get("content", "")
                meta_str = hit.entity.get("metadata", "{}")
                try:
                    metadata = json.loads(meta_str)
                except:
                    metadata = {}
                
                search_results.append({
                    "document_id": hit.entity.get("document_id", ""),
                    "chunk_id": hit.entity.get("chunk_id", ""),
                    "content": content,
                    "score": hit.score,
                    "metadata": metadata,
                })
        
        return search_results
        
    except Exception as e:
        logger.error("Milvus search failed: %s", e)
        return [
            {
                "document_id": str(uuid4()),
                "chunk_id": str(uuid4()),
                "content": "知识库检索暂时不可用",
                "score": 0.0,
                "metadata": {"error": str(e)},
            }
        ]


async def vector_search_cached(query: str, top_k: int = 5) -> list[dict]:
    """带缓存的向量搜索"""
    # 尝试搜索缓存
    cached = await _search_cache.get(query, top_k)
    if cached:
        logger.debug("Search cache hit for query: %s...", query[:30])
        return cached
    
    # 缓存未命中，执行搜索
    results = vector_search_sync(query, top_k)
    
    # 写入缓存
    await _search_cache.set(query, top_k, results)
    
    return results


def build_search_context(search_results: list[dict]) -> tuple[str, list[SourceDocument]]:
    """构建搜索上下文和源文档列表"""
    context_parts = []
    sources = []
    
    for i, result in enumerate(search_results):
        if result["score"] > 0.1:
            context_parts.append("[文档%d] %s" % (i + 1, result["content"]))
            sources.append(
                SourceDocument(
                    document_id=result["document_id"],
                    chunk_id=result["chunk_id"],
                    content=result["content"][:200],
                    score=result["score"],
                    metadata=result.get("metadata", {}),
                )
            )
    
    context = "\n\n".join(context_parts) if context_parts else "（知识库暂无相关内容）"
    return context, sources


# ── DeepSeek AI 生成 ────────────────────────────────────────────
async def call_deepseek(query: str, context: str = "") -> str:
    """Call DeepSeek API for answer generation (非流式)."""
    settings = get_settings()

    system_prompt = """你是一个企业知识库问答助手。根据用户的问题和提供的上下文信息，给出准确、专业的回答。
如果上下文中没有相关信息，请说明你无法从知识库中找到相关内容，但可以提供一般性的回答。
回答要简洁明了，使用中文。"""

    user_prompt = f"""上下文信息：
{context if context else '（暂无相关文档，请基于你的知识回答）'}

用户问题：{query}

请基于以上信息回答问题："""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                DEEPSEEK_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
            )
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI 生成失败: {str(e)}"


async def stream_deepseek(query: str, context: str = ""):
    """Call DeepSeek API for answer generation (流式)."""
    settings = get_settings()

    system_prompt = """你是一个企业知识库问答助手。根据用户的问题和提供的上下文信息，给出准确、专业的回答。
如果上下文中没有相关信息，请说明你无法从知识库中找到相关内容，但可以提供一般性的回答。
回答要简洁明了，使用中文。"""

    user_prompt = f"""上下文信息：
{context if context else '（暂无相关文档，请基于你的知识回答）'}

用户问题：{query}

请基于以上信息回答问题："""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                DEEPSEEK_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "stream": True,
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
    """Execute a RAG query (非流式): Qwen Embedding → Milvus → DeepSeek AI."""
    start_time = time.time()

    # Step 1: 向量搜索 (带缓存)
    search_results = await vector_search_cached(request.query, top_k=request.top_k)

    # Step 2: 构建上下文
    context, sources = build_search_context(search_results)

    # Step 3: DeepSeek AI 生成
    answer = await call_deepseek(request.query, context)

    latency_ms = int((time.time() - start_time) * 1000)

    _query_log.append({
        "user_id": current_user["id"],
        "query": request.query,
        "latency_ms": latency_ms,
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
    """Execute a RAG query with SSE streaming response."""
    start_time = time.time()

    async def event_generator():
        """SSE 事件生成器"""
        try:
            # Step 1: 搜索 (带缓存)
            search_start = time.time()
            search_results = await vector_search_cached(request.query, top_k=request.top_k)
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

