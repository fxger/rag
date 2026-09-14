"""
Prometheus Metrics 定义和 /metrics 端点.

提供:
- 应用层指标 (HTTP请求、RAG查询)
- 并发控制指标 (Semaphore状态)
- 连接池指标 (Milvus/Redis)
- 缓存指标 (命中率、淘汰)
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import logging
from fastapi import APIRouter, Response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Metrics"])

# ═══════════════════════════════════════════════════════════════════
# 应用信息
# ═══════════════════════════════════════════════════════════════════

APP_INFO = Info(
    "rag_app",
    "Enterprise RAG System application info"
)

# ═══════════════════════════════════════════════════════════════════
# HTTP 请求指标 (由中间件自动采集)
# ═══════════════════════════════════════════════════════════════════

HTTP_REQUESTS_TOTAL = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "rag_http_requests_in_progress",
    "Number of HTTP requests currently in progress",
    ["method", "endpoint"],
)

# ═══════════════════════════════════════════════════════════════════
# RAG 查询指标
# ═══════════════════════════════════════════════════════════════════

RAG_QUERY_TOTAL = Counter(
    "rag_query_total",
    "Total RAG queries",
    ["status"],  # success, error
)

RAG_QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "RAG query total duration in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

RAG_RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG retrieval phase duration",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

RAG_RERANK_DURATION = Histogram(
    "rag_rerank_duration_seconds",
    "RAG reranking phase duration",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

RAG_GENERATION_DURATION = Histogram(
    "rag_generation_duration_seconds",
    "RAG AI generation duration",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

RAG_RETRIEVAL_RESULTS = Histogram(
    "rag_retrieval_results_count",
    "Number of results returned by retrieval",
    buckets=[1, 2, 5, 10, 20, 50],
)

# ═══════════════════════════════════════════════════════════════════
# 并发控制指标 (从 SingletonManager 实时采集)
# ═══════════════════════════════════════════════════════════════════

SEMAPHORE_ACTIVE = Gauge(
    "rag_semaphore_active",
    "Number of active inference tasks",
)

SEMAPHORE_WAITING = Gauge(
    "rag_semaphore_waiting",
    "Number of waiting inference tasks",
)

SEMAPHORE_MAX_CONCURRENT = Gauge(
    "rag_semaphore_max_concurrent",
    "Maximum concurrent inference limit",
)

# ═══════════════════════════════════════════════════════════════════
# 重排序器指标
# ═══════════════════════════════════════════════════════════════════

RERANKER_REQUESTS_TOTAL = Gauge(
    "rag_reranker_requests_total",
    "Total reranker requests",
)

RERANKER_INFERENCE_TOTAL = Gauge(
    "rag_reranker_inference_total",
    "Total actual reranker inferences (excluding cache hits)",
)

RERANKER_CACHE_HITS_TOTAL = Gauge(
    "rag_reranker_cache_hits_total",
    "Total reranker cache hits",
)

RERANKER_CACHE_SIZE = Gauge(
    "rag_reranker_cache_size",
    "Current reranker cache size",
)

# ═══════════════════════════════════════════════════════════════════
# 连接池指标
# ═══════════════════════════════════════════════════════════════════

MILVUS_POOL_SIZE = Gauge(
    "rag_milvus_pool_size",
    "Milvus connection pool size",
)

MILVUS_POOL_ACTIVE = Gauge(
    "rag_milvus_pool_active",
    "Milvus active connections",
)

REDIS_POOL_SIZE = Gauge(
    "rag_redis_pool_size",
    "Redis connection pool size",
)

REDIS_POOL_ACTIVE = Gauge(
    "rag_redis_pool_active",
    "Redis active connections",
)

# ═══════════════════════════════════════════════════════════════════
# 文档处理指标
# ═══════════════════════════════════════════════════════════════════

DOCUMENTS_UPLOADED_TOTAL = Counter(
    "rag_documents_uploaded_total",
    "Total documents uploaded",
)

DOCUMENTS_PROCESSED_TOTAL = Counter(
    "rag_documents_processed_total",
    "Total documents processed successfully",
)

DOCUMENTS_FAILED_TOTAL = Counter(
    "rag_documents_failed_total",
    "Total document processing failures",
)

DOCUMENT_PROCESSING_DURATION = Histogram(
    "rag_document_processing_duration_seconds",
    "Document processing duration",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# ═══════════════════════════════════════════════════════════════════
# AI 模型指标
# ═══════════════════════════════════════════════════════════════════

AI_GENERATION_TOTAL = Counter(
    "rag_ai_generation_total",
    "Total AI generation calls",
    ["status"],
)

AI_GENERATION_TOKENS = Counter(
    "rag_ai_generation_tokens_total",
    "Total tokens generated by AI",
)

# ═══════════════════════════════════════════════════════════════════
# 缓存指标
# ═══════════════════════════════════════════════════════════════════

CACHE_HITS_TOTAL = Counter(
    "rag_cache_hits_total",
    "Total cache hits",
    ["cache_type"],  # reranker, query, embedding
)

CACHE_MISSES_TOTAL = Counter(
    "rag_cache_misses_total",
    "Total cache misses",
    ["cache_type"],
)

CACHE_SIZE = Gauge(
    "rag_cache_size",
    "Current cache size",
    ["cache_type"],
)


# ═══════════════════════════════════════════════════════════════════
# 指标更新函数 - 从 SingletonManager 采集实时数据
# ═══════════════════════════════════════════════════════════════════

def update_concurrent_metrics():
    """从 SingletonManager 采集并发控制指标"""
    try:
        from src.core.concurrent_singleton import SingletonManager
        stats = SingletonManager.get_stats()
        
        # 更新重排序器指标
        for key, instance in stats.items():
            if isinstance(instance, dict) and "semaphore" in instance:
                sem_stats = instance["semaphore"]
                SEMAPHORE_ACTIVE.set(sem_stats.get("active", 0))
                SEMAPHORE_WAITING.set(sem_stats.get("waiting", 0))
                SEMAPHORE_MAX_CONCURRENT.set(sem_stats.get("max_concurrent", 4))
                
                RERANKER_REQUESTS_TOTAL.set(instance.get("total_requests", 0))
                RERANKER_INFERENCE_TOTAL.set(instance.get("inference_count", 0))
                RERANKER_CACHE_HITS_TOTAL.set(instance.get("cache_hits", 0))
                RERANKER_CACHE_SIZE.set(instance.get("cache_size", 0))
        
        # 更新连接池指标
        pools = stats.get("pools", {})
        for key, pool_stats in pools.items():
            if "milvus" in key:
                MILVUS_POOL_SIZE.set(pool_stats.get("pool_size", 0))
                MILVUS_POOL_ACTIVE.set(pool_stats.get("active", 0))
            elif "redis" in key:
                REDIS_POOL_SIZE.set(pool_stats.get("pool_size", 0))
                REDIS_POOL_ACTIVE.set(pool_stats.get("active", 0))
                
    except Exception as e:
        logger.warning(f"Failed to update concurrent metrics: {e}")


# ═══════════════════════════════════════════════════════════════════
# /metrics 端点
# ═══════════════════════════════════════════════════════════════════

@router.get("/metrics", include_in_schema=False)
async def metrics():
    """
    Prometheus 指标端点.
    
    返回所有注册的指标，格式为 Prometheus exposition format.
    供 Prometheus Server 定期拉取 (scrape).
    """
    # 更新实时指标
    update_concurrent_metrics()
    
    # 生成指标输出
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def init_app_info():
    """初始化应用信息指标"""
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        APP_INFO.info({
            "version": settings.APP_VERSION,
            "environment": "production" if not settings.DEBUG else "development",
        })
    except Exception:
        APP_INFO.info({
            "version": "unknown",
            "environment": "unknown",
        })
