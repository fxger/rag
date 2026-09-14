"""
Enterprise RAG System — FastAPI Application Entry Point.

Manages application lifespan (startup/shutdown), CORS, and API router mounting.
"""

import logging
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def warmup_milvus() -> bool:
    """预加载 Milvus 集合到内存"""
    try:
        from src.config.milvus import get_milvus_client, get_or_create_collection, load_collection
        
        start = time.time()
        client = get_milvus_client()
        collection_name = get_or_create_collection()
        
        # 加载集合到内存
        load_collection(collection_name)
        
        elapsed = (time.time() - start) * 1000
        logger.info("✅ Milvus collection preloaded: %s (%.1fms)", collection_name, elapsed)
        return True
    except Exception as e:
        logger.warning("⚠️ Milvus preload failed: %s", e)
        return False


async def warmup_embedding_connection() -> bool:
    """预热 Embedding API 连接"""
    try:
        import httpx
        
        start = time.time()
        qwen_url = "http://192.168.86.132:8000"
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{qwen_url}/health")
            
        elapsed = (time.time() - start) * 1000
        logger.info("✅ Embedding connection warmed up (%.1fms)", elapsed)
        return True
    except Exception as e:
        logger.warning("⚠️ Embedding warmup failed (service may be unavailable): %s", e)
        return False


async def warmup_redis() -> bool:
    """预热 Redis 连接"""
    try:
        from src.config.redis import get_cache_redis
        
        start = time.time()
        redis = await get_cache_redis()
        await redis.ping()
        
        elapsed = (time.time() - start) * 1000
        logger.info("✅ Redis connection warmed up (%.1fms)", elapsed)
        return True
    except Exception as e:
        logger.warning("⚠️ Redis warmup failed: %s", e)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events."""
    # ── Startup ──────────────────────────────────────────────────────
    startup_start = time.time()
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    # 初始化数据库
    try:
        from src.config.database import init_db
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database initialization skipped (unreachable): %s", type(e).__name__)

    # 预热各服务连接
    logger.info("🔥 Warming up services...")
    
    await warmup_redis()
    await warmup_milvus()
    await warmup_embedding_connection()
    
    # 初始化 Prometheus 指标
    try:
        from src.api.metrics import init_app_info
        init_app_info()
        logger.info("📊 Prometheus metrics initialized")
    except Exception as e:
        logger.warning("⚠️ Prometheus metrics init failed: %s", e)
    
    startup_elapsed = (time.time() - startup_start) * 1000
    logger.info("🚀 Startup complete (%.1fms)", startup_elapsed)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down %s", settings.APP_NAME)
    try:
        from src.config.database import close_db
        await close_db()
    except Exception:
        pass
    try:
        from src.config.redis import close_redis
        await close_redis()
    except Exception:
        pass
    try:
        from src.config.milvus import close_milvus
        close_milvus()
    except Exception:
        pass
    logger.info("All connections closed")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Enterprise-grade Retrieval-Augmented Generation System",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # ── CORS Middleware ──────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Metrics Middleware (新增) ────────────────────────────────────
    from src.api.middleware.metrics import MetricsMiddleware
    app.add_middleware(MetricsMiddleware)

    # ── Register API Routers ────────────────────────────────────────
    from src.api.v1.auth import router as auth_router
    from src.api.v1.query import router as query_router
    from src.api.v1.documents import router as documents_router
    from src.api.v1.admin import router as admin_router
    from src.api.health import router as health_router
    from src.api.metrics import router as metrics_router  # 新增

    app.include_router(health_router)
    app.include_router(metrics_router)  # 新增: /metrics 端点
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(query_router, prefix="/api/v1/query", tags=["Query"])
    app.include_router(documents_router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(admin_router, prefix="/api/v1/admin", tags=["Admin"])

    return app


# Create the app instance for uvicorn
app = create_app()


# ── 静态文件服务 (前端) ──────────────────────────────────────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# 前端文件路径
FRONTEND_DIR = r"D:\Desktop\RAG\frontend"

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """提供前端页面"""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# 挂载静态文件
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
