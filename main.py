"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import api_router
from src.api.middleware.cors import configure_cors
from src.api.middleware.rate_limiter import RateLimiterMiddleware
from src.config.settings import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    print("Starting RAG API server...")
    # Initialize database connections, Redis, etc.
    yield
    # Shutdown
    print("Shutting down RAG API server...")
    # Close connections


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Enterprise RAG System API",
        description="Retrieval-Augmented Generation API for enterprise document search and Q&A",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Configure CORS
    configure_cors(
        app,
        allowed_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
    )
    
    # Add rate limiting middleware
    app.add_middleware(
        RateLimiterMiddleware,
        requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
    )
    
    # Include API router
    app.include_router(api_router)
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
    )
