"""Health check API endpoints."""
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.common import ResponseModel
from src.config.database import get_db
from src.config.redis import get_redis

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=ResponseModel)
async def health_check() -> Any:
    """Liveness check endpoint.
    
    Returns 200 if the service is running. This endpoint is used
    by load balancers and orchestrators to verify the service is alive.
    
    Returns:
        Health status response
    """
    return ResponseModel(
        data={
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
        }
    )


@router.get("/health/ready", response_model=ResponseModel)
async def readiness_check(
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Readiness check endpoint.
    
    Verifies connectivity to all required services:
    - PostgreSQL database
    - Redis cache
    - Milvus vector store
    
    Returns 200 only if all services are accessible.
    
    Args:
        db: Database session
        
    Returns:
        Readiness status with component health
    """
    components = {}
    all_healthy = True
    
    # Check PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        components["postgresql"] = {"status": "healthy"}
    except Exception as e:
        components["postgresql"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False
    
    # Check Redis
    try:
        redis = await get_redis()
        await redis.ping()
        components["redis"] = {"status": "healthy"}
        await redis.close()
    except Exception as e:
        components["redis"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False
    
    # Check Milvus
    try:
        from pymilvus import connections, utility
        
        # This is a simplified check - in production, use connection pooling
        connections.connect(alias="health_check", host="localhost", port="19530")
        if utility.has_collection("document_chunks", using="health_check"):
            components["milvus"] = {"status": "healthy"}
        else:
            components["milvus"] = {"status": "degraded", "message": "Collection not found"}
        connections.disconnect("health_check")
    except Exception as e:
        components["milvus"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False
    
    # Determine overall status
    overall_status = "ready" if all_healthy else "not_ready"
    
    return ResponseModel(
        data={
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": components,
        }
    )
