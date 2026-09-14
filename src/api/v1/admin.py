"""Admin API endpoints — simplified for demo."""

from typing import Any

from fastapi import APIRouter, Depends

from src.api.middleware.auth import get_current_user, require_admin
from src.api.schemas.common import ResponseModel

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/stats")
async def get_stats(
    current_user: dict = Depends(require_admin),
) -> Any:
    """Get system statistics."""
    return ResponseModel(
        data={
            "users": 3,
            "documents": 0,
            "queries": 0,
            "avg_latency_ms": 0,
            "status": "healthy",
        }
    )


@router.get("/users")
async def list_users(
    current_user: dict = Depends(require_admin),
) -> Any:
    """List all users."""
    from src.api.middleware.auth import _IN_MEMORY_USERS
    users = [
        {"id": u["id"], "username": u["username"], "email": u["email"], "role": u.get("role", "user")}
        for u in _IN_MEMORY_USERS.values()
    ]
    return ResponseModel(data=users)


@router.post("/reindex")
async def reindex(
    current_user: dict = Depends(require_admin),
) -> Any:
    """Trigger reindexing (demo)."""
    return ResponseModel(message="Reindexing triggered (demo mode)")


@router.get("/audit-logs")
async def audit_logs(
    current_user: dict = Depends(require_admin),
) -> Any:
    """Get audit logs (demo)."""
    return ResponseModel(data=[])