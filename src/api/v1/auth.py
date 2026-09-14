"""Authentication API endpoints — with in-memory fallback."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.middleware.auth import (
    _IN_MEMORY_USERS,
    _find_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from src.api.schemas.auth import (
    TokenRefreshRequest,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
)
from src.api.schemas.common import ResponseModel

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=ResponseModel[TokenResponse])
async def login(credentials: UserLogin) -> Any:
    """Authenticate user and return JWT tokens."""
    user = await _find_user(credentials.username)
    
    if not user or not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.get("is_active"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive")
    
    token_data = {"sub": user["id"], "username": user["username"], "role": user.get("role", "user")}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    return ResponseModel(data=TokenResponse(access_token=access_token, refresh_token=refresh_token))


@router.post("/register", response_model=ResponseModel[UserResponse])
async def register(user_data: UserCreate) -> Any:
    """Register a new user."""
    existing = await _find_user(user_data.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    import uuid
    user = {
        "id": str(uuid.uuid4()),
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": hash_password(user_data.password),
        "role": "user",
        "department": "",
        "is_active": True,
        "full_name": user_data.full_name or "",
        "created_at": datetime.now(timezone.utc),
        "last_login": None,
    }
    _IN_MEMORY_USERS[user_data.username] = user
    
    return ResponseModel(data=_to_user_response(user))


@router.post("/refresh", response_model=ResponseModel[TokenResponse])
async def refresh_token(body: TokenRefreshRequest) -> Any:
    """Refresh access token."""
    payload = decode_token(body.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    username = payload.get("username")
    user = await _find_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    token_data = {"sub": user["id"], "username": user["username"], "role": user.get("role", "user")}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    return ResponseModel(data=TokenResponse(access_token=access_token, refresh_token=refresh_token))


@router.get("/me", response_model=ResponseModel[UserResponse])
async def get_me(current_user: dict = Depends(get_current_user)) -> Any:
    """Get current user profile."""
    return ResponseModel(data=_to_user_response(current_user))


@router.put("/me", response_model=ResponseModel[UserResponse])
async def update_me(user_data: UserUpdate, current_user: dict = Depends(get_current_user)) -> Any:
    """Update current user profile."""
    if user_data.email:
        current_user["email"] = user_data.email
    if user_data.full_name:
        current_user["full_name"] = user_data.full_name
    return ResponseModel(data=_to_user_response(current_user))


def _to_user_response(user: dict) -> UserResponse:
    """Convert user dict to UserResponse schema."""
    return UserResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        full_name=user.get("full_name", ""),
        role=user.get("role", "user"),
        is_active=user.get("is_active", True),
        created_at=user.get("created_at", datetime.now(timezone.utc)),
        last_login=user.get("last_login"),
    )