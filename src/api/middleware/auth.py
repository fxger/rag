"""JWT authentication middleware — with in-memory fallback when DB is unavailable."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from src.config.settings import get_settings

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

# ── In-memory user store (fallback when DB is unavailable) ────────────
_IN_MEMORY_USERS: dict[str, dict] = {}


def _get_or_create_memory_user(username: str, email: str, password: str, role: str = "user") -> dict:
    """Get or create a user in the in-memory store."""
    if username in _IN_MEMORY_USERS:
        return _IN_MEMORY_USERS[username]
    user = {
        "id": str(uuid.uuid4()),
        "username": username,
        "email": email,
        "hashed_password": pwd_context.hash(password),
        "role": role,
        "department": "Demo",
        "is_active": True,
    }
    _IN_MEMORY_USERS[username] = user
    return user


# ── Bootstrap demo users ─────────────────────────────────────────────
_get_or_create_memory_user("admin", "admin@company.com", "Admin123!", "admin")
_get_or_create_memory_user("user1", "user1@company.com", "User123!", "user")
_get_or_create_memory_user("analyst1", "analyst1@company.com", "Analyst123!", "analyst")


# ── Token helpers ─────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


# ── Password helpers ──────────────────────────────────────────────────

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# ── User lookup ───────────────────────────────────────────────────────

async def _find_user(username: str) -> Optional[dict]:
    """Find user by username — try DB first, fall back to memory."""
    try:
        from src.config.database import async_session_factory
        from src.models import User
        from sqlalchemy import select
        async with async_session_factory() as session:
            result = await session.execute(select(User).where(User.username == username))
            user = result.scalar_one_or_none()
            if user:
                return {
                    "id": str(user.id),
                    "username": user.username,
                    "email": user.email,
                    "hashed_password": user.hashed_password,
                    "role": user.role.value if hasattr(user.role, 'value') else user.role,
                    "department": user.department or "",
                    "is_active": user.is_active,
                }
    except Exception:
        pass
    return _IN_MEMORY_USERS.get(username)


# ── FastAPI dependencies ──────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Extract and validate JWT token, return user dict."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        username: str = payload.get("username")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await _find_user(username)
    if user is None or not user.get("is_active"):
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Require admin role."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user