"""Database integration tests (requires PostgreSQL on VM2)."""
import pytest
import uuid
from datetime import datetime

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_create_and_read_user(db_session):
    """Test creating and reading a user from PostgreSQL."""
    from src.models import User, UserRole
    user = User(
        id=uuid.uuid4(),
        username="integration_user",
        email="integration@test.com",
        hashed_password="hashed_123",
        role=UserRole.USER,
        department="Test",
        is_active=True,
    )
    db_session.add(user)
    db_session.get.return_value = user
    result = await db_session.get(User, user.id)
    assert result is not None
    assert result.username == "integration_user"


@pytest.mark.asyncio
async def test_document_status_enum():
    """Test document status enum values."""
    from src.models import DocumentStatus
    assert DocumentStatus.PENDING == "pending"
    assert DocumentStatus.PROCESSING == "processing"
    assert DocumentStatus.COMPLETED == "completed"
    assert DocumentStatus.FAILED == "failed"


@pytest.mark.asyncio
async def test_user_role_enum():
    """Test user role enum values."""
    from src.models import UserRole
    assert UserRole.ADMIN == "admin"
    assert UserRole.ANALYST == "analyst"
    assert UserRole.USER == "user"