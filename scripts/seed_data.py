"""
Seed test data script.

Creates sample users and documents for development.
Run: python scripts/seed_data.py
"""

import asyncio
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def seed():
    from src.config.database import async_session_factory
    from src.models import User, UserRole, Document, DocumentStatus, FileType
    from passlib.hash import bcrypt

    print("Seeding test data...")
    async with async_session_factory() as session:
        # Create test users
        users_data = [
            {"username": "analyst1", "email": "analyst1@test.com", "role": UserRole.ANALYST, "department": "Data Science"},
            {"username": "user1", "email": "user1@test.com", "role": UserRole.USER, "department": "Engineering"},
        ]

        created_users = []
        for data in users_data:
            user = User(
                id=uuid.uuid4(),
                username=data["username"],
                email=data["email"],
                hashed_password=bcrypt.hash("password123"),
                role=data["role"],
                department=data["department"],
                is_active=True,
            )
            session.add(user)
            created_users.append(user)

        await session.flush()

        # Create sample documents
        for user in created_users[:1]:
            for i in range(3):
                doc = Document(
                    id=uuid.uuid4(),
                    title=f"Sample Document {i+1}",
                    content_hash=f"hash_{uuid.uuid4().hex[:16]}",
                    file_path=f"docs/sample_{i+1}.pdf",
                    file_type=FileType.PDF,
                    file_size=1024 * (i + 1),
                    status=DocumentStatus.COMPLETED,
                    chunk_count=10 * (i + 1),
                    owner_id=user.id,
                )
                session.add(doc)

        await session.commit()
        print("✅ Test data seeded successfully")


if __name__ == "__main__":
    asyncio.run(seed())