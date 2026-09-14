"""
Database initialization script.

Creates all tables and an initial admin user.
Run: python scripts/init_db.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def init():
    from src.config.database import init_db, async_session_factory
    from src.models import User, UserRole
    from passlib.hash import bcrypt
    import uuid

    print("Creating database tables...")
    await init_db()
    print("✅ Tables created")

    # Create admin user
    print("Creating admin user...")
    async with async_session_factory() as session:
        from sqlalchemy import select
        result = await session.execute(select(User).where(User.username == "admin"))
        if result.scalar_one_or_none() is None:
            admin = User(
                id=uuid.uuid4(),
                username="admin",
                email="admin@enterprise.com",
                hashed_password=bcrypt.hash("admin123"),
                role=UserRole.ADMIN,
                department="Administration",
                is_active=True,
            )
            session.add(admin)
            await session.commit()
            print("✅ Admin user created (username: admin, password: admin123)")
        else:
            print("ℹ️  Admin user already exists")


if __name__ == "__main__":
    asyncio.run(init())