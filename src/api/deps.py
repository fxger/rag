"""FastAPI dependency injection for services."""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from src.config.database import async_session_factory, get_db
from src.config.redis import get_redis
from src.api.middleware.auth import get_current_user
from src.services.rag_service import RAGService
from src.services.document_service import DocumentService


async def get_rag_service(
    db: AsyncSession = None,
) -> RAGService:
    """Get RAG service instance.
    
    Args:
        db: Optional database session (will create if not provided)
        
    Returns:
        Configured RAGService instance
    """
    return RAGService()


async def get_document_service(
    db: AsyncSession = None,
) -> DocumentService:
    """Get document service instance.
    
    Args:
        db: Optional database session (will create if not provided)
        
    Returns:
        Configured DocumentService instance
    """
    return DocumentService()


async def get_rag_service_with_db(
    db: AsyncSession = None,
) -> RAGService:
    """Get RAG service with database session.
    
    Args:
        db: Database session
        
    Returns:
        RAGService with database access
    """
    if db is None:
        async with async_session_factory() as session:
            return RAGService(db=session)
    return RAGService(db=db)


async def get_document_service_with_db(
    db: AsyncSession = None,
) -> DocumentService:
    """Get document service with database session.
    
    Args:
        db: Database session
        
    Returns:
        DocumentService with database access
    """
    if db is None:
        async with async_session_factory() as session:
            return DocumentService(db=session)
    return DocumentService(db=db)
