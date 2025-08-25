"""Database connection and session management."""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from .config import get_settings

logger = logging.getLogger(__name__)

# Global engine and session maker
_engine = None
_async_session_maker = None


def get_database_url() -> str:
    """Get the database URL from settings."""
    settings = get_settings()
    return settings.database.url


async def init_database():
    """Initialize the database connection and create session maker."""
    global _engine, _async_session_maker
    
    if _engine is not None:
        logger.warning("Database already initialized")
        return
    
    database_url = get_database_url()
    logger.info(f"Initializing database connection: {database_url.split('@')[-1] if '@' in database_url else database_url}")
    
    # Create async engine
    _engine = create_async_engine(
        database_url,
        echo=False,  # Set to True for SQL debugging
        poolclass=NullPool,  # Use NullPool for development
        pool_pre_ping=True,
        future=True
    )
    
    # Create session maker
    _async_session_maker = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False
    )
    
    logger.info("Database connection initialized")


async def close_database():
    """Close the database connection."""
    global _engine, _async_session_maker
    
    if _engine is None:
        return
    
    logger.info("Closing database connection")
    
    await _engine.dispose()
    _engine = None
    _async_session_maker = None
    
    logger.info("Database connection closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    if _async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with _async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_health() -> bool:
    """Check if database is accessible."""
    if _engine is None:
        return False
    
    try:
        async with _engine.begin() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False