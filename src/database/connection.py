# Database Connection Module
# ============================================================================
# Async database connection management with SQLAlchemy 2.0
# Implements connection pooling with asyncpg for high load stability
# ============================================================================

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from src.core.config import get_settings
from src.core.logging import logger


# Base class for all models
Base = declarative_base()

# Global engine instance
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """
    Get the SQLAlchemy async engine instance.
    
    Returns:
        AsyncEngine: Configured async engine with connection pooling
        
    Raises:
        RuntimeError: If engine hasn't been initialized
    """
    global _engine
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call init_db() first.")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get the async session factory.
    
    Returns:
        async_sessionmaker: Configured session factory
        
    Raises:
        RuntimeError: If session factory hasn't been initialized
    """
    global _session_factory
    if _session_factory is None:
        raise RuntimeError("Session factory not initialized. Call init_db() first.")
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session.
    
    This is the primary way to get a database session in handlers.
    
    Yields:
        AsyncSession: Database session for operations
        
    Example:
        ```python
        async with get_session() as session:
            await session.execute(select(User).where(User.id == 1))
        ```
    """
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> AsyncEngine:
    """
    Initialize the database connection and create tables.
    
    Sets up:
    - Async engine with connection pooling (asyncpg driver)
    - Session factory with proper configuration
    - All database tables
    
    Returns:
        AsyncEngine: Initialized engine instance
        
    Raises:
        Exception: If database connection fails
    """
    global _engine, _session_factory
    
    settings = get_settings()
    db_settings = settings.database
    
    logger.info(
        f"Initializing database connection to {db_settings.host}:{db_settings.port}/{db_settings.database}"
    )
    
    # Create async engine with connection pooling
    _engine = create_async_engine(
        db_settings.async_url,
        echo=settings.debug,
        pool_size=db_settings.pool_size,
        max_overflow=db_settings.max_overflow,
        pool_timeout=db_settings.pool_timeout,
        pool_pre_ping=True,  # Enable connection health checks
        pool_recycle=3600,   # Recycle connections after 1 hour
    )
    
    # Create session factory
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    
    # Test connection
    try:
        async with _engine.begin() as conn:
            await conn.execute("SELECT 1")
        logger.info("✅ Database connection established successfully")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise
    
    # Create tables if they don't exist
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables initialized")
    
    return _engine


async def close_db() -> None:
    """
    Close the database connection and cleanup resources.
    
    Should be called on application shutdown.
    """
    global _engine, _session_factory
    
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("🔒 Database connection closed")


async def check_db_health() -> bool:
    """
    Check database connection health.
    
    Returns:
        bool: True if database is healthy, False otherwise
    """
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# Alias for backward compatibility
AsyncSessionLocal = get_session
async_session_factory = get_session_factory
