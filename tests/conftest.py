"""Test configuration and fixtures for Meta MCP."""

import asyncio
import os
import pytest
from typing import AsyncGenerator
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text

# Set test environment
os.environ["ENV"] = "test"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test_meta_mcp"

from app.main import app
from app.core.database import init_database, close_database, get_db
from app.models.origin import Origin, OriginStatus, AuthType
from app.models.base import Base


# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/test_meta_mcp"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create a test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_pre_ping=True
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Drop all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture(scope="function")
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session_maker = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        # Start a transaction
        await session.begin()
        
        yield session
        
        # Rollback the transaction
        await session.rollback()


@pytest.fixture(scope="function") 
async def client(db_session: AsyncSession) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a test HTTP client with database override."""
    
    # Override the get_db dependency
    async def get_test_db():
        yield db_session
    
    app.dependency_overrides[get_db] = get_test_db
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture
async def sample_origin(db_session: AsyncSession) -> Origin:
    """Create a sample origin for testing."""
    origin = Origin(
        id=uuid4(),
        name="Sample Origin",
        url="https://sample.com/mcp",
        auth_type=AuthType.NONE,
        status=OriginStatus.ACTIVE,
        tls_verify=True,
        meta={
            "description": "Sample MCP server for testing",
            "tags": ["test", "sample"],
            "refresh_interval": 3600
        }
    )
    
    db_session.add(origin)
    await db_session.commit()
    await db_session.refresh(origin)
    
    return origin


@pytest.fixture
def sample_origin_id(sample_origin: Origin) -> str:
    """Get the ID of the sample origin as a string."""
    return str(sample_origin.id)


@pytest.fixture(scope="session", autouse=True)
async def setup_test_database():
    """Set up test database before running tests."""
    # This will run once before all tests
    
    # Create test database if it doesn't exist
    admin_engine = create_async_engine(
        "postgresql+asyncpg://test:test@localhost:5432/postgres",
        isolation_level="AUTOCOMMIT"
    )
    
    try:
        async with admin_engine.connect() as conn:
            # Check if test database exists
            result = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = 'test_meta_mcp'")
            )
            
            if not result.fetchone():
                # Create test database
                await conn.execute(text("CREATE DATABASE test_meta_mcp"))
                
        # Enable pgvector extension
        test_engine = create_async_engine(TEST_DATABASE_URL)
        async with test_engine.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await test_engine.dispose()
        
    except Exception as e:
        print(f"Warning: Could not set up test database: {e}")
        print("Make sure PostgreSQL is running with a 'test' user")
    
    finally:
        await admin_engine.dispose()


# Test configuration overrides
@pytest.fixture(autouse=True)
def test_config():
    """Set test configuration."""
    # Override configuration for testing
    os.environ.update({
        "DATABASE_URL": TEST_DATABASE_URL,
        "LOG_LEVEL": "DEBUG",
        "MODELS_EMBEDDINGS_PROVIDER": "off",
        "MODELS_RERANKER_PROVIDER": "off", 
        "MODELS_PLANNER_PROVIDER": "off"
    })


# Skip slow tests by default
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# Add async test support
@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio backend for anyio."""
    return "asyncio"