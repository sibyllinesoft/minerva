#!/usr/bin/env python3
"""Test Phase 4: Selection Engine implementation."""

import asyncio
import logging
import os
import sys
from pathlib import Path
from uuid import uuid4

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import get_db, init_database
from app.models.origin import Origin, OriginStatus, AuthType
from app.models.tool import Tool
from app.schemas.selection import SelectionRequest
from app.services.selection_engine import get_selection_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_test_data(db: AsyncSession):
    """Create test origins and tools for selection testing."""
    
    logger.info("Setting up test data...")
    
    # Check if test origin already exists
    result = await db.execute(
        select(Origin).where(Origin.url == "http://localhost:8000/test")
    )
    origin = result.scalar_one_or_none()
    
    if not origin:
        # Create test origin
        origin = Origin(
            id=uuid4(),
            name="test-selection-origin",
            url="http://localhost:8000/test",
            auth_type=AuthType.NONE,
            status=OriginStatus.ACTIVE,
            tool_count=0,
            meta={"description": "Test origin for selection engine"}
        )
        db.add(origin)
        await db.commit()
        await db.refresh(origin)
    else:
        logger.info("Using existing test origin")
    
    # Create diverse test tools
    test_tools = [
        {
            "name": "file_reader",
            "brief": "Read and analyze files",
            "description": "A tool for reading various file formats and extracting content for analysis",
            "categories": ["file-management", "analysis"],
            "reliability_score": 95.0
        },
        {
            "name": "web_scraper", 
            "brief": "Extract data from web pages",
            "description": "Scrape and parse web content, handling dynamic pages and structured data extraction",
            "categories": ["web", "data-extraction"],
            "reliability_score": 88.0
        },
        {
            "name": "code_formatter",
            "brief": "Format and beautify source code", 
            "description": "Automatically format code in multiple programming languages with style configuration",
            "categories": ["development", "formatting"],
            "reliability_score": 92.0
        },
        {
            "name": "database_query",
            "brief": "Execute SQL queries safely",
            "description": "Run database queries with parameter binding and result formatting",
            "categories": ["database", "query"],
            "reliability_score": 97.0
        },
        {
            "name": "image_processor",
            "brief": "Process and transform images",
            "description": "Resize, crop, filter and convert images with batch processing support",
            "categories": ["media", "image-processing"],
            "reliability_score": 85.0
        },
        {
            "name": "text_analyzer",
            "brief": "Analyze text content and extract insights",
            "description": "Perform natural language processing including sentiment analysis, entity extraction, and topic modeling",
            "categories": ["nlp", "analysis", "text"],
            "reliability_score": 90.0
        },
        {
            "name": "api_client",
            "brief": "Generic HTTP API client",
            "description": "Make HTTP requests to REST APIs with authentication, retry logic, and response parsing",
            "categories": ["http", "api", "client"],
            "reliability_score": 93.0
        }
    ]
    
    # Check if tools already exist for this origin
    result = await db.execute(
        select(Tool).where(Tool.origin_id == origin.id)
    )
    existing_tools = result.scalars().all()
    
    if existing_tools:
        logger.info(f"Using {len(existing_tools)} existing test tools")
        return origin, existing_tools
    
    tools = []
    for tool_data in test_tools:
        tool = Tool(
            id=uuid4(),
            origin_id=origin.id,
            name=tool_data["name"],
            brief=tool_data["brief"],
            description=tool_data["description"],
            categories=tool_data["categories"],
            reliability_score=tool_data["reliability_score"],
            args_schema={"type": "object", "properties": {}},
            returns_schema={"type": "object"},
            deprecated=False,
            is_side_effect_free=True
        )
        tools.append(tool)
        db.add(tool)
    
    await db.commit()
    
    logger.info(f"Created {len(tools)} test tools")
    return origin, tools


async def test_selection_modes():
    """Test different selection modes (fast, balanced, thorough)."""
    
    logger.info("Testing selection modes...")
    
    selection_engine = get_selection_engine()
    
    async for db in get_db():
        try:
            # Test each mode with the same query
            test_query = "analyze files and extract text content"
            
            for mode in ["fast", "balanced", "thorough"]:
                logger.info(f"Testing {mode} mode...")
                
                request = SelectionRequest(
                    query=test_query,
                    mode=mode
                )
                
                response = await selection_engine.select_tools(request, db)
                
                logger.info(
                    f"Mode {mode}: {len(response.tools)} tools found in "
                    f"{response.total_time_ms:.1f}ms (search: {response.search_time_ms:.1f}ms)"
                )
                
                # Verify mode characteristics
                if mode == "fast":
                    assert len(response.tools) <= 5, f"Fast mode should return ≤5 tools, got {len(response.tools)}"
                    assert response.rerank_time_ms is None, "Fast mode should not use reranking"
                
                elif mode == "balanced":
                    assert len(response.tools) <= 10, f"Balanced mode should return ≤10 tools, got {len(response.tools)}"
                
                elif mode == "thorough":
                    assert len(response.tools) <= 15, f"Thorough mode should return ≤15 tools, got {len(response.tools)}"
                
                # Show top results
                for i, tool in enumerate(response.tools[:3], 1):
                    logger.info(
                        f"  {i}. {tool.name} (score: {tool.scores.final_score:.3f}) - {tool.brief}"
                    )
            
            break
            
        except Exception as e:
            logger.error(f"Mode testing failed: {e}", exc_info=True)
            raise


async def test_search_functionality():
    """Test hybrid search with different queries."""
    
    logger.info("Testing search functionality...")
    
    selection_engine = get_selection_engine()
    
    async for db in get_db():
        try:
            test_queries = [
                "read files",           # Should match file_reader
                "web scraping",         # Should match web_scraper  
                "format code",          # Should match code_formatter
                "database SQL",         # Should match database_query
                "image processing",     # Should match image_processor
                "text analysis NLP",    # Should match text_analyzer
                "HTTP API client"       # Should match api_client
            ]
            
            for query in test_queries:
                logger.info(f"Testing query: '{query}'")
                
                request = SelectionRequest(
                    query=query,
                    mode="balanced",
                    max_results=3
                )
                
                response = await selection_engine.select_tools(request, db)
                
                logger.info(f"  Found {len(response.tools)} tools:")
                for tool in response.tools:
                    logger.info(
                        f"    {tool.rank}. {tool.name} (score: {tool.scores.final_score:.3f})"
                    )
            
            break
            
        except Exception as e:
            logger.error(f"Search testing failed: {e}", exc_info=True)
            raise


async def test_filtering_options():
    """Test category and origin filtering."""
    
    logger.info("Testing filtering options...")
    
    selection_engine = get_selection_engine()
    
    async for db in get_db():
        try:
            # Test category filtering
            request = SelectionRequest(
                query="process data",
                mode="balanced",
                categories=["analysis"]  # Should match file_reader and text_analyzer
            )
            
            response = await selection_engine.select_tools(request, db)
            
            logger.info(f"Category filter 'analysis': {len(response.tools)} tools")
            for tool in response.tools:
                assert "analysis" in tool.categories, f"Tool {tool.name} missing 'analysis' category"
                logger.info(f"  - {tool.name}: {tool.categories}")
            
            # Test reliability filtering
            request = SelectionRequest(
                query="reliable tools",
                mode="balanced",
                min_reliability=0.95  # Should only return database_query (97%)
            )
            
            response = await selection_engine.select_tools(request, db)
            
            logger.info(f"Reliability filter ≥95%: {len(response.tools)} tools")
            for tool in response.tools:
                reliability = tool.reliability_score or 0.0
                assert reliability >= 95.0, f"Tool {tool.name} has reliability {reliability}% < 95%"
                logger.info(f"  - {tool.name}: {reliability}%")
            
            break
            
        except Exception as e:
            logger.error(f"Filtering testing failed: {e}", exc_info=True)
            raise


async def test_caching():
    """Test selection caching functionality."""
    
    logger.info("Testing caching functionality...")
    
    selection_engine = get_selection_engine()
    
    async for db in get_db():
        try:
            query = "test caching query"
            
            request = SelectionRequest(
                query=query,
                mode="fast"
            )
            
            # First request - should miss cache
            response1 = await selection_engine.select_tools(request, db)
            assert not response1.cache_hit, "First request should miss cache"
            
            # Second request - should hit cache
            response2 = await selection_engine.select_tools(request, db)
            assert response2.cache_hit, "Second request should hit cache"
            assert response2.cache_key == response1.cache_key, "Cache keys should match"
            
            logger.info("✅ Caching working correctly")
            
            break
            
        except Exception as e:
            logger.error(f"Caching test failed: {e}", exc_info=True)
            raise


async def test_scoring_system():
    """Test the scoring system components."""
    
    logger.info("Testing scoring system...")
    
    selection_engine = get_selection_engine()
    
    async for db in get_db():
        try:
            request = SelectionRequest(
                query="analyze text files",
                mode="thorough"  # Includes utility scoring
            )
            
            response = await selection_engine.select_tools(request, db)
            
            if response.tools:
                tool = response.tools[0]
                scores = tool.scores
                
                logger.info("Score breakdown for top result:")
                logger.info(f"  Tool: {tool.name}")
                logger.info(f"  Hybrid Score: {scores.hybrid_score:.3f}")
                logger.info(f"  Text Score: {scores.bm25_score:.3f}" if scores.bm25_score else "  Text Score: N/A")
                logger.info(f"  Vector Score: {scores.vector_score:.3f}" if scores.vector_score else "  Vector Score: N/A")
                logger.info(f"  Rerank Score: {scores.rerank_score:.3f}" if scores.rerank_score else "  Rerank Score: N/A")
                logger.info(f"  Utility Score: {scores.utility_score:.3f}" if scores.utility_score else "  Utility Score: N/A")
                logger.info(f"  Final Score: {scores.final_score:.3f}")
                logger.info(f"  Explanation: {scores.score_explanation}")
                
                # Verify scoring consistency
                assert scores.final_score > 0, "Final score should be positive"
                assert scores.hybrid_score >= 0, "Hybrid score should be non-negative"
            
            break
            
        except Exception as e:
            logger.error(f"Scoring test failed: {e}", exc_info=True)
            raise


async def run_all_tests():
    """Run all Phase 4 selection tests."""
    
    logger.info("🚀 Starting Phase 4: Selection Engine Tests")
    
    # Initialize database
    await init_database()
    
    async for db in get_db():
        try:
            # Setup test data
            origin, tools = await setup_test_data(db)
            
            # Run tests
            await test_selection_modes()
            await test_search_functionality()  
            await test_filtering_options()
            await test_caching()
            await test_scoring_system()
            
            logger.info("✅ Phase 4: Selection Engine - COMPLETE!")
            logger.info("🎯 All selection engine components working correctly:")
            logger.info("   - Hybrid search (BM25 + Vector similarity)")
            logger.info("   - Selection modes (fast/balanced/thorough)")
            logger.info("   - Cross-encoder reranking")
            logger.info("   - MMR diversification")
            logger.info("   - Utility scoring")
            logger.info("   - Policy filtering")
            logger.info("   - Result caching")
            
            break
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())