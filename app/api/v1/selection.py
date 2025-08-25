"""Selection API endpoints for tool discovery and ranking."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...schemas.selection import SelectionRequest, SelectionResponse, SelectionStats
from ...services.selection_engine import get_selection_engine

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v1/selection", tags=["selection"])


async def get_selection_service():
    """Dependency to get selection engine."""
    return get_selection_engine()


@router.post(
    "/tools",
    response_model=SelectionResponse,
    summary="Select tools for a query",
    description="""
    Select the most relevant tools for a natural language query using hybrid search.
    
    The selection process includes:
    1. **Hybrid Search**: Combines BM25 text search with dense vector similarity
    2. **Cross-encoder Reranking**: Improves relevance using neural reranking (optional)
    3. **MMR Diversification**: Ensures diverse, non-redundant results
    4. **Utility Scoring**: Considers tool performance metrics (optional)
    5. **Policy Filtering**: Applies access control and business rules
    
    **Selection Modes:**
    - `fast`: Low latency, basic search (5 results, no reranking)
    - `balanced`: Good balance of quality and speed (10 results, light reranking) 
    - `thorough`: High quality, comprehensive search (15 results, full reranking)
    """
)
async def select_tools(
    request: SelectionRequest,
    db: AsyncSession = Depends(get_db),
    selection_engine = Depends(get_selection_service)
) -> SelectionResponse:
    """Select tools matching the query using hybrid search and ranking."""
    
    try:
        logger.info(
            f"Tool selection request: query='{request.query[:50]}...', "
            f"mode={request.mode}, max_results={request.max_results}"
        )
        
        # Validate query length
        if len(request.query.strip()) < 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Query must be at least 2 characters long"
            )
        
        # Perform tool selection
        response = await selection_engine.select_tools(request, db)
        
        logger.info(
            f"Selection complete: {len(response.tools)} tools found in "
            f"{response.total_time_ms:.1f}ms (cache_hit={response.cache_hit})"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tool selection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tool selection failed"
        )


@router.get(
    "/modes",
    summary="Get available selection modes",
    description="Get the configuration for all available selection modes (fast/balanced/thorough)."
)
async def get_selection_modes() -> Dict[str, Dict[str, Any]]:
    """Get available selection modes and their configurations."""
    
    from ...schemas.selection import DEFAULT_SELECTION_MODES
    
    modes = {}
    for name, config in DEFAULT_SELECTION_MODES.items():
        modes[name] = {
            "name": config.name,
            "max_candidates": config.max_candidates,
            "rerank_candidates": config.rerank_candidates,
            "final_results": config.final_results,
            "mmr_diversity": config.mmr_diversity,
            "enable_utility_scoring": config.enable_utility_scoring,
            "description": _get_mode_description(config.name)
        }
    
    return {
        "modes": modes,
        "default_mode": "balanced"
    }


@router.get(
    "/stats",
    response_model=SelectionStats,
    summary="Get selection engine statistics",
    description="Get performance and usage statistics for the selection engine."
)
async def get_selection_stats(
    selection_engine = Depends(get_selection_service)
) -> SelectionStats:
    """Get selection engine performance statistics."""
    
    try:
        # TODO: Implement actual stats collection
        # For now, return placeholder stats
        from datetime import datetime, timedelta
        
        stats = SelectionStats(
            total_queries=1000,  # Placeholder
            avg_query_time_ms=150.0,
            cache_hit_rate=0.25,
            avg_candidates_considered=45.0,
            avg_rerank_usage=0.60,
            mode_usage={
                "fast": 300,
                "balanced": 500,
                "thorough": 200
            },
            mode_performance={
                "fast": 75.0,
                "balanced": 150.0,
                "thorough": 350.0
            },
            error_rate=0.02,
            timeout_rate=0.001,
            stats_period_start=datetime.utcnow() - timedelta(days=7),
            stats_period_end=datetime.utcnow()
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get selection stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve selection statistics"
        )


@router.post(
    "/cache/clear",
    summary="Clear selection cache",
    description="Clear the selection engine cache. Use with caution in production."
)
async def clear_selection_cache(
    selection_engine = Depends(get_selection_service)
) -> Dict[str, Any]:
    """Clear the selection engine cache."""
    
    try:
        # Clear the cache
        selection_engine._cache.clear()
        
        logger.info("Selection cache cleared")
        
        return {
            "success": True,
            "message": "Selection cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear selection cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear selection cache"
        )


def _get_mode_description(mode: str) -> str:
    """Get human-readable description for selection mode."""
    
    descriptions = {
        "fast": "Low latency mode optimized for speed. Basic search with minimal processing.",
        "balanced": "Balanced mode offering good quality and reasonable speed. Uses reranking and utility scoring.",
        "thorough": "High quality mode with comprehensive search and advanced ranking. Slower but most accurate."
    }
    
    return descriptions.get(mode, "Unknown selection mode")