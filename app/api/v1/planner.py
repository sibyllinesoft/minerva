"""Planner API endpoints for tool execution planning."""

import logging
from typing import List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...schemas.planner import (
    PlanRequest, PlanResponse, PlannerStats, PlannerConfig,
    ExecutionPlan, PlannerMode
)
from ...services.planner_service import get_planner_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/planner")


@router.post("/generate", response_model=PlanResponse)
async def generate_plan(
    request: PlanRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate an execution plan for the given query.
    
    Creates a directed acyclic graph (DAG) of tool calls to fulfill the user's request.
    Plans are automatically validated and repaired if needed.
    """
    try:
        planner = get_planner_service()
        response = await planner.generate_plan(request, db)
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": response.error_code,
                    "message": response.error_message,
                    "planning_time_ms": response.planning_time_ms
                }
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Plan generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plan generation failed: {str(e)}"
        )


@router.get("/plan/{plan_id}", response_model=ExecutionPlan)
async def get_plan(plan_id: str):
    """
    Retrieve a previously generated execution plan by ID.
    
    Note: This would require plan storage in a real implementation.
    Currently returns a not implemented error.
    """
    # TODO: Implement plan storage and retrieval
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Plan storage not yet implemented"
    )


@router.post("/validate")
async def validate_plan(plan: ExecutionPlan):
    """
    Validate an execution plan without generating a new one.
    
    Checks for cycles, missing dependencies, and other structural issues.
    """
    try:
        planner = get_planner_service()
        errors = planner._validate_plan(plan)
        
        return {
            "valid": len(errors) == 0,
            "errors": [
                {
                    "type": error.error_type,
                    "message": error.message,
                    "tool_call_id": error.tool_call_id,
                    "suggested_fix": error.suggested_fix
                }
                for error in errors
            ]
        }
        
    except Exception as e:
        logger.error(f"Plan validation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plan validation failed: {str(e)}"
        )


@router.get("/modes", response_model=List[Dict[str, Any]])
async def get_planning_modes():
    """
    Get available planning modes and their characteristics.
    
    Returns information about trivial, simple, and complex planning modes.
    """
    return [
        {
            "mode": PlannerMode.TRIVIAL,
            "description": "Single tool execution with no dependencies",
            "characteristics": [
                "Fastest generation time",
                "Highest reliability", 
                "Limited capability",
                "No parallel execution"
            ],
            "typical_use_cases": [
                "Simple queries requiring one tool",
                "Fallback when complex planning fails"
            ],
            "max_tools": 1,
            "supports_parallel": False
        },
        {
            "mode": PlannerMode.SIMPLE,
            "description": "Linear sequence of 2-3 tools with simple dependencies",
            "characteristics": [
                "Fast generation time",
                "Good reliability",
                "Sequential execution",
                "Basic dependency handling"
            ],
            "typical_use_cases": [
                "Multi-step workflows",
                "Data processing pipelines",
                "Simple automation sequences"
            ],
            "max_tools": 3,
            "supports_parallel": False
        },
        {
            "mode": PlannerMode.COMPLEX,
            "description": "Full DAG with parallel execution and complex dependencies",
            "characteristics": [
                "Longer generation time",
                "Advanced capabilities",
                "Parallel execution support",
                "Complex dependency management"
            ],
            "typical_use_cases": [
                "Complex workflows",
                "Parallel data processing",
                "Sophisticated automation",
                "Multi-domain tasks"
            ],
            "max_tools": 10,
            "supports_parallel": True
        }
    ]


@router.get("/stats", response_model=PlannerStats)
async def get_planner_stats():
    """
    Get planner performance statistics.
    
    Returns metrics about planning success rates, timing, and mode usage.
    """
    try:
        planner = get_planner_service()
        stats = await planner.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get planner stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get planner stats: {str(e)}"
        )


@router.post("/test", response_model=PlanResponse)
async def test_planning(
    query: str = Query(..., description="Test query for planning"),
    mode: PlannerMode = Query(PlannerMode.SIMPLE, description="Planning mode to test"),
    max_tools: int = Query(5, ge=1, le=10, description="Maximum tools in plan"),
    db: AsyncSession = Depends(get_db)
):
    """
    Test endpoint for quickly generating plans with different parameters.
    
    Useful for development and debugging of the planning system.
    """
    try:
        request = PlanRequest(
            query=query,
            mode=mode,
            max_tools=max_tools,
            timeout_ms=30000  # Short timeout for testing
        )
        
        planner = get_planner_service()
        response = await planner.generate_plan(request, db)
        
        return response
        
    except Exception as e:
        logger.error(f"Test planning failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test planning failed: {str(e)}"
        )


@router.get("/config", response_model=PlannerConfig)
async def get_planner_config():
    """
    Get current planner configuration.
    
    Returns the configuration settings used for plan generation.
    """
    try:
        planner = get_planner_service()
        return planner.config
        
    except Exception as e:
        logger.error(f"Failed to get planner config: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get planner config: {str(e)}"
        )


@router.post("/config")
async def update_planner_config(config: PlannerConfig):
    """
    Update planner configuration.
    
    Allows runtime adjustment of planning parameters like timeouts,
    repair iterations, and caching settings.
    """
    try:
        planner = get_planner_service()
        planner.config = config
        
        return {
            "success": True,
            "message": "Planner configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to update planner config: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update planner config: {str(e)}"
        )


@router.post("/clear-cache")
async def clear_plan_cache():
    """
    Clear the plan generation cache.
    
    Forces regeneration of all future plans until cache rebuilds.
    """
    try:
        planner = get_planner_service()
        planner._cache.clear()
        
        return {
            "success": True,
            "message": "Plan cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear plan cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear plan cache: {str(e)}"
        )