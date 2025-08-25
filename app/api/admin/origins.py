"""Admin API endpoints for origin management."""

import logging
from typing import Dict, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...models.origin import OriginStatus, AuthType
from ...schemas.origins import (
    OriginCreate, OriginUpdate, OriginResponse, OriginListRequest, 
    OriginListResponse, OriginHealthCheck
)
from ...services.origin_manager import OriginManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v1/admin/origins", tags=["admin", "origins"])


async def get_origin_manager(db: AsyncSession = Depends(get_db)) -> OriginManager:
    """Dependency to get origin manager."""
    return OriginManager(db)


@router.post(
    "",
    response_model=OriginResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new origin",
    description="Create a new upstream MCP server origin for crawling and tool discovery."
)
async def create_origin(
    origin_data: OriginCreate,
    origin_manager: OriginManager = Depends(get_origin_manager)
) -> OriginResponse:
    """Create a new origin."""
    try:
        logger.info(f"Creating origin: {origin_data.name}")
        origin = await origin_manager.create_origin(origin_data)
        return OriginResponse.from_orm(origin)
    
    except ValueError as e:
        logger.warning(f"Origin creation validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create origin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create origin"
        )


@router.get(
    "",
    response_model=OriginListResponse,
    summary="List origins",
    description="List origins with filtering, pagination, and sorting options."
)
async def list_origins(
    status_filter: OriginStatus = Query(None, alias="status", description="Filter by status"),
    auth_type_filter: AuthType = Query(None, alias="auth_type", description="Filter by auth type"),
    search: str = Query(None, max_length=255, description="Search in name, url, or tags"),
    limit: int = Query(50, ge=1, le=1000, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    origin_manager: OriginManager = Depends(get_origin_manager)
) -> OriginListResponse:
    """List origins with filtering and pagination."""
    try:
        logger.debug(f"Listing origins with filters: status={status_filter}, limit={limit}")
        
        filters = OriginListRequest(
            status=status_filter,
            auth_type=auth_type_filter,
            search=search,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        origins, total = await origin_manager.list_origins(filters)
        
        return OriginListResponse(
            origins=[OriginResponse.from_orm(origin) for origin in origins],
            total=total,
            limit=limit,
            offset=offset
        )
    
    except ValueError as e:
        logger.warning(f"Origin listing validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to list origins: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list origins"
        )


@router.get(
    "/{origin_id}",
    response_model=OriginResponse,
    summary="Get origin by ID",
    description="Retrieve a specific origin by its unique identifier."
)
async def get_origin(
    origin_id: UUID,
    origin_manager: OriginManager = Depends(get_origin_manager)
) -> OriginResponse:
    """Get a specific origin by ID."""
    try:
        logger.debug(f"Getting origin {origin_id}")
        origin = await origin_manager.get_origin(origin_id)
        
        if not origin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Origin with ID {origin_id} not found"
            )
        
        return OriginResponse.from_orm(origin)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get origin {origin_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve origin"
        )


@router.put(
    "/{origin_id}",
    response_model=OriginResponse,
    summary="Update origin",
    description="Update an existing origin's configuration and settings."
)
async def update_origin(
    origin_id: UUID,
    update_data: OriginUpdate,
    origin_manager: OriginManager = Depends(get_origin_manager)
) -> OriginResponse:
    """Update an existing origin."""
    try:
        logger.info(f"Updating origin {origin_id}")
        origin = await origin_manager.update_origin(origin_id, update_data)
        
        if not origin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Origin with ID {origin_id} not found"
            )
        
        return OriginResponse.from_orm(origin)
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Origin update validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to update origin {origin_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update origin"
        )


@router.delete(
    "/{origin_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete origin",
    description="Soft delete an origin by marking it as deprecated."
)
async def delete_origin(
    origin_id: UUID,
    origin_manager: OriginManager = Depends(get_origin_manager)
):
    """Soft delete an origin."""
    try:
        logger.info(f"Deleting origin {origin_id}")
        success = await origin_manager.delete_origin(origin_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Origin with ID {origin_id} not found"
            )
        
        return None  # 204 No Content
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete origin {origin_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete origin"
        )


@router.post(
    "/{origin_id}/health",
    response_model=OriginHealthCheck,
    summary="Health check origin",
    description="Perform a health check on a specific origin to test connectivity and responsiveness."
)
async def health_check_origin(
    origin_id: UUID,
    origin_manager: OriginManager = Depends(get_origin_manager)
) -> OriginHealthCheck:
    """Perform a health check on an origin."""
    try:
        logger.info(f"Health checking origin {origin_id}")
        origin = await origin_manager.get_origin(origin_id)
        
        if not origin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Origin with ID {origin_id} not found"
            )
        
        health_check = await origin_manager.health_check_origin(origin)
        return health_check
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to health check origin {origin_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform health check"
        )


@router.get(
    "/{origin_id}/stats",
    response_model=Dict[str, Any],
    summary="Get origin statistics",
    description="Retrieve detailed statistics and performance metrics for an origin."
)
async def get_origin_stats(
    origin_id: UUID,
    origin_manager: OriginManager = Depends(get_origin_manager)
) -> Dict[str, Any]:
    """Get detailed statistics for an origin."""
    try:
        logger.debug(f"Getting stats for origin {origin_id}")
        origin = await origin_manager.get_origin(origin_id)
        
        if not origin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Origin with ID {origin_id} not found"
            )
        
        # Build comprehensive stats
        stats = {
            "origin_id": origin.id,
            "name": origin.name,
            "url": origin.url,
            "status": origin.status,
            "tool_count": origin.tool_count,
            "avg_response_time_ms": origin.avg_response_time_ms,
            "success_rate": origin.success_rate,
            "last_crawled_at": origin.last_crawled_at,
            "last_error": origin.last_error,
            "created_at": origin.created_at,
            "updated_at": origin.updated_at,
            "meta": origin.meta
        }
        
        return stats
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats for origin {origin_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve origin statistics"
        )