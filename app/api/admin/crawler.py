"""Admin API endpoints for crawler management."""

import logging
from typing import Dict, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...services.crawler import get_crawler, get_scheduler
from ...services.origin_manager import OriginManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v1/admin/crawler", tags=["admin", "crawler"])


async def get_origin_manager(db: AsyncSession = Depends(get_db)) -> OriginManager:
    """Dependency to get origin manager."""
    return OriginManager(db)


@router.post(
    "/crawl",
    summary="Trigger manual crawl",
    description="Trigger a manual crawl of all active origins to discover tools."
)
async def trigger_manual_crawl(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Trigger a manual crawl of all active origins."""
    try:
        logger.info("Manual crawl triggered via API")
        
        crawler = get_crawler()
        
        # Run crawl in background
        background_tasks.add_task(crawler.crawl_all_origins, db)
        
        return {
            "success": True,
            "message": "Manual crawl started in background",
            "status": "started"
        }
    
    except Exception as e:
        logger.error(f"Failed to trigger manual crawl: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start crawl"
        )


@router.post(
    "/crawl/{origin_id}",
    summary="Crawl specific origin",
    description="Trigger a crawl for a specific origin to discover tools."
)
async def crawl_origin(
    origin_id: UUID,
    background_tasks: BackgroundTasks,
    origin_manager: OriginManager = Depends(get_origin_manager)
) -> Dict[str, Any]:
    """Crawl a specific origin."""
    try:
        logger.info(f"Manual origin crawl triggered for {origin_id}")
        
        # Get the origin
        origin = await origin_manager.get_origin(origin_id)
        if not origin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Origin with ID {origin_id} not found"
            )
        
        crawler = get_crawler()
        
        # Run crawl in background
        async def crawl_task():
            from ...core.database import get_db
            async for session in get_db():
                try:
                    result = await crawler.crawl_origin(origin, session)
                    logger.info(f"Origin {origin_id} crawl completed: {result}")
                    break
                except Exception as e:
                    logger.error(f"Crawl task failed: {e}")
                    break
        
        background_tasks.add_task(crawl_task)
        
        return {
            "success": True,
            "message": f"Crawl started for origin {origin_id}",
            "origin_id": str(origin_id),
            "origin_name": origin.name,
            "status": "started"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger origin crawl {origin_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start origin crawl"
        )


@router.get(
    "/status",
    summary="Get crawler status",
    description="Get the current status of the crawler and scheduler."
)
async def get_crawler_status() -> Dict[str, Any]:
    """Get crawler and scheduler status."""
    try:
        scheduler = get_scheduler()
        
        return {
            "crawler": {
                "available": True,
                "max_concurrent": get_crawler().max_concurrent,
                "timeout": get_crawler().timeout
            },
            "scheduler": {
                "running": scheduler._running,
                "task_active": scheduler._task is not None and not scheduler._task.done() if scheduler._task else False
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get crawler status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve crawler status"
        )


@router.post(
    "/scheduler/start",
    summary="Start crawler scheduler",
    description="Start the periodic crawler scheduler."
)
async def start_scheduler(
    interval_hours: float = 24.0,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Start the crawler scheduler."""
    try:
        if interval_hours < 0.1 or interval_hours > 168:  # Min 6 minutes, max 1 week
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Interval must be between 0.1 and 168 hours"
            )
        
        scheduler = get_scheduler()
        
        if scheduler._running:
            return {
                "success": True,
                "message": "Scheduler is already running",
                "interval_hours": interval_hours,
                "status": "already_running"
            }
        
        await scheduler.start_scheduler(db, interval_hours)
        
        logger.info(f"Crawler scheduler started with {interval_hours}h interval")
        
        return {
            "success": True,
            "message": f"Scheduler started with {interval_hours}h interval",
            "interval_hours": interval_hours,
            "status": "started"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start scheduler"
        )


@router.post(
    "/scheduler/stop",
    summary="Stop crawler scheduler",
    description="Stop the periodic crawler scheduler."
)
async def stop_scheduler() -> Dict[str, Any]:
    """Stop the crawler scheduler."""
    try:
        scheduler = get_scheduler()
        
        if not scheduler._running:
            return {
                "success": True,
                "message": "Scheduler is not running",
                "status": "already_stopped"
            }
        
        await scheduler.stop_scheduler()
        
        logger.info("Crawler scheduler stopped")
        
        return {
            "success": True,
            "message": "Scheduler stopped",
            "status": "stopped"
        }
    
    except Exception as e:
        logger.error(f"Failed to stop scheduler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop scheduler"
        )


@router.get(
    "/stats",
    summary="Get crawl statistics",
    description="Get statistics about recent crawling activity."
)
async def get_crawl_stats(
    origin_manager: OriginManager = Depends(get_origin_manager)
) -> Dict[str, Any]:
    """Get crawling statistics."""
    try:
        # Get basic stats from origins
        from sqlalchemy import select, func
        from ...models.origin import Origin, OriginStatus
        from ...models.tool import Tool
        
        # This would normally be done through the origin manager
        # For now, return basic stats
        origins = await origin_manager.get_active_origins()
        
        active_origins = len(origins)
        total_tools = sum(origin.tool_count for origin in origins)
        
        # Calculate some basic stats
        origins_with_errors = len([o for o in origins if o.status == OriginStatus.ERROR])
        recently_crawled = len([o for o in origins if o.last_crawled_at])
        
        return {
            "origins": {
                "total_active": active_origins,
                "recently_crawled": recently_crawled,
                "with_errors": origins_with_errors,
                "success_rate": (recently_crawled - origins_with_errors) / max(recently_crawled, 1)
            },
            "tools": {
                "total_discovered": total_tools,
                "average_per_origin": total_tools / max(active_origins, 1)
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get crawl stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve crawl statistics"
        )