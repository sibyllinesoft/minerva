"""Origin management service with CRUD operations and business logic."""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

import httpx
from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from ..models.origin import Origin, OriginStatus, AuthType
from ..schemas.origins import (
    OriginCreate, OriginUpdate, OriginListRequest, OriginHealthCheck
)

logger = logging.getLogger(__name__)


class OriginManager:
    """Service for managing origin CRUD operations and business logic."""
    
    def __init__(self, db: AsyncSession):
        """Initialize the origin manager with database session."""
        self.db = db
    
    async def create_origin(self, origin_data: OriginCreate) -> Origin:
        """Create a new origin with validation."""
        logger.info(f"Creating origin: {origin_data.name} at {origin_data.url}")
        
        # Check for existing URL
        existing = await self.get_origin_by_url(str(origin_data.url))
        if existing:
            raise ValueError(f"Origin with URL {origin_data.url} already exists")
        
        # Create the origin
        origin = Origin(
            name=origin_data.name,
            url=str(origin_data.url),
            auth_type=origin_data.auth_type,
            status=origin_data.status,
            tls_verify=origin_data.tls_verify,
            tls_pinning=origin_data.tls_pinning,
            meta=origin_data.meta or {}
        )
        
        try:
            self.db.add(origin)
            await self.db.commit()
            await self.db.refresh(origin)
            logger.info(f"Created origin {origin.id}: {origin.name}")
            return origin
        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Failed to create origin: {e}")
            raise ValueError("Origin with this URL already exists") from e
    
    async def get_origin(self, origin_id: UUID) -> Optional[Origin]:
        """Get origin by ID."""
        result = await self.db.execute(
            select(Origin).where(Origin.id == origin_id)
        )
        return result.scalar_one_or_none()
    
    async def get_origin_by_url(self, url: str) -> Optional[Origin]:
        """Get origin by URL."""
        result = await self.db.execute(
            select(Origin).where(Origin.url == url)
        )
        return result.scalar_one_or_none()
    
    async def update_origin(self, origin_id: UUID, update_data: OriginUpdate) -> Optional[Origin]:
        """Update an existing origin."""
        origin = await self.get_origin(origin_id)
        if not origin:
            return None
        
        logger.info(f"Updating origin {origin_id}: {origin.name}")
        
        # Check for URL conflicts if URL is being updated
        if update_data.url and str(update_data.url) != origin.url:
            existing = await self.get_origin_by_url(str(update_data.url))
            if existing and existing.id != origin_id:
                raise ValueError(f"Origin with URL {update_data.url} already exists")
        
        # Update fields
        update_dict = update_data.dict(exclude_unset=True)
        for field, value in update_dict.items():
            if field == 'url' and value:
                setattr(origin, field, str(value))
            else:
                setattr(origin, field, value)
        
        origin.updated_at = datetime.utcnow()
        
        try:
            await self.db.commit()
            await self.db.refresh(origin)
            logger.info(f"Updated origin {origin_id}")
            return origin
        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Failed to update origin {origin_id}: {e}")
            raise ValueError("Origin with this URL already exists") from e
    
    async def delete_origin(self, origin_id: UUID) -> bool:
        """Soft delete an origin by setting status to deprecated."""
        origin = await self.get_origin(origin_id)
        if not origin:
            return False
        
        logger.info(f"Soft deleting origin {origin_id}: {origin.name}")
        
        origin.status = OriginStatus.DEPRECATED
        origin.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(origin)
        
        logger.info(f"Soft deleted origin {origin_id}")
        return True
    
    async def list_origins(
        self, 
        filters: OriginListRequest
    ) -> Tuple[List[Origin], int]:
        """List origins with filtering, pagination, and sorting."""
        logger.debug(f"Listing origins with filters: {filters}")
        
        # Build base query
        query = select(Origin)
        count_query = select(func.count(Origin.id))
        
        # Apply filters
        conditions = []
        
        if filters.status:
            conditions.append(Origin.status == filters.status)
        
        if filters.auth_type:
            conditions.append(Origin.auth_type == filters.auth_type)
        
        if filters.search:
            search_term = f"%{filters.search}%"
            search_conditions = [
                Origin.name.ilike(search_term),
                Origin.url.ilike(search_term),
                Origin.meta['tags'].astext.ilike(search_term)  # Search in JSON tags
            ]
            conditions.append(or_(*search_conditions))
        
        if conditions:
            where_clause = and_(*conditions)
            query = query.where(where_clause)
            count_query = count_query.where(where_clause)
        
        # Get total count
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        # Apply sorting
        sort_column = getattr(Origin, filters.sort_by, Origin.created_at)
        if filters.sort_order == "asc":
            query = query.order_by(sort_column.asc())
        else:
            query = query.order_by(sort_column.desc())
        
        # Apply pagination
        query = query.offset(filters.offset).limit(filters.limit)
        
        # Execute query
        result = await self.db.execute(query)
        origins = result.scalars().all()
        
        logger.debug(f"Found {len(origins)} origins (total: {total})")
        return list(origins), total
    
    async def update_origin_stats(
        self, 
        origin_id: UUID, 
        tool_count: Optional[int] = None,
        avg_response_time_ms: Optional[float] = None,
        success_rate: Optional[float] = None,
        last_error: Optional[str] = None
    ) -> Optional[Origin]:
        """Update origin statistics."""
        origin = await self.get_origin(origin_id)
        if not origin:
            return None
        
        logger.debug(f"Updating stats for origin {origin_id}")
        
        if tool_count is not None:
            origin.tool_count = tool_count
        
        if avg_response_time_ms is not None:
            origin.avg_response_time_ms = avg_response_time_ms
        
        if success_rate is not None:
            origin.success_rate = success_rate
        
        if last_error is not None:
            origin.last_error = last_error
            if last_error and origin.status != OriginStatus.ERROR:
                origin.status = OriginStatus.ERROR
        
        origin.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(origin)
        
        return origin
    
    async def mark_crawl_complete(
        self, 
        origin_id: UUID, 
        success: bool = True, 
        error: Optional[str] = None
    ) -> Optional[Origin]:
        """Mark a crawl as complete and update status."""
        origin = await self.get_origin(origin_id)
        if not origin:
            return None
        
        logger.info(f"Marking crawl complete for origin {origin_id}: success={success}")
        
        if success:
            origin.last_crawled_at = datetime.utcnow()
            if origin.status == OriginStatus.ERROR:
                origin.status = OriginStatus.ACTIVE
            origin.last_error = None
        else:
            origin.status = OriginStatus.ERROR
            origin.last_error = error
        
        origin.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(origin)
        
        return origin
    
    async def health_check_origin(self, origin: Origin) -> OriginHealthCheck:
        """Perform a basic health check on an origin."""
        logger.debug(f"Health checking origin {origin.id}: {origin.url}")
        
        start_time = datetime.utcnow()
        health_check = OriginHealthCheck(
            origin_id=origin.id,
            url=origin.url,
            status="error",
            timestamp=start_time
        )
        
        try:
            # Build headers for auth if needed
            headers = {}
            if origin.auth_type == AuthType.BEARER and origin.meta.get('bearer_token'):
                headers['Authorization'] = f"Bearer {origin.meta['bearer_token']}"
            elif origin.auth_type == AuthType.API_KEY and origin.meta.get('api_key'):
                api_key = origin.meta['api_key']
                key_header = origin.meta.get('api_key_header', 'X-API-Key')
                headers[key_header] = api_key
            
            # Build client config
            client_config = {
                'verify': origin.tls_verify,
                'timeout': origin.meta.get('timeout', 30.0),
                'headers': headers
            }
            
            # Try to make a basic request (HEAD or GET to /health, /status, or root)
            health_urls = [
                f"{origin.url.rstrip('/')}/health",
                f"{origin.url.rstrip('/')}/status", 
                f"{origin.url.rstrip('/')}/",
            ]
            
            async with httpx.AsyncClient(**client_config) as client:
                last_error = None
                
                for url in health_urls:
                    try:
                        response = await client.head(url)
                        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                        
                        if response.status_code < 400:
                            health_check.status = "healthy"
                            health_check.response_time_ms = response_time
                            break
                        else:
                            last_error = f"HTTP {response.status_code}"
                    except Exception as e:
                        last_error = str(e)
                        continue
                
                if health_check.status == "error":
                    health_check.error = last_error or "All health check endpoints failed"
        
        except Exception as e:
            logger.warning(f"Health check failed for origin {origin.id}: {e}")
            health_check.error = str(e)
        
        return health_check
    
    async def get_active_origins(self) -> List[Origin]:
        """Get all active origins for crawling."""
        result = await self.db.execute(
            select(Origin)
            .where(Origin.status == OriginStatus.ACTIVE)
            .order_by(Origin.last_crawled_at.asc().nulls_first())
        )
        return list(result.scalars().all())