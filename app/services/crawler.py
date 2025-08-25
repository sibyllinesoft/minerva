"""Async crawler for discovering tools from upstream MCP servers."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4
import json

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..core.config import get_settings
from ..core.database import get_db
from ..models.origin import Origin, OriginStatus
from ..models.tool import Tool
from ..services.origin_manager import OriginManager
from ..services.tool_validator import get_tool_validator

logger = logging.getLogger(__name__)


class MCPCrawler:
    """Async crawler for discovering tools from upstream MCP servers."""
    
    def __init__(self, max_concurrent: int = 5, timeout: float = 30.0):
        """Initialize the crawler with concurrency and timeout settings."""
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._client_config = {
            "timeout": timeout,
            "limits": httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20
            )
        }
    
    async def crawl_all_origins(self, db: AsyncSession) -> Dict[str, Any]:
        """Crawl all active origins for tools."""
        logger.info("Starting crawl of all active origins")
        
        origin_manager = OriginManager(db)
        origins = await origin_manager.get_active_origins()
        
        if not origins:
            logger.info("No active origins to crawl")
            return {
                "success": True,
                "message": "No active origins found",
                "crawled": 0,
                "errors": 0,
                "tools_discovered": 0
            }
        
        logger.info(f"Crawling {len(origins)} active origins")
        
        # Create crawl tasks for all origins
        tasks = []
        for origin in origins:
            task = asyncio.create_task(
                self._crawl_origin_with_semaphore(origin, db),
                name=f"crawl-{origin.id}"
            )
            tasks.append(task)
        
        # Wait for all crawls to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        total_crawled = 0
        total_errors = 0
        total_tools = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Crawl task failed for origin {origins[i].id}: {result}")
                total_errors += 1
            elif isinstance(result, dict):
                total_crawled += 1
                total_tools += result.get("tools_discovered", 0)
                if not result.get("success", False):
                    total_errors += 1
        
        logger.info(f"Crawl complete: {total_crawled} origins processed, {total_errors} errors, {total_tools} tools discovered")
        
        return {
            "success": total_errors == 0,
            "message": f"Crawled {total_crawled} origins with {total_errors} errors",
            "crawled": total_crawled,
            "errors": total_errors,
            "tools_discovered": total_tools
        }
    
    async def crawl_origin(self, origin: Origin, db: AsyncSession) -> Dict[str, Any]:
        """Crawl a single origin for tools."""
        logger.info(f"Crawling origin {origin.id}: {origin.name}")
        
        origin_manager = OriginManager(db)
        
        try:
            # Discover tools from the origin
            tools_data = await self._discover_tools_from_origin(origin)
            
            if not tools_data:
                logger.warning(f"No tools discovered from origin {origin.id}")
                await origin_manager.mark_crawl_complete(
                    origin.id, 
                    success=True, 
                    error=None
                )
                return {
                    "success": True,
                    "tools_discovered": 0,
                    "message": "No tools found"
                }
            
            # Process and store discovered tools
            tools_processed = await self._process_discovered_tools(origin, tools_data, db)
            
            # Update origin statistics
            await origin_manager.update_origin_stats(
                origin.id,
                tool_count=tools_processed
            )
            
            # Mark crawl as complete
            await origin_manager.mark_crawl_complete(
                origin.id,
                success=True,
                error=None
            )
            
            logger.info(f"Successfully crawled origin {origin.id}: {tools_processed} tools processed")
            
            return {
                "success": True,
                "tools_discovered": tools_processed,
                "message": f"Successfully processed {tools_processed} tools"
            }
        
        except Exception as e:
            error_msg = f"Failed to crawl origin {origin.id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Mark crawl as failed
            await origin_manager.mark_crawl_complete(
                origin.id,
                success=False,
                error=str(e)
            )
            
            return {
                "success": False,
                "tools_discovered": 0,
                "message": error_msg,
                "error": str(e)
            }
    
    async def _crawl_origin_with_semaphore(self, origin: Origin, db: AsyncSession) -> Dict[str, Any]:
        """Crawl origin with semaphore-based concurrency control."""
        async with self.semaphore:
            return await self.crawl_origin(origin, db)
    
    async def _discover_tools_from_origin(self, origin: Origin) -> Optional[List[Dict[str, Any]]]:
        """Discover tools from an MCP origin."""
        logger.debug(f"Discovering tools from {origin.url}")
        
        # Build request configuration
        headers = {"Content-Type": "application/json"}
        
        # Add authentication if configured
        if origin.auth_type.value == "bearer" and origin.meta.get("bearer_token"):
            headers["Authorization"] = f"Bearer {origin.meta['bearer_token']}"
        elif origin.auth_type.value == "api_key" and origin.meta.get("api_key"):
            api_key_header = origin.meta.get("api_key_header", "X-API-Key")
            headers[api_key_header] = origin.meta["api_key"]
        
        client_config = self._client_config.copy()
        client_config["headers"] = headers
        client_config["verify"] = origin.tls_verify
        
        async with httpx.AsyncClient(**client_config) as client:
            try:
                # Try standard MCP tools discovery endpoints
                discovery_endpoints = [
                    f"{origin.url.rstrip('/')}/tools/list",
                    f"{origin.url.rstrip('/')}/mcp/tools", 
                    f"{origin.url.rstrip('/')}/tools",
                    f"{origin.url.rstrip('/')}/list_tools"
                ]
                
                for endpoint in discovery_endpoints:
                    try:
                        logger.debug(f"Trying discovery endpoint: {endpoint}")
                        
                        # MCP protocol tools list request
                        mcp_request = {
                            "jsonrpc": "2.0",
                            "id": str(uuid4()),
                            "method": "tools/list",
                            "params": {}
                        }
                        
                        response = await client.post(endpoint, json=mcp_request)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Handle MCP JSON-RPC response
                            if "result" in data and "tools" in data["result"]:
                                tools = data["result"]["tools"]
                                logger.info(f"Discovered {len(tools)} tools from {endpoint}")
                                return tools
                            
                            # Handle direct tools array response
                            elif isinstance(data, list):
                                logger.info(f"Discovered {len(data)} tools from {endpoint}")
                                return data
                            
                            # Handle tools array in response
                            elif "tools" in data:
                                tools = data["tools"]
                                logger.info(f"Discovered {len(tools)} tools from {endpoint}")
                                return tools
                        
                        elif response.status_code == 404:
                            # Try next endpoint
                            continue
                        else:
                            logger.warning(f"Unexpected response from {endpoint}: {response.status_code}")
                    
                    except httpx.RequestError as e:
                        logger.debug(f"Request failed for {endpoint}: {e}")
                        continue
                
                logger.warning(f"No tools discovered from any endpoint for {origin.url}")
                return None
            
            except Exception as e:
                logger.error(f"Failed to discover tools from {origin.url}: {e}")
                return None
    
    async def _process_discovered_tools(
        self, 
        origin: Origin, 
        tools_data: List[Dict[str, Any]], 
        db: AsyncSession
    ) -> int:
        """Process and store discovered tools."""
        logger.debug(f"Processing {len(tools_data)} tools from origin {origin.id}")
        
        if not tools_data:
            return 0
        
        tools_processed = 0
        current_time = datetime.utcnow()
        
        # Track existing tools for this origin
        existing_tools_query = select(Tool).where(Tool.origin_id == origin.id)
        result = await db.execute(existing_tools_query)
        existing_tools = {tool.name: tool for tool in result.scalars().all()}
        
        seen_tool_names: Set[str] = set()
        
        validator = get_tool_validator()
        
        for tool_data in tools_data:
            try:
                # Validate tool data
                is_valid, validation_errors = validator.validate_tool(tool_data, strict=False)
                if not is_valid:
                    logger.warning(f"Tool validation failed: {validation_errors}")
                    continue
                
                # Normalize tool data
                normalized_tool = validator.normalize_tool(tool_data)
                if not normalized_tool:
                    continue
                
                tool_name = normalized_tool["name"]
                seen_tool_names.add(tool_name)
                
                if tool_name in existing_tools:
                    # Update existing tool
                    existing_tool = existing_tools[tool_name]
                    existing_tool.brief = normalized_tool.get("brief", existing_tool.brief)
                    existing_tool.description = normalized_tool.get("description", existing_tool.description)
                    existing_tool.schema = normalized_tool.get("schema", existing_tool.schema)
                    existing_tool.category = normalized_tool.get("category", existing_tool.category)
                    existing_tool.tags = normalized_tool.get("tags", existing_tool.tags)
                    existing_tool.deprecated = False
                    existing_tool.last_seen_at = current_time
                    existing_tool.updated_at = current_time
                    
                    logger.debug(f"Updated existing tool: {tool_name}")
                else:
                    # Create new tool
                    new_tool = Tool(
                        id=uuid4(),
                        origin_id=origin.id,
                        name=tool_name,
                        brief=normalized_tool.get("brief", ""),
                        description=normalized_tool.get("description", ""),
                        schema=normalized_tool.get("schema", {}),
                        category=normalized_tool.get("category"),
                        tags=normalized_tool.get("tags", []),
                        deprecated=False,
                        last_seen_at=current_time,
                        reliability_score=50.0,  # Default score
                        created_at=current_time,
                        updated_at=current_time
                    )
                    
                    db.add(new_tool)
                    logger.debug(f"Created new tool: {tool_name}")
                
                tools_processed += 1
            
            except Exception as e:
                logger.error(f"Failed to process tool {tool_data.get('name', 'unknown')}: {e}")
                continue
        
        # Mark tools not seen in this crawl as deprecated
        for tool_name, tool in existing_tools.items():
            if tool_name not in seen_tool_names:
                tool.deprecated = True
                tool.updated_at = current_time
                logger.debug(f"Marked tool as deprecated: {tool_name}")
        
        # Commit all changes
        await db.commit()
        
        logger.info(f"Processed {tools_processed} tools from origin {origin.id}")
        return tools_processed


class CrawlScheduler:
    """Scheduler for periodic crawling of origins."""
    
    def __init__(self, crawler: MCPCrawler):
        """Initialize the scheduler with a crawler instance."""
        self.crawler = crawler
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start_scheduler(self, db: AsyncSession, interval_hours: float = 24.0):
        """Start the periodic crawling scheduler."""
        if self._running:
            logger.warning("Scheduler is already running")
            return
        
        self._running = True
        interval_seconds = interval_hours * 3600
        
        logger.info(f"Starting crawl scheduler with {interval_hours}h interval")
        
        self._task = asyncio.create_task(
            self._schedule_loop(db, interval_seconds),
            name="crawl-scheduler"
        )
    
    async def stop_scheduler(self):
        """Stop the crawling scheduler."""
        self._running = False
        
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Crawl scheduler stopped")
    
    async def trigger_manual_crawl(self, db: AsyncSession) -> Dict[str, Any]:
        """Trigger a manual crawl of all origins."""
        logger.info("Manual crawl triggered")
        return await self.crawler.crawl_all_origins(db)
    
    async def _schedule_loop(self, db: AsyncSession, interval_seconds: float):
        """Main scheduling loop."""
        try:
            while self._running:
                try:
                    logger.info("Starting scheduled crawl")
                    result = await self.crawler.crawl_all_origins(db)
                    logger.info(f"Scheduled crawl completed: {result}")
                    
                    # Wait for next interval
                    await asyncio.sleep(interval_seconds)
                
                except Exception as e:
                    logger.error(f"Error in scheduled crawl: {e}", exc_info=True)
                    # Wait a shorter time on error before retrying
                    await asyncio.sleep(min(3600, interval_seconds / 4))  # Max 1 hour
        
        except asyncio.CancelledError:
            logger.info("Crawl scheduler cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in crawl scheduler: {e}", exc_info=True)
            raise


# Global instances
_crawler: Optional[MCPCrawler] = None
_scheduler: Optional[CrawlScheduler] = None


def get_crawler() -> MCPCrawler:
    """Get the global crawler instance."""
    global _crawler
    if _crawler is None:
        settings = get_settings()
        _crawler = MCPCrawler(
            max_concurrent=settings.ingestion.concurrent_origins,
            timeout=settings.ingestion.request_timeout_seconds
        )
    return _crawler


def get_scheduler() -> CrawlScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = CrawlScheduler(get_crawler())
    return _scheduler