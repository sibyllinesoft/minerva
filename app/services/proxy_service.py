"""
Proxy service for safe upstream tool execution with circuit breakers and observability.
"""
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
from uuid import UUID
import logging
from contextlib import asynccontextmanager
from collections import defaultdict, deque

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from fastapi import HTTPException

from app.core.database import get_db_session
from app.models.origin import Origin
from app.models.tool import Tool
from app.models.session import Trace
from app.schemas.proxy import (
    ToolCallRequest, ToolCallResponse, ExecutionStatus, ExecutionTrace,
    ProxyStats, ExecutionBatchRequest, ExecutionBatchResponse,
    StreamingToolCall, StreamingChunk
)
from app.services.circuit_breaker import circuit_breaker_registry, CircuitBreakerError


logger = logging.getLogger(__name__)


class ProxyService:
    """Service for proxying tool calls to upstream MCP servers."""
    
    def __init__(self, max_concurrent_calls: int = 50, default_timeout: int = 30):
        self.max_concurrent_calls = max_concurrent_calls
        self.default_timeout = default_timeout
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_calls)
        
        # Statistics tracking
        self._stats_lock = asyncio.Lock()
        self._stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'circuit_breaker_openings': 0,
            'active_executions': 0,
            'response_times': deque(maxlen=1000),  # Keep last 1000 response times
            'last_updated': datetime.utcnow()
        }
        
        # HTTP client for upstream calls
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Active executions tracking
        self._active_executions: Dict[str, datetime] = {}
        
        logger.info(f"ProxyService initialized with max_concurrent_calls={max_concurrent_calls}")
    
    async def startup(self):
        """Initialize the proxy service."""
        timeout_config = httpx.Timeout(
            connect=10.0,
            read=self.default_timeout,
            write=10.0,
            pool=5.0
        )
        
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30
        )
        
        self._http_client = httpx.AsyncClient(
            timeout=timeout_config,
            limits=limits,
            follow_redirects=True,
            verify=True  # Enable TLS verification
        )
        logger.info("ProxyService HTTP client initialized")
    
    async def shutdown(self):
        """Cleanup the proxy service."""
        if self._http_client:
            await self._http_client.aclose()
        logger.info("ProxyService shut down")
    
    @asynccontextmanager
    async def _execution_context(self, execution_id: str):
        """Context manager for tracking active executions."""
        async with self._semaphore:
            async with self._stats_lock:
                self._active_executions[execution_id] = datetime.utcnow()
                self._stats['active_executions'] = len(self._active_executions)
            
            try:
                yield
            finally:
                async with self._stats_lock:
                    self._active_executions.pop(execution_id, None)
                    self._stats['active_executions'] = len(self._active_executions)
    
    async def execute_tool(
        self,
        request: ToolCallRequest,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> ToolCallResponse:
        """Execute a single tool call with circuit breaker protection."""
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(f"Starting tool execution {execution_id}: {request.tool_name}")
        
        async with self._execution_context(execution_id):
            try:
                # Find appropriate origin and tool
                async with get_db_session() as db:
                    origin, tool = await self._find_tool(db, request.tool_name, request.origin_id)
                
                # Get circuit breaker for origin
                breaker = await circuit_breaker_registry.get_breaker(origin.id, origin.base_url)
                
                # Execute with circuit breaker protection
                result = await breaker.call(
                    self._execute_upstream_call,
                    origin, tool, request, execution_id
                )
                
                end_time = datetime.utcnow()
                duration_ms = int((end_time - start_time).total_seconds() * 1000)
                
                # Update statistics
                await self._update_stats(True, duration_ms)
                
                # Create audit trace
                await self._create_trace(
                    execution_id=execution_id,
                    tool_name=request.tool_name,
                    origin=origin,
                    request_data=request.dict(),
                    response_data=result,
                    status=ExecutionStatus.SUCCESS,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    trace_id=request.trace_id
                )
                
                return ToolCallResponse(
                    execution_id=execution_id,
                    tool_name=request.tool_name,
                    status=ExecutionStatus.SUCCESS,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    origin_id=origin.id,
                    origin_url=origin.base_url,
                    metadata={'trace_id': request.trace_id}
                )
                
            except CircuitBreakerError as e:
                logger.warning(f"Circuit breaker blocked execution {execution_id}: {e}")
                await self._update_stats(False, 0, circuit_breaker_opened=True)
                
                # Create failed trace
                await self._create_trace(
                    execution_id=execution_id,
                    tool_name=request.tool_name,
                    request_data=request.dict(),
                    status=ExecutionStatus.CIRCUIT_OPEN,
                    start_time=start_time,
                    end_time=datetime.utcnow(),
                    error_details={'error': str(e), 'type': 'CircuitBreakerError'},
                    client_ip=client_ip,
                    user_agent=user_agent,
                    trace_id=request.trace_id
                )
                
                return ToolCallResponse(
                    execution_id=execution_id,
                    tool_name=request.tool_name,
                    status=ExecutionStatus.CIRCUIT_OPEN,
                    error=str(e),
                    error_code='CIRCUIT_BREAKER_OPEN',
                    start_time=start_time,
                    end_time=datetime.utcnow()
                )
                
            except asyncio.TimeoutError:
                logger.error(f"Execution {execution_id} timed out")
                await self._update_stats(False, 0)
                
                return ToolCallResponse(
                    execution_id=execution_id,
                    tool_name=request.tool_name,
                    status=ExecutionStatus.TIMEOUT,
                    error="Execution timed out",
                    error_code='TIMEOUT',
                    start_time=start_time,
                    end_time=datetime.utcnow()
                )
                
            except Exception as e:
                logger.error(f"Execution {execution_id} failed: {e}", exc_info=True)
                await self._update_stats(False, 0)
                
                return ToolCallResponse(
                    execution_id=execution_id,
                    tool_name=request.tool_name,
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                    error_code='EXECUTION_ERROR',
                    start_time=start_time,
                    end_time=datetime.utcnow()
                )
    
    async def execute_batch(
        self,
        request: ExecutionBatchRequest,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> ExecutionBatchResponse:
        """Execute multiple tool calls concurrently."""
        batch_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(f"Starting batch execution {batch_id}: {len(request.calls)} calls")
        
        # Create semaphore for batch concurrency control
        batch_semaphore = asyncio.Semaphore(request.max_concurrency or 5)
        
        async def execute_single(call_request: ToolCallRequest) -> ToolCallResponse:
            async with batch_semaphore:
                return await self.execute_tool(call_request, client_ip, user_agent)
        
        # Execute all calls concurrently
        if request.fail_fast:
            # Stop on first failure
            results = []
            for call_request in request.calls:
                result = await execute_single(call_request)
                results.append(result)
                if result.status == ExecutionStatus.FAILED:
                    logger.warning(f"Batch {batch_id} stopping on first failure")
                    break
        else:
            # Execute all calls regardless of failures
            tasks = [execute_single(call_request) for call_request in request.calls]
            results = await asyncio.gather(*tasks, return_exceptions=False)
        
        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        successful_calls = sum(1 for r in results if r.status == ExecutionStatus.SUCCESS)
        failed_calls = len(results) - successful_calls
        
        batch_status = ExecutionStatus.SUCCESS if failed_calls == 0 else ExecutionStatus.FAILED
        
        return ExecutionBatchResponse(
            batch_id=batch_id,
            total_calls=len(request.calls),
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            results=results,
            batch_status=batch_status,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms
        )
    
    async def _find_tool(
        self, 
        db: AsyncSession, 
        tool_name: str, 
        origin_id: Optional[UUID] = None
    ) -> tuple[Origin, Tool]:
        """Find the appropriate origin and tool for execution."""
        query = select(Tool).join(Origin).where(
            and_(
                Tool.name == tool_name,
                Tool.status == "active",
                Origin.status == "active"
            )
        )
        
        if origin_id:
            query = query.where(Origin.id == origin_id)
        
        # Prefer origins with better reliability
        query = query.order_by(Origin.reliability_score.desc())
        
        result = await db.execute(query)
        tool = result.scalar_one_or_none()
        
        if not tool:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{tool_name}' not found or not available"
            )
        
        return tool.origin, tool
    
    async def _execute_upstream_call(
        self,
        origin: Origin,
        tool: Tool,
        request: ToolCallRequest,
        execution_id: str
    ) -> Any:
        """Execute the actual upstream call to the MCP server."""
        if not self._http_client:
            raise RuntimeError("ProxyService not initialized")
        
        # Prepare request payload
        payload = {
            "jsonrpc": "2.0",
            "id": execution_id,
            "method": "tools/call",
            "params": {
                "name": tool.name,
                "arguments": request.arguments
            }
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "meta-mcp-proxy/1.0",
        }
        
        # Add authentication if configured
        if origin.auth_type == "bearer" and origin.auth_config.get("token"):
            headers["Authorization"] = f"Bearer {origin.auth_config['token']}"
        elif origin.auth_type == "api_key":
            key = origin.auth_config.get("key", "X-API-Key")
            value = origin.auth_config.get("value")
            if value:
                headers[key] = value
        elif origin.auth_type == "basic":
            username = origin.auth_config.get("username")
            password = origin.auth_config.get("password")
            if username and password:
                import base64
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        # Execute request with timeout
        timeout = request.timeout or self.default_timeout
        logger.debug(f"Executing upstream call to {origin.base_url} for tool {tool.name}")
        
        try:
            response = await self._http_client.post(
                origin.base_url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            
            # Parse JSON-RPC response
            response_data = response.json()
            
            if "error" in response_data:
                error = response_data["error"]
                raise HTTPException(
                    status_code=400,
                    detail=f"Upstream error: {error.get('message', 'Unknown error')}"
                )
            
            return response_data.get("result")
            
        except httpx.TimeoutException:
            logger.error(f"Upstream call timed out: {origin.base_url}")
            raise asyncio.TimeoutError(f"Upstream call timed out after {timeout}s")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Upstream HTTP error {e.response.status_code}: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Upstream HTTP error: {e.response.status_code}"
            )
    
    async def _create_trace(
        self,
        execution_id: str,
        tool_name: str,
        request_data: Dict[str, Any],
        status: ExecutionStatus,
        start_time: datetime,
        origin: Optional[Origin] = None,
        response_data: Optional[Any] = None,
        end_time: Optional[datetime] = None,
        duration_ms: Optional[int] = None,
        error_details: Optional[Dict[str, Any]] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        trace_id: Optional[str] = None,
        retry_count: int = 0
    ):
        """Create an audit trace for the execution."""
        try:
            async with get_db_session() as db:
                trace = Trace(
                    id=uuid.uuid4(),
                    execution_id=execution_id,
                    trace_id=trace_id,
                    tool_name=tool_name,
                    origin_id=origin.id if origin else None,
                    origin_url=origin.base_url if origin else None,
                    request_payload=request_data,
                    response_payload=response_data,
                    status=status.value,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    error_details=error_details,
                    retry_count=retry_count,
                    client_ip=client_ip,
                    user_agent=user_agent
                )
                
                db.add(trace)
                await db.commit()
                
        except Exception as e:
            logger.error(f"Failed to create trace: {e}", exc_info=True)
    
    async def _update_stats(
        self,
        success: bool,
        duration_ms: int,
        circuit_breaker_opened: bool = False
    ):
        """Update proxy service statistics."""
        async with self._stats_lock:
            self._stats['total_executions'] += 1
            self._stats['last_updated'] = datetime.utcnow()
            
            if success:
                self._stats['successful_executions'] += 1
                self._stats['response_times'].append(duration_ms)
            else:
                self._stats['failed_executions'] += 1
                
            if circuit_breaker_opened:
                self._stats['circuit_breaker_openings'] += 1
    
    async def get_stats(self) -> ProxyStats:
        """Get current proxy service statistics."""
        async with self._stats_lock:
            response_times = list(self._stats['response_times'])
            
            if response_times:
                avg_response = sum(response_times) / len(response_times)
                sorted_times = sorted(response_times)
                p95_idx = int(len(sorted_times) * 0.95)
                p99_idx = int(len(sorted_times) * 0.99)
                p95_response = sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0
                p99_response = sorted_times[p99_idx] if p99_idx < len(sorted_times) else 0
            else:
                avg_response = p95_response = p99_response = 0.0
            
            circuit_stats = circuit_breaker_registry.get_all_stats()
            origins_circuit_open = sum(1 for stats in circuit_stats.values() 
                                     if stats.state.value == "open")
            origins_available = len(circuit_stats) - origins_circuit_open
            
            return ProxyStats(
                total_executions=self._stats['total_executions'],
                successful_executions=self._stats['successful_executions'],
                failed_executions=self._stats['failed_executions'],
                circuit_breaker_openings=self._stats['circuit_breaker_openings'],
                active_executions=self._stats['active_executions'],
                avg_response_time_ms=avg_response,
                p95_response_time_ms=p95_response,
                p99_response_time_ms=p99_response,
                origins_available=origins_available,
                origins_circuit_open=origins_circuit_open,
                last_updated=self._stats['last_updated']
            )
    
    async def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently active executions."""
        now = datetime.utcnow()
        async with self._stats_lock:
            return {
                execution_id: {
                    'started_at': started_at.isoformat(),
                    'duration_seconds': int((now - started_at).total_seconds())
                }
                for execution_id, started_at in self._active_executions.items()
            }
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution (best effort)."""
        # Note: This is a simplified implementation.
        # In a full implementation, you'd need to track tasks and cancel them.
        async with self._stats_lock:
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]
                self._stats['active_executions'] = len(self._active_executions)
                logger.info(f"Cancelled execution {execution_id}")
                return True
            return False


# Global proxy service instance
proxy_service = ProxyService()