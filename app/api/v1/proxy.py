"""
API endpoints for proxy/executor service.
"""
from typing import Dict, Any, List
from uuid import UUID
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.schemas.proxy import (
    ToolCallRequest, ToolCallResponse, ProxyStats, ExecutionBatchRequest,
    ExecutionBatchResponse, StreamingToolCall, StreamingChunk
)
from app.services.proxy_service import proxy_service
from app.services.circuit_breaker import circuit_breaker_registry


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/proxy", tags=["proxy"])


@router.post("/execute", response_model=ToolCallResponse, status_code=status.HTTP_200_OK)
async def execute_tool(
    request: ToolCallRequest,
    http_request: Request
) -> ToolCallResponse:
    """Execute a single tool call through the proxy."""
    client_ip = http_request.client.host if http_request.client else None
    user_agent = http_request.headers.get("user-agent")
    
    logger.info(f"Tool execution request: {request.tool_name}")
    
    return await proxy_service.execute_tool(
        request=request,
        client_ip=client_ip,
        user_agent=user_agent
    )


@router.post("/batch", response_model=ExecutionBatchResponse, status_code=status.HTTP_200_OK)
async def execute_batch(
    request: ExecutionBatchRequest,
    http_request: Request
) -> ExecutionBatchResponse:
    """Execute multiple tool calls in a batch."""
    client_ip = http_request.client.host if http_request.client else None
    user_agent = http_request.headers.get("user-agent")
    
    logger.info(f"Batch execution request: {len(request.calls)} calls")
    
    return await proxy_service.execute_batch(
        request=request,
        client_ip=client_ip,
        user_agent=user_agent
    )


@router.get("/stats", response_model=ProxyStats)
async def get_proxy_stats() -> ProxyStats:
    """Get proxy service statistics."""
    return await proxy_service.get_stats()


@router.get("/executions/active")
async def get_active_executions() -> Dict[str, Any]:
    """Get information about currently active executions."""
    return {
        "active_executions": await proxy_service.get_active_executions(),
        "total_active": len(await proxy_service.get_active_executions())
    }


@router.delete("/executions/{execution_id}")
async def cancel_execution(execution_id: str) -> Dict[str, Any]:
    """Cancel an active execution."""
    success = await proxy_service.cancel_execution(execution_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Execution {execution_id} not found or already completed"
        )
    
    return {"message": f"Execution {execution_id} cancelled", "success": True}


@router.get("/circuit-breakers")
async def get_circuit_breaker_stats():
    """Get circuit breaker statistics for all origins."""
    stats = circuit_breaker_registry.get_all_stats()
    
    return {
        "circuit_breakers": [stats_obj.dict() for stats_obj in stats.values()],
        "summary": {
            "total_origins": len(stats),
            "open_circuits": circuit_breaker_registry.get_open_count(),
            "available_origins": circuit_breaker_registry.get_available_count()
        }
    }


@router.post("/circuit-breakers/reset")
async def reset_circuit_breakers() -> Dict[str, Any]:
    """Reset all circuit breakers to closed state."""
    await circuit_breaker_registry.reset_all()
    return {"message": "All circuit breakers reset", "success": True}


@router.post("/circuit-breakers/{origin_id}/reset")
async def reset_circuit_breaker(origin_id: UUID) -> Dict[str, Any]:
    """Reset specific circuit breaker to closed state."""
    stats = circuit_breaker_registry.get_all_stats()
    
    if origin_id not in stats:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit breaker for origin {origin_id} not found"
        )
    
    breaker = await circuit_breaker_registry.get_breaker(origin_id, "")
    await breaker.reset()
    
    return {"message": f"Circuit breaker for origin {origin_id} reset", "success": True}


@router.post("/circuit-breakers/{origin_id}/open")
async def force_open_circuit_breaker(origin_id: UUID) -> Dict[str, Any]:
    """Force open a specific circuit breaker."""
    stats = circuit_breaker_registry.get_all_stats()
    
    if origin_id not in stats:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit breaker for origin {origin_id} not found"
        )
    
    breaker = await circuit_breaker_registry.get_breaker(origin_id, "")
    await breaker.force_open()
    
    return {"message": f"Circuit breaker for origin {origin_id} forced open", "success": True}


# Streaming endpoints (for future implementation)
@router.post("/stream")
async def stream_tool_execution(request: StreamingToolCall, http_request: Request):
    """Execute a tool call with streaming response (placeholder)."""
    # This is a placeholder for streaming implementation
    # Would require server-sent events or websockets
    
    raise HTTPException(
        status_code=501,
        detail="Streaming execution not yet implemented"
    )


@router.get("/health")
async def proxy_health() -> Dict[str, Any]:
    """Health check for proxy service."""
    stats = await proxy_service.get_stats()
    circuit_stats = circuit_breaker_registry.get_all_stats()
    
    # Determine health status
    total_origins = len(circuit_stats)
    open_circuits = circuit_breaker_registry.get_open_count()
    availability_ratio = (total_origins - open_circuits) / max(total_origins, 1)
    
    is_healthy = availability_ratio >= 0.5  # At least 50% of origins available
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "service": "proxy",
        "stats": {
            "total_executions": stats.total_executions,
            "success_rate": stats.successful_executions / max(stats.total_executions, 1),
            "active_executions": stats.active_executions,
            "avg_response_time_ms": stats.avg_response_time_ms,
            "availability_ratio": availability_ratio
        },
        "circuit_breakers": {
            "total_origins": total_origins,
            "available_origins": stats.origins_available,
            "open_circuits": open_circuits
        }
    }