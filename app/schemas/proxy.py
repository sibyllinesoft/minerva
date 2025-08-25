"""
Pydantic schemas for proxy/executor service.
"""
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator
from uuid import UUID


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ExecutionStatus(str, Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    CANCELLED = "cancelled"


class ToolCallRequest(BaseModel):
    """Request to execute a tool call."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    origin_id: Optional[UUID] = Field(None, description="Specific origin to use")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="Execution timeout in seconds")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    trace_id: Optional[str] = Field(None, description="Distributed tracing ID")
    
    @validator('tool_name')
    def validate_tool_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Tool name cannot be empty")
        if len(v) > 100:
            raise ValueError("Tool name too long (max 100 chars)")
        return v.strip()


class ToolCallResponse(BaseModel):
    """Response from tool execution."""
    execution_id: str = Field(..., description="Unique execution identifier")
    tool_name: str = Field(..., description="Name of executed tool")
    status: ExecutionStatus = Field(..., description="Execution status")
    result: Optional[Any] = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Structured error code")
    start_time: datetime = Field(..., description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution completion time")
    duration_ms: Optional[int] = Field(None, description="Execution duration in milliseconds")
    origin_id: Optional[UUID] = Field(None, description="Origin that executed the tool")
    origin_url: Optional[str] = Field(None, description="Origin URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")


class ExecutionTrace(BaseModel):
    """Detailed execution trace for audit."""
    execution_id: str = Field(..., description="Unique execution identifier")
    trace_id: Optional[str] = Field(None, description="Distributed tracing ID")
    tool_name: str = Field(..., description="Tool that was executed")
    origin_id: Optional[UUID] = Field(None, description="Origin ID")
    origin_url: Optional[str] = Field(None, description="Origin URL")
    request_payload: Dict[str, Any] = Field(..., description="Request payload sent to origin")
    response_payload: Optional[Dict[str, Any]] = Field(None, description="Response from origin")
    status: ExecutionStatus = Field(..., description="Final execution status")
    start_time: datetime = Field(..., description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution completion time")
    duration_ms: Optional[int] = Field(None, description="Total duration in milliseconds")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
    circuit_breaker_state: Optional[CircuitBreakerState] = Field(None, description="Circuit breaker state during execution")
    retry_count: int = Field(0, description="Number of retry attempts")
    client_ip: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")


class CircuitBreakerStats(BaseModel):
    """Circuit breaker statistics."""
    origin_id: UUID = Field(..., description="Origin ID")
    origin_url: str = Field(..., description="Origin URL")
    state: CircuitBreakerState = Field(..., description="Current state")
    failure_count: int = Field(0, description="Current failure count")
    success_count: int = Field(0, description="Recent success count")
    last_failure_time: Optional[datetime] = Field(None, description="Last failure timestamp")
    last_success_time: Optional[datetime] = Field(None, description="Last success timestamp")
    next_attempt_time: Optional[datetime] = Field(None, description="Next allowed attempt time")
    failure_threshold: int = Field(5, description="Failure threshold for opening circuit")
    recovery_timeout: int = Field(60, description="Recovery timeout in seconds")
    half_open_max_calls: int = Field(3, description="Max calls in half-open state")


class ProxyStats(BaseModel):
    """Proxy service statistics."""
    total_executions: int = Field(0, description="Total tool executions")
    successful_executions: int = Field(0, description="Successful executions")
    failed_executions: int = Field(0, description="Failed executions")
    circuit_breaker_openings: int = Field(0, description="Circuit breaker openings")
    active_executions: int = Field(0, description="Currently active executions")
    avg_response_time_ms: float = Field(0.0, description="Average response time")
    p95_response_time_ms: float = Field(0.0, description="95th percentile response time")
    p99_response_time_ms: float = Field(0.0, description="99th percentile response time")
    origins_available: int = Field(0, description="Available origins")
    origins_circuit_open: int = Field(0, description="Origins with circuit breaker open")
    last_updated: datetime = Field(..., description="Last statistics update")


class ExecutionBatchRequest(BaseModel):
    """Request for batch tool execution."""
    calls: List[ToolCallRequest] = Field(..., description="List of tool calls to execute")
    max_concurrency: Optional[int] = Field(5, ge=1, le=20, description="Max concurrent executions")
    fail_fast: bool = Field(False, description="Stop on first failure")
    timeout: Optional[int] = Field(60, ge=1, le=600, description="Total batch timeout in seconds")
    
    @validator('calls')
    def validate_calls(cls, v):
        if not v:
            raise ValueError("At least one tool call required")
        if len(v) > 50:
            raise ValueError("Too many calls in batch (max 50)")
        return v


class ExecutionBatchResponse(BaseModel):
    """Response from batch tool execution."""
    batch_id: str = Field(..., description="Unique batch identifier")
    total_calls: int = Field(..., description="Total number of calls in batch")
    successful_calls: int = Field(0, description="Number of successful calls")
    failed_calls: int = Field(0, description="Number of failed calls")
    results: List[ToolCallResponse] = Field(..., description="Individual call results")
    batch_status: ExecutionStatus = Field(..., description="Overall batch status")
    start_time: datetime = Field(..., description="Batch start time")
    end_time: Optional[datetime] = Field(None, description="Batch completion time")
    duration_ms: Optional[int] = Field(None, description="Total batch duration")


class StreamingToolCall(BaseModel):
    """Streaming tool call request."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    origin_id: Optional[UUID] = Field(None, description="Specific origin to use")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    stream: bool = Field(True, description="Enable streaming response")
    timeout: Optional[int] = Field(60, ge=1, le=300, description="Stream timeout in seconds")


class StreamingChunk(BaseModel):
    """Streaming response chunk."""
    execution_id: str = Field(..., description="Execution identifier")
    chunk_id: int = Field(..., description="Chunk sequence number")
    chunk_type: str = Field(..., description="Chunk type (data, error, done)")
    data: Optional[Any] = Field(None, description="Chunk data")
    timestamp: datetime = Field(..., description="Chunk timestamp")
    is_final: bool = Field(False, description="Whether this is the final chunk")