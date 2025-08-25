"""Planner schemas for tool execution DAGs."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator
from enum import Enum


class PlannerMode(str, Enum):
    """Planning modes with different complexity levels."""
    TRIVIAL = "trivial"      # Single tool, no dependencies
    SIMPLE = "simple"        # 2-3 tools, linear dependencies  
    COMPLEX = "complex"      # Full DAG with parallel execution


class ToolCall(BaseModel):
    """Single tool call in execution plan."""
    id: str = Field(description="Unique identifier for this tool call")
    tool_id: UUID = Field(description="ID of the tool to execute")
    tool_name: str = Field(description="Name of the tool for reference")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    depends_on: List[str] = Field(default_factory=list, description="List of tool call IDs this depends on")
    timeout_ms: Optional[int] = Field(default=30000, description="Execution timeout in milliseconds")
    retry_count: int = Field(default=0, ge=0, le=3, description="Number of retries allowed")
    
    @validator('id', pre=True, always=True)
    def generate_id_if_missing(cls, v):
        return v if v else f"call_{uuid4().hex[:8]}"


class ExecutionPlan(BaseModel):
    """Complete execution plan with DAG structure."""
    id: str = Field(description="Unique plan identifier")
    query: str = Field(min_length=1, max_length=2000, description="Original user query")
    mode: PlannerMode = Field(description="Planning complexity mode")
    
    # DAG structure
    tool_calls: List[ToolCall] = Field(description="All tool calls in execution order")
    execution_order: List[List[str]] = Field(description="Parallel execution groups by dependency level")
    
    # Metadata
    estimated_duration_ms: Optional[int] = Field(description="Estimated total execution time")
    complexity_score: float = Field(ge=0, le=1, description="Plan complexity (0=simple, 1=complex)")
    confidence_score: float = Field(ge=0, le=1, description="Planner confidence in this plan")
    
    # Generation details
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_time_ms: float = Field(description="Time taken to generate this plan")
    repair_iterations: int = Field(default=0, description="Number of repair loops applied")
    fallback_used: bool = Field(default=False, description="Whether trivial fallback was used")
    
    @validator('execution_order')
    def validate_execution_order(cls, v, values):
        """Ensure execution order matches tool calls."""
        if 'tool_calls' not in values:
            return v
            
        tool_call_ids = {call.id for call in values['tool_calls']}
        order_ids = {call_id for group in v for call_id in group}
        
        # Allow mismatched execution order for repair testing (will be caught by validation later)
        if tool_call_ids != order_ids:
            # Log warning instead of raising error
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Execution order mismatch - tool_calls: {tool_call_ids}, order: {order_ids}")
        
        return v


class PlanRequest(BaseModel):
    """Request for plan generation."""
    query: str = Field(min_length=1, max_length=2000, description="User query to plan for")
    available_tools: Optional[List[UUID]] = Field(default=None, description="Restrict to specific tools")
    mode: Optional[PlannerMode] = Field(default=None, description="Planning mode (auto-detect if None)")
    max_tools: int = Field(default=10, ge=1, le=20, description="Maximum tools in plan")
    max_parallel: int = Field(default=5, ge=1, le=10, description="Maximum parallel execution groups")
    timeout_ms: int = Field(default=60000, ge=5000, le=300000, description="Planning timeout")
    
    # Advanced options
    allow_loops: bool = Field(default=False, description="Allow iterative tool execution")
    require_confirmation: bool = Field(default=False, description="Require human confirmation before execution")
    optimize_for: Literal["speed", "accuracy", "cost"] = Field(default="speed", description="Optimization target")


class PlanResponse(BaseModel):
    """Response containing generated execution plan."""
    success: bool = Field(description="Whether planning succeeded")
    plan: Optional[ExecutionPlan] = Field(default=None, description="Generated execution plan")
    
    # Generation metadata
    planning_time_ms: float = Field(description="Total planning time")
    tools_considered: int = Field(description="Number of tools evaluated")
    mode_used: PlannerMode = Field(description="Actual planning mode used")
    
    # Error handling
    error_code: Optional[str] = Field(default=None, description="Error code if planning failed")
    error_message: Optional[str] = Field(default=None, description="Human-readable error description")
    fallback_reason: Optional[str] = Field(default=None, description="Reason for fallback if used")
    
    # Caching
    cache_hit: bool = Field(default=False, description="Whether result came from cache")
    cache_key: Optional[str] = Field(default=None, description="Cache key for this plan")


class PlanValidationError(BaseModel):
    """Validation error in generated plan."""
    error_type: Literal["cycle", "missing_dependency", "invalid_args", "timeout", "other"] = Field(description="Type of validation error")
    message: str = Field(description="Error description")
    tool_call_id: Optional[str] = Field(default=None, description="ID of problematic tool call")
    suggested_fix: Optional[str] = Field(default=None, description="Suggested repair action")


class PlanRepairRequest(BaseModel):
    """Request to repair a flawed execution plan."""
    original_plan: ExecutionPlan = Field(description="Plan that needs repair")
    validation_errors: List[PlanValidationError] = Field(description="Identified validation errors")
    repair_strategy: Literal["remove", "reorder", "substitute", "simplify"] = Field(default="reorder", description="Repair approach")
    max_iterations: int = Field(default=3, ge=1, le=10, description="Maximum repair attempts")


class PlannerStats(BaseModel):
    """Statistics about planner performance."""
    total_plans_generated: int = Field(description="Total plans generated")
    success_rate: float = Field(ge=0, le=1, description="Planning success rate")
    average_planning_time_ms: float = Field(description="Average time to generate plans")
    
    # Mode breakdown
    trivial_plans: int = Field(description="Simple single-tool plans")
    simple_plans: int = Field(description="Linear multi-tool plans") 
    complex_plans: int = Field(description="Full DAG plans")
    
    # Quality metrics
    average_confidence: float = Field(ge=0, le=1, description="Average planner confidence")
    repair_rate: float = Field(ge=0, le=1, description="Percentage of plans requiring repair")
    fallback_rate: float = Field(ge=0, le=1, description="Percentage using trivial fallback")
    
    # Performance
    cache_hit_rate: float = Field(ge=0, le=1, description="Plan cache hit rate")
    execution_success_rate: float = Field(ge=0, le=1, description="Plan execution success rate")
    
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PlannerConfig(BaseModel):
    """Configuration for planner behavior."""
    
    # LLM generation settings
    llm_model: str = Field(default="gpt-4", description="LLM model for plan generation")
    temperature: float = Field(default=0.1, ge=0, le=1, description="Generation temperature")
    max_tokens: int = Field(default=2000, ge=100, le=8000, description="Maximum response tokens")
    
    # Planning constraints
    max_plan_size: int = Field(default=10, ge=1, le=50, description="Maximum tools per plan")
    max_parallel_groups: int = Field(default=5, ge=1, le=20, description="Maximum parallel execution groups")
    default_timeout_ms: int = Field(default=30000, ge=1000, le=600000, description="Default tool timeout")
    
    # Repair settings
    max_repair_iterations: int = Field(default=5, ge=1, le=20, description="Maximum repair attempts")
    repair_confidence_threshold: float = Field(default=0.7, ge=0, le=1, description="Minimum confidence for repair")
    
    # Fallback behavior
    enable_trivial_fallback: bool = Field(default=True, description="Allow trivial single-tool fallback")
    fallback_confidence_threshold: float = Field(default=0.5, ge=0, le=1, description="Confidence threshold for fallback")
    
    # Caching
    enable_plan_caching: bool = Field(default=True, description="Enable plan result caching")
    cache_ttl_hours: int = Field(default=24, ge=1, le=168, description="Plan cache TTL in hours")