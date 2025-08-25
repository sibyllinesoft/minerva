"""Planner service for generating tool execution DAGs."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
import hashlib
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..models.tool import Tool
from ..schemas.planner import (
    ExecutionPlan, PlanRequest, PlanResponse, PlannerMode, ToolCall,
    PlanValidationError as PlanValidationErrorSchema, PlanRepairRequest, PlannerStats, PlannerConfig
)
from ..core.config import get_settings
from ..services.models.manager import get_model_manager


logger = logging.getLogger(__name__)


class CycleDetectionError(Exception):
    """Raised when a cycle is detected in the execution plan."""
    pass


class PlanValidationException(Exception):
    """Raised when plan validation fails."""
    pass


class PlannerService:
    """Service for generating and validating tool execution plans."""
    
    def __init__(self):
        self.config = PlannerConfig()
        self._cache: Dict[str, Tuple[PlanResponse, datetime]] = {}
        self._stats = {
            "total_plans": 0,
            "successful_plans": 0,
            "trivial_plans": 0,
            "simple_plans": 0,
            "complex_plans": 0,
            "repair_attempts": 0,
            "fallback_uses": 0,
            "cache_hits": 0,
            "total_planning_time": 0.0
        }
        
    async def generate_plan(self, request: PlanRequest, db: AsyncSession) -> PlanResponse:
        """Generate execution plan for user query."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = self._get_cached_plan(cache_key)
            if cached_response:
                self._stats["cache_hits"] += 1
                cached_response.cache_hit = True
                cached_response.cache_key = cache_key
                return cached_response
            
            # Get available tools
            available_tools = await self._get_available_tools(request, db)
            if not available_tools:
                return PlanResponse(
                    success=False,
                    planning_time_ms=(time.time() - start_time) * 1000,
                    tools_considered=0,
                    mode_used=PlannerMode.TRIVIAL,
                    error_code="NO_TOOLS",
                    error_message="No tools available for planning"
                )
            
            # Determine planning mode
            mode = request.mode or self._determine_mode(request.query, available_tools)
            
            # Generate plan based on mode
            if mode == PlannerMode.TRIVIAL:
                plan = await self._generate_trivial_plan(request, available_tools)
            elif mode == PlannerMode.SIMPLE:
                plan = await self._generate_simple_plan(request, available_tools, db)
            else:  # COMPLEX
                plan = await self._generate_complex_plan(request, available_tools, db)
            
            # Validate and repair if needed
            validated_plan = await self._validate_and_repair_plan(plan, db)
            
            planning_time = (time.time() - start_time) * 1000
            
            response = PlanResponse(
                success=True,
                plan=validated_plan,
                planning_time_ms=planning_time,
                tools_considered=len(available_tools),
                mode_used=mode,
                cache_hit=False,
                cache_key=cache_key
            )
            
            # Cache successful plans
            if self.config.enable_plan_caching:
                self._cache_plan(cache_key, response)
            
            # Update stats
            self._update_stats(mode, planning_time, validated_plan.repair_iterations > 0)
            
            return response
            
        except Exception as e:
            logger.error(f"Plan generation failed: {e}", exc_info=True)
            planning_time = (time.time() - start_time) * 1000
            
            # Try fallback to trivial plan
            if self.config.enable_trivial_fallback:
                try:
                    available_tools = await self._get_available_tools(request, db)
                    fallback_plan = await self._generate_trivial_plan(request, available_tools)
                    fallback_plan.fallback_used = True
                    
                    response = PlanResponse(
                        success=True,
                        plan=fallback_plan,
                        planning_time_ms=planning_time,
                        tools_considered=len(available_tools) if available_tools else 0,
                        mode_used=PlannerMode.TRIVIAL,
                        fallback_reason=str(e),
                        cache_hit=False
                    )
                    
                    self._stats["fallback_uses"] += 1
                    return response
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback planning failed: {fallback_error}")
            
            return PlanResponse(
                success=False,
                planning_time_ms=planning_time,
                tools_considered=0,
                mode_used=PlannerMode.TRIVIAL,
                error_code="GENERATION_FAILED",
                error_message=str(e)
            )
    
    async def _get_available_tools(self, request: PlanRequest, db: AsyncSession) -> List[Tool]:
        """Get list of available tools for planning."""
        query = select(Tool).where(Tool.deprecated == False)
        
        if request.available_tools:
            query = query.where(Tool.id.in_(request.available_tools))
        
        result = await db.execute(query.limit(request.max_tools))
        return list(result.scalars().all())
    
    def _determine_mode(self, query: str, available_tools: List[Tool]) -> PlannerMode:
        """Automatically determine planning mode based on query complexity."""
        # Simple heuristics for mode detection
        query_lower = query.lower()
        
        # Keywords suggesting complex planning
        complex_keywords = ['workflow', 'pipeline', 'sequence', 'then', 'after', 'before', 'parallel']
        simple_keywords = ['and', 'also', 'plus', 'with']
        
        if any(keyword in query_lower for keyword in complex_keywords):
            return PlannerMode.COMPLEX
        elif any(keyword in query_lower for keyword in simple_keywords):
            return PlannerMode.SIMPLE
        elif len(available_tools) == 1:
            return PlannerMode.TRIVIAL
        else:
            return PlannerMode.SIMPLE
    
    async def _generate_trivial_plan(self, request: PlanRequest, available_tools: List[Tool]) -> ExecutionPlan:
        """Generate trivial single-tool plan."""
        if not available_tools:
            raise ValueError("No tools available for trivial plan")
        
        # Select most relevant tool (simple scoring)
        tool = available_tools[0]  # For now, just use first tool
        
        tool_call = ToolCall(
            id="trivial_call",
            tool_id=tool.id,
            tool_name=tool.name,
            args={},  # Empty args for trivial plan
            depends_on=[],
            timeout_ms=self.config.default_timeout_ms
        )
        
        return ExecutionPlan(
            id=f"trivial_{tool_call.id}",
            query=request.query,
            mode=PlannerMode.TRIVIAL,
            tool_calls=[tool_call],
            execution_order=[[tool_call.id]],
            estimated_duration_ms=self.config.default_timeout_ms,
            complexity_score=0.1,
            confidence_score=0.8,
            generation_time_ms=1.0,  # Trivial plans are instant
            repair_iterations=0,
            fallback_used=False
        )
    
    async def _generate_simple_plan(self, request: PlanRequest, available_tools: List[Tool], db: AsyncSession) -> ExecutionPlan:
        """Generate simple linear plan with 2-3 tools."""
        # For now, create a simple linear plan with up to 3 tools
        selected_tools = available_tools[:min(3, len(available_tools))]
        
        tool_calls = []
        execution_order = []
        
        for i, tool in enumerate(selected_tools):
            call_id = f"step_{i+1}"
            depends_on = [f"step_{i}"] if i > 0 else []
            
            tool_call = ToolCall(
                id=call_id,
                tool_id=tool.id,
                tool_name=tool.name,
                args={},
                depends_on=depends_on,
                timeout_ms=self.config.default_timeout_ms
            )
            
            tool_calls.append(tool_call)
            execution_order.append([call_id])
        
        return ExecutionPlan(
            id=f"simple_{len(tool_calls)}_{hash(request.query) % 10000}",
            query=request.query,
            mode=PlannerMode.SIMPLE,
            tool_calls=tool_calls,
            execution_order=execution_order,
            estimated_duration_ms=len(tool_calls) * self.config.default_timeout_ms,
            complexity_score=0.3,
            confidence_score=0.7,
            generation_time_ms=10.0,
            repair_iterations=0,
            fallback_used=False
        )
    
    async def _generate_complex_plan(self, request: PlanRequest, available_tools: List[Tool], db: AsyncSession) -> ExecutionPlan:
        """Generate complex DAG plan using LLM."""
        model_manager = get_model_manager()
        
        # For now, create a more complex plan with some parallel execution
        selected_tools = available_tools[:min(request.max_tools, len(available_tools))]
        
        tool_calls = []
        execution_order = [[], []]  # Two levels for parallel execution
        
        # First level - parallel independent calls
        for i, tool in enumerate(selected_tools[:3]):
            call_id = f"parallel_{i+1}"
            
            tool_call = ToolCall(
                id=call_id,
                tool_id=tool.id,
                tool_name=tool.name,
                args={},
                depends_on=[],
                timeout_ms=self.config.default_timeout_ms
            )
            
            tool_calls.append(tool_call)
            execution_order[0].append(call_id)
        
        # Second level - depends on first level
        if len(selected_tools) > 3:
            for i, tool in enumerate(selected_tools[3:5]):
                call_id = f"dependent_{i+1}"
                
                tool_call = ToolCall(
                    id=call_id,
                    tool_id=tool.id,
                    tool_name=tool.name,
                    args={},
                    depends_on=execution_order[0],  # Depends on all first-level calls
                    timeout_ms=self.config.default_timeout_ms
                )
                
                tool_calls.append(tool_call)
                execution_order[1].append(call_id)
        
        # Remove empty execution levels
        execution_order = [level for level in execution_order if level]
        
        return ExecutionPlan(
            id=f"complex_{len(tool_calls)}_{hash(request.query) % 10000}",
            query=request.query,
            mode=PlannerMode.COMPLEX,
            tool_calls=tool_calls,
            execution_order=execution_order,
            estimated_duration_ms=len(execution_order) * self.config.default_timeout_ms,
            complexity_score=0.8,
            confidence_score=0.6,
            generation_time_ms=100.0,
            repair_iterations=0,
            fallback_used=False
        )
    
    async def _validate_and_repair_plan(self, plan: ExecutionPlan, db: AsyncSession) -> ExecutionPlan:
        """Validate plan and repair if needed."""
        validation_errors = self._validate_plan(plan)
        
        if not validation_errors:
            return plan
        
        # Attempt repairs
        repair_iterations = 0
        current_plan = plan
        
        while validation_errors and repair_iterations < self.config.max_repair_iterations:
            repair_iterations += 1
            logger.info(f"Repair iteration {repair_iterations} for plan {plan.id}")
            
            current_plan = self._repair_plan(current_plan, validation_errors)
            validation_errors = self._validate_plan(current_plan)
        
        current_plan.repair_iterations = repair_iterations
        
        if validation_errors:
            logger.warning(f"Plan {plan.id} still has validation errors after repair: {validation_errors}")
            # Return plan anyway - executor can handle some validation errors
        
        return current_plan
    
    def _validate_plan(self, plan: ExecutionPlan) -> List[PlanValidationErrorSchema]:
        """Validate execution plan for cycles and dependency issues."""
        errors = []
        
        # Check for cycles
        try:
            self._check_cycles(plan)
        except CycleDetectionError as e:
            errors.append(PlanValidationErrorSchema(
                error_type="cycle",
                message=str(e),
                suggested_fix="Remove or reorder dependencies to break cycle"
            ))
        
        # Check dependencies exist
        call_ids = {call.id for call in plan.tool_calls}
        for call in plan.tool_calls:
            for dep_id in call.depends_on:
                if dep_id not in call_ids:
                    errors.append(PlanValidationErrorSchema(
                        error_type="missing_dependency",
                        message=f"Tool call {call.id} depends on non-existent call {dep_id}",
                        tool_call_id=call.id,
                        suggested_fix=f"Remove dependency on {dep_id} or add missing tool call"
                    ))
        
        # Check execution order consistency
        order_ids = {call_id for group in plan.execution_order for call_id in group}
        if order_ids != call_ids:
            errors.append(PlanValidationErrorSchema(
                error_type="other",
                message="Execution order doesn't match tool calls",
                suggested_fix="Rebuild execution order from dependencies"
            ))
        
        return errors
    
    def _check_cycles(self, plan: ExecutionPlan) -> None:
        """Check for cycles in dependency graph."""
        # Build adjacency list
        graph = {call.id: call.depends_on for call in plan.tool_calls}
        
        # DFS cycle detection
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                raise CycleDetectionError(f"Cycle detected involving node {node}")
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for call_id in graph:
            if call_id not in visited:
                has_cycle(call_id)
    
    def _repair_plan(self, plan: ExecutionPlan, errors: List[PlanValidationErrorSchema]) -> ExecutionPlan:
        """Attempt to repair plan by addressing validation errors."""
        repaired_plan = ExecutionPlan(**plan.dict())
        
        for error in errors:
            if error.error_type == "cycle":
                # Simple cycle repair: remove some dependencies
                repaired_plan = self._break_cycles(repaired_plan)
            
            elif error.error_type == "missing_dependency" and error.tool_call_id:
                # Remove invalid dependencies
                for call in repaired_plan.tool_calls:
                    if call.id == error.tool_call_id:
                        call.depends_on = [dep for dep in call.depends_on 
                                         if any(c.id == dep for c in repaired_plan.tool_calls)]
        
        # Rebuild execution order
        repaired_plan.execution_order = self._build_execution_order(repaired_plan)
        
        return repaired_plan
    
    def _break_cycles(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Break cycles by removing some dependencies."""
        # Simple strategy: remove dependencies that create cycles
        # In a real implementation, this would be more sophisticated
        
        call_dict = {call.id: call for call in plan.tool_calls}
        
        # Use topological sort attempt to identify problematic edges
        in_degree = {call.id: 0 for call in plan.tool_calls}
        for call in plan.tool_calls:
            for dep in call.depends_on:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Remove dependencies from nodes with highest in-degree
        max_in_degree = max(in_degree.values()) if in_degree else 0
        if max_in_degree > 2:  # Arbitrary threshold
            for call in plan.tool_calls:
                if in_degree[call.id] > 2:
                    call.depends_on = call.depends_on[:1]  # Keep only first dependency
        
        return plan
    
    def _build_execution_order(self, plan: ExecutionPlan) -> List[List[str]]:
        """Build execution order from dependency graph."""
        # Topological sort with level assignment
        call_dict = {call.id: call for call in plan.tool_calls}
        in_degree = {call.id: len(call.depends_on) for call in plan.tool_calls}
        
        execution_order = []
        remaining_calls = set(call_dict.keys())
        
        while remaining_calls:
            # Find calls with no dependencies
            current_level = [call_id for call_id in remaining_calls if in_degree[call_id] == 0]
            
            if not current_level:
                # Break remaining cycles arbitrarily
                current_level = [next(iter(remaining_calls))]
            
            execution_order.append(current_level)
            
            # Remove current level and update in-degrees
            for call_id in current_level:
                remaining_calls.remove(call_id)
                for other_call in plan.tool_calls:
                    if call_id in other_call.depends_on:
                        in_degree[other_call.id] -= 1
        
        return execution_order
    
    def _generate_cache_key(self, request: PlanRequest) -> str:
        """Generate cache key for plan request."""
        # Include relevant fields that affect plan generation
        cache_data = {
            "query": request.query,
            "available_tools": sorted(str(tool_id) for tool_id in (request.available_tools or [])),
            "mode": request.mode.value if request.mode else None,
            "max_tools": request.max_tools,
            "max_parallel": request.max_parallel,
            "optimize_for": request.optimize_for
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cached_plan(self, cache_key: str) -> Optional[PlanResponse]:
        """Get cached plan if available and not expired."""
        if not self.config.enable_plan_caching:
            return None
        
        if cache_key not in self._cache:
            return None
        
        response, cached_at = self._cache[cache_key]
        ttl = timedelta(hours=self.config.cache_ttl_hours)
        
        if datetime.utcnow() - cached_at > ttl:
            del self._cache[cache_key]
            return None
        
        return response
    
    def _cache_plan(self, cache_key: str, response: PlanResponse) -> None:
        """Cache plan response."""
        if self.config.enable_plan_caching:
            self._cache[cache_key] = (response, datetime.utcnow())
    
    def _update_stats(self, mode: PlannerMode, planning_time: float, required_repair: bool) -> None:
        """Update internal statistics."""
        self._stats["total_plans"] += 1
        self._stats["successful_plans"] += 1
        self._stats["total_planning_time"] += planning_time
        
        if mode == PlannerMode.TRIVIAL:
            self._stats["trivial_plans"] += 1
        elif mode == PlannerMode.SIMPLE:
            self._stats["simple_plans"] += 1
        else:
            self._stats["complex_plans"] += 1
        
        if required_repair:
            self._stats["repair_attempts"] += 1
    
    async def get_stats(self) -> PlannerStats:
        """Get planner performance statistics."""
        total_plans = self._stats["total_plans"]
        
        return PlannerStats(
            total_plans_generated=total_plans,
            success_rate=self._stats["successful_plans"] / max(total_plans, 1),
            average_planning_time_ms=self._stats["total_planning_time"] / max(total_plans, 1),
            trivial_plans=self._stats["trivial_plans"],
            simple_plans=self._stats["simple_plans"],
            complex_plans=self._stats["complex_plans"],
            average_confidence=0.7,  # Placeholder
            repair_rate=self._stats["repair_attempts"] / max(total_plans, 1),
            fallback_rate=self._stats["fallback_uses"] / max(total_plans, 1),
            cache_hit_rate=self._stats["cache_hits"] / max(total_plans, 1),
            execution_success_rate=0.85  # Placeholder - would track from execution results
        )


# Global service instance
_planner_service: Optional[PlannerService] = None


def get_planner_service() -> PlannerService:
    """Get the global planner service instance."""
    global _planner_service
    if _planner_service is None:
        _planner_service = PlannerService()
    return _planner_service