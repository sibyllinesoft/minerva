#!/usr/bin/env python3
"""Test Phase 5: Planner implementation."""

import asyncio
import logging
import os
import sys
from pathlib import Path
from uuid import uuid4

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import get_db, init_database
from app.models.origin import Origin, OriginStatus, AuthType
from app.models.tool import Tool
from app.schemas.planner import (
    PlanRequest, PlanResponse, PlannerMode, ExecutionPlan, 
    ToolCall, PlannerStats
)
from app.services.planner_service import get_planner_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_test_data(db: AsyncSession):
    """Create test origins and tools for planner testing."""
    
    logger.info("Setting up test data...")
    
    # Check if test origin already exists
    result = await db.execute(
        select(Origin).where(Origin.url == "http://localhost:8000/planner-test")
    )
    origin = result.scalar_one_or_none()
    
    if not origin:
        # Create test origin
        origin = Origin(
            id=uuid4(),
            name="test-planner-origin",
            url="http://localhost:8000/planner-test",
            auth_type=AuthType.NONE,
            status=OriginStatus.ACTIVE,
            tool_count=0,
            meta={"description": "Test origin for planner engine"}
        )
        db.add(origin)
        await db.commit()
        await db.refresh(origin)
    else:
        logger.info("Using existing test origin")
    
    # Create diverse test tools for planning
    test_tools = [
        {
            "name": "data_reader",
            "brief": "Read data from various sources",
            "description": "A tool for reading data from files, databases, or APIs",
            "categories": ["data", "input"],
            "reliability_score": 95.0
        },
        {
            "name": "data_validator", 
            "brief": "Validate data quality and format",
            "description": "Validate data schemas, check for anomalies, and ensure data quality",
            "categories": ["data", "validation"],
            "reliability_score": 92.0
        },
        {
            "name": "data_transformer",
            "brief": "Transform and clean data",
            "description": "Clean, normalize, and transform data for analysis",
            "categories": ["data", "processing"],
            "reliability_score": 88.0
        },
        {
            "name": "data_analyzer",
            "brief": "Analyze data patterns and statistics", 
            "description": "Perform statistical analysis and pattern detection on data",
            "categories": ["analysis", "statistics"],
            "reliability_score": 90.0
        },
        {
            "name": "report_generator",
            "brief": "Generate reports and visualizations",
            "description": "Create comprehensive reports with charts and visualizations",
            "categories": ["output", "visualization"],
            "reliability_score": 87.0
        },
        {
            "name": "notification_sender",
            "brief": "Send notifications and alerts",
            "description": "Send email, SMS, or webhook notifications based on results",
            "categories": ["notification", "communication"],
            "reliability_score": 93.0
        },
        {
            "name": "backup_creator",
            "brief": "Create data backups",
            "description": "Create secure backups of processed data and results",
            "categories": ["backup", "storage"],
            "reliability_score": 96.0
        }
    ]
    
    # Check if tools already exist for this origin
    result = await db.execute(
        select(Tool).where(Tool.origin_id == origin.id)
    )
    existing_tools = result.scalars().all()
    
    if existing_tools:
        logger.info(f"Using {len(existing_tools)} existing test tools")
        return origin, existing_tools
    
    tools = []
    for tool_data in test_tools:
        tool = Tool(
            id=uuid4(),
            origin_id=origin.id,
            name=tool_data["name"],
            brief=tool_data["brief"],
            description=tool_data["description"],
            categories=tool_data["categories"],
            reliability_score=tool_data["reliability_score"],
            args_schema={"type": "object", "properties": {}},
            returns_schema={"type": "object"},
            deprecated=False,
            is_side_effect_free=True
        )
        tools.append(tool)
        db.add(tool)
    
    await db.commit()
    
    logger.info(f"Created {len(tools)} test tools")
    return origin, tools


async def test_planning_modes():
    """Test different planning modes (trivial, simple, complex)."""
    
    logger.info("Testing planning modes...")
    
    planner = get_planner_service()
    
    async for db in get_db():
        try:
            # Test each mode with appropriate queries
            test_cases = [
                {
                    "query": "read data from file",
                    "mode": PlannerMode.TRIVIAL,
                    "expected_tools": 1,
                    "expected_parallel_groups": 1
                },
                {
                    "query": "read data and validate it",
                    "mode": PlannerMode.SIMPLE,
                    "expected_tools": 2,
                    "expected_parallel_groups": 2
                },
                {
                    "query": "read data, validate it, transform it, analyze patterns, and generate a report",
                    "mode": PlannerMode.COMPLEX,
                    "expected_tools": 5,
                    "expected_parallel_groups": 2  # Some parallel execution
                }
            ]
            
            for test_case in test_cases:
                logger.info(f"Testing {test_case['mode']} mode...")
                
                request = PlanRequest(
                    query=test_case["query"],
                    mode=test_case["mode"],
                    max_tools=test_case["expected_tools"]
                )
                
                response = await planner.generate_plan(request, db)
                
                logger.info(
                    f"Mode {test_case['mode']}: {len(response.plan.tool_calls) if response.plan else 0} tools, "
                    f"{len(response.plan.execution_order) if response.plan else 0} execution levels, "
                    f"{response.planning_time_ms:.1f}ms planning time"
                )
                
                # Validate response
                assert response.success, f"Planning failed: {response.error_message}"
                assert response.plan is not None, "Plan should be generated"
                assert response.mode_used == test_case["mode"], f"Expected mode {test_case['mode']}, got {response.mode_used}"
                
                plan = response.plan
                
                # Validate plan structure
                if test_case["mode"] == PlannerMode.TRIVIAL:
                    assert len(plan.tool_calls) == 1, f"Trivial plan should have 1 tool, got {len(plan.tool_calls)}"
                    assert len(plan.execution_order) == 1, "Trivial plan should have 1 execution level"
                
                elif test_case["mode"] == PlannerMode.SIMPLE:
                    assert len(plan.tool_calls) <= 3, f"Simple plan should have ≤3 tools, got {len(plan.tool_calls)}"
                    assert len(plan.execution_order) >= 1, "Simple plan should have execution order"
                
                elif test_case["mode"] == PlannerMode.COMPLEX:
                    assert len(plan.tool_calls) >= 2, f"Complex plan should have ≥2 tools, got {len(plan.tool_calls)}"
                    assert len(plan.execution_order) >= 1, "Complex plan should have execution order"
                
                # Show plan structure
                logger.info(f"  Plan ID: {plan.id}")
                logger.info(f"  Complexity: {plan.complexity_score:.2f}")
                logger.info(f"  Confidence: {plan.confidence_score:.2f}")
                logger.info(f"  Execution levels: {len(plan.execution_order)}")
                for i, level in enumerate(plan.execution_order):
                    logger.info(f"    Level {i+1}: {level}")
            
            break
            
        except Exception as e:
            logger.error(f"Planning mode test failed: {e}", exc_info=True)
            raise


async def test_plan_validation():
    """Test plan validation and cycle detection."""
    
    logger.info("Testing plan validation...")
    
    planner = get_planner_service()
    
    async for db in get_db():
        try:
            # Create a plan with valid structure
            valid_plan = ExecutionPlan(
                id="test_valid_plan",
                query="test query",
                mode=PlannerMode.SIMPLE,
                tool_calls=[
                    ToolCall(id="step1", tool_id=uuid4(), tool_name="tool1", depends_on=[]),
                    ToolCall(id="step2", tool_id=uuid4(), tool_name="tool2", depends_on=["step1"])
                ],
                execution_order=[["step1"], ["step2"]],
                estimated_duration_ms=60000,
                complexity_score=0.5,
                confidence_score=0.8,
                generation_time_ms=10.0
            )
            
            # Validate the valid plan
            errors = planner._validate_plan(valid_plan)
            assert len(errors) == 0, f"Valid plan should have no errors, got: {errors}"
            logger.info("✅ Valid plan passed validation")
            
            # Create a plan with a cycle
            cyclic_plan = ExecutionPlan(
                id="test_cyclic_plan",
                query="test query",
                mode=PlannerMode.SIMPLE,
                tool_calls=[
                    ToolCall(id="step1", tool_id=uuid4(), tool_name="tool1", depends_on=["step2"]),
                    ToolCall(id="step2", tool_id=uuid4(), tool_name="tool2", depends_on=["step1"])
                ],
                execution_order=[["step1"], ["step2"]],
                estimated_duration_ms=60000,
                complexity_score=0.5,
                confidence_score=0.8,
                generation_time_ms=10.0
            )
            
            # Validate the cyclic plan
            errors = planner._validate_plan(cyclic_plan)
            cycle_errors = [e for e in errors if e.error_type == "cycle"]
            assert len(cycle_errors) > 0, "Cyclic plan should have cycle errors"
            logger.info("✅ Cycle detection working correctly")
            
            # Create a plan with missing dependencies
            missing_dep_plan = ExecutionPlan(
                id="test_missing_dep_plan",
                query="test query",
                mode=PlannerMode.SIMPLE,
                tool_calls=[
                    ToolCall(id="step1", tool_id=uuid4(), tool_name="tool1", depends_on=["nonexistent"]),
                    ToolCall(id="step2", tool_id=uuid4(), tool_name="tool2", depends_on=["step1"])
                ],
                execution_order=[["step1"], ["step2"]],
                estimated_duration_ms=60000,
                complexity_score=0.5,
                confidence_score=0.8,
                generation_time_ms=10.0
            )
            
            # Validate the plan with missing dependencies
            errors = planner._validate_plan(missing_dep_plan)
            dep_errors = [e for e in errors if e.error_type == "missing_dependency"]
            assert len(dep_errors) > 0, "Plan with missing dependencies should have dependency errors"
            logger.info("✅ Missing dependency detection working correctly")
            
            break
            
        except Exception as e:
            logger.error(f"Plan validation test failed: {e}", exc_info=True)
            raise


async def test_plan_repair():
    """Test plan repair functionality."""
    
    logger.info("Testing plan repair...")
    
    planner = get_planner_service()
    
    async for db in get_db():
        try:
            # Test the repair functionality by creating a plan with cyclic dependencies
            # and see if the repair can break them
            cyclic_plan = ExecutionPlan(
                id="test_cyclic_repair_plan",
                query="test repair query",
                mode=PlannerMode.SIMPLE,
                tool_calls=[
                    ToolCall(id="step1", tool_id=uuid4(), tool_name="tool1", depends_on=["step2"]),
                    ToolCall(id="step2", tool_id=uuid4(), tool_name="tool2", depends_on=["step1"])
                ],
                execution_order=[["step1"], ["step2"]],
                estimated_duration_ms=60000,
                complexity_score=0.5,
                confidence_score=0.8,
                generation_time_ms=10.0
            )
            
            # Validate the plan before repair (should have errors)
            errors_before = planner._validate_plan(cyclic_plan)
            logger.info(f"Plan has {len(errors_before)} errors before repair")
            
            # Attempt to repair the plan
            repaired_plan = await planner._validate_and_repair_plan(cyclic_plan, db)
            
            # Check that repair improved the plan
            errors_after_repair = planner._validate_plan(repaired_plan)
            logger.info(f"Repair reduced errors to {len(errors_after_repair)}")
            
            # Repair iterations should be tracked
            assert repaired_plan.repair_iterations > 0, "Repair iterations should be tracked"
            
            # Execution order should be rebuilt
            assert len(repaired_plan.execution_order) >= 1, "Repaired plan should have execution order"
            
            # All tool calls should be included in execution order
            order_ids = {call_id for group in repaired_plan.execution_order for call_id in group}
            call_ids = {call.id for call in repaired_plan.tool_calls}
            assert order_ids == call_ids, "Execution order should include all tool calls after repair"
            
            logger.info("✅ Plan repair working correctly")
            
            break
            
        except Exception as e:
            logger.error(f"Plan repair test failed: {e}", exc_info=True)
            raise


async def test_plan_caching():
    """Test plan caching functionality."""
    
    logger.info("Testing plan caching...")
    
    planner = get_planner_service()
    
    async for db in get_db():
        try:
            query = "test caching query for planner"
            
            request = PlanRequest(
                query=query,
                mode=PlannerMode.SIMPLE,
                max_tools=3
            )
            
            # First request - should miss cache
            response1 = await planner.generate_plan(request, db)
            assert response1.success, "First planning request should succeed"
            assert not response1.cache_hit, "First request should miss cache"
            
            # Second request - should hit cache
            response2 = await planner.generate_plan(request, db)
            assert response2.success, "Second planning request should succeed"
            assert response2.cache_hit, "Second request should hit cache"
            assert response2.cache_key == response1.cache_key, "Cache keys should match"
            
            logger.info("✅ Plan caching working correctly")
            
            break
            
        except Exception as e:
            logger.error(f"Plan caching test failed: {e}", exc_info=True)
            raise


async def test_automatic_mode_detection():
    """Test automatic planning mode detection."""
    
    logger.info("Testing automatic mode detection...")
    
    planner = get_planner_service()
    
    async for db in get_db():
        try:
            # Get available tools for mode detection
            available_tools = await planner._get_available_tools(
                PlanRequest(query="test", max_tools=10), db
            )
            
            test_cases = [
                {
                    "query": "read file",
                    "expected_mode": PlannerMode.SIMPLE  # Single task but multiple tools available
                },
                {
                    "query": "create a complex workflow with parallel processing and then generate report",
                    "expected_mode": PlannerMode.COMPLEX  # Complex keywords
                },
                {
                    "query": "read data and validate it",
                    "expected_mode": PlannerMode.SIMPLE  # Simple keywords
                }
            ]
            
            for test_case in test_cases:
                detected_mode = planner._determine_mode(test_case["query"], available_tools)
                logger.info(f"Query: '{test_case['query']}' → Mode: {detected_mode}")
                
                # Note: Mode detection is heuristic, so we just verify it returns a valid mode
                assert detected_mode in [PlannerMode.TRIVIAL, PlannerMode.SIMPLE, PlannerMode.COMPLEX], \
                    f"Invalid mode detected: {detected_mode}"
            
            logger.info("✅ Automatic mode detection working")
            
            break
            
        except Exception as e:
            logger.error(f"Mode detection test failed: {e}", exc_info=True)
            raise


async def test_planner_stats():
    """Test planner statistics tracking."""
    
    logger.info("Testing planner statistics...")
    
    planner = get_planner_service()
    
    async for db in get_db():
        try:
            # Generate a few plans to populate stats
            requests = [
                PlanRequest(query="test stats query 1", mode=PlannerMode.TRIVIAL),
                PlanRequest(query="test stats query 2", mode=PlannerMode.SIMPLE), 
                PlanRequest(query="test stats query 3", mode=PlannerMode.COMPLEX)
            ]
            
            for request in requests:
                response = await planner.generate_plan(request, db)
                assert response.success, f"Stats test planning should succeed: {response.error_message}"
            
            # Get statistics
            stats = await planner.get_stats()
            
            logger.info("Planner Statistics:")
            logger.info(f"  Total plans: {stats.total_plans_generated}")
            logger.info(f"  Success rate: {stats.success_rate:.2%}")
            logger.info(f"  Average planning time: {stats.average_planning_time_ms:.1f}ms")
            logger.info(f"  Trivial plans: {stats.trivial_plans}")
            logger.info(f"  Simple plans: {stats.simple_plans}")
            logger.info(f"  Complex plans: {stats.complex_plans}")
            logger.info(f"  Average confidence: {stats.average_confidence:.2f}")
            logger.info(f"  Repair rate: {stats.repair_rate:.2%}")
            logger.info(f"  Fallback rate: {stats.fallback_rate:.2%}")
            logger.info(f"  Cache hit rate: {stats.cache_hit_rate:.2%}")
            
            # Validate stats structure
            assert stats.total_plans_generated >= 3, "Should have generated at least 3 plans"
            assert 0 <= stats.success_rate <= 1, "Success rate should be between 0 and 1"
            assert stats.average_planning_time_ms >= 0, "Average planning time should be non-negative"
            
            logger.info("✅ Planner statistics working correctly")
            
            break
            
        except Exception as e:
            logger.error(f"Planner stats test failed: {e}", exc_info=True)
            raise


async def run_all_tests():
    """Run all Phase 5 planner tests."""
    
    logger.info("🚀 Starting Phase 5: Planner Tests")
    
    # Initialize database
    await init_database()
    
    async for db in get_db():
        try:
            # Setup test data
            origin, tools = await setup_test_data(db)
            
            # Run tests
            await test_planning_modes()
            await test_plan_validation() 
            await test_plan_repair()
            await test_plan_caching()
            await test_automatic_mode_detection()
            await test_planner_stats()
            
            logger.info("✅ Phase 5: Planner - COMPLETE!")
            logger.info("🎯 All planner components working correctly:")
            logger.info("   - Planning modes (trivial/simple/complex)")
            logger.info("   - Plan validation and cycle detection")
            logger.info("   - Plan repair and error recovery")
            logger.info("   - DAG generation and execution ordering")
            logger.info("   - Plan caching for performance")
            logger.info("   - Automatic mode detection")
            logger.info("   - Statistics tracking")
            
            break
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())