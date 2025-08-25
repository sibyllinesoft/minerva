"""LLM planner service with JSON grammar enforcement and repair."""

import asyncio
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Union
import httpx

from .types import (
    PlannerServiceInterface,
    ModelConfig,
    ModelProvider,
    PlanRequest,
    PlanResult,
    PlanStep,
    ModelServiceError,
    ModelNotInitializedError,
    ModelLoadError,
    InferenceError,
)

logger = logging.getLogger(__name__)


class PlannerService(PlannerServiceInterface):
    """LLM-based planner with JSON grammar enforcement and repair."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None
        self._json_schema = self._get_plan_schema()
        self._repair_examples = self._get_repair_examples()

    async def initialize(self) -> None:
        """Initialize the planner service."""
        try:
            if self.config.provider == ModelProvider.OFF:
                self._initialized = True
                logger.info("Planner service disabled")
                return

            if self.config.provider == ModelProvider.API_REMOTE:
                await self._init_api()
            else:
                raise ModelServiceError(f"Unsupported planner provider: {self.config.provider}")

            self._initialized = True
            logger.info(f"Planner service initialized with {self.config.provider.value}")

        except Exception as e:
            logger.error(f"Failed to initialize planner service: {e}")
            raise ModelLoadError(f"Planner service initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False
        logger.info("Planner service shutdown complete")

    async def warmup(self, sample_input=None) -> None:
        """Warm up the planner service."""
        if not self.is_initialized:
            raise ModelNotInitializedError("Service not initialized")

        if self.config.provider == ModelProvider.OFF:
            return

        try:
            logger.info("Warming up planner service...")
            start_time = time.time()

            # Use provided sample or default
            if sample_input and hasattr(sample_input, 'goal_text'):
                request = sample_input
            else:
                from .types import PlanRequest
                request = PlanRequest(
                    goal_text="Test planning warmup",
                    context="This is a warmup test",
                    max_steps=2
                )

            # Perform warmup planning
            result = await self.plan(request)

            warmup_time = (time.time() - start_time) * 1000
            logger.info(f"Planner warmup complete: {warmup_time:.2f}ms, success={result.success}")

        except Exception as e:
            logger.warning(f"Planner warmup failed: {e}")

    async def plan(self, request: PlanRequest) -> PlanResult:
        """Generate a plan for the given goal."""
        if not self.is_initialized:
            raise ModelNotInitializedError("Service not initialized")

        if self.config.provider == ModelProvider.OFF:
            # Return a trivial plan when planner is disabled
            return self._create_trivial_plan(request)

        start_time = time.time()

        try:
            # Generate plan with retry logic
            plan_result = await self._generate_plan_with_retry(request)
            
            processing_time_ms = (time.time() - start_time) * 1000
            plan_result.processing_time_ms = processing_time_ms

            return plan_result

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback to trivial plan on failure
            return self._create_trivial_plan(request, error=str(e))

    async def validate_plan(self, plan: List[PlanStep]) -> bool:
        """Validate a plan structure."""
        try:
            # Basic structural validation
            if not plan:
                return False

            for step in plan:
                if not step.action or not step.tool_name:
                    return False
                
                if not isinstance(step.parameters, dict):
                    return False

            # Additional semantic validation could be added here
            return True

        except Exception as e:
            logger.error(f"Plan validation failed: {e}")
            return False

    async def _init_api(self) -> None:
        """Initialize API client for LLM planning."""
        if not self.config.api_key:
            raise ModelLoadError("API key required for LLM planner")

        self._client = httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
        )

        logger.info(f"API planner configured: model={self.config.model_name}")

    async def _generate_plan_with_retry(self, request: PlanRequest) -> PlanResult:
        """Generate plan with parse-repair-fallback retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Generate raw response
                raw_response = await self._call_llm(request, attempt)
                
                # Try to parse JSON
                plan_data = self._parse_json_response(raw_response)
                
                if plan_data:
                    # Convert to plan steps
                    plan_steps = self._convert_to_plan_steps(plan_data)
                    
                    # Validate plan
                    if await self.validate_plan(plan_steps):
                        return PlanResult(
                            success=True,
                            plan=plan_steps,
                            reasoning=plan_data.get("reasoning", "")
                        )

                # If we get here, parsing/validation failed
                last_error = "Invalid plan structure"

            except Exception as e:
                last_error = str(e)
                
                # Try to repair JSON on parse errors
                if attempt < self.config.max_retries:
                    logger.warning(f"Planning attempt {attempt + 1} failed: {e}, trying repair")
                    continue

        # All attempts failed, return fallback
        logger.warning(f"All planning attempts failed, using trivial plan: {last_error}")
        return self._create_trivial_plan(request, error=last_error)

    async def _call_llm(self, request: PlanRequest, attempt: int) -> str:
        """Call the LLM API to generate a plan."""
        # Build prompt with JSON grammar enforcement
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(request)
        
        # Add repair context for retry attempts
        if attempt > 0:
            user_prompt += f"\n\nPrevious attempts failed. Please ensure JSON is valid and follows the exact schema.\n{self._get_repair_examples()}"

        try:
            if "openai" in self.config.model_name.lower():
                return await self._call_openai(system_prompt, user_prompt)
            elif "anthropic" in self.config.model_name.lower() or "claude" in self.config.model_name.lower():
                return await self._call_anthropic(system_prompt, user_prompt)
            else:
                return await self._call_generic_api(system_prompt, user_prompt)

        except Exception as e:
            raise InferenceError(f"LLM API call failed: {e}")

    async def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API."""
        response = await self._client.post(
            f"{self.config.api_base or 'https://api.openai.com'}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.config.model_name or "gpt-4-turbo-preview",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 2048
            }
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic API."""
        response = await self._client.post(
            f"{self.config.api_base or 'https://api.anthropic.com'}/v1/messages",
            headers={
                "x-api-key": self.config.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": self.config.model_name or "claude-3-sonnet-20240229",
                "max_tokens": 2048,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1
            }
        )
        response.raise_for_status()
        
        data = response.json()
        return data["content"][0]["text"]

    async def _call_generic_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call generic OpenAI-compatible API."""
        response = await self._client.post(
            f"{self.config.api_base}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2048
            }
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _build_system_prompt(self) -> str:
        """Build system prompt for plan generation."""
        return f"""You are a planning assistant that creates execution plans for tool-based tasks.

You MUST respond with valid JSON following this exact schema:
{json.dumps(self._json_schema, indent=2)}

Rules:
1. Always respond with valid JSON
2. Include reasoning for the plan
3. Keep plans concise but complete
4. Use available tools effectively
5. Maximum {5} steps per plan
6. Each step must be executable"""

    def _build_user_prompt(self, request: PlanRequest) -> str:
        """Build user prompt with request details."""
        prompt = f"Goal: {request.goal_text}\n"
        
        if request.context:
            prompt += f"Context: {request.context}\n"
        
        if request.available_tools:
            tools_str = "\n".join([f"- {tool['name']}: {tool.get('description', '')}" 
                                 for tool in request.available_tools])
            prompt += f"\nAvailable tools:\n{tools_str}\n"
        
        prompt += f"\nCreate a plan with maximum {request.max_steps} steps."
        return prompt

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response with error handling."""
        try:
            # First, try direct parsing
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or other formatting
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find any JSON-like structure
            brace_match = re.search(r'\{.*\}', response, re.DOTALL)
            if brace_match:
                try:
                    return json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            logger.warning(f"Failed to parse JSON from response: {response[:200]}...")
            return None

    def _convert_to_plan_steps(self, plan_data: Dict[str, Any]) -> List[PlanStep]:
        """Convert parsed plan data to PlanStep objects."""
        steps = []
        plan_steps = plan_data.get("plan", [])
        
        for step_data in plan_steps:
            step = PlanStep(
                action=step_data.get("action", ""),
                tool_name=step_data.get("tool_name", ""),
                parameters=step_data.get("parameters", {}),
                reasoning=step_data.get("reasoning")
            )
            steps.append(step)
        
        return steps

    def _create_trivial_plan(self, request: PlanRequest, error: Optional[str] = None) -> PlanResult:
        """Create a trivial fallback plan."""
        # Simple plan that exposes available tools without complex planning
        trivial_steps = []
        
        if request.available_tools:
            # Use the first available tool as a fallback
            tool = request.available_tools[0]
            step = PlanStep(
                action="execute",
                tool_name=tool["name"],
                parameters={},
                reasoning="Trivial fallback plan using available tool"
            )
            trivial_steps.append(step)
        
        return PlanResult(
            success=bool(trivial_steps),
            plan=trivial_steps,
            reasoning="Fallback to trivial plan",
            error=error
        )

    def _get_plan_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for plan responses."""
        return {
            "type": "object",
            "required": ["plan", "reasoning"],
            "properties": {
                "plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["action", "tool_name", "parameters"],
                        "properties": {
                            "action": {"type": "string"},
                            "tool_name": {"type": "string"}, 
                            "parameters": {"type": "object"},
                            "reasoning": {"type": "string"}
                        }
                    }
                },
                "reasoning": {"type": "string"}
            }
        }

    def _get_repair_examples(self) -> str:
        """Get examples for JSON repair guidance."""
        return """
Example valid response:
{
  "plan": [
    {
      "action": "search",
      "tool_name": "web_search",
      "parameters": {"query": "example"},
      "reasoning": "Need to find information"
    }
  ],
  "reasoning": "This plan searches for the requested information"
}"""