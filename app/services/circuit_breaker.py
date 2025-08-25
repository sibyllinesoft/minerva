"""
Circuit breaker implementation for upstream service protection.
"""
import asyncio
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Callable, Any
from uuid import UUID
import logging

from app.schemas.proxy import CircuitBreakerState, CircuitBreakerStats


logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Circuit breaker is open, blocking requests."""
    pass


class CircuitBreaker:
    """Circuit breaker for individual origins with configurable thresholds."""
    
    def __init__(
        self,
        origin_id: UUID,
        origin_url: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        success_threshold: int = 2
    ):
        self.origin_id = origin_id
        self.origin_url = origin_url
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        self.half_open_calls = 0
        
        # Thread safety
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection."""
        async with self._lock:
            await self._check_state()
            
            if self.state == CircuitBreakerState.OPEN:
                logger.warning(f"Circuit breaker OPEN for origin {self.origin_id}, blocking call")
                raise CircuitBreakerError(f"Circuit breaker is open for origin {self.origin_url}")
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    logger.warning(f"Half-open call limit reached for origin {self.origin_id}")
                    raise CircuitBreakerError(f"Half-open call limit reached for {self.origin_url}")
                self.half_open_calls += 1
        
        # Execute the function outside the lock to avoid blocking other calls
        try:
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            await self._on_success(duration)
            return result
            
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _check_state(self):
        """Check and update circuit breaker state based on time and conditions."""
        now = datetime.utcnow()
        
        if self.state == CircuitBreakerState.OPEN:
            if self.next_attempt_time and now >= self.next_attempt_time:
                logger.info(f"Circuit breaker transitioning to HALF_OPEN for origin {self.origin_id}")
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Half-open state is managed by call success/failure
            pass
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Check if we should open due to failures
            if self.failure_count >= self.failure_threshold:
                await self._open_circuit()
    
    async def _on_success(self, duration: float):
        """Handle successful call."""
        async with self._lock:
            self.last_success_time = datetime.utcnow()
            self.success_count += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= self.success_threshold:
                    logger.info(f"Circuit breaker transitioning to CLOSED for origin {self.origin_id}")
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.half_open_calls = 0
            
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                if self.failure_count > 0:
                    self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self, error: Exception):
        """Handle failed call."""
        async with self._lock:
            self.last_failure_time = datetime.utcnow()
            self.failure_count += 1
            
            logger.warning(f"Circuit breaker failure for origin {self.origin_id}: {error}")
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Failure in half-open state should open the circuit
                await self._open_circuit()
            
            elif self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    await self._open_circuit()
    
    async def _open_circuit(self):
        """Open the circuit breaker."""
        logger.error(f"Opening circuit breaker for origin {self.origin_id} after {self.failure_count} failures")
        self.state = CircuitBreakerState.OPEN
        self.next_attempt_time = datetime.utcnow() + timedelta(seconds=self.recovery_timeout)
        self.half_open_calls = 0
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics."""
        return CircuitBreakerStats(
            origin_id=self.origin_id,
            origin_url=self.origin_url,
            state=self.state,
            failure_count=self.failure_count,
            success_count=self.success_count,
            last_failure_time=self.last_failure_time,
            last_success_time=self.last_success_time,
            next_attempt_time=self.next_attempt_time,
            failure_threshold=self.failure_threshold,
            recovery_timeout=self.recovery_timeout,
            half_open_max_calls=self.half_open_max_calls
        )
    
    async def reset(self):
        """Reset circuit breaker to closed state."""
        async with self._lock:
            logger.info(f"Resetting circuit breaker for origin {self.origin_id}")
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.last_failure_time = None
            self.next_attempt_time = None
    
    async def force_open(self):
        """Force circuit breaker to open state."""
        async with self._lock:
            logger.warning(f"Force opening circuit breaker for origin {self.origin_id}")
            await self._open_circuit()


class CircuitBreakerRegistry:
    """Registry for managing circuit breakers across origins."""
    
    def __init__(self):
        self._breakers: Dict[UUID, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
        self.default_config = {
            'failure_threshold': 5,
            'recovery_timeout': 60,
            'half_open_max_calls': 3,
            'success_threshold': 2
        }
    
    async def get_breaker(self, origin_id: UUID, origin_url: str, **config) -> CircuitBreaker:
        """Get or create circuit breaker for origin."""
        if origin_id not in self._breakers:
            async with self._lock:
                if origin_id not in self._breakers:
                    breaker_config = {**self.default_config, **config}
                    self._breakers[origin_id] = CircuitBreaker(
                        origin_id=origin_id,
                        origin_url=origin_url,
                        **breaker_config
                    )
                    logger.info(f"Created circuit breaker for origin {origin_id}")
        
        return self._breakers[origin_id]
    
    async def remove_breaker(self, origin_id: UUID):
        """Remove circuit breaker for origin."""
        async with self._lock:
            if origin_id in self._breakers:
                del self._breakers[origin_id]
                logger.info(f"Removed circuit breaker for origin {origin_id}")
    
    def get_all_stats(self) -> Dict[UUID, CircuitBreakerStats]:
        """Get statistics for all circuit breakers."""
        return {
            origin_id: breaker.get_stats() 
            for origin_id, breaker in self._breakers.items()
        }
    
    async def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()
        logger.info("Reset all circuit breakers")
    
    def get_open_count(self) -> int:
        """Get count of open circuit breakers."""
        return sum(1 for breaker in self._breakers.values() 
                  if breaker.state == CircuitBreakerState.OPEN)
    
    def get_available_count(self) -> int:
        """Get count of available (closed or half-open) origins."""
        return sum(1 for breaker in self._breakers.values() 
                  if breaker.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN])


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()