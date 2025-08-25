"""Model service manager with warmup hooks and lifecycle management."""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from ...core.config import get_settings
from .registry import ModelRegistry
from .types import ModelServiceError

logger = logging.getLogger(__name__)


class ModelServiceManager:
    """Manager for all model services with warmup and lifecycle management."""

    def __init__(self):
        self._registry: Optional[ModelRegistry] = None
        self._initialized = False
        self._warmup_completed = False

    async def initialize(self) -> None:
        """Initialize all model services."""
        if self._initialized:
            return

        try:
            logger.info("Initializing model service manager...")
            settings = get_settings()
            
            # Initialize registry
            self._registry = ModelRegistry()
            
            # Get model configuration
            models_config = {}
            if hasattr(settings, 'models') and settings.models:
                models_config = {
                    "embeddings": settings.models.embeddings.model_dump(),
                    "reranker": settings.models.reranker.model_dump(), 
                    "planner": settings.models.planner.model_dump(),
                }

            await self._registry.initialize(models_config)
            self._initialized = True
            
            logger.info("Model service manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model service manager: {e}")
            await self.shutdown()
            raise ModelServiceError(f"Model service manager initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown all model services."""
        if self._registry:
            await self._registry.shutdown()
            self._registry = None
        
        self._initialized = False
        self._warmup_completed = False
        logger.info("Model service manager shutdown complete")

    async def warmup(self, timeout_seconds: float = 60.0) -> Dict[str, Any]:
        """Warm up all model services with timeout."""
        if not self._initialized:
            raise ModelServiceError("Manager not initialized")

        if self._warmup_completed:
            logger.info("Model services already warmed up")
            return self.get_warmup_status()

        logger.info("Starting model service warmup...")
        start_time = time.time()
        warmup_results = {}

        try:
            # Warmup with timeout
            warmup_task = asyncio.create_task(self._registry.warmup_all())
            await asyncio.wait_for(warmup_task, timeout=timeout_seconds)
            
            self._warmup_completed = True
            warmup_time = (time.time() - start_time) * 1000
            
            warmup_results = {
                "success": True,
                "warmup_time_ms": warmup_time,
                "services": self._registry.get_service_info(),
                "message": f"All services warmed up in {warmup_time:.2f}ms"
            }
            
            logger.info(f"Model warmup completed: {warmup_time:.2f}ms")

        except asyncio.TimeoutError:
            warmup_time = (time.time() - start_time) * 1000
            warmup_results = {
                "success": False,
                "warmup_time_ms": warmup_time,
                "services": self._registry.get_service_info(),
                "message": f"Warmup timed out after {timeout_seconds}s",
                "error": "timeout"
            }
            logger.warning(f"Model warmup timed out after {timeout_seconds}s")

        except Exception as e:
            warmup_time = (time.time() - start_time) * 1000
            warmup_results = {
                "success": False,
                "warmup_time_ms": warmup_time,
                "services": self._registry.get_service_info() if self._registry else {},
                "message": f"Warmup failed: {e}",
                "error": str(e)
            }
            logger.error(f"Model warmup failed: {e}")

        return warmup_results

    def get_registry(self) -> ModelRegistry:
        """Get the model registry."""
        if not self._registry:
            raise ModelServiceError("Registry not initialized")
        return self._registry

    def get_warmup_status(self) -> Dict[str, Any]:
        """Get current warmup status."""
        if not self._initialized:
            return {
                "initialized": False,
                "warmed_up": False,
                "services": {},
                "message": "Manager not initialized"
            }

        return {
            "initialized": self._initialized,
            "warmed_up": self._warmup_completed,
            "services": self._registry.get_service_info() if self._registry else {},
            "message": "Ready" if self._warmup_completed else "Initialized but not warmed up"
        }

    @property
    def is_ready(self) -> bool:
        """Check if manager is ready for inference."""
        return self._initialized and self._warmup_completed

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        if not self._initialized:
            return {
                "healthy": False,
                "message": "Manager not initialized",
                "services": {}
            }

        service_health = {}
        overall_healthy = True

        try:
            # Check each service
            services = self._registry.get_service_info()
            for service_name, service_info in services.items():
                try:
                    service = self._registry.get_service(service_name)
                    if service and service.is_initialized:
                        # Quick health check - could be expanded
                        service_health[service_name] = {
                            "healthy": True,
                            "initialized": True,
                            "provider": service_info.get("provider"),
                            "device": service_info.get("device")
                        }
                    else:
                        service_health[service_name] = {
                            "healthy": False,
                            "initialized": False,
                            "error": "Service not initialized"
                        }
                        overall_healthy = False

                except Exception as e:
                    service_health[service_name] = {
                        "healthy": False,
                        "error": str(e)
                    }
                    overall_healthy = False

        except Exception as e:
            overall_healthy = False
            logger.error(f"Health check failed: {e}")

        return {
            "healthy": overall_healthy,
            "message": "All services healthy" if overall_healthy else "Some services unhealthy",
            "services": service_health,
            "warmed_up": self._warmup_completed
        }

    async def reload_config(self) -> Dict[str, Any]:
        """Reload configuration and reinitialize services."""
        try:
            logger.info("Reloading model service configuration...")
            
            # Shutdown current services
            await self.shutdown()
            
            # Reinitialize with new config
            await self.initialize()
            
            return {
                "success": True,
                "message": "Configuration reloaded successfully",
                "services": self._registry.get_service_info() if self._registry else {}
            }

        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
            return {
                "success": False,
                "message": f"Configuration reload failed: {e}",
                "error": str(e)
            }


# Global manager instance
_manager: Optional[ModelServiceManager] = None


def get_model_manager() -> ModelServiceManager:
    """Get the global model service manager."""
    global _manager
    if _manager is None:
        _manager = ModelServiceManager()
    return _manager


@asynccontextmanager
async def model_service_lifespan():
    """Context manager for model service lifecycle."""
    manager = get_model_manager()
    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.shutdown()


async def ensure_model_services_ready(warmup_timeout: float = 60.0) -> None:
    """Ensure model services are initialized and warmed up."""
    manager = get_model_manager()
    
    if not manager._initialized:
        await manager.initialize()
    
    if not manager._warmup_completed:
        warmup_result = await manager.warmup(warmup_timeout)
        if not warmup_result["success"]:
            logger.warning(f"Model warmup incomplete: {warmup_result['message']}")
            # Continue anyway - services may still be functional


async def create_model_services_context():
    """Create async context for model services (for FastAPI lifespan)."""
    manager = get_model_manager()
    
    # Initialize services
    await manager.initialize()
    
    # Start warmup in background (don't block startup)
    warmup_task = asyncio.create_task(manager.warmup())
    
    try:
        yield {
            "model_manager": manager,
            "warmup_task": warmup_task
        }
    finally:
        # Cancel warmup if still running
        if not warmup_task.done():
            warmup_task.cancel()
            try:
                await warmup_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown services
        await manager.shutdown()