"""Model registry for managing model providers and instances."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Type, Any
import platform

from .types import (
    BaseModelService,
    EmbeddingServiceInterface, 
    RerankerServiceInterface,
    PlannerServiceInterface,
    ModelConfig,
    ModelProvider,
    DeviceType,
    ModelServiceError,
    DeviceNotAvailableError,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing model services and their lifecycle."""

    def __init__(self):
        self._services: Dict[str, BaseModelService] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._initialized = False

    async def initialize(self, models_config: Dict[str, Any]) -> None:
        """Initialize all model services from configuration."""
        try:
            logger.info("Initializing model registry...")
            
            # Parse configuration
            embedding_config = self._parse_model_config(
                models_config.get("embeddings", {}), "embeddings"
            )
            reranker_config = self._parse_model_config(
                models_config.get("reranker", {}), "reranker" 
            )
            planner_config = self._parse_model_config(
                models_config.get("planner", {}), "planner"
            )

            self._configs = {
                "embeddings": embedding_config,
                "reranker": reranker_config, 
                "planner": planner_config,
            }

            # Initialize services
            tasks = []
            
            # Embeddings service (always required)
            if embedding_config.provider != ModelProvider.OFF:
                tasks.append(self._init_embedding_service(embedding_config))

            # Reranker service (always required)
            if reranker_config.provider != ModelProvider.OFF:
                tasks.append(self._init_reranker_service(reranker_config))

            # Planner service (optional)
            if planner_config.provider != ModelProvider.OFF:
                tasks.append(self._init_planner_service(planner_config))

            # Initialize all services concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            self._initialized = True
            logger.info("Model registry initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model registry: {e}")
            await self.shutdown()
            raise ModelServiceError(f"Model registry initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown all model services."""
        logger.info("Shutting down model registry...")
        
        shutdown_tasks = []
        for service in self._services.values():
            if service.is_initialized:
                shutdown_tasks.append(service.shutdown())
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self._services.clear()
        self._initialized = False
        logger.info("Model registry shutdown complete")

    async def warmup_all(self) -> None:
        """Warm up all initialized services."""
        if not self._initialized:
            raise ModelServiceError("Registry not initialized")

        logger.info("Warming up all model services...")
        warmup_tasks = []

        for service_name, service in self._services.items():
            if service.is_initialized:
                sample_input = self._get_sample_input_for_service(service_name)
                warmup_tasks.append(service.warmup(sample_input))

        if warmup_tasks:
            await asyncio.gather(*warmup_tasks, return_exceptions=True)

        logger.info("Model warmup complete")

    def get_service(self, service_name: str) -> Optional[BaseModelService]:
        """Get a model service by name."""
        return self._services.get(service_name)

    def get_embedding_service(self) -> Optional[EmbeddingServiceInterface]:
        """Get the embeddings service."""
        service = self._services.get("embeddings")
        if service and isinstance(service, EmbeddingServiceInterface):
            return service
        return None

    def get_reranker_service(self) -> Optional[RerankerServiceInterface]:
        """Get the reranker service.""" 
        service = self._services.get("reranker")
        if service and isinstance(service, RerankerServiceInterface):
            return service
        return None

    def get_planner_service(self) -> Optional[PlannerServiceInterface]:
        """Get the planner service."""
        service = self._services.get("planner")
        if service and isinstance(service, PlannerServiceInterface):
            return service
        return None

    @property
    def is_initialized(self) -> bool:
        """Check if registry is initialized."""
        return self._initialized

    def _parse_model_config(self, config_data: Dict[str, Any], service_name: str) -> ModelConfig:
        """Parse model configuration from data."""
        provider_str = config_data.get("provider", "off")
        provider = ModelProvider(provider_str)
        
        device_str = config_data.get("device", "auto")
        device = DeviceType(device_str)
        
        # Auto-detect device if requested
        if device == DeviceType.AUTO:
            device = self._detect_best_device()

        return ModelConfig(
            provider=provider,
            model_path=config_data.get("model_path"),
            dimensions=config_data.get("dimensions"),
            batch_size=config_data.get("batch_size", 32),
            device=device,
            api_key=config_data.get("api_key"),
            api_base=config_data.get("api_base"),
            model_name=config_data.get("model", config_data.get("model_name")),
            max_retries=config_data.get("max_retries", 3),
            timeout_seconds=config_data.get("timeout_seconds", 30),
        )

    def _detect_best_device(self) -> DeviceType:
        """Auto-detect the best available device."""
        try:
            # Check for NVIDIA GPU
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            if "CUDAExecutionProvider" in providers:
                logger.info("Detected CUDA device")
                return DeviceType.CUDA
            elif "CoreMLExecutionProvider" in providers and platform.system() == "Darwin":
                logger.info("Detected MPS device")
                return DeviceType.MPS
            else:
                logger.info("Using CPU device")
                return DeviceType.CPU
                
        except ImportError:
            logger.warning("ONNXRuntime not available, defaulting to CPU")
            return DeviceType.CPU

    async def _init_embedding_service(self, config: ModelConfig) -> None:
        """Initialize embedding service."""
        from .embeddings import EmbeddingService
        
        try:
            service = EmbeddingService(config)
            await service.initialize()
            self._services["embeddings"] = service
            logger.info(f"Initialized embeddings service with {config.provider.value}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings service: {e}")
            raise

    async def _init_reranker_service(self, config: ModelConfig) -> None:
        """Initialize reranker service."""
        from .reranker import RerankerService
        
        try:
            service = RerankerService(config) 
            await service.initialize()
            self._services["reranker"] = service
            logger.info(f"Initialized reranker service with {config.provider.value}")
        except Exception as e:
            logger.error(f"Failed to initialize reranker service: {e}")
            raise

    async def _init_planner_service(self, config: ModelConfig) -> None:
        """Initialize planner service."""
        from .planner import PlannerService
        
        try:
            service = PlannerService(config)
            await service.initialize() 
            self._services["planner"] = service
            logger.info(f"Initialized planner service with {config.provider.value}")
        except Exception as e:
            logger.error(f"Failed to initialize planner service: {e}")
            raise

    def _get_sample_input_for_service(self, service_name: str) -> Any:
        """Get appropriate sample input for warmup."""
        if service_name == "embeddings":
            from .types import EmbeddingBatch
            return EmbeddingBatch(texts=["Hello world", "This is a test"])
        elif service_name == "reranker":
            from .types import RerankQuery
            return RerankQuery(
                query_text="test query",
                candidate_texts=["candidate 1", "candidate 2"]
            )
        elif service_name == "planner":
            from .types import PlanRequest
            return PlanRequest(
                goal_text="test goal",
                context="test context"
            )
        return None

    def get_service_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all services."""
        info = {}
        for name, service in self._services.items():
            config = self._configs.get(name)
            info[name] = {
                "initialized": service.is_initialized,
                "provider": config.provider.value if config else "unknown",
                "device": config.device.value if config else "unknown",
                "model_path": config.model_path if config else None,
            }
        return info