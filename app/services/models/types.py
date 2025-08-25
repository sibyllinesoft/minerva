"""Type definitions for model services."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np


class DeviceType(Enum):
    """Supported device types for model inference."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class ModelProvider(Enum):
    """Model provider types."""
    ONNX_LOCAL = "onnx_local"
    API_REMOTE = "api_remote"
    OFF = "off"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    provider: ModelProvider
    model_path: Optional[str] = None
    dimensions: Optional[int] = None
    batch_size: int = 32
    device: DeviceType = DeviceType.AUTO
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_name: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 30


@dataclass
class EmbeddingBatch:
    """A batch of texts for embedding."""
    texts: List[str]
    batch_id: Optional[str] = None


@dataclass
class EmbeddingResult:
    """Result of embedding computation."""
    embeddings: np.ndarray  # Shape: (batch_size, dimensions)
    batch_id: Optional[str] = None
    model_name: str = ""
    processing_time_ms: float = 0.0


@dataclass
class RerankQuery:
    """A query for reranking."""
    query_text: str
    candidate_texts: List[str]
    batch_id: Optional[str] = None


@dataclass
class RerankResult:
    """Result of reranking computation."""
    scores: List[float]  # Relevance scores for each candidate
    batch_id: Optional[str] = None
    model_name: str = ""
    processing_time_ms: float = 0.0


@dataclass
class PlanRequest:
    """Request for LLM planning."""
    goal_text: str
    context: Optional[str] = None
    available_tools: Optional[List[Dict[str, Any]]] = None
    max_steps: int = 5


@dataclass
class PlanStep:
    """A single step in a plan."""
    action: str
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: Optional[str] = None


@dataclass
class PlanResult:
    """Result of LLM planning."""
    success: bool
    plan: List[PlanStep]
    reasoning: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0


class BaseModelService(ABC):
    """Base class for all model services."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model service."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the model service and cleanup resources."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized

    @abstractmethod
    async def warmup(self, sample_input: Any = None) -> None:
        """Warm up the model with sample input."""
        pass


class EmbeddingServiceInterface(BaseModelService):
    """Interface for embedding services."""

    @abstractmethod
    async def embed_batch(self, batch: EmbeddingBatch) -> EmbeddingResult:
        """Compute embeddings for a batch of texts."""
        pass

    @abstractmethod
    async def embed_single(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        pass


class RerankerServiceInterface(BaseModelService):
    """Interface for reranking services."""

    @abstractmethod
    async def rerank(self, query: RerankQuery) -> RerankResult:
        """Rerank candidates based on relevance to query."""
        pass


class PlannerServiceInterface(BaseModelService):
    """Interface for planning services."""

    @abstractmethod
    async def plan(self, request: PlanRequest) -> PlanResult:
        """Generate a plan for the given goal."""
        pass

    @abstractmethod
    async def validate_plan(self, plan: List[PlanStep]) -> bool:
        """Validate a plan structure."""
        pass


# Error types
class ModelServiceError(Exception):
    """Base exception for model service errors."""
    pass


class ModelNotInitializedError(ModelServiceError):
    """Model service not initialized."""
    pass


class ModelLoadError(ModelServiceError):
    """Failed to load model."""
    pass


class InferenceError(ModelServiceError):
    """Error during model inference."""
    pass


class DeviceNotAvailableError(ModelServiceError):
    """Requested device not available."""
    pass


class BatchSizeExceededError(ModelServiceError):
    """Batch size exceeds maximum limit."""
    pass