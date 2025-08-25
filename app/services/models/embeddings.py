"""Embedding service with ONNX and API providers."""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

from .types import (
    EmbeddingServiceInterface,
    ModelConfig,
    ModelProvider,
    DeviceType,
    EmbeddingBatch,
    EmbeddingResult,
    ModelServiceError,
    ModelNotInitializedError,
    ModelLoadError,
    InferenceError,
    DeviceNotAvailableError,
    BatchSizeExceededError,
)

logger = logging.getLogger(__name__)


class EmbeddingService(EmbeddingServiceInterface):
    """Embedding service supporting ONNX local and API remote providers."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._session = None
        self._dimensions = None
        self._max_batch_size = config.batch_size
        self._device_providers = []
        
    async def initialize(self) -> None:
        """Initialize the embedding service based on provider."""
        try:
            if self.config.provider == ModelProvider.ONNX_LOCAL:
                await self._init_onnx()
            elif self.config.provider == ModelProvider.API_REMOTE:
                await self._init_api()
            else:
                raise ModelServiceError(f"Unsupported provider: {self.config.provider}")
                
            self._initialized = True
            logger.info(f"Embedding service initialized with {self.config.provider.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise ModelLoadError(f"Embedding service initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        if self._session:
            # ONNX sessions don't have explicit cleanup
            self._session = None
        self._model = None
        self._tokenizer = None
        self._initialized = False
        logger.info("Embedding service shutdown complete")

    async def warmup(self, sample_input=None) -> None:
        """Warm up the model with sample input."""
        if not self.is_initialized:
            raise ModelNotInitializedError("Service not initialized")

        try:
            logger.info("Warming up embedding service...")
            start_time = time.time()
            
            # Use provided sample or default
            if sample_input and hasattr(sample_input, 'texts'):
                texts = sample_input.texts
            else:
                texts = ["Hello world", "This is a warmup test"]

            # Perform warmup embedding
            result = await self.embed_batch(EmbeddingBatch(texts=texts))
            
            warmup_time = (time.time() - start_time) * 1000
            logger.info(f"Embedding warmup complete: {warmup_time:.2f}ms, shape={result.embeddings.shape}")
            
        except Exception as e:
            logger.warning(f"Embedding warmup failed: {e}")

    async def embed_batch(self, batch: EmbeddingBatch) -> EmbeddingResult:
        """Compute embeddings for a batch of texts."""
        if not self.is_initialized:
            raise ModelNotInitializedError("Service not initialized")

        if len(batch.texts) > self._max_batch_size:
            raise BatchSizeExceededError(
                f"Batch size {len(batch.texts)} exceeds maximum {self._max_batch_size}"
            )

        start_time = time.time()
        
        try:
            if self.config.provider == ModelProvider.ONNX_LOCAL:
                embeddings = await self._embed_onnx(batch.texts)
            elif self.config.provider == ModelProvider.API_REMOTE:
                embeddings = await self._embed_api(batch.texts)
            else:
                raise InferenceError(f"Unsupported provider: {self.config.provider}")

            processing_time_ms = (time.time() - start_time) * 1000

            return EmbeddingResult(
                embeddings=embeddings,
                batch_id=batch.batch_id,
                model_name=self.config.model_name or "unknown",
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            logger.error(f"Embedding inference failed: {e}")
            raise InferenceError(f"Embedding computation failed: {e}")

    async def embed_single(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        batch = EmbeddingBatch(texts=[text])
        result = await self.embed_batch(batch)
        return result.embeddings[0]

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if not self._dimensions:
            raise ModelNotInitializedError("Service not initialized or dimensions unknown")
        return self._dimensions

    async def _init_onnx(self) -> None:
        """Initialize ONNX model."""
        try:
            import onnxruntime as ort
            
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise ModelLoadError(f"Model file not found: {model_path}")

            # Configure execution providers based on device
            providers = self._get_onnx_providers()
            
            # Create inference session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            self._session = ort.InferenceSession(
                str(model_path),
                providers=providers,
                sess_options=session_options
            )
            
            # Get output shape to determine dimensions
            output_info = self._session.get_outputs()[0]
            if len(output_info.shape) >= 2:
                self._dimensions = output_info.shape[-1]
            else:
                # Fallback to config or default
                self._dimensions = self.config.dimensions or 384

            logger.info(f"ONNX embedding model loaded: {model_path}, dims={self._dimensions}")
            logger.info(f"Using providers: {self._session.get_providers()}")

        except ImportError:
            raise ModelLoadError("onnxruntime not installed")
        except Exception as e:
            raise ModelLoadError(f"Failed to load ONNX model: {e}")

    async def _init_api(self) -> None:
        """Initialize API-based embedding service."""
        # For API-based embeddings, we just validate config
        if not self.config.api_key:
            raise ModelLoadError("API key required for remote embedding provider")
        
        # Set dimensions from config or use default
        self._dimensions = self.config.dimensions or 1536  # Common OpenAI dimension
        
        logger.info(f"API embedding service configured: dims={self._dimensions}")

    def _get_onnx_providers(self) -> List[str]:
        """Get ONNX execution providers based on device configuration."""
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
        except ImportError:
            return ["CPUExecutionProvider"]

        providers = []
        
        if self.config.device == DeviceType.CUDA:
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                
        elif self.config.device == DeviceType.MPS:
            if "CoreMLExecutionProvider" in available_providers:
                providers.append("CoreMLExecutionProvider") 
            else:
                logger.warning("MPS requested but not available, falling back to CPU")

        # Always add CPU as fallback
        providers.append("CPUExecutionProvider")
        
        return providers

    async def _embed_onnx(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings using ONNX model."""
        try:
            # For now, implement a simple tokenization approach
            # In production, you'd use the actual tokenizer for your model
            inputs = self._prepare_onnx_inputs(texts)
            
            # Run inference
            outputs = self._session.run(None, inputs)
            embeddings = outputs[0]  # Assume first output is embeddings
            
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings

        except Exception as e:
            raise InferenceError(f"ONNX inference failed: {e}")

    def _prepare_onnx_inputs(self, texts: List[str]) -> dict:
        """Prepare inputs for ONNX model."""
        # This is a simplified implementation
        # Real implementation would use proper tokenization
        max_length = 512
        vocab_size = 30522  # BERT vocab size as example
        
        batch_size = len(texts)
        input_ids = np.zeros((batch_size, max_length), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_length), dtype=np.int64)
        
        # Simplified tokenization (just use text length as proxy)
        for i, text in enumerate(texts):
            length = min(len(text.split()), max_length)
            input_ids[i, :length] = np.random.randint(1, vocab_size, length)
            attention_mask[i, :length] = 1
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    async def _embed_api(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings using API provider."""
        try:
            import httpx
            
            # Example for OpenAI API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.api_base or 'https://api.openai.com'}/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.config.model_name or "text-embedding-ada-002",
                        "input": texts
                    },
                    timeout=self.config.timeout_seconds
                )
                response.raise_for_status()
                
                data = response.json()
                embeddings = np.array([item["embedding"] for item in data["data"]])
                
                return embeddings

        except Exception as e:
            raise InferenceError(f"API embedding failed: {e}")

    def get_batch_size_for_texts(self, texts: List[str]) -> int:
        """Get appropriate batch size for given texts."""
        # Simple heuristic: reduce batch size for longer texts
        avg_length = sum(len(text) for text in texts) / len(texts)
        
        if avg_length > 1000:
            return min(8, self._max_batch_size)
        elif avg_length > 500:
            return min(16, self._max_batch_size) 
        else:
            return min(self._max_batch_size, len(texts))

    async def embed_large_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a large batch by splitting into smaller batches."""
        if len(texts) <= self._max_batch_size:
            batch = EmbeddingBatch(texts=texts)
            result = await self.embed_batch(batch)
            return result.embeddings

        # Split into smaller batches
        embeddings_list = []
        for i in range(0, len(texts), self._max_batch_size):
            batch_texts = texts[i:i + self._max_batch_size]
            batch = EmbeddingBatch(texts=batch_texts)
            result = await self.embed_batch(batch)
            embeddings_list.append(result.embeddings)

        return np.vstack(embeddings_list)