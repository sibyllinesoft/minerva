"""Cross-encoder reranking service with dynamic batching."""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from .types import (
    RerankerServiceInterface,
    ModelConfig,
    ModelProvider,
    DeviceType,
    RerankQuery,
    RerankResult,
    ModelServiceError,
    ModelNotInitializedError,
    ModelLoadError,
    InferenceError,
    BatchSizeExceededError,
)

logger = logging.getLogger(__name__)


class RerankerService(RerankerServiceInterface):
    """Cross-encoder reranking service with ONNX and API providers."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._session = None
        self._max_batch_size = config.batch_size
        self._early_exit_margin = 0.1  # Early exit if score margin is clear
        self._device_providers = []

    async def initialize(self) -> None:
        """Initialize the reranker service."""
        try:
            if self.config.provider == ModelProvider.ONNX_LOCAL:
                await self._init_onnx()
            elif self.config.provider == ModelProvider.API_REMOTE:
                await self._init_api()
            else:
                raise ModelServiceError(f"Unsupported provider: {self.config.provider}")

            self._initialized = True
            logger.info(f"Reranker service initialized with {self.config.provider.value}")

        except Exception as e:
            logger.error(f"Failed to initialize reranker service: {e}")
            raise ModelLoadError(f"Reranker service initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        if self._session:
            self._session = None
        self._initialized = False
        logger.info("Reranker service shutdown complete")

    async def warmup(self, sample_input=None) -> None:
        """Warm up the reranker model."""
        if not self.is_initialized:
            raise ModelNotInitializedError("Service not initialized")

        try:
            logger.info("Warming up reranker service...")
            start_time = time.time()

            # Use provided sample or default
            if sample_input and hasattr(sample_input, 'query_text'):
                query = sample_input
            else:
                from .types import RerankQuery
                query = RerankQuery(
                    query_text="test query",
                    candidate_texts=["candidate 1", "candidate 2", "candidate 3"]
                )

            # Perform warmup reranking
            result = await self.rerank(query)

            warmup_time = (time.time() - start_time) * 1000
            logger.info(f"Reranker warmup complete: {warmup_time:.2f}ms, {len(result.scores)} scores")

        except Exception as e:
            logger.warning(f"Reranker warmup failed: {e}")

    async def rerank(self, query: RerankQuery) -> RerankResult:
        """Rerank candidates based on relevance to query."""
        if not self.is_initialized:
            raise ModelNotInitializedError("Service not initialized")

        if len(query.candidate_texts) > self._max_batch_size:
            # Handle large batches by splitting
            return await self._rerank_large_batch(query)

        start_time = time.time()

        try:
            if self.config.provider == ModelProvider.ONNX_LOCAL:
                scores = await self._rerank_onnx(query)
            elif self.config.provider == ModelProvider.API_REMOTE:
                scores = await self._rerank_api(query)
            else:
                raise InferenceError(f"Unsupported provider: {self.config.provider}")

            processing_time_ms = (time.time() - start_time) * 1000

            return RerankResult(
                scores=scores,
                batch_id=query.batch_id,
                model_name=self.config.model_name or "unknown",
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise InferenceError(f"Reranking computation failed: {e}")

    async def _init_onnx(self) -> None:
        """Initialize ONNX reranker model."""
        try:
            import onnxruntime as ort

            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise ModelLoadError(f"Model file not found: {model_path}")

            # Configure execution providers
            providers = self._get_onnx_providers()

            # Create inference session with optimizations
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Enable int8 optimization if available
            if self.config.device in [DeviceType.CPU, DeviceType.CUDA]:
                session_options.inter_op_num_threads = 1
                session_options.intra_op_num_threads = 4

            self._session = ort.InferenceSession(
                str(model_path),
                providers=providers,
                sess_options=session_options
            )

            logger.info(f"ONNX reranker model loaded: {model_path}")
            logger.info(f"Using providers: {self._session.get_providers()}")

        except ImportError:
            raise ModelLoadError("onnxruntime not installed")
        except Exception as e:
            raise ModelLoadError(f"Failed to load ONNX reranker model: {e}")

    async def _init_api(self) -> None:
        """Initialize API-based reranker service."""
        if not self.config.api_key:
            raise ModelLoadError("API key required for remote reranker provider")

        logger.info("API reranker service configured")

    def _get_onnx_providers(self) -> List[str]:
        """Get ONNX execution providers for reranker."""
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
        except ImportError:
            return ["CPUExecutionProvider"]

        providers = []

        if self.config.device == DeviceType.CUDA:
            if "CUDAExecutionProvider" in available_providers:
                # Configure CUDA provider for cross-encoder
                providers.append(("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
                }))
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")

        elif self.config.device == DeviceType.MPS:
            if "CoreMLExecutionProvider" in available_providers:
                providers.append("CoreMLExecutionProvider")
            else:
                logger.warning("MPS requested but not available, falling back to CPU")

        # CPU provider with optimizations
        providers.append(("CPUExecutionProvider", {
            "intra_op_num_threads": 4,
            "inter_op_num_threads": 1,
        }))

        return providers

    async def _rerank_onnx(self, query: RerankQuery) -> List[float]:
        """Perform reranking using ONNX model."""
        try:
            # Prepare input pairs (query, candidate)
            pairs = [(query.query_text, candidate) for candidate in query.candidate_texts]
            
            # Dynamic batching with early exit optimization
            scores = []
            batch_size = min(len(pairs), self._max_batch_size)
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_inputs = self._prepare_reranker_inputs(batch_pairs)
                
                # Run inference
                outputs = self._session.run(None, batch_inputs)
                batch_scores = self._extract_scores(outputs)
                scores.extend(batch_scores)
                
                # Early exit optimization: if we have a clear winner, can stop early
                if self._should_early_exit(scores, len(query.candidate_texts)):
                    # Pad remaining scores with low values
                    remaining = len(query.candidate_texts) - len(scores)
                    scores.extend([-1.0] * remaining)
                    break

            return scores[:len(query.candidate_texts)]

        except Exception as e:
            raise InferenceError(f"ONNX reranking failed: {e}")

    def _prepare_reranker_inputs(self, pairs: List[Tuple[str, str]]) -> dict:
        """Prepare inputs for ONNX cross-encoder model."""
        # Simplified implementation - real models would use proper tokenization
        max_length = 512
        vocab_size = 30522
        
        batch_size = len(pairs)
        input_ids = np.zeros((batch_size, max_length), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_length), dtype=np.int64)
        token_type_ids = np.zeros((batch_size, max_length), dtype=np.int64)
        
        # Simplified tokenization for query-candidate pairs
        for i, (query, candidate) in enumerate(pairs):
            # Combine query and candidate with [SEP] token
            combined_text = f"{query} [SEP] {candidate}"
            length = min(len(combined_text.split()), max_length)
            
            input_ids[i, :length] = np.random.randint(1, vocab_size, length)
            attention_mask[i, :length] = 1
            
            # Mark candidate part with token_type_ids
            sep_pos = len(query.split()) + 1
            if sep_pos < max_length:
                token_type_ids[i, sep_pos:length] = 1
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }

    def _extract_scores(self, outputs) -> List[float]:
        """Extract relevance scores from model outputs."""
        # Assume output is logits, apply softmax to get probabilities
        logits = outputs[0]  # Shape: (batch_size, num_classes)
        
        if logits.shape[-1] == 1:
            # Single score output
            scores = logits.flatten().tolist()
        else:
            # Multi-class output, take positive class probability
            exp_logits = np.exp(logits)
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            scores = probabilities[:, 1].tolist()  # Assume class 1 is "relevant"
        
        return scores

    def _should_early_exit(self, current_scores: List[float], total_candidates: int) -> bool:
        """Determine if we can exit early based on score margins."""
        if len(current_scores) < 2:
            return False
        
        # Sort current scores to find margin between top candidates
        sorted_scores = sorted(current_scores, reverse=True)
        top_score = sorted_scores[0]
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        
        margin = top_score - second_score
        
        # Early exit if margin is significant and we've processed enough candidates
        min_processed = min(8, total_candidates // 2)
        return (margin > self._early_exit_margin and 
                len(current_scores) >= min_processed)

    async def _rerank_api(self, query: RerankQuery) -> List[float]:
        """Perform reranking using API provider."""
        try:
            import httpx

            # Example API call (adapt for your provider)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.api_base or 'https://api.example.com'}/v1/rerank",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.config.model_name or "rerank-base",
                        "query": query.query_text,
                        "documents": query.candidate_texts,
                        "return_documents": False
                    },
                    timeout=self.config.timeout_seconds
                )
                response.raise_for_status()

                data = response.json()
                scores = [result["relevance_score"] for result in data["results"]]
                
                return scores

        except Exception as e:
            raise InferenceError(f"API reranking failed: {e}")

    async def _rerank_large_batch(self, query: RerankQuery) -> RerankResult:
        """Handle large batches by splitting into smaller ones."""
        start_time = time.time()
        all_scores = []

        # Split candidates into smaller batches
        for i in range(0, len(query.candidate_texts), self._max_batch_size):
            batch_candidates = query.candidate_texts[i:i + self._max_batch_size]
            batch_query = RerankQuery(
                query_text=query.query_text,
                candidate_texts=batch_candidates,
                batch_id=f"{query.batch_id}_batch_{i}" if query.batch_id else None
            )
            
            batch_result = await self.rerank(batch_query)
            all_scores.extend(batch_result.scores)

        processing_time_ms = (time.time() - start_time) * 1000

        return RerankResult(
            scores=all_scores,
            batch_id=query.batch_id,
            model_name=self.config.model_name or "unknown",
            processing_time_ms=processing_time_ms,
        )

    def get_optimal_batch_size(self, query_length: int, avg_candidate_length: int) -> int:
        """Get optimal batch size based on input characteristics."""
        # Adjust batch size based on text lengths to manage memory usage
        total_length = query_length + avg_candidate_length
        
        if total_length > 1000:
            return min(4, self._max_batch_size)
        elif total_length > 500:
            return min(8, self._max_batch_size)
        else:
            return self._max_batch_size