"""Tool selection engine implementing hybrid search with reranking and policy filtering."""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from uuid import UUID
import numpy as np

from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..core.config import get_settings
from ..models.tool import Tool, ToolEmbedding, ToolStats
from ..models.origin import Origin
from ..models.policy import Policy
from ..schemas.selection import (
    SelectionRequest, SelectionResponse, SelectedTool, ToolScore,
    DEFAULT_SELECTION_MODES, SelectionMode
)
from ..services.models.manager import get_model_manager

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Hybrid search combining BM25 (Tantivy) and dense vector retrieval."""
    
    def __init__(self):
        self.settings = get_settings()
        # Cache for search index - in production this would use external search engine
        self._text_search_cache = {}
    
    async def search_tools(
        self, 
        query: str, 
        session: AsyncSession,
        max_candidates: int = 50,
        categories: Optional[List[str]] = None,
        origins: Optional[List[UUID]] = None,
        exclude_deprecated: bool = True
    ) -> List[Tuple[Tool, float]]:
        """Perform hybrid search combining text and vector similarity."""
        
        # Build base query with filters
        base_query = select(Tool).options(selectinload(Tool.origin))
        filters = []
        
        if exclude_deprecated:
            filters.append(Tool.deprecated == False)
        
        if categories:
            # PostgreSQL array overlap using && operator
            filters.append(Tool.categories.op('&&')(categories))
        
        if origins:
            filters.append(Tool.origin_id.in_(origins))
        
        if filters:
            base_query = base_query.where(and_(*filters))
        
        # Get embeddings if model service is available
        model_manager = get_model_manager()
        vector_results = []
        
        embeddings_service = None
        if model_manager and model_manager._registry:
            embeddings_service = model_manager._registry.get_embeddings_service()
        
        if embeddings_service:
            try:
                # Generate query embedding
                query_embedding = await embeddings_service.embed_text(query)
                if query_embedding is not None:
                    vector_results = await self._vector_search(
                        query_embedding, session, base_query, max_candidates
                    )
            except Exception as e:
                logger.warning(f"Vector search failed, using text-only: {e}")
        
        # Perform text search (simplified BM25-like scoring)
        text_results = await self._text_search(query, session, base_query, max_candidates)
        
        # Combine and deduplicate results
        combined_results = self._combine_search_results(
            text_results, vector_results, max_candidates
        )
        
        return combined_results
    
    async def _vector_search(
        self,
        query_embedding: np.ndarray,
        session: AsyncSession,
        base_query,
        max_candidates: int
    ) -> List[Tuple[Tool, float]]:
        """Perform dense vector similarity search using pgvector."""
        
        try:
            # Convert numpy array to list for PostgreSQL
            embedding_list = query_embedding.tolist()
            
            # Query for vector similarity using L2 distance
            # Note: pgvector uses <-> for L2 distance (lower is more similar)
            vector_query = (
                base_query
                .join(ToolEmbedding, Tool.id == ToolEmbedding.tool_id)
                .order_by(text(f"tool_embeddings.embedding <-> '[{','.join(map(str, embedding_list))}]'"))
                .limit(max_candidates)
            )
            
            result = await session.execute(vector_query)
            tools = result.scalars().all()
            
            # Convert L2 distances to similarity scores (0-1)
            # Note: In production, we'd get the actual distances from the query
            vector_results = []
            for i, tool in enumerate(tools):
                # Simulate similarity score - in practice get from query
                similarity = 1.0 - (i * 0.01)  # Decreasing similarity
                vector_results.append((tool, max(0.0, similarity)))
            
            return vector_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _text_search(
        self,
        query: str,
        session: AsyncSession,
        base_query,
        max_candidates: int
    ) -> List[Tuple[Tool, float]]:
        """Perform text search with BM25-like scoring."""
        
        # Simplified text search using PostgreSQL full-text search
        # In production, this would use Tantivy or Elasticsearch
        
        try:
            # Create text search vector from query
            search_terms = query.lower().split()
            
            # Build similarity conditions
            similarity_conditions = []
            for term in search_terms:
                # Search in name, brief, and description
                term_pattern = f"%{term}%"
                similarity_conditions.extend([
                    func.lower(Tool.name).like(term_pattern),
                    func.lower(Tool.brief).like(term_pattern),
                    func.lower(Tool.description).like(term_pattern)
                ])
            
            if similarity_conditions:
                text_query = (
                    base_query
                    .where(or_(*similarity_conditions))
                    .limit(max_candidates)
                )
            else:
                # No search terms, return top tools by reliability
                text_query = (
                    base_query
                    .order_by(Tool.reliability_score.desc().nullslast())
                    .limit(max_candidates)
                )
            
            result = await session.execute(text_query)
            tools = result.scalars().all()
            
            # Calculate BM25-like scores
            text_results = []
            for tool in tools:
                score = self._calculate_text_score(query, tool)
                text_results.append((tool, score))
            
            # Sort by score descending
            text_results.sort(key=lambda x: x[1], reverse=True)
            return text_results[:max_candidates]
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def _calculate_text_score(self, query: str, tool: Tool) -> float:
        """Calculate BM25-like text similarity score."""
        
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0
        
        # Combine tool text content
        content = (
            f"{tool.name} {tool.brief} {tool.description}"
            f" {' '.join(tool.categories or [])}"
        ).lower()
        
        content_terms = set(content.split())
        
        # Simple term frequency scoring
        matches = query_terms.intersection(content_terms)
        if not matches:
            return 0.0
        
        # Basic BM25-like scoring
        match_ratio = len(matches) / len(query_terms)
        
        # Boost for name matches
        name_matches = sum(1 for term in query_terms if term in tool.name.lower())
        name_boost = 2.0 if name_matches > 0 else 1.0
        
        # Boost for exact phrase matches
        phrase_boost = 1.5 if query.lower() in content else 1.0
        
        return match_ratio * name_boost * phrase_boost
    
    def _combine_search_results(
        self,
        text_results: List[Tuple[Tool, float]],
        vector_results: List[Tuple[Tool, float]],
        max_candidates: int
    ) -> List[Tuple[Tool, float]]:
        """Combine and deduplicate search results with hybrid scoring."""
        
        # Combine results with weighted hybrid scoring
        text_weight = 0.4
        vector_weight = 0.6
        
        # Create dictionaries for efficient lookup
        text_scores = {tool.id: score for tool, score in text_results}
        vector_scores = {tool.id: score for tool, score in vector_results}
        
        # Get all unique tools
        all_tools = {}
        for tool, _ in text_results:
            all_tools[tool.id] = tool
        for tool, _ in vector_results:
            all_tools[tool.id] = tool
        
        # Calculate hybrid scores
        hybrid_results = []
        for tool_id, tool in all_tools.items():
            text_score = text_scores.get(tool_id, 0.0)
            vector_score = vector_scores.get(tool_id, 0.0)
            
            # Hybrid score calculation
            hybrid_score = (text_weight * text_score + vector_weight * vector_score)
            
            # Store individual scores for debugging
            tool._debug_text_score = text_score
            tool._debug_vector_score = vector_score
            
            hybrid_results.append((tool, hybrid_score))
        
        # Sort by hybrid score and limit
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        return hybrid_results[:max_candidates]


class MMRDiversifier:
    """Maximal Marginal Relevance (MMR) for result diversification."""
    
    def __init__(self, lambda_param: float = 0.5):
        self.lambda_param = lambda_param  # Relevance vs diversity tradeoff
    
    async def diversify_results(
        self,
        candidates: List[Tuple[Tool, float]],
        max_results: int,
        embeddings_service=None
    ) -> List[Tuple[Tool, float]]:
        """Apply MMR diversification to candidate tools."""
        
        if not candidates or max_results <= 0:
            return []
        
        if len(candidates) <= max_results:
            return candidates
        
        # Initialize with highest scored candidate
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        # If no embeddings available, fall back to simple category diversification
        if not embeddings_service:
            return self._category_based_diversification(candidates, max_results)
        
        try:
            # Get embeddings for all candidates
            selected_embeddings = []
            remaining_embeddings = []
            
            # For now, use simple category-based diversification
            # In production, this would use actual embeddings
            return await self._semantic_diversification(candidates, max_results)
            
        except Exception as e:
            logger.warning(f"MMR diversification failed, using category-based: {e}")
            return self._category_based_diversification(candidates, max_results)
    
    def _category_based_diversification(
        self, 
        candidates: List[Tuple[Tool, float]], 
        max_results: int
    ) -> List[Tuple[Tool, float]]:
        """Diversify based on tool categories."""
        
        selected = []
        used_categories = set()
        
        # First pass: select tools with unique primary categories
        for tool, score in candidates:
            if len(selected) >= max_results:
                break
            
            primary_category = tool.categories[0] if tool.categories else "uncategorized"
            
            if primary_category not in used_categories:
                selected.append((tool, score))
                used_categories.add(primary_category)
        
        # Second pass: fill remaining slots with highest scores
        for tool, score in candidates:
            if len(selected) >= max_results:
                break
            
            if not any(t.id == tool.id for t, _ in selected):
                selected.append((tool, score))
        
        return selected
    
    async def _semantic_diversification(
        self,
        candidates: List[Tuple[Tool, float]],
        max_results: int
    ) -> List[Tuple[Tool, float]]:
        """Diversify based on semantic similarity (placeholder)."""
        
        # Placeholder for actual semantic diversification
        # In production, this would use tool embeddings to calculate
        # semantic distance and apply MMR algorithm
        
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        while len(selected) < max_results and remaining:
            best_candidate = None
            best_mmr_score = -1
            
            for i, (candidate_tool, relevance_score) in enumerate(remaining):
                # Calculate semantic similarity with selected tools
                # For now, use category overlap as proxy
                avg_similarity = self._calculate_category_similarity(
                    candidate_tool, [tool for tool, _ in selected]
                )
                
                # MMR score: lambda * relevance - (1-lambda) * similarity
                mmr_score = (
                    self.lambda_param * relevance_score -
                    (1 - self.lambda_param) * avg_similarity
                )
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = (candidate_tool, relevance_score)
            
            if best_candidate:
                selected.append(best_candidate)
                remaining = [item for item in remaining if item[0].id != best_candidate[0].id]
        
        return selected
    
    def _calculate_category_similarity(self, tool: Tool, selected_tools: List[Tool]) -> float:
        """Calculate category-based similarity."""
        
        if not selected_tools:
            return 0.0
        
        tool_categories = set(tool.categories or [])
        
        similarities = []
        for selected_tool in selected_tools:
            selected_categories = set(selected_tool.categories or [])
            
            if not tool_categories and not selected_categories:
                similarity = 1.0  # Both uncategorized
            elif not tool_categories or not selected_categories:
                similarity = 0.0  # One uncategorized
            else:
                # Jaccard similarity
                intersection = len(tool_categories.intersection(selected_categories))
                union = len(tool_categories.union(selected_categories))
                similarity = intersection / union if union > 0 else 0.0
            
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities)


class UtilityScorer:
    """Performance-based utility scoring for tool selection."""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def calculate_utility_scores(
        self,
        candidates: List[Tuple[Tool, float]],
        session: AsyncSession
    ) -> Dict[UUID, Dict[str, float]]:
        """Calculate utility scores based on performance metrics."""
        
        tool_ids = [tool.id for tool, _ in candidates]
        if not tool_ids:
            return {}
        
        # Get recent performance stats
        stats_query = (
            select(ToolStats)
            .where(ToolStats.tool_id.in_(tool_ids))
            .order_by(ToolStats.window_start.desc())
        )
        
        result = await session.execute(stats_query)
        stats = result.scalars().all()
        
        # Group stats by tool
        tool_stats = {}
        for stat in stats:
            if stat.tool_id not in tool_stats:
                tool_stats[stat.tool_id] = []
            tool_stats[stat.tool_id].append(stat)
        
        # Calculate utility scores
        utility_scores = {}
        for tool, _ in candidates:
            scores = self._calculate_tool_utility(tool, tool_stats.get(tool.id, []))
            utility_scores[tool.id] = scores
        
        return utility_scores
    
    def _calculate_tool_utility(
        self, 
        tool: Tool, 
        stats: List[ToolStats]
    ) -> Dict[str, float]:
        """Calculate individual tool utility metrics."""
        
        if not stats:
            # No performance data - use reliability score
            base_score = (tool.reliability_score or 50.0) / 100.0
            return {
                "success_rate": base_score,
                "latency_score": base_score,
                "reliability_score": base_score,
                "overall_utility": base_score
            }
        
        # Calculate weighted averages of recent performance
        recent_stats = stats[:5]  # Use last 5 windows
        
        # Success rate score
        success_rates = [s.success_rate for s in recent_stats if s.success_rate is not None]
        success_score = (sum(success_rates) / len(success_rates) / 100.0) if success_rates else 0.5
        
        # Latency score (inverse of p95 latency, normalized)
        p95_values = [s.p95_ms for s in recent_stats if s.p95_ms is not None]
        if p95_values:
            avg_p95 = sum(p95_values) / len(p95_values)
            # Normalize: lower latency = higher score (max at 100ms = 1.0, 1000ms = 0.1)
            latency_score = max(0.1, min(1.0, (1000 - avg_p95) / 900))
        else:
            latency_score = 0.5
        
        # Reliability penalty for high error rates
        reliability_score = min(1.0, (tool.reliability_score or 50.0) / 100.0)
        
        # Combined utility score
        overall_utility = (
            0.4 * success_score +
            0.3 * latency_score +
            0.3 * reliability_score
        )
        
        return {
            "success_rate": success_score,
            "latency_score": latency_score, 
            "reliability_score": reliability_score,
            "overall_utility": overall_utility
        }


class SelectionEngine:
    """Main tool selection engine coordinating all components."""
    
    def __init__(self):
        self.search_engine = HybridSearchEngine()
        self.mmr_diversifier = MMRDiversifier()
        self.utility_scorer = UtilityScorer()
        self.settings = get_settings()
        # Simple in-memory cache - production would use Redis
        self._cache = {}
    
    async def select_tools(
        self,
        request: SelectionRequest,
        session: AsyncSession
    ) -> SelectionResponse:
        """Main tool selection pipeline."""
        
        start_time = time.time()
        
        # Get selection mode configuration
        mode_config = DEFAULT_SELECTION_MODES.get(request.mode, DEFAULT_SELECTION_MODES["balanced"])
        
        # Override mode settings if specified in request
        if request.max_results is not None:
            mode_config.final_results = request.max_results
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check cache
        if cache_key in self._cache:
            logger.info(f"Cache hit for query: {request.query[:50]}...")
            cached_response = self._cache[cache_key]
            cached_response.cache_hit = True
            return cached_response
        
        try:
            # Step 1: Hybrid search (BM25 + Vector)
            search_start = time.time()
            candidates = await self.search_engine.search_tools(
                query=request.query,
                session=session,
                max_candidates=mode_config.max_candidates,
                categories=request.categories,
                origins=request.origins,
                exclude_deprecated=request.exclude_deprecated
            )
            search_time = (time.time() - search_start) * 1000
            
            if not candidates:
                # No candidates found
                return SelectionResponse(
                    tools=[],
                    query=request.query,
                    mode=request.mode,
                    total_candidates=0,
                    search_time_ms=search_time,
                    rerank_time_ms=None,
                    total_time_ms=(time.time() - start_time) * 1000,
                    cache_hit=False,
                    cache_key=None,
                    policy_filters_applied=[],
                    pre_filter_count=0
                )
            
            logger.info(f"Hybrid search found {len(candidates)} candidates")
            
            # Step 2: Policy filtering (pre-reranking)
            filtered_candidates = await self._apply_policy_filters(
                candidates, request, session, pre_rerank=True
            )
            pre_filter_count = len(candidates)
            
            # Step 3: Cross-encoder reranking (if enabled)
            rerank_time = None
            if (request.enable_reranking is not False and 
                mode_config.rerank_candidates > 0 and 
                len(filtered_candidates) > 1):
                
                rerank_start = time.time()
                rerank_candidates = filtered_candidates[:mode_config.rerank_candidates]
                reranked = await self._apply_reranking(request.query, rerank_candidates)
                rerank_time = (time.time() - rerank_start) * 1000
                
                # Merge reranked with remaining candidates
                remaining = filtered_candidates[mode_config.rerank_candidates:]
                filtered_candidates = reranked + remaining
                
                logger.info(f"Reranked top {len(rerank_candidates)} candidates")
            
            # Step 4: Utility scoring (if enabled)
            utility_scores = {}
            if mode_config.enable_utility_scoring:
                utility_scores = await self.utility_scorer.calculate_utility_scores(
                    filtered_candidates, session
                )
            
            # Step 5: MMR diversification
            # Set lambda parameter and then diversify
            self.mmr_diversifier.lambda_param = mode_config.mmr_diversity
            diversified = await self.mmr_diversifier.diversify_results(
                filtered_candidates,
                mode_config.final_results
            )
            
            # Step 6: Policy filtering (post-diversification)
            final_tools = await self._apply_policy_filters(
                diversified, request, session, pre_rerank=False
            )
            
            # Step 7: Create response
            selected_tools = []
            for rank, (tool, hybrid_score) in enumerate(final_tools, 1):
                
                tool_utility = utility_scores.get(tool.id, {})
                
                # Create scoring breakdown
                scores = ToolScore(
                    bm25_score=getattr(tool, '_debug_text_score', None),
                    vector_score=getattr(tool, '_debug_vector_score', None),
                    hybrid_score=hybrid_score,
                    rerank_score=getattr(tool, '_rerank_score', None),
                    mmr_score=hybrid_score,  # MMR adjusts the main score
                    utility_score=tool_utility.get('overall_utility'),
                    performance_factors=tool_utility if tool_utility else None,
                    final_score=hybrid_score,
                    score_explanation=self._generate_score_explanation(
                        hybrid_score, tool_utility
                    )
                )
                
                selected_tool = SelectedTool(
                    id=tool.id,
                    origin_id=tool.origin_id,
                    name=tool.name,
                    brief=tool.brief,
                    description=tool.description,
                    categories=tool.categories or [],
                    args_schema=tool.args_schema or {},
                    returns_schema=tool.returns_schema or {},
                    reliability_score=tool.reliability_score,
                    is_side_effect_free=tool.is_side_effect_free,
                    scores=scores,
                    rank=rank,
                    origin_name=tool.origin.name if tool.origin else "Unknown",
                    origin_url=tool.origin.url if tool.origin else ""
                )
                
                selected_tools.append(selected_tool)
            
            total_time = (time.time() - start_time) * 1000
            
            response = SelectionResponse(
                tools=selected_tools,
                query=request.query,
                mode=request.mode,
                total_candidates=len(candidates),
                search_time_ms=search_time,
                rerank_time_ms=rerank_time,
                total_time_ms=total_time,
                cache_hit=False,
                cache_key=cache_key,
                policy_filters_applied=[],  # TODO: Track applied filters
                pre_filter_count=pre_filter_count
            )
            
            # Cache the response
            self._cache[cache_key] = response
            
            logger.info(
                f"Selection complete: {len(selected_tools)} tools in {total_time:.1f}ms"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Selection failed for query '{request.query}': {e}", exc_info=True)
            raise
    
    def _generate_cache_key(self, request: SelectionRequest) -> str:
        """Generate cache key for the selection request."""
        
        # Create hash from request parameters
        key_data = {
            "query": request.query,
            "mode": request.mode,
            "max_results": request.max_results,
            "categories": sorted(request.categories) if request.categories else None,
            "origins": sorted(str(o) for o in request.origins) if request.origins else None,
            "exclude_deprecated": request.exclude_deprecated,
            "enable_reranking": request.enable_reranking,
            "min_reliability": request.min_reliability
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    async def _apply_policy_filters(
        self,
        candidates: List[Tuple[Tool, float]],
        request: SelectionRequest,
        session: AsyncSession,
        pre_rerank: bool = True
    ) -> List[Tuple[Tool, float]]:
        """Apply policy-based filtering to candidates."""
        
        # TODO: Implement policy system
        # For now, just apply basic reliability filtering
        
        if request.min_reliability is not None:
            filtered = []
            for tool, score in candidates:
                reliability = tool.reliability_score or 0.0
                if reliability >= (request.min_reliability * 100):
                    filtered.append((tool, score))
            return filtered
        
        return candidates
    
    async def _apply_reranking(
        self,
        query: str,
        candidates: List[Tuple[Tool, float]]
    ) -> List[Tuple[Tool, float]]:
        """Apply cross-encoder reranking to candidates."""
        
        # TODO: Implement actual cross-encoder reranking
        # For now, return candidates with simulated rerank scores
        
        model_manager = get_model_manager()
        reranker_service = None
        if model_manager and model_manager._registry:
            reranker_service = model_manager._registry.get_reranker_service()
            
        if not reranker_service:
            logger.warning("Reranker service not available, skipping reranking")
            return candidates
        
        try:
            # Prepare texts for reranking
            texts = []
            for tool, _ in candidates:
                tool_text = f"{tool.name}: {tool.brief}"
                texts.append(tool_text)
            
            # Get reranking scores
            rerank_scores = await reranker_service.rerank(
                query, texts
            )
            
            # Combine with original candidates and add rerank scores
            reranked = []
            for i, (tool, original_score) in enumerate(candidates):
                rerank_score = rerank_scores[i] if i < len(rerank_scores) else original_score
                tool._rerank_score = rerank_score  # Store for debugging
                
                # Weighted combination of original and rerank scores
                final_score = 0.6 * rerank_score + 0.4 * original_score
                reranked.append((tool, final_score))
            
            # Sort by reranked scores
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates
    
    def _generate_score_explanation(
        self, 
        hybrid_score: float, 
        utility_scores: Dict[str, float]
    ) -> str:
        """Generate human-readable score explanation."""
        
        explanation = f"Relevance: {hybrid_score:.3f}"
        
        if utility_scores:
            utility = utility_scores.get('overall_utility', 0)
            explanation += f", Performance: {utility:.3f}"
        
        return explanation


# Global instance
_selection_engine: Optional[SelectionEngine] = None


def get_selection_engine() -> SelectionEngine:
    """Get the global selection engine instance."""
    global _selection_engine
    if _selection_engine is None:
        _selection_engine = SelectionEngine()
    return _selection_engine