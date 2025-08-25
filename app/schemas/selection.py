"""Selection request and response schemas for the Meta MCP selection engine."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, validator


class SelectionMode(BaseModel):
    """Configuration for selection latency/quality tradeoffs."""
    
    name: Literal["fast", "balanced", "thorough"] = Field(
        description="Selection mode affecting speed vs quality tradeoff"
    )
    max_candidates: int = Field(
        description="Maximum candidates from hybrid search (before reranking)"
    )
    rerank_candidates: int = Field(
        description="Number of candidates to send to reranker"
    )
    final_results: int = Field(
        description="Final number of tools to return"
    )
    mmr_diversity: float = Field(
        ge=0.0, le=1.0,
        description="MMR diversity parameter (0=relevance only, 1=diversity only)"
    )
    enable_utility_scoring: bool = Field(
        description="Whether to include performance-based utility scoring"
    )


class SelectionRequest(BaseModel):
    """Request for tool selection from the catalog."""
    
    query: str = Field(
        min_length=1, max_length=1000,
        description="Natural language query for tool search"
    )
    
    mode: Literal["fast", "balanced", "thorough"] = Field(
        default="balanced",
        description="Selection mode affecting latency/quality tradeoff"
    )
    
    max_results: Optional[int] = Field(
        default=None, ge=1, le=50,
        description="Override default result count for this query"
    )
    
    # Filtering options
    categories: Optional[List[str]] = Field(
        default=None,
        description="Filter to specific tool categories"
    )
    
    origins: Optional[List[UUID]] = Field(
        default=None,
        description="Filter to specific origin servers"
    )
    
    exclude_deprecated: bool = Field(
        default=True,
        description="Exclude deprecated tools from results"
    )
    
    # Advanced options
    enable_reranking: Optional[bool] = Field(
        default=None,
        description="Override mode's reranking setting"
    )
    
    min_reliability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Minimum reliability score threshold"
    )
    
    # Context for personalization/policy
    session_id: Optional[UUID] = Field(
        default=None,
        description="Session ID for policy enforcement and personalization"
    )
    
    client_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional client context for policy decisions"
    )


class ToolScore(BaseModel):
    """Detailed scoring breakdown for a selected tool."""
    
    # Search scores
    bm25_score: Optional[float] = Field(
        description="BM25 sparse retrieval score"
    )
    
    vector_score: Optional[float] = Field(
        description="Dense vector similarity score"
    )
    
    hybrid_score: float = Field(
        description="Combined hybrid search score"
    )
    
    # Reranking and diversification
    rerank_score: Optional[float] = Field(
        description="Cross-encoder reranking score"
    )
    
    mmr_score: Optional[float] = Field(
        description="MMR diversification score"
    )
    
    # Utility scoring
    utility_score: Optional[float] = Field(
        description="Performance-based utility score"
    )
    
    performance_factors: Optional[Dict[str, float]] = Field(
        description="Individual performance metrics (success_rate, latency, etc.)"
    )
    
    # Final score
    final_score: float = Field(
        description="Final ranking score (combination of all factors)"
    )
    
    # Metadata
    score_explanation: Optional[str] = Field(
        description="Human-readable explanation of scoring"
    )


class SelectedTool(BaseModel):
    """A tool selected by the engine with scoring details."""
    
    # Tool identification
    id: UUID = Field(description="Tool unique identifier")
    origin_id: UUID = Field(description="Origin server ID")
    name: str = Field(description="Tool name")
    
    # Tool metadata
    brief: str = Field(description="Brief description")
    description: str = Field(description="Detailed description")
    categories: List[str] = Field(description="Tool categories")
    
    # Schema information
    args_schema: Dict[str, Any] = Field(description="Tool arguments schema")
    returns_schema: Dict[str, Any] = Field(description="Tool returns schema")
    
    # Quality indicators
    reliability_score: Optional[float] = Field(description="Tool reliability score")
    is_side_effect_free: bool = Field(description="Whether tool has side effects")
    
    # Selection metadata
    scores: ToolScore = Field(description="Detailed scoring breakdown")
    rank: int = Field(ge=1, description="Final ranking position (1-based)")
    
    # Origin information
    origin_name: str = Field(description="Origin server name")
    origin_url: str = Field(description="Origin server URL")


class SelectionResponse(BaseModel):
    """Response from the tool selection engine."""
    
    # Results
    tools: List[SelectedTool] = Field(description="Selected tools ranked by relevance")
    
    # Query metadata
    query: str = Field(description="Original query")
    mode: str = Field(description="Selection mode used")
    total_candidates: int = Field(description="Total candidates considered")
    
    # Timing information
    search_time_ms: float = Field(description="Hybrid search time in milliseconds")
    rerank_time_ms: Optional[float] = Field(description="Reranking time in milliseconds")
    total_time_ms: float = Field(description="Total selection time in milliseconds")
    
    # Cache information
    cache_hit: bool = Field(description="Whether result was served from cache")
    cache_key: Optional[str] = Field(description="Cache key used (if cached)")
    
    # Policy and filtering
    policy_filters_applied: List[str] = Field(
        description="Policy filters that were applied"
    )
    pre_filter_count: int = Field(
        description="Candidate count before policy filtering"
    )
    
    # Debug information (only in development)
    debug_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Debug information for development/troubleshooting"
    )


class SelectionStats(BaseModel):
    """Statistics about the selection engine performance."""
    
    # Query statistics
    total_queries: int = Field(description="Total queries processed")
    avg_query_time_ms: float = Field(description="Average query processing time")
    cache_hit_rate: float = Field(description="Cache hit rate percentage")
    
    # Search statistics
    avg_candidates_considered: float = Field(description="Average candidates per query")
    avg_rerank_usage: float = Field(description="Percentage of queries using reranking")
    
    # Performance by mode
    mode_usage: Dict[str, int] = Field(description="Usage count by selection mode")
    mode_performance: Dict[str, float] = Field(description="Average latency by mode")
    
    # Error statistics
    error_rate: float = Field(description="Query error rate percentage")
    timeout_rate: float = Field(description="Query timeout rate percentage")
    
    # Time window
    stats_period_start: datetime = Field(description="Statistics period start")
    stats_period_end: datetime = Field(description="Statistics period end")


# Default mode configurations
DEFAULT_SELECTION_MODES = {
    "fast": SelectionMode(
        name="fast",
        max_candidates=20,
        rerank_candidates=0,  # No reranking
        final_results=5,
        mmr_diversity=0.1,  # Minimal diversity
        enable_utility_scoring=False
    ),
    "balanced": SelectionMode(
        name="balanced",
        max_candidates=50,
        rerank_candidates=20,
        final_results=10,
        mmr_diversity=0.3,
        enable_utility_scoring=True
    ),
    "thorough": SelectionMode(
        name="thorough",
        max_candidates=100,
        rerank_candidates=50,
        final_results=15,
        mmr_diversity=0.5,  # High diversity
        enable_utility_scoring=True
    )
}