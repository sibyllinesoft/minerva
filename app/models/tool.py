"""Tool models for MCP capabilities."""

from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

from sqlalchemy import (
    String, Text, Boolean, JSON, Integer, Float, 
    ForeignKey, UniqueConstraint, Index, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from .base import Base, UUIDMixin, TimestampMixin


class Tool(Base, UUIDMixin, TimestampMixin):
    """Represents a tool/capability from an upstream MCP server.
    
    This is the core entity representing individual tools that can be
    selected and executed through the Meta MCP system.
    """
    
    __tablename__ = "tools"
    
    # Foreign key to origin
    origin_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("origins.id", ondelete="CASCADE"),
        nullable=False,
        comment="Origin server that provides this tool"
    )
    
    # Tool identification  
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Tool name (unique within origin)"
    )
    
    version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="1.0.0",
        comment="Tool version"
    )
    
    # Documentation and description
    brief: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Brief description for search and selection"
    )
    
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Detailed description and usage"
    )
    
    # Schema definitions
    args_schema: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="JSON Schema for tool arguments"
    )
    
    returns_schema: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="JSON Schema for tool return values"
    )
    
    # Categorization and examples
    categories: Mapped[List[str]] = mapped_column(
        ARRAY(String(100)),
        nullable=False,
        default=list,
        comment="Tool categories for filtering and organization"
    )
    
    examples: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Usage examples and test cases"
    )
    
    # Lifecycle management
    last_seen_at: Mapped[datetime] = mapped_column(
        nullable=False,
        server_default="now()",
        comment="Last time this tool was seen during crawling"
    )
    
    deprecated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether this tool is deprecated"
    )
    
    deprecated_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Reason for deprecation if applicable"
    )
    
    # Quality and reliability indicators
    is_side_effect_free: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether tool has side effects (safe for probing)"
    )
    
    reliability_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Computed reliability score (0.0-1.0)"
    )
    
    # Search and ranking metadata
    search_embedding_version: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Version of embedding model used"
    )
    
    # Relationships
    origin = relationship("Origin", back_populates="tools")
    embeddings = relationship(
        "ToolEmbedding",
        back_populates="tool",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    stats = relationship(
        "ToolStats",
        back_populates="tool", 
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    
    # Constraints and indexes
    __table_args__ = (
        # Unique tool name within each origin
        UniqueConstraint("origin_id", "name", name="uq_tools_origin_name"),
        
        # Search and filtering indexes
        Index("ix_tools_name", "name"),
        Index("ix_tools_categories", "categories", postgresql_using="gin"),
        Index("ix_tools_deprecated", "deprecated"),
        Index("ix_tools_last_seen_at", "last_seen_at"),
        Index("ix_tools_reliability_score", "reliability_score"),
        
        # Full-text search index on description content
        Index(
            "ix_tools_search_content", 
            "brief", "description",
            postgresql_using="gin",
            postgresql_ops={
                "brief": "gin_trgm_ops",
                "description": "gin_trgm_ops"
            }
        ),
        
        # Compound indexes for common queries
        Index("ix_tools_origin_deprecated", "origin_id", "deprecated"),
        Index("ix_tools_origin_last_seen", "origin_id", "last_seen_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Tool(name='{self.name}', origin_id='{self.origin_id}', deprecated={self.deprecated})>"


class ToolEmbedding(Base):
    """Vector embeddings for tools used in dense retrieval.
    
    Separate table to optimize vector operations and allow multiple 
    embedding models per tool.
    """
    
    __tablename__ = "tool_embeddings"
    
    # Primary key is tool_id + model combination
    tool_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tools.id", ondelete="CASCADE"),
        primary_key=True,
        comment="Tool this embedding represents"
    )
    
    model: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
        comment="Embedding model identifier"
    )
    
    # Vector embedding (defaulting to 768 dimensions)
    embedding: Mapped[Vector] = mapped_column(
        Vector(768),
        nullable=False,
        comment="Dense vector embedding"
    )
    
    # Metadata
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False,
        server_default="now()",
        onupdate="now()",
        comment="When this embedding was last updated"
    )
    
    embedding_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Version of the content used to generate embedding"
    )
    
    # Relationships
    tool = relationship("Tool", back_populates="embeddings")
    
    # Vector similarity index (HNSW with L2 distance)
    __table_args__ = (
        Index(
            "ix_tool_embeddings_vector_l2",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={
                "m": 16,
                "ef_construction": 200
            },
            postgresql_ops={"embedding": "vector_l2_ops"}
        ),
        Index("ix_tool_embeddings_model", "model"),
        Index("ix_tool_embeddings_updated_at", "updated_at"),
    )
    
    def __repr__(self) -> str:
        return f"<ToolEmbedding(tool_id='{self.tool_id}', model='{self.model}')>"


class ToolStats(Base, UUIDMixin):
    """Performance and usage statistics for tools.
    
    Stores aggregated metrics over time windows for tool selection
    and quality assessment.
    """
    
    __tablename__ = "tool_stats"
    
    # Foreign key to tool
    tool_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tools.id", ondelete="CASCADE"),
        nullable=False,
        comment="Tool these stats represent"
    )
    
    # Time window
    window_start: Mapped[datetime] = mapped_column(
        nullable=False,
        comment="Start of the statistics window"
    )
    
    window_len: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Length of window in minutes"
    )
    
    # Performance metrics
    p50_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="50th percentile response time in milliseconds"
    )
    
    p95_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="95th percentile response time in milliseconds"
    )
    
    p99_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="99th percentile response time in milliseconds"
    )
    
    # Success metrics
    success_rate: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Success rate as percentage (0.0-100.0)"
    )
    
    call_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of calls in this window"
    )
    
    error_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of errors in this window"
    )
    
    # Error taxonomy
    error_taxonomy: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Breakdown of error types and frequencies"
    )
    
    # Relationships
    tool = relationship("Tool", back_populates="stats")
    
    # Indexes for time-series queries
    __table_args__ = (
        Index("ix_tool_stats_tool_window", "tool_id", "window_start"),
        Index("ix_tool_stats_window_start", "window_start"),
        Index("ix_tool_stats_success_rate", "success_rate"),
        Index("ix_tool_stats_p95_ms", "p95_ms"),
        
        # Compound index for recent performance queries
        Index("ix_tool_stats_tool_recent", "tool_id", "window_start"),
    )
    
    def __repr__(self) -> str:
        return f"<ToolStats(tool_id='{self.tool_id}', window_start='{self.window_start}', success_rate={self.success_rate})>"