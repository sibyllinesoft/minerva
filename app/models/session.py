"""Session and trace models for observability."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

from sqlalchemy import String, Text, Integer, Float, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, ENUM
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDMixin, TimestampMixin


class SessionStatus(str, Enum):
    """Status of a session."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


class TraceSpanType(str, Enum):
    """Type of trace span."""
    SELECTION = "selection"
    PLANNING = "planning"
    PROXY_CALL = "proxy_call"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    POLICY_CHECK = "policy_check"


class Session(Base, UUIDMixin, TimestampMixin):
    """Represents a client session with the Meta MCP server.
    
    Sessions track user interactions and tool selections over time,
    enabling context-aware tool selection and usage analytics.
    """
    
    __tablename__ = "sessions"
    
    # Client identification
    client_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Client identifier if available"
    )
    
    user_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="User identifier if available"
    )
    
    # Session metadata
    status: Mapped[SessionStatus] = mapped_column(
        ENUM(SessionStatus, name="session_status_enum"),
        nullable=False,
        default=SessionStatus.ACTIVE,
        comment="Current session status"
    )
    
    selection_mode: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Selection mode used (fast/balanced/thorough)"
    )
    
    # Context and state
    context: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Session context and conversation state"
    )
    
    # Statistics
    tool_calls: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of tool calls in this session"
    )
    
    successful_calls: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of successful tool calls"
    )
    
    total_latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Total latency across all calls"
    )
    
    # Lifecycle
    started_at: Mapped[datetime] = mapped_column(
        nullable=False,
        server_default="now()",
        comment="Session start time"
    )
    
    last_activity_at: Mapped[datetime] = mapped_column(
        nullable=False,
        server_default="now()",
        comment="Last activity timestamp"
    )
    
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Session end time"
    )
    
    # Relationships
    traces = relationship(
        "Trace",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Trace.created_at"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_sessions_client_id", "client_id"),
        Index("ix_sessions_user_id", "user_id"),
        Index("ix_sessions_status", "status"),
        Index("ix_sessions_started_at", "started_at"),
        Index("ix_sessions_last_activity", "last_activity_at"),
        Index("ix_sessions_selection_mode", "selection_mode"),
        
        # Compound indexes for common queries
        Index("ix_sessions_client_status", "client_id", "status"),
        Index("ix_sessions_user_status", "user_id", "status"),
        Index("ix_sessions_active_recent", "status", "last_activity_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Session(client_id='{self.client_id}', status='{self.status}', tool_calls={self.tool_calls})>"


class Trace(Base, UUIDMixin, TimestampMixin):
    """Individual trace spans within a session.
    
    Provides detailed observability into tool selection, planning,
    and execution steps.
    """
    
    __tablename__ = "traces"
    
    # Session relationship
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        comment="Session this trace belongs to"
    )
    
    # Trace identification
    span_type: Mapped[TraceSpanType] = mapped_column(
        ENUM(TraceSpanType, name="trace_span_type_enum"),
        nullable=False,
        comment="Type of operation being traced"
    )
    
    operation_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Name of the operation"
    )
    
    parent_span_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("traces.id"),
        nullable=True,
        comment="Parent span for nested operations"
    )
    
    # Timing
    started_at: Mapped[datetime] = mapped_column(
        nullable=False,
        server_default="now()",
        comment="Operation start time"
    )
    
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Operation end time"
    )
    
    duration_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Operation duration in milliseconds"
    )
    
    # Status and results
    success: Mapped[Optional[bool]] = mapped_column(
        nullable=True,
        comment="Whether the operation succeeded"
    )
    
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if operation failed"
    )
    
    # Operation details
    input_data: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Input parameters for the operation"
    )
    
    output_data: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Output results from the operation"
    )
    
    trace_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional metadata and tags"
    )
    
    # Tool-specific fields
    tool_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tools.id"),
        nullable=True,
        comment="Tool ID if this trace involves a specific tool"
    )
    
    origin_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("origins.id"),
        nullable=True,
        comment="Origin ID if this trace involves a specific origin"
    )
    
    # Relationships
    session = relationship("Session", back_populates="traces")
    parent_span = relationship("Trace", remote_side="Trace.id")
    child_spans = relationship("Trace", back_populates="parent_span")
    tool = relationship("Tool")
    origin = relationship("Origin")
    
    # Indexes
    __table_args__ = (
        Index("ix_traces_session_id", "session_id"),
        Index("ix_traces_span_type", "span_type"),
        Index("ix_traces_parent_span_id", "parent_span_id"),
        Index("ix_traces_started_at", "started_at"),
        Index("ix_traces_duration_ms", "duration_ms"),
        Index("ix_traces_success", "success"),
        Index("ix_traces_tool_id", "tool_id"),
        Index("ix_traces_origin_id", "origin_id"),
        
        # Compound indexes for observability queries
        Index("ix_traces_session_started", "session_id", "started_at"),
        Index("ix_traces_session_type", "session_id", "span_type"),
        Index("ix_traces_type_started", "span_type", "started_at"),
        Index("ix_traces_success_duration", "success", "duration_ms"),
        
        # Performance analysis indexes
        Index("ix_traces_tool_performance", "tool_id", "success", "duration_ms"),
        Index("ix_traces_origin_performance", "origin_id", "success", "duration_ms"),
    )
    
    def __repr__(self) -> str:
        return f"<Trace(operation='{self.operation_name}', span_type='{self.span_type}', success={self.success})>"