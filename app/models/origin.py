"""Origin model for upstream MCP servers."""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

from sqlalchemy import String, Text, Boolean, JSON, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDMixin, TimestampMixin


class OriginStatus(str, Enum):
    """Status of an upstream MCP server origin."""
    ACTIVE = "active"
    INACTIVE = "inactive"  
    ERROR = "error"
    DEPRECATED = "deprecated"


class AuthType(str, Enum):
    """Authentication type for upstream origins."""
    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    MTLS = "mtls"


class Origin(Base, UUIDMixin, TimestampMixin):
    """Represents an upstream MCP server origin.
    
    This table stores information about MCP servers that we crawl
    for tools and capabilities.
    """
    
    __tablename__ = "origins"
    
    # Core identification
    url: Mapped[str] = mapped_column(
        String(2048),
        unique=True,
        nullable=False,
        comment="Base URL of the upstream MCP server"
    )
    
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable name for the origin"
    )
    
    # Authentication and security
    auth_type: Mapped[AuthType] = mapped_column(
        ENUM(AuthType, name="auth_type_enum"),
        nullable=False,
        default=AuthType.NONE,
        comment="Authentication method required"
    )
    
    # TLS configuration
    tls_verify: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether to verify TLS certificates"
    )
    
    tls_pinning: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Optional TLS certificate pinning configuration"
    )
    
    # Status and health
    status: Mapped[OriginStatus] = mapped_column(
        ENUM(OriginStatus, name="origin_status_enum"),
        nullable=False,
        default=OriginStatus.ACTIVE,
        comment="Current status of the origin"
    )
    
    last_crawled_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Last successful crawl timestamp"
    )
    
    last_error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Last error message if status is ERROR"
    )
    
    # Metadata and configuration
    meta: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional metadata and configuration"
    )
    
    # Stats
    tool_count: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        comment="Number of tools from this origin"
    )
    
    avg_response_time_ms: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Average response time in milliseconds"
    )
    
    success_rate: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Success rate as a percentage (0.0-100.0)"
    )
    
    # Relationships
    tools = relationship(
        "Tool", 
        back_populates="origin",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_origins_url", "url"),
        Index("ix_origins_status", "status"),
        Index("ix_origins_auth_type", "auth_type"),
        Index("ix_origins_last_crawled_at", "last_crawled_at"),
        UniqueConstraint("url", name="uq_origins_url"),
    )
    
    def __repr__(self) -> str:
        return f"<Origin(name='{self.name}', url='{self.url}', status='{self.status}')>"