"""Policy and RBAC models."""

from typing import Dict, List, Any, Optional

from sqlalchemy import String, JSON, Boolean, Index, UniqueConstraint, ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, UUIDMixin, TimestampMixin


class Policy(Base, UUIDMixin, TimestampMixin):
    """RBAC/ACL policies for tool access control.
    
    Defines which tools/origins are allowed or denied for specific
    organizations and roles.
    """
    
    __tablename__ = "policies"
    
    # Policy identification
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable policy name"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
        comment="Policy description and purpose"
    )
    
    # Scope
    organization: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Organization this policy applies to"
    )
    
    role: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Role this policy applies to (admin, user, etc.)"
    )
    
    # Access control lists
    allow_origins: Mapped[List[str]] = mapped_column(
        ARRAY(String(255)),
        nullable=False,
        default=list,
        comment="List of allowed origin URLs or patterns"
    )
    
    deny_origins: Mapped[List[str]] = mapped_column(
        ARRAY(String(255)),
        nullable=False,
        default=list,
        comment="List of denied origin URLs or patterns"
    )
    
    allow_tools: Mapped[List[str]] = mapped_column(
        ARRAY(String(255)),
        nullable=False,
        default=list,
        comment="List of allowed tool names or patterns"
    )
    
    deny_tools: Mapped[List[str]] = mapped_column(
        ARRAY(String(255)),
        nullable=False,
        default=list,
        comment="List of denied tool names or patterns"
    )
    
    allow_categories: Mapped[List[str]] = mapped_column(
        ARRAY(String(100)),
        nullable=False,
        default=list,
        comment="List of allowed tool categories"
    )
    
    deny_categories: Mapped[List[str]] = mapped_column(
        ARRAY(String(100)),
        nullable=False,
        default=list,
        comment="List of denied tool categories"
    )
    
    # Policy behavior
    default_allow: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Default behavior when no explicit rules match"
    )
    
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether this policy is active"
    )
    
    priority: Mapped[int] = mapped_column(
        nullable=False,
        default=100,
        comment="Policy priority (lower numbers = higher priority)"
    )
    
    # Additional metadata
    meta: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional policy configuration"
    )
    
    # Version for atomic updates
    version: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
        comment="Policy version for atomic updates"
    )
    
    # Audit fields
    created_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="User who created this policy"
    )
    
    updated_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="User who last updated this policy"
    )
    
    # Indexes and constraints
    __table_args__ = (
        # Unique policy per org/role combination
        UniqueConstraint("organization", "role", "name", name="uq_policies_org_role_name"),
        
        # Query optimization indexes
        Index("ix_policies_org", "organization"),
        Index("ix_policies_role", "role"),
        Index("ix_policies_enabled", "enabled"),
        Index("ix_policies_priority", "priority"),
        Index("ix_policies_version", "version"),
        
        # Compound indexes for common queries
        Index("ix_policies_org_role", "organization", "role"),
        Index("ix_policies_org_role_enabled", "organization", "role", "enabled"),
        Index("ix_policies_priority_enabled", "priority", "enabled"),
        
        # Array indexes for ACL lists
        Index("ix_policies_allow_origins", "allow_origins", postgresql_using="gin"),
        Index("ix_policies_deny_origins", "deny_origins", postgresql_using="gin"),
        Index("ix_policies_allow_tools", "allow_tools", postgresql_using="gin"),
        Index("ix_policies_deny_tools", "deny_tools", postgresql_using="gin"),
        Index("ix_policies_allow_categories", "allow_categories", postgresql_using="gin"),
        Index("ix_policies_deny_categories", "deny_categories", postgresql_using="gin"),
    )
    
    def __repr__(self) -> str:
        return f"<Policy(name='{self.name}', org='{self.organization}', role='{self.role}', enabled={self.enabled})>"