"""Secret storage models for encrypted credentials."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import String, Text, Boolean, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, UUIDMixin, TimestampMixin


class SecretType(str, Enum):
    """Type of secret stored."""
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2_TOKEN = "oauth2_token"
    TLS_CERT = "tls_cert"
    PRIVATE_KEY = "private_key"
    GENERIC = "generic"


class Secret(Base, UUIDMixin, TimestampMixin):
    """Encrypted storage for sensitive credentials and secrets.
    
    This table stores encrypted secrets used for authenticating
    with upstream MCP servers and other external services.
    """
    
    __tablename__ = "secrets"
    
    # Secret identification
    name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        comment="Unique name/key for the secret"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Human-readable description"
    )
    
    # Secret type and metadata
    secret_type: Mapped[SecretType] = mapped_column(
        ENUM(SecretType, name="secret_type_enum"),
        nullable=False,
        comment="Type of secret stored"
    )
    
    # Encrypted data (AES-256-GCM)
    encrypted_value: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Encrypted secret value (base64 encoded)"
    )
    
    # Encryption metadata
    encryption_key_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="ID of the encryption key used"
    )
    
    encryption_algorithm: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="AES-256-GCM",
        comment="Encryption algorithm used"
    )
    
    initialization_vector: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Initialization vector (base64 encoded)"
    )
    
    authentication_tag: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Authentication tag for integrity (base64 encoded)"
    )
    
    # Access control
    owner: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Owner of this secret"
    )
    
    # Lifecycle management
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="When this secret expires"
    )
    
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Last time this secret was accessed"
    )
    
    rotation_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether this secret needs rotation"
    )
    
    # Audit fields
    created_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="User who created this secret"
    )
    
    updated_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="User who last updated this secret"
    )
    
    # Security metadata
    access_count: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        comment="Number of times this secret has been accessed"
    )
    
    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("name", name="uq_secrets_name"),
        
        Index("ix_secrets_name", "name"),
        Index("ix_secrets_type", "secret_type"),
        Index("ix_secrets_owner", "owner"),
        Index("ix_secrets_expires_at", "expires_at"),
        Index("ix_secrets_rotation_required", "rotation_required"),
        Index("ix_secrets_encryption_key_id", "encryption_key_id"),
        
        # Audit and monitoring indexes
        Index("ix_secrets_created_by", "created_by"),
        Index("ix_secrets_last_used_at", "last_used_at"),
        Index("ix_secrets_access_count", "access_count"),
    )
    
    def __repr__(self) -> str:
        return f"<Secret(name='{self.name}', type='{self.secret_type}', expires_at='{self.expires_at}')>"