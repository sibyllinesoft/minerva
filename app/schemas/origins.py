"""Pydantic schemas for origin management."""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, validator, root_validator

from ..models.origin import OriginStatus, AuthType


class OriginBase(BaseModel):
    """Base origin schema with common fields."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name for the origin"
    )
    
    url: HttpUrl = Field(
        ...,
        description="Base URL of the upstream MCP server"
    )
    
    auth_type: AuthType = Field(
        default=AuthType.NONE,
        description="Authentication method required"
    )
    
    tls_verify: bool = Field(
        default=True,
        description="Whether to verify TLS certificates"
    )
    
    tls_pinning: Optional[str] = Field(
        default=None,
        description="Optional TLS certificate pinning configuration"
    )
    
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata and configuration"
    )
    
    @validator('url')
    def validate_url(cls, v):
        """Validate URL format and requirements."""
        url_str = str(v)
        if not any(url_str.startswith(scheme) for scheme in ['http://', 'https://']):
            raise ValueError('URL must use http or https scheme')
        return v
    
    @validator('meta')
    def validate_meta(cls, v):
        """Validate metadata structure."""
        if not isinstance(v, dict):
            raise ValueError('Meta must be a dictionary')
        
        # Validate specific meta fields if present
        if 'refresh_interval' in v:
            interval = v['refresh_interval']
            if not isinstance(interval, int) or interval < 3600:  # Min 1 hour
                raise ValueError('refresh_interval must be at least 3600 seconds (1 hour)')
        
        if 'tags' in v:
            tags = v['tags']
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError('tags must be a list of strings')
        
        return v


class OriginCreate(OriginBase):
    """Schema for creating a new origin."""
    
    status: OriginStatus = Field(
        default=OriginStatus.ACTIVE,
        description="Initial status of the origin"
    )


class OriginUpdate(BaseModel):
    """Schema for updating an existing origin."""
    
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Human-readable name for the origin"
    )
    
    url: Optional[HttpUrl] = Field(
        None,
        description="Base URL of the upstream MCP server"
    )
    
    auth_type: Optional[AuthType] = Field(
        None,
        description="Authentication method required"
    )
    
    status: Optional[OriginStatus] = Field(
        None,
        description="Current status of the origin"
    )
    
    tls_verify: Optional[bool] = Field(
        None,
        description="Whether to verify TLS certificates"
    )
    
    tls_pinning: Optional[str] = Field(
        None,
        description="Optional TLS certificate pinning configuration"
    )
    
    meta: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata and configuration"
    )
    
    @validator('url')
    def validate_url(cls, v):
        """Validate URL format and requirements."""
        if v is not None:
            url_str = str(v)
            if not any(url_str.startswith(scheme) for scheme in ['http://', 'https://']):
                raise ValueError('URL must use http or https scheme')
        return v
    
    @validator('meta')
    def validate_meta(cls, v):
        """Validate metadata structure."""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError('Meta must be a dictionary')
            
            # Validate specific meta fields if present
            if 'refresh_interval' in v:
                interval = v['refresh_interval']
                if not isinstance(interval, int) or interval < 3600:  # Min 1 hour
                    raise ValueError('refresh_interval must be at least 3600 seconds (1 hour)')
            
            if 'tags' in v:
                tags = v['tags']
                if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                    raise ValueError('tags must be a list of strings')
        
        return v


class OriginResponse(OriginBase):
    """Schema for origin responses."""
    
    id: UUID = Field(
        ...,
        description="Unique identifier for the origin"
    )
    
    status: OriginStatus = Field(
        ...,
        description="Current status of the origin"
    )
    
    last_crawled_at: Optional[datetime] = Field(
        None,
        description="Last successful crawl timestamp"
    )
    
    last_error: Optional[str] = Field(
        None,
        description="Last error message if status is ERROR"
    )
    
    tool_count: int = Field(
        default=0,
        description="Number of tools from this origin"
    )
    
    avg_response_time_ms: Optional[float] = Field(
        None,
        description="Average response time in milliseconds"
    )
    
    success_rate: Optional[float] = Field(
        None,
        description="Success rate as a percentage (0.0-100.0)"
    )
    
    created_at: datetime = Field(
        ...,
        description="Timestamp when origin was created"
    )
    
    updated_at: datetime = Field(
        ...,
        description="Timestamp when origin was last updated"
    )
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True


class OriginListRequest(BaseModel):
    """Schema for listing origins with filtering."""
    
    status: Optional[OriginStatus] = Field(
        None,
        description="Filter by origin status"
    )
    
    auth_type: Optional[AuthType] = Field(
        None,
        description="Filter by authentication type"
    )
    
    search: Optional[str] = Field(
        None,
        max_length=255,
        description="Search in name, url, or tags"
    )
    
    limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Number of results to return"
    )
    
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of results to skip"
    )
    
    sort_by: str = Field(
        default="created_at",
        description="Field to sort by"
    )
    
    sort_order: str = Field(
        default="desc",
        pattern="^(asc|desc)$",
        description="Sort order (asc or desc)"
    )
    
    @validator('sort_by')
    def validate_sort_by(cls, v):
        """Validate sort_by field."""
        allowed_fields = {
            'created_at', 'updated_at', 'name', 'url', 'status', 
            'last_crawled_at', 'tool_count', 'avg_response_time_ms', 'success_rate'
        }
        if v not in allowed_fields:
            raise ValueError(f'sort_by must be one of: {", ".join(sorted(allowed_fields))}')
        return v


class OriginListResponse(BaseModel):
    """Schema for origin list response."""
    
    origins: List[OriginResponse] = Field(
        ...,
        description="List of origins"
    )
    
    total: int = Field(
        ...,
        description="Total number of origins matching filters"
    )
    
    limit: int = Field(
        ...,
        description="Number of results requested"
    )
    
    offset: int = Field(
        ...,
        description="Number of results skipped"
    )


class OriginHealthCheck(BaseModel):
    """Schema for origin health check response."""
    
    origin_id: UUID = Field(
        ...,
        description="Origin identifier"
    )
    
    url: str = Field(
        ...,
        description="URL checked"
    )
    
    status: str = Field(
        ...,
        description="Health check status (healthy, unhealthy, error)"
    )
    
    response_time_ms: Optional[float] = Field(
        None,
        description="Response time in milliseconds"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error message if health check failed"
    )
    
    timestamp: datetime = Field(
        ...,
        description="When the health check was performed"
    )