"""Tests for the Origins API endpoints."""

import pytest
from uuid import uuid4
from httpx import AsyncClient

from app.models.origin import OriginStatus, AuthType


class TestOriginsAPI:
    """Test suite for Origins API endpoints."""

    async def test_create_origin_success(self, client: AsyncClient):
        """Test successful origin creation."""
        origin_data = {
            "name": "Test Origin",
            "url": "https://example.com/mcp",
            "auth_type": AuthType.NONE,
            "tls_verify": True,
            "meta": {
                "description": "Test MCP server",
                "tags": ["test", "example"],
                "refresh_interval": 3600
            }
        }
        
        response = await client.post("/v1/admin/origins", json=origin_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Origin"
        assert data["url"] == "https://example.com/mcp"
        assert data["auth_type"] == AuthType.NONE
        assert data["status"] == OriginStatus.ACTIVE
        assert data["tool_count"] == 0
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    async def test_create_origin_duplicate_url(self, client: AsyncClient):
        """Test creating origin with duplicate URL fails."""
        origin_data = {
            "name": "Test Origin",
            "url": "https://example.com/mcp",
            "auth_type": AuthType.NONE
        }
        
        # Create first origin
        response1 = await client.post("/v1/admin/origins", json=origin_data)
        assert response1.status_code == 201
        
        # Try to create duplicate
        origin_data["name"] = "Duplicate Origin"
        response2 = await client.post("/v1/admin/origins", json=origin_data)
        assert response2.status_code == 409
        assert "already exists" in response2.json()["detail"]

    async def test_create_origin_invalid_url(self, client: AsyncClient):
        """Test creating origin with invalid URL fails."""
        origin_data = {
            "name": "Invalid URL Origin",
            "url": "not-a-url",
            "auth_type": AuthType.NONE
        }
        
        response = await client.post("/v1/admin/origins", json=origin_data)
        assert response.status_code == 422

    async def test_create_origin_invalid_refresh_interval(self, client: AsyncClient):
        """Test creating origin with invalid refresh interval fails."""
        origin_data = {
            "name": "Invalid Refresh Origin",
            "url": "https://example.com/mcp2",
            "auth_type": AuthType.NONE,
            "meta": {
                "refresh_interval": 1800  # Less than 1 hour
            }
        }
        
        response = await client.post("/v1/admin/origins", json=origin_data)
        assert response.status_code == 422

    async def test_get_origin_success(self, client: AsyncClient, sample_origin_id: str):
        """Test successful origin retrieval."""
        response = await client.get(f"/v1/admin/origins/{sample_origin_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_origin_id
        assert data["name"] == "Sample Origin"

    async def test_get_origin_not_found(self, client: AsyncClient):
        """Test getting non-existent origin returns 404."""
        fake_id = str(uuid4())
        response = await client.get(f"/v1/admin/origins/{fake_id}")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    async def test_list_origins_empty(self, client: AsyncClient):
        """Test listing origins when none exist."""
        response = await client.get("/v1/admin/origins")
        
        assert response.status_code == 200
        data = response.json()
        assert data["origins"] == []
        assert data["total"] == 0
        assert data["limit"] == 50
        assert data["offset"] == 0

    async def test_list_origins_with_results(self, client: AsyncClient, sample_origin_id: str):
        """Test listing origins with results."""
        response = await client.get("/v1/admin/origins")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["origins"]) == 1
        assert data["total"] == 1
        assert data["origins"][0]["id"] == sample_origin_id

    async def test_list_origins_filtering(self, client: AsyncClient, sample_origin_id: str):
        """Test listing origins with filters."""
        # Filter by status
        response = await client.get("/v1/admin/origins?status=active")
        assert response.status_code == 200
        data = response.json()
        assert len(data["origins"]) == 1
        
        # Filter by non-existent status
        response = await client.get("/v1/admin/origins?status=error")
        assert response.status_code == 200
        data = response.json()
        assert len(data["origins"]) == 0

    async def test_list_origins_pagination(self, client: AsyncClient, sample_origin_id: str):
        """Test listing origins with pagination."""
        response = await client.get("/v1/admin/origins?limit=10&offset=0")
        
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert len(data["origins"]) <= 10

    async def test_list_origins_sorting(self, client: AsyncClient, sample_origin_id: str):
        """Test listing origins with sorting."""
        response = await client.get("/v1/admin/origins?sort_by=name&sort_order=asc")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["origins"]) >= 1

    async def test_update_origin_success(self, client: AsyncClient, sample_origin_id: str):
        """Test successful origin update."""
        update_data = {
            "name": "Updated Origin Name",
            "meta": {
                "description": "Updated description",
                "tags": ["updated", "test"]
            }
        }
        
        response = await client.put(f"/v1/admin/origins/{sample_origin_id}", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Origin Name"
        assert data["meta"]["description"] == "Updated description"

    async def test_update_origin_not_found(self, client: AsyncClient):
        """Test updating non-existent origin returns 404."""
        fake_id = str(uuid4())
        update_data = {"name": "New Name"}
        
        response = await client.put(f"/v1/admin/origins/{fake_id}", json=update_data)
        assert response.status_code == 404

    async def test_update_origin_duplicate_url(self, client: AsyncClient):
        """Test updating origin to duplicate URL fails."""
        # Create two origins
        origin1_data = {
            "name": "Origin 1",
            "url": "https://origin1.com/mcp",
            "auth_type": AuthType.NONE
        }
        origin2_data = {
            "name": "Origin 2", 
            "url": "https://origin2.com/mcp",
            "auth_type": AuthType.NONE
        }
        
        response1 = await client.post("/v1/admin/origins", json=origin1_data)
        response2 = await client.post("/v1/admin/origins", json=origin2_data)
        
        assert response1.status_code == 201
        assert response2.status_code == 201
        
        origin1_id = response1.json()["id"]
        
        # Try to update origin1 to use origin2's URL
        update_data = {"url": "https://origin2.com/mcp"}
        response = await client.put(f"/v1/admin/origins/{origin1_id}", json=update_data)
        
        assert response.status_code == 409

    async def test_delete_origin_success(self, client: AsyncClient):
        """Test successful origin deletion (soft delete)."""
        # Create origin to delete
        origin_data = {
            "name": "To Delete Origin",
            "url": "https://delete.com/mcp",
            "auth_type": AuthType.NONE
        }
        
        create_response = await client.post("/v1/admin/origins", json=origin_data)
        assert create_response.status_code == 201
        origin_id = create_response.json()["id"]
        
        # Delete the origin
        response = await client.delete(f"/v1/admin/origins/{origin_id}")
        assert response.status_code == 204
        
        # Verify it's soft deleted (status changed to deprecated)
        get_response = await client.get(f"/v1/admin/origins/{origin_id}")
        assert get_response.status_code == 200
        assert get_response.json()["status"] == OriginStatus.DEPRECATED

    async def test_delete_origin_not_found(self, client: AsyncClient):
        """Test deleting non-existent origin returns 404."""
        fake_id = str(uuid4())
        response = await client.delete(f"/v1/admin/origins/{fake_id}")
        
        assert response.status_code == 404

    async def test_health_check_origin_success(self, client: AsyncClient, sample_origin_id: str):
        """Test origin health check."""
        response = await client.post(f"/v1/admin/origins/{sample_origin_id}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["origin_id"] == sample_origin_id
        assert "status" in data
        assert "timestamp" in data
        assert data["url"] == "https://sample.com/mcp"

    async def test_health_check_origin_not_found(self, client: AsyncClient):
        """Test health check on non-existent origin returns 404."""
        fake_id = str(uuid4())
        response = await client.post(f"/v1/admin/origins/{fake_id}/health")
        
        assert response.status_code == 404

    async def test_get_origin_stats(self, client: AsyncClient, sample_origin_id: str):
        """Test getting origin statistics."""
        response = await client.get(f"/v1/admin/origins/{sample_origin_id}/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["origin_id"] == sample_origin_id
        assert "name" in data
        assert "url" in data
        assert "status" in data
        assert "tool_count" in data
        assert "created_at" in data