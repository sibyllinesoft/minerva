"""Main FastAPI application for Meta MCP server."""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvloop
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.core.config import get_settings, Settings
from app.core.database import init_database, close_database, check_database_health
from app.services.models.manager import get_model_manager, create_model_services_context
from app.api.admin.origins import router as origins_router
from app.api.admin.crawler import router as crawler_router
from app.api.v1.selection import router as selection_router
from app.api.v1.planner import router as planner_router
from app.api.v1.proxy import router as proxy_router
from app.services.proxy_service import proxy_service


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Meta MCP server...")
    
    try:
        # Initialize settings
        settings = get_settings()
        
        # Validate configuration
        config_issues = settings.validate_config()
        if config_issues:
            logger.warning("Configuration issues detected", issues=config_issues)
        
        # Initialize model services
        logger.info("Initializing model services...")
        model_manager = get_model_manager()
        await model_manager.initialize()
        
        # Initialize proxy service
        logger.info("Initializing proxy service...")
        await proxy_service.startup()
        
        # Start model warmup in background (non-blocking)
        warmup_task = asyncio.create_task(model_manager.warmup(timeout_seconds=120.0))
        
        # Initialize database connection
        logger.info("Initializing database connection...")
        await init_database()
        
        # Store manager in app state
        app.state.model_manager = model_manager
        app.state.warmup_task = warmup_task
        app.state.proxy_service = proxy_service
        
        # TODO: Initialize search indices
        
        logger.info("Meta MCP server started successfully", 
                   version=app.version,
                   host=settings.server.host,
                   port=settings.server.port)
        
    except Exception as e:
        logger.error("Failed to start Meta MCP server", error=str(e))
        sys.exit(1)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Meta MCP server...")
    
    # Cancel warmup if still running
    if hasattr(app.state, 'warmup_task') and not app.state.warmup_task.done():
        app.state.warmup_task.cancel()
        try:
            await app.state.warmup_task
        except asyncio.CancelledError:
            pass
    
    # Shutdown proxy service
    if hasattr(app.state, 'proxy_service'):
        await app.state.proxy_service.shutdown()
    
    # Shutdown model services
    if hasattr(app.state, 'model_manager'):
        await app.state.model_manager.shutdown()
    
    # Close database connection
    await close_database()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Use uvloop for better async performance on Unix
    if sys.platform != "win32":
        uvloop.install()
    
    settings = get_settings()
    
    app = FastAPI(
        title="Meta MCP",
        description="A local-first Meta MCP server for aggregating multiple MCP servers",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(origins_router, tags=["admin"])
    app.include_router(crawler_router, tags=["admin"])
    app.include_router(selection_router, tags=["selection"])
    app.include_router(planner_router, tags=["planner"])
    app.include_router(proxy_router, prefix="/v1", tags=["proxy"])
    
    # Health endpoints
    @app.get("/healthz")
    async def health_check() -> Dict[str, Any]:
        """Kubernetes-style liveness check."""
        return {
            "status": "healthy",
            "service": "meta-mcp",
            "version": "0.1.0"
        }
    
    @app.get("/readyz") 
    async def readiness_check() -> Dict[str, Any]:
        """Kubernetes-style readiness check."""
        settings = get_settings()
        
        # Check model service status
        model_status = "initializing"
        if hasattr(app.state, 'model_manager'):
            health = await app.state.model_manager.health_check()
            model_status = "healthy" if health["healthy"] else "unhealthy"
        
        # Check database connectivity
        db_status = "healthy" if await check_database_health() else "unhealthy"
        
        # TODO: Check search indices status
        
        all_healthy = model_status == "healthy" and db_status == "healthy"
        
        return {
            "status": "ready" if all_healthy else "not_ready",
            "service": "meta-mcp",
            "version": "0.1.0",
            "checks": {
                "database": db_status,
                "indices": "healthy",   # TODO: Real check
                "models": model_status
            }
        }
    
    @app.get("/status")
    async def status() -> Dict[str, Any]:
        """Detailed status information."""
        settings = get_settings()
        
        # Get model service status
        model_runtime = {"initialized": False, "services": {}}
        if hasattr(app.state, 'model_manager'):
            warmup_status = app.state.model_manager.get_warmup_status()
            model_runtime.update(warmup_status)
        
        return {
            "service": "meta-mcp",
            "version": "0.1.0",
            "git_sha": os.getenv("GIT_SHA", "unknown"),
            "config": {
                "selection_modes": list(settings.selection_modes.keys()),
                "models": {
                    "embeddings": settings.models.embeddings.provider,
                    "reranker": settings.models.reranker.provider,
                    "planner": settings.models.planner.provider
                }
            },
            "runtime": {
                "models": model_runtime
            },
            # TODO: Add more runtime statistics
            "stats": {
                "origins": 0,
                "tools": 0,
                "catalog_version": "unknown"
            }
        }
    
    # API v1 routes placeholder
    @app.get("/v1/info")
    async def api_info() -> Dict[str, Any]:
        """API information endpoint."""
        return {
            "api_version": "v1",
            "service": "meta-mcp", 
            "description": "Meta MCP aggregation and orchestration API",
            "endpoints": {
                "health": "/healthz",
                "readiness": "/readyz", 
                "status": "/status",
                "models": "/v1/models/*",
                "selection": "/v1/selection/*",
                "admin": "/v1/admin/*",
                "debug": "/v1/debug/*",
                "mcp": "/mcp"
            }
        }

    # Model service endpoints
    @app.get("/v1/models/status")
    async def models_status() -> Dict[str, Any]:
        """Get detailed model service status."""
        if not hasattr(app.state, 'model_manager'):
            raise HTTPException(status_code=503, detail="Model services not initialized")
        
        health = await app.state.model_manager.health_check()
        warmup = app.state.model_manager.get_warmup_status()
        
        return {
            "health": health,
            "warmup": warmup,
            "ready": app.state.model_manager.is_ready
        }

    @app.post("/v1/models/warmup")
    async def trigger_warmup() -> Dict[str, Any]:
        """Trigger model warmup."""
        if not hasattr(app.state, 'model_manager'):
            raise HTTPException(status_code=503, detail="Model services not initialized")
        
        result = await app.state.model_manager.warmup(timeout_seconds=120.0)
        return result

    @app.post("/v1/models/reload")
    async def reload_models() -> Dict[str, Any]:
        """Reload model configuration and reinitialize services."""
        if not hasattr(app.state, 'model_manager'):
            raise HTTPException(status_code=503, detail="Model services not initialized")
        
        result = await app.state.model_manager.reload_config()
        return result
    
    # MCP protocol endpoint placeholder  
    @app.post("/mcp")
    async def mcp_endpoint():
        """Main MCP protocol endpoint (placeholder)."""
        raise HTTPException(status_code=501, detail="MCP endpoint not implemented yet")
    
    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content={"error": "Bad Request", "detail": str(exc)}
        )
    
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request, exc):
        return JSONResponse(
            status_code=404,
            content={"error": "Not Found", "detail": str(exc)}
        )
    
    # TODO: Add more API routes
    # TODO: Add admin UI routes
    # TODO: Add observability middleware
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=settings.server.workers,
        log_level=settings.server.log_level.lower(),
        access_log=True
    )