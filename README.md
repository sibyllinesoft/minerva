# Minerva

An intelligent MCP (Model Context Protocol) orchestration system that discovers, aggregates, and intelligently orchestrates multiple upstream MCP servers.

## Overview

Minerva is a sophisticated intelligent orchestration system that provides:

- **Discovery & Aggregation**: Automatically discover and normalize tools from multiple upstream MCP servers
- **Intelligent Selection**: Hybrid search using BM25 (Tantivy) + dense embeddings (ONNX) with cross-encoder reranking
- **Optional Planning**: LLM-powered DAG generation for complex multi-tool workflows
- **Secure Execution**: Proxy tool calls with RBAC/ACL controls and observability
- **Admin Interface**: Web UI for managing origins, policies, and monitoring
- **Production Ready**: Full observability with OpenTelemetry, circuit breakers, and degradation modes

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 14+ with pgvector extension

### Development Setup

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd minerva
   pip install -e ".[dev]"
   ```

2. **Start PostgreSQL**:
   ```bash
   docker-compose up postgres -d
   ```

3. **Run the server**:
   ```bash
   python -m app.main
   # Or with uvicorn directly:
   uvicorn app.main:app --reload
   ```

4. **Access the application**:
   - API: http://localhost:8000
   - Health: http://localhost:8000/healthz  
   - Docs: http://localhost:8000/docs
   - Admin UI: http://localhost:8000/admin (coming in Phase 9)

### Full Stack with Docker

```bash
# Start everything (PostgreSQL + Minerva + Redis)
docker-compose --profile full up -d

# Check health
curl http://localhost:8000/healthz
```

## Architecture

### Core Components

- **Ingestion Engine**: Crawls upstream MCP servers, validates schemas, stores capability cards
- **Search System**: Hybrid BM25 + vector search with cross-encoder reranking
- **Model Runtime**: Pluggable ONNX/API models for embeddings, reranking, and planning
- **Proxy Layer**: Secure tool execution with observability and circuit breakers
- **Policy Engine**: RBAC/ACL with audit trails
- **Admin Interface**: Management UI and debugging tools

### Selection Modes

- **Fast** (200ms): 24 BM25 + 16 dense candidates → 16 rerank → 5 tools
- **Balanced** (500ms): 48 BM25 + 32 dense candidates → 32 rerank → 8 tools  
- **Thorough** (2s): 96 BM25 + 64 dense candidates → 64 rerank → 10 tools + planning

## Configuration

Configuration is managed through YAML files with environment variable substitution:

```yaml
# config/development.yaml
server:
  host: "0.0.0.0"
  port: 8000

database:
  url: "postgresql+asyncpg://minerva:minerva_dev@localhost:5432/minerva"

selection_modes:
  fast:
    bm25_candidates: 24
    dense_candidates: 16
    expose_tools: 5
```

Environment variables can be substituted using `${VAR_NAME}` or `${VAR_NAME:-default}` syntax.

## Development

### Project Structure

```
├── app/                    # Main application code
│   ├── core/              # Core configuration and utilities
│   ├── models/            # Database models and schemas  
│   ├── api/               # FastAPI routes and endpoints
│   ├── services/          # Business logic services
│   └── utils/             # Utility functions
├── admin_ui/              # Admin web interface
├── migrations/            # Database migrations (Alembic)
├── config/                # Configuration files
├── scripts/               # Setup and utility scripts
├── tests/                 # Test suites
└── docs/                  # Documentation
```

### Development Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black app/ tests/
ruff check app/ tests/

# Type checking
mypy app/

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload
```

## Implementation Status

This project is implemented in phases:

- ✅ **Phase 0**: Repository bootstrap & foundations
- 🚧 **Phase 1**: Schema, migrations, and indexing (In Progress)
- ⏳ **Phase 2**: Model runtime (ONNX embeddings, reranker, planner)
- ⏳ **Phase 3**: Ingestion (discovery, validation, snapshots)
- ⏳ **Phase 4**: Selection (hybrid search, rerank, MMR, policy)
- ⏳ **Phase 5**: Planner (optional DAG generation)
- ⏳ **Phase 6**: Proxy/executor (safe calls, observability)
- ⏳ **Phase 7**: RBAC/ACL policy engine
- ⏳ **Phase 8**: Secrets management
- ⏳ **Phase 9**: Admin UI
- ⏳ **Phase 10**: Observability & reliability
- ⏳ **Phase 11**: Evaluation scaffold
- ⏳ **Phase 12**: Packaging & distribution
- ⏳ **Phase 13**: Cloud-ready upgrades
- ⏳ **Phase 14**: Milestones & acceptance gates
- ⏳ **Phase 15**: Runbooks & rollbacks

## License

MIT License - see LICENSE file for details.