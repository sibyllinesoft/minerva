"""Configuration management with YAML and environment variable support."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "INFO"
    cors_origins: List[str] = ["http://localhost:3000"]


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql+asyncpg://metamcp:metamcp_dev@localhost:5432/metamcp"))
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


class SelectionModeConfig(BaseModel):
    """Configuration for a selection mode (fast/balanced/thorough)."""
    bm25_candidates: int
    dense_candidates: int
    rerank_candidates: int
    expose_tools: int
    enable_planner: bool
    timeout_ms: int

    @validator('timeout_ms')
    def validate_timeout(cls, v):
        if v < 100 or v > 30000:
            raise ValueError('timeout_ms must be between 100 and 30000')
        return v


class ModelConfig(BaseModel):
    """Model configuration."""
    provider: str = "onnx_local"
    model_path: Optional[str] = None
    dimensions: Optional[int] = None
    batch_size: int = 32
    device: str = "auto"


class ModelsConfig(BaseModel):
    """Models configuration."""
    embeddings: ModelConfig
    reranker: ModelConfig
    planner: ModelConfig


class UtilityScoringConfig(BaseModel):
    """Utility scoring configuration."""
    relevance_weight: float = 0.5
    success_weight: float = 0.2
    latency_penalty: float = 0.2
    risk_penalty: float = 0.1

    @validator('relevance_weight', 'success_weight', 'latency_penalty', 'risk_penalty')
    def validate_weights(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Weight must be between 0.0 and 1.0')
        return v


class SearchConfig(BaseModel):
    """Search and ranking configuration."""
    bm25_weight: float = 0.7
    dense_weight: float = 0.3
    mmr_lambda: float = 0.3
    utility_scoring: UtilityScoringConfig = UtilityScoringConfig()
    cache_ttl_seconds: int = 300
    cache_size_mb: int = 64

    @validator('bm25_weight', 'dense_weight')
    def validate_search_weights(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Search weight must be between 0.0 and 1.0')
        return v


class StorageConfig(BaseModel):
    """Storage configuration."""
    data_dir: str = "./data"
    bm25_index_dir: str = "./data/bm25"
    snapshots_dir: str = "./data/snapshots"
    logs_dir: str = "./logs"

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        for dir_path in [self.data_dir, self.bm25_index_dir, 
                        self.snapshots_dir, self.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


class SecretsConfig(BaseModel):
    """Secrets configuration."""
    provider: str = "file_vault"
    vault_path: str = "./data/secrets.vault"


class SecurityConfig(BaseModel):
    """Security configuration."""
    admin_username: str = Field(default_factory=lambda: os.getenv("ADMIN_USERNAME", "admin"))
    admin_password: str = Field(default_factory=lambda: os.getenv("ADMIN_PASSWORD", ""))
    secret_key: str = Field(default_factory=lambda: os.getenv("SECRET_KEY", ""))
    secrets: SecretsConfig = SecretsConfig()


class IngestionConfig(BaseModel):
    """Ingestion configuration."""
    refresh_interval_hours: int = 24
    concurrent_origins: int = 5
    request_timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_backoff_seconds: int = 2
    enable_probes: bool = True
    probe_timeout_seconds: int = 10
    max_probe_concurrency: int = 3


class OtelConfig(BaseModel):
    """OpenTelemetry configuration."""
    service_name: str = "meta-mcp"
    service_version: str = "0.1.0"


class MetricsConfig(BaseModel):
    """Metrics configuration."""
    endpoint: str = "/metrics"
    buckets: List[float] = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]


class ObservabilityConfig(BaseModel):
    """Observability configuration."""
    enable_tracing: bool = True
    enable_metrics: bool = True
    otel: OtelConfig = OtelConfig()
    metrics: MetricsConfig = MetricsConfig()


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    timeout_seconds: int = 60
    expected_recovery_seconds: int = 30


class Settings(BaseSettings):
    """Main application settings."""
    
    # Core configuration sections
    server: ServerConfig = ServerConfig()
    database: DatabaseConfig = DatabaseConfig()
    selection_modes: Dict[str, SelectionModeConfig] = {}
    models: Optional[ModelsConfig] = None
    search: SearchConfig = SearchConfig()
    storage: StorageConfig = StorageConfig()
    security: SecurityConfig = SecurityConfig()
    ingestion: IngestionConfig = IngestionConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    circuit_breakers: Dict[str, CircuitBreakerConfig] = {
        "default": CircuitBreakerConfig()
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False

    @classmethod
    def load_from_yaml(cls, config_path: str) -> "Settings":
        """Load configuration from YAML file with environment variable substitution."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Substitute environment variables
        config_data = cls._substitute_env_vars(config_data)
        
        # Convert selection_modes to proper format
        if 'selection_modes' in config_data:
            selection_modes = {}
            for mode_name, mode_config in config_data['selection_modes'].items():
                selection_modes[mode_name] = SelectionModeConfig(**mode_config)
            config_data['selection_modes'] = selection_modes
        
        return cls(**config_data)

    @classmethod
    def _substitute_env_vars(cls, data: Any) -> Any:
        """Recursively substitute environment variables in configuration data."""
        if isinstance(data, dict):
            return {k: cls._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Handle ${VAR} and ${VAR:-default} patterns
            if data.startswith("${") and data.endswith("}"):
                var_expr = data[2:-1]  # Remove ${ and }
                if ":-" in var_expr:
                    var_name, default_value = var_expr.split(":-", 1)
                    return os.getenv(var_name, default_value)
                else:
                    return os.getenv(var_expr, data)
            return data
        else:
            return data

    def get_selection_mode(self, mode: str) -> SelectionModeConfig:
        """Get selection mode configuration."""
        if mode not in self.selection_modes:
            raise ValueError(f"Unknown selection mode: {mode}")
        return self.selection_modes[mode]

    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return any issues."""
        issues = {}
        
        # Validate selection modes
        if not self.selection_modes:
            issues["selection_modes"] = "No selection modes configured"
        
        # Validate model paths
        if self.models.embeddings.provider == "onnx_local":
            if not self.models.embeddings.model_path:
                issues["embeddings_model"] = "ONNX embeddings provider requires model_path"
        
        if self.models.reranker.provider == "onnx_local":
            if not self.models.reranker.model_path:
                issues["reranker_model"] = "ONNX reranker provider requires model_path"
        
        return issues


# Global settings instance
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global settings
    if settings is None:
        config_path = os.getenv("CONFIG_PATH", "config/development.yaml")
        settings = Settings.load_from_yaml(config_path)
    return settings


def init_settings(config_path: str) -> Settings:
    """Initialize settings from configuration file."""
    global settings
    settings = Settings.load_from_yaml(config_path)
    return settings