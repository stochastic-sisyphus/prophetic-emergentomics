"""
Configuration management for Prophetic Emergentomics.

Supports environment variables, .env files, and programmatic configuration.
Focused on ML/DL emergence detection with GDELT as behavioral data source.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GDELTSettings(BaseSettings):
    """GDELT API configuration - behavioral data source."""

    model_config = SettingsConfigDict(env_prefix="GDELT_")

    base_url: str = "https://api.gdeltproject.org/api/v2"
    doc_api_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    geo_api_url: str = "https://api.gdeltproject.org/api/v2/geo/geo"
    tv_api_url: str = "https://api.gdeltproject.org/api/v2/tv/tv"

    # Rate limiting
    requests_per_minute: int = 60
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Data collection
    update_interval_minutes: int = 15
    lookback_hours: int = 24
    max_records_per_query: int = 250

    # Economic focus filters
    economic_themes: list[str] = Field(
        default=[
            "ECON_",
            "TAX_",
            "TRADE_",
            "INFLATION",
            "UNEMPLOYMENT",
            "GDP",
            "INTEREST_RATE",
            "CENTRAL_BANK",
            "STOCK_MARKET",
            "CURRENCY",
            "RECESSION",
            "ECONOMIC_GROWTH",
            "LABOR_MARKET",
            "SUPPLY_CHAIN",
            "MANUFACTURING",
            "TECH_",
            "AI_",
            "AUTOMATION",
        ]
    )


class MLSettings(BaseSettings):
    """ML/DL model configuration."""

    model_config = SettingsConfigDict(env_prefix="ML_")

    # Anomaly detection
    anomaly_contamination: float = Field(default=0.1, description="Expected anomaly rate")
    anomaly_method: str = "isolation_forest"

    # Clustering
    clustering_method: str = "hdbscan"
    min_cluster_size: int = 5
    min_samples: int = 3

    # Dimensionality reduction
    reduction_method: str = "umap"
    n_components: int = 2
    n_neighbors: int = 15

    # GNN settings
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10

    # Thresholds
    emergence_threshold: float = 0.7
    phase_transition_threshold: float = 0.85
    novelty_threshold: float = 0.6


class DataSettings(BaseSettings):
    """Data source configuration."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("data/models")
    cache_dir: Path = Path("data/cache")

    # Data retention
    raw_retention_days: int = 90
    processed_retention_days: int = 365

    # Processing
    batch_size: int = 1000
    parallel_workers: int = 4

    # Time series
    default_lookback_days: int = 365
    resampling_frequency: str = "1D"


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application metadata
    app_name: str = "Prophetic Emergentomics"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Component settings
    gdelt: GDELTSettings = Field(default_factory=GDELTSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    data: DataSettings = Field(default_factory=DataSettings)

    # Feature flags
    enable_gdelt_integration: bool = True
    enable_anomaly_detection: bool = True
    enable_clustering: bool = True
    enable_gnn: bool = True
    enable_simulation: bool = True

    @field_validator("log_level", mode="before")
    @classmethod
    def uppercase_log_level(cls, v: str) -> str:
        return v.upper() if isinstance(v, str) else v


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


def reset_settings() -> None:
    """Reset cached settings (useful for testing)."""
    get_settings.cache_clear()
