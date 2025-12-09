"""
Configuration management for Prophetic Emergentomics.

Supports environment variables, .env files, and programmatic configuration.
Designed for modularity and extensibility.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GDELTSettings(BaseSettings):
    """GDELT API configuration."""

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


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: Literal["anthropic", "openai", "litellm"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"

    # API keys (loaded from environment)
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    # Generation parameters
    max_tokens: int = 4096
    temperature: float = 0.3
    top_p: float = 0.95

    # Rate limiting
    requests_per_minute: int = 50
    max_retries: int = 3


class CAGSettings(BaseSettings):
    """Context Augmented Generation configuration."""

    model_config = SettingsConfigDict(env_prefix="CAG_")

    # Context window management
    max_context_tokens: int = 100000
    context_priority_decay: float = 0.9

    # Synthesis parameters
    synthesis_depth: Literal["shallow", "moderate", "deep"] = "moderate"
    include_historical_context: bool = True
    include_theoretical_framework: bool = True

    # Pattern detection
    emergence_threshold: float = 0.7
    anomaly_sensitivity: float = 0.8
    trend_momentum_window_days: int = 7


class MedallionSettings(BaseSettings):
    """Medallion architecture configuration."""

    model_config = SettingsConfigDict(env_prefix="MEDALLION_")

    data_dir: Path = Path("data")
    bronze_dir: Path = Path("data/bronze")
    silver_dir: Path = Path("data/silver")
    gold_dir: Path = Path("data/gold")
    cache_dir: Path = Path("data/cache")

    # Data retention
    bronze_retention_days: int = 90
    silver_retention_days: int = 365
    gold_retention_days: int = -1  # Infinite

    # Processing
    batch_size: int = 1000
    parallel_workers: int = 4


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
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cag: CAGSettings = Field(default_factory=CAGSettings)
    medallion: MedallionSettings = Field(default_factory=MedallionSettings)

    # Feature flags for modular enhancement
    enable_gdelt_integration: bool = True
    enable_llm_synthesis: bool = True
    enable_pattern_detection: bool = True
    enable_opportunity_alerts: bool = True

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
