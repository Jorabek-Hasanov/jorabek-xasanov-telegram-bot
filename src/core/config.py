# Core Configuration Module
# ============================================================================
# Centralized configuration management using pydantic-settings
# Follows the "Fail Fast" principle for environment validation
# ============================================================================

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_",
        extra="ignore"
    )
    
    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    user: str = Field(default="bot_user", description="PostgreSQL username")
    password: str = Field(default="secure_password", description="PostgreSQL password")
    database: str = Field(default="bot_db", description="PostgreSQL database name")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")
    pool_timeout: int = Field(default=30, ge=1, le=300, description="Pool timeout in seconds")
    
    @property
    def async_url(self) -> str:
        """Generate async database URL for SQLAlchemy."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def sync_url(self) -> str:
        """Generate sync database URL for SQLAlchemy."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and rate limiting."""
    
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        extra="ignore"
    )
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    database: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    max_connections: int = Field(default=50, description="Max Redis connections")
    socket_timeout: int = Field(default=5, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=5, description="Connection timeout in seconds")
    
    @property
    def url(self) -> str:
        """Generate Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"


class AISettings(BaseSettings):
    """Groq AI configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="AI_",
        extra="ignore"
    )
    
    model: str = Field(default="llama-3.3-70b-versatile", description="AI model name")
    api_key: str = Field(default="", description="Groq API key")
    temperature: float = Field(default=0.72, ge=0.0, le=2.0, description="Temperature parameter")
    max_tokens: int = Field(default=4096, ge=1, le=16384, description="Max tokens in response")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p parameter")
    frequency_penalty: float = Field(default=0.1, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.1, ge=-2.0, le=2.0, description="Presence penalty")
    request_timeout: int = Field(default=35, ge=1, le=120, description="Request timeout in seconds")
    context_history_limit: int = Field(default=15, description="Max conversation history messages")
    summary_threshold: int = Field(default=20, description="Messages before summarization")


class SecuritySettings(BaseSettings):
    """Security and encryption settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        extra="ignore"
    )
    
    encryption_key: str = Field(default="", description="Fernet encryption key")
    rate_limit_requests: int = Field(default=5, description="Max requests per window")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    admin_chat_id: int = Field(default=0, description="Admin Telegram chat ID for alerts")
    prompt_injection_patterns: List[str] = Field(
        default_factory=lambda: [
            r"ignore.*previous.*instructions",
            r"forget.*everything",
            r"system.*prompt",
            r"you.*are.*now",
            r"act.*as.*",
            r"jailbreak",
            r"developer.*mode",
        ],
        description="Patterns for prompt injection detection"
    )


class FileProcessingSettings(BaseSettings):
    """File processing configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="FILE_",
        extra="ignore"
    )
    
    max_file_size_mb: int = Field(default=20, ge=1, le=100, description="Max file size in MB")
    allowed_mime_types: List[str] = Field(
        default_factory=lambda: [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/gif",
            "image/webp",
        ],
        description="Allowed MIME types for file uploads"
    )
    temp_dir: str = Field(default="/tmp/bot_uploads", description="Temporary file directory")
    ocr_engine: str = Field(default="easyocr", description="OCR engine (easyocr or tesseract)")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="MONITORING_",
        extra="ignore"
    )
    
    error_rate_threshold: float = Field(default=10.0, ge=0.0, le=100.0, description="Error rate threshold %")
    error_rate_window: int = Field(default=300, ge=60, le=3600, description="Error rate window in seconds")
    log_format: str = Field(default="json", description="Log format (json or plain)")
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    healthcheck_interval: int = Field(default=30, description="Healthcheck interval in seconds")


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="TELEGRAM_",
        extra="ignore"
    )
    
    bot_token: str = Field(default="", description="Telegram bot token")
    parse_mode: str = Field(default="HTML", description="Telegram parse mode")
    bot_username: Optional[str] = Field(default=None, description="Bot username")


class AppSettings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Subsections
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ai: AISettings = Field(default_factory=AISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    file_processing: FileProcessingSettings = Field(default_factory=FileProcessingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    
    # Application settings
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    app_version: str = Field(default="2.0.0", description="Application version")
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug


@lru_cache()
def get_settings() -> AppSettings:
    """
    Get application settings with caching.
    
    This function uses lru_cache to ensure settings are only loaded once,
    following the singleton pattern for configuration management.
    
    Returns:
        AppSettings: Application configuration instance
    """
    return AppSettings()
