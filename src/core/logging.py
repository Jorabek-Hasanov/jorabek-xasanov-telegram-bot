# Structured Logging Module
# ============================================================================
# Professional logging with JSON format for ELK Stack integration
# Implements structured logging following observability best practices
# ============================================================================

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import get_settings


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs logs in JSON format for easy parsing by ELK Stack,
    Loki, or other log aggregation systems.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        # Add session info if present
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Console formatter with colors for human-readable output.
    
    Used in development mode for better readability.
    """
    
    RESET = "\033[0m"
    COLORS = {
        "DEBUG": "\033[94m",    # Blue
        "INFO": "\033[92m",     # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",    # Red
        "CRITICAL": "\033[95m", # Purple
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # Create simple format
        format_str = "%(asctime)s | %(levelname)-8s | %(message)s"
        
        # Add location info in debug mode
        if record.levelno <= logging.DEBUG:
            format_str = "%(asctime)s | %(levelname)-8s | [%(name)s:%(lineno)d] | %(message)s"
        
        return super().format(record)


class StructuredLogger:
    """
    Structured logging service for the application.
    
    Provides:
    - JSON logging for production (ELK Stack compatible)
    - Colored console logging for development
    - Multiple handlers (file, console)
    - Metric logging for performance tracking
    """
    
    def __init__(self, name: str = "jorabek_bot") -> None:
        """Initialize the structured logger."""
        self.logger = logging.getLogger(name)
        self.settings = get_settings()
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Configure logger with appropriate handlers and formatters."""
        self.logger.setLevel(getattr(logging, self.settings.log_level))
        self.logger.handlers.clear()
        
        # Create logs directory
        logs_dir = self.settings.project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if self.settings.monitoring.log_format == "json":
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(ColoredConsoleFormatter())
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # Debug file handler (all logs)
        debug_file = logs_dir / "bot_debug.log"
        debug_handler = logging.FileHandler(debug_file, encoding="utf-8")
        debug_handler.setFormatter(JSONFormatter())
        debug_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(debug_handler)
        
        # Error file handler (errors only)
        error_file = logs_dir / "bot_error.log"
        error_handler = logging.FileHandler(error_file, encoding="utf-8")
        error_handler.setFormatter(JSONFormatter())
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
        
        # AI Usage log
        ai_file = logs_dir / "ai_usage.log"
        ai_handler = logging.FileHandler(ai_file, encoding="utf-8")
        ai_handler.setFormatter(JSONFormatter())
        ai_handler.setLevel(logging.INFO)
        self.logger.addHandler(ai_handler)
        
        # Performance metrics log
        perf_file = logs_dir / "performance_metrics.log"
        perf_handler = logging.FileHandler(perf_file, encoding="utf-8")
        perf_handler.setFormatter(JSONFormatter())
        perf_handler.setLevel(logging.INFO)
        self.logger.addHandler(perf_handler)
        
        # Security audit log
        security_file = logs_dir / "security_audit.log"
        security_handler = logging.FileHandler(security_file, encoding="utf-8")
        security_handler.setFormatter(JSONFormatter())
        security_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(security_handler)
    
    def debug(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=self._prepare_extra(extra, **kwargs))
    
    def info(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Log info message."""
        self.logger.info(message, extra=self._prepare_extra(extra, **kwargs))
    
    def warning(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=self._prepare_extra(extra, **kwargs))
    
    def error(
        self,
        message: str,
        exc_info: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Log error message."""
        if exc_info:
            self.logger.error(message, exc_info=exc_info, extra=self._prepare_extra(extra, **kwargs))
        else:
            self.logger.error(message, extra=self._prepare_extra(extra, **kwargs))
    
    def critical(
        self,
        message: str,
        exc_info: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Log critical message."""
        if exc_info:
            self.logger.critical(message, exc_info=exc_info, extra=self._prepare_extra(extra, **kwargs))
        else:
            self.logger.critical(message, extra=self._prepare_extra(extra, **kwargs))
    
    def log_ai_usage(
        self,
        user_id: int,
        model: str,
        tokens_used: int,
        latency_ms: float,
        session_id: str,
        cache_hit: bool = False
    ) -> None:
        """Log AI service usage metrics."""
        self.logger.info(
            f"AI Request | User:{user_id} | Model:{model} | "
            f"Tokens:{tokens_used} | Latency:{latency_ms:.2f}ms | "
            f"Cache:{cache_hit}",
            extra={
                "event_type": "ai_usage",
                "user_id": user_id,
                "model": model,
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
                "session_id": session_id,
                "cache_hit": cache_hit,
            }
        )
    
    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[int],
        details: Dict[str, Any]
    ) -> None:
        """Log security-related events."""
        self.logger.info(
            f"Security Event | Type:{event_type} | User:{user_id}",
            extra={
                "event_type": "security",
                "security_event_type": event_type,
                "user_id": user_id,
                **details,
            }
        )
    
    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log performance metrics."""
        self.logger.info(
            f"Performance | Operation:{operation} | Duration:{duration_ms:.2f}ms",
            extra={
                "event_type": "performance",
                "operation": operation,
                "duration_ms": duration_ms,
                **(details or {}),
            }
        )
    
    def _prepare_extra(
        self,
        extra: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Prepare extra fields for logging."""
        result = kwargs.copy()
        if extra:
            result.update(extra)
        return {"extra_data": result} if result else None


# Global logger instance
logger = StructuredLogger()


def get_logger(name: str) -> StructuredLogger:
    """
    Get a named logger instance.
    
    Args:
        name: Name for the logger
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)
