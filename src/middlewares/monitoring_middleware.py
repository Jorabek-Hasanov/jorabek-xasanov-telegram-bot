# Monitoring Middleware
# ============================================================================
# Prometheus metrics collection and structured logging for observability
# ============================================================================

import asyncio
import psutil
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Optional

from aiogram import BaseMiddleware
from aiogram.types import Update

from src.core.config import get_settings
from src.core.logging import logger


@dataclass
class Metrics:
    """Bot performance metrics."""
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    db_query_count: int = 0
    db_query_latency_ms: float = 0.0
    ai_request_count: int = 0
    ai_request_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100


class MetricsCollector:
    """
    In-memory metrics collector for Prometheus integration.
    
    Provides real-time metrics for monitoring and alerting.
    
    Example:
        ```python
        collector = MetricsCollector()
        collector.record_request(latency_ms=150)
        print(collector.get_error_rate())
        ```
    """
    
    def __init__(
        self,
        error_rate_threshold: float = 10.0,
        window_seconds: int = 300,
    ) -> None:
        """
        Initialize metrics collector.
        
        Args:
            error_rate_threshold: Alert threshold percentage
            window_seconds: Time window for error rate calculation
        """
        self.error_rate_threshold = error_rate_threshold
        self.window_seconds = window_seconds
        
        self.metrics = Metrics()
        self._lock = asyncio.Lock()
        
        # Sliding window for error tracking
        self._error_timestamps: Deque[float] = deque()
        self._request_timestamps: Deque[float] = deque()
        
        # Latency tracking
        self._latencies: Deque[float] = deque(maxlen=1000)
        
        logger.info("Metrics Collector initialized")
    
    async def record_request(
        self,
        latency_ms: float,
        success: bool = True,
        user_id: Optional[int] = None,
    ) -> None:
        """
        Record a request for metrics.
        
        Args:
            latency_ms: Request latency in milliseconds
            success: Whether request was successful
            user_id: User ID for logging
        """
        async with self._lock:
            now = time.time()
            
            self.metrics.request_count += 1
            self.metrics.total_latency_ms += latency_ms
            self._latencies.append(latency_ms)
            self._request_timestamps.append(now)
            
            if not success:
                self.metrics.error_count += 1
                self._error_timestamps.append(now)
            
            # Cleanup old timestamps
            self._cleanup_timestamps(now)
            
            # Check for critical error rate
            error_rate = self.get_error_rate()
            if error_rate > self.error_rate_threshold:
                logger.critical(
                    f"🚨 CRITICAL ALERT: Error rate {error_rate:.1f}% exceeds threshold {self.error_rate_threshold}%",
                    extra={
                        "event_type": "critical_alert",
                        "error_rate": error_rate,
                        "threshold": self.error_rate_threshold,
                    }
                )
    
    async def record_db_query(self, latency_ms: float) -> None:
        """Record database query metrics."""
        async with self._lock:
            self.metrics.db_query_count += 1
            self.metrics.db_query_latency_ms += latency_ms
    
    async def record_ai_request(self, latency_ms: float) -> None:
        """Record AI request metrics."""
        async with self._lock:
            self.metrics.ai_request_count += 1
            self.metrics.ai_request_latency_ms += latency_ms
    
    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.metrics.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.metrics.cache_misses += 1
    
    def _cleanup_timestamps(self, now: float) -> None:
        """Remove timestamps outside the window."""
        cutoff = now - self.window_seconds
        
        while self._error_timestamps and self._error_timestamps[0] < cutoff:
            self._error_timestamps.popleft()
        
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()
    
    def get_error_rate(self) -> float:
        """Get current error rate percentage."""
        if len(self._request_timestamps) == 0:
            return 0.0
        return (len(self._error_timestamps) / len(self._request_timestamps)) * 100
    
    def get_latency_percentile(self, percentile: float) -> float:
        """Get latency at given percentile."""
        if not self._latencies:
            return 0.0
        
        sorted_latencies = sorted(self._latencies)
        index = int(len(sorted_latencies) * percentile / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "request_count": self.metrics.request_count,
            "error_count": self.metrics.error_count,
            "error_rate_percent": round(self.get_error_rate(), 2),
            "avg_latency_ms": round(self.metrics.avg_latency_ms, 2),
            "p50_latency_ms": round(self.get_latency_percentile(50), 2),
            "p95_latency_ms": round(self.get_latency_percentile(95), 2),
            "p99_latency_ms": round(self.get_latency_percentile(99), 2),
            "db_query_count": self.metrics.db_query_count,
            "db_query_avg_ms": round(
                self.metrics.db_query_latency_ms / max(1, self.metrics.db_query_count), 2
            ),
            "ai_request_count": self.metrics.ai_request_count,
            "ai_request_avg_ms": round(
                self.metrics.ai_request_latency_ms / max(1, self.metrics.ai_request_count), 2
            ),
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate_percent": round(
                (self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)) * 100, 2
            ),
        }
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus metrics format."""
        summary = self.get_metrics_summary()
        system_metrics = self.get_system_metrics()
        
        lines = [
            "# HELP bot_requests_total Total number of requests",
            "# TYPE bot_requests_total counter",
            f"bot_requests_total {summary['request_count']}",
            "",
            "# HELP bot_errors_total Total number of errors",
            "# TYPE bot_errors_total counter",
            f"bot_errors_total {summary['error_count']}",
            "",
            "# HELP bot_error_rate_percent Current error rate percentage",
            "# TYPE bot_error_rate_percent gauge",
            f"bot_error_rate_percent {summary['error_rate_percent']}",
            "",
            "# HELP bot_request_latency_ms Average request latency",
            "# TYPE bot_request_latency_ms gauge",
            f"bot_request_latency_ms {summary['avg_latency_ms']}",
            "",
            "# HELP bot_p50_latency_ms P50 request latency",
            "# TYPE bot_p50_latency_ms gauge",
            f"bot_p50_latency_ms {summary['p50_latency_ms']}",
            "",
            "# HELP bot_p95_latency_ms P95 request latency",
            "# TYPE bot_p95_latency_ms gauge",
            f"bot_p95_latency_ms {summary['p95_latency_ms']}",
            "",
            "# HELP bot_p99_latency_ms P99 request latency",
            "# TYPE bot_p99_latency_ms gauge",
            f"bot_p99_latency_ms {summary['p99_latency_ms']}",
            "",
            "# HELP bot_db_queries_total Total database queries",
            "# TYPE bot_db_queries_total counter",
            f"bot_db_queries_total {summary['db_query_count']}",
            "",
            "# HELP bot_ai_requests_total Total AI requests",
            "# TYPE bot_ai_requests_total counter",
            f"bot_ai_requests_total {summary['ai_request_count']}",
            "",
            "# HELP bot_cache_hits_total Total cache hits",
            "# TYPE bot_cache_hits_total counter",
            f"bot_cache_hits_total {summary['cache_hits']}",
            "",
            "# HELP bot_cache_misses_total Total cache misses",
            "# TYPE bot_cache_misses_total counter",
            f"bot_cache_misses_total {summary['cache_misses']}",
            "",
            "# HELP bot_cache_hit_rate_percent Cache hit rate percentage",
            "# TYPE bot_cache_hit_rate_percent gauge",
            f"bot_cache_hit_rate_percent {summary['cache_hit_rate_percent']}",
            "",
            "# HELP bot_cpu_percent CPU usage percentage",
            "# TYPE bot_cpu_percent gauge",
            f"bot_cpu_percent {system_metrics['cpu_percent']}",
            "",
            "# HELP bot_memory_percent Memory usage percentage",
            "# TYPE bot_memory_percent gauge",
            f"bot_memory_percent {system_metrics['memory_percent']}",
            "",
            "# HELP bot_disk_percent Disk usage percentage",
            "# TYPE bot_disk_percent gauge",
            f"bot_disk_percent {system_metrics['disk_percent']}",
        ]
        
        return "\n".join(lines)
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get system metrics (CPU, memory, disk usage)."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
            }
        except Exception:
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_percent": 0.0,
            }


class MonitoringMiddleware(BaseMiddleware):
    """
    Middleware for collecting request metrics.
    
    Records timing and error counts for all updates.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None) -> None:
        """Initialize monitoring middleware."""
        self.metrics = metrics_collector or MetricsCollector()
        logger.info("Monitoring Middleware initialized")
    
    async def __call__(
        self,
        handler: Callable[[Update, Dict[str, Any]], Any],
        event: Update,
        data: Dict[str, Any],
    ) -> Any:
        """
        Process update with metrics collection.
        
        Args:
            handler: Next handler in chain
            event: Incoming update
            data: Handler data dictionary
            
        Returns:
            Handler result
        """
        start_time = time.perf_counter()
        
        try:
            result = await handler(event, data)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            user_id = event.message.from_user.id if event.message else None
            
            await self.metrics.record_request(latency_ms, success=True, user_id=user_id)
            
            return result
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            user_id = event.message.from_user.id if event.message else None
            
            await self.metrics.record_request(latency_ms, success=False, user_id=user_id)
            
            logger.error(
                f"Request failed: {e}",
                extra={"event_type": "request_error", "user_id": user_id}
            )
            
            raise


# Global metrics collector instance
metrics_collector = MetricsCollector()
