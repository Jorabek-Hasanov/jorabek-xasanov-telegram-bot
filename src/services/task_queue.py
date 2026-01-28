# Task Queue Service
# ============================================================================
# Redis-based asynchronous task queue with exponential backoff
# Offloads heavy tasks from the main event loop
# ============================================================================

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import redis.asyncio as redis

from src.core.config import get_settings
from src.core.logging import logger


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class TaskInfo:
    """Task information."""
    task_id: str
    queue_name: str
    status: TaskStatus
    payload: Dict[str, Any]
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    result: Optional[Any] = None


class TaskHandler(ABC):
    """Abstract base class for task handlers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Task handler name."""
        pass
    
    @abstractmethod
    async def process(self, payload: Dict[str, Any]) -> Any:
        """Process the task."""
        pass
    
    @property
    def max_retries(self) -> int:
        """Maximum retry attempts."""
        return 3
    
    @property
    def retry_delay(self) -> float:
        """Base delay between retries (seconds)."""
        return 1.0
    
    @property
    def exponential_base(self) -> float:
        """Exponential backoff base."""
        return 2.0


class TaskQueueService:
    """
    Redis-based task queue service.
    
    Features:
    - Asynchronous task processing
    - Exponential backoff retry policy
    - Status tracking and progress updates
    - Multiple queue support
    
    Example:
        ```python
        queue = TaskQueueService()
        await queue.enqueue("process_file", {"user_id": 123, "file_id": 456})
        ```
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        """Initialize task queue service."""
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.handlers: Dict[str, TaskHandler] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        self.settings = get_settings()
    
    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            await self.redis.ping()
            logger.info("✅ Task queue Redis connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Task queue Redis connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            logger.info("Task queue Redis connection closed")
    
    def register_handler(self, handler: TaskHandler) -> None:
        """Register a task handler."""
        self.handlers[handler.name] = handler
        logger.info(f"Registered task handler: {handler.name}")
    
    def _get_queue_key(self, queue_name: str) -> str:
        """Get Redis key for queue."""
        return f"taskqueue:{queue_name}"
    
    def _get_task_key(self, task_id: str) -> str:
        """Get Redis key for task info."""
        return f"taskqueue:task:{task_id}"
    
    def _get_processing_key(self, queue_name: str) -> str:
        """Get Redis key for processing set."""
        return f"taskqueue:processing:{queue_name}"
    
    async def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        queue_name: str = "default",
        priority: int = 0,
    ) -> str:
        """
        Enqueue a task for background processing.
        
        Args:
            task_type: Type of task (handler name)
            payload: Task payload data
            queue_name: Queue name
            priority: Priority (higher = processed first)
            
        Returns:
            Task ID
        """
        if not self.redis:
            raise RuntimeError("Task queue not connected")
        
        task_id = f"{task_type}:{int(time.time() * 1000)}:{id(payload)}"
        
        task_info = TaskInfo(
            task_id=task_id,
            queue_name=queue_name,
            status=TaskStatus.PENDING,
            payload={"type": task_type, **payload},
            created_at=time.time(),
        )
        
        # Store task info
        task_key = self._get_task_key(task_id)
        await self.redis.hset(task_key, mapping={
            "task_id": task_info.task_id,
            "queue_name": task_info.queue_name,
            "status": task_info.status.value,
            "payload": json.dumps(task_info.payload),
            "created_at": str(task_info.created_at),
            "retry_count": "0",
            "max_retries": str(self.handlers.get(task_type, TaskHandler()).max_retries),
        })
        await self.redis.expire(task_key, 86400)  # 24h expiry
        
        # Add to queue with priority
        queue_key = self._get_queue_key(queue_name)
        score = -(priority * 1000 + task_info.created_at)
        await self.redis.zadd(queue_key, {task_id: score})
        
        logger.info(
            f"Task enqueued: {task_id} ({task_type})",
            extra={"event_type": "task_enqueued", "task_id": task_id}
        )
        
        return task_id
    
    async def dequeue(self, queue_name: str = "default") -> Optional[TaskInfo]:
        """
        Dequeue the next task from the queue.
        
        Args:
            queue_name: Queue name
            
        Returns:
            TaskInfo or None if queue is empty
        """
        if not self.redis:
            return None
        
        queue_key = self._get_queue_key(queue_name)
        processing_key = self._get_processing_key(queue_name)
        
        # Get next task
        task_ids = await self.redis.zrange(queue_key, 0, 0)
        if not task_ids:
            return None
        
        task_id = task_ids[0]
        
        # Move to processing
        pipe = self.redis.pipeline()
        pipe.zrem(queue_key, task_id)
        pipe.zadd(processing_key, {task_id: time.time()})
        pipe.hset(self._get_task_key(task_id), "status", TaskStatus.PROCESSING.value)
        pipe.hset(self._get_task_key(task_id), "started_at", str(time.time()))
        await pipe.execute()
        
        # Get full task info
        task_key = self._get_task_key(task_id)
        task_data = await self.redis.hgetall(task_key)
        
        if not task_data:
            return None
        
        return TaskInfo(
            task_id=task_data["task_id"],
            queue_name=task_data["queue_name"],
            status=TaskStatus.PROCESSING,
            payload=json.loads(task_data["payload"]),
            created_at=float(task_data["created_at"]),
            started_at=float(task_data["started_at"]),
            retry_count=int(task_data.get("retry_count", 0)),
            max_retries=int(task_data.get("max_retries", 3)),
        )
    
    async def process_task(self, task_info: TaskInfo) -> bool:
        """
        Process a single task.
        
        Args:
            task_info: Task to process
            
        Returns:
            True if successful, False otherwise
        """
        handler = self.handlers.get(task_info.payload.get("type"))
        if not handler:
            logger.error(f"No handler for task type: {task_info.payload.get('type')}")
            return False
        
        try:
            result = await handler.process(task_info.payload)
            
            # Mark as completed
            await self._complete_task(task_info.task_id, result, success=True)
            
            logger.info(
                f"Task completed: {task_info.task_id}",
                extra={"event_type": "task_completed", "task_id": task_info.task_id}
            )
            return True
            
        except Exception as e:
            logger.error(
                f"Task failed: {task_info.task_id} | Error: {e}",
                extra={"event_type": "task_failed", "task_id": task_info.task_id}
            )
            
            # Check if should retry
            if task_info.retry_count < task_info.max_retries:
                await self._retry_task(task_info, handler, e)
                return False
            else:
                await self._complete_task(task_info.task_id, None, success=False, error=str(e))
                return False
    
    async def _complete_task(
        self,
        task_id: str,
        result: Any,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Mark task as completed or failed."""
        processing_key = self._get_processing_key("default")  # Get from task info in real impl
        
        pipe = self.redis.pipeline()
        pipe.zrem(processing_key, task_id)
        pipe.hset(self._get_task_key(task_id), "status", TaskStatus.COMPLETED.value if success else TaskStatus.FAILED.value)
        pipe.hset(self._get_task_key(task_id), "completed_at", str(time.time()))
        if result is not None:
            pipe.hset(self._get_task_key(task_id), "result", json.dumps(result))
        if error:
            pipe.hset(self._get_task_key(task_id), "last_error", error)
        await pipe.execute()
    
    async def _retry_task(
        self,
        task_info: TaskInfo,
        handler: TaskHandler,
        error: Exception,
    ) -> None:
        """Schedule task for retry with exponential backoff."""
        retry_delay = handler.retry_delay * (handler.exponential_base ** task_info.retry_count)
        
        # Update task info
        pipe = self.redis.pipeline()
        pipe.hset(self._get_task_key(task_info.task_id), "status", TaskStatus.RETRYING.value)
        pipe.hset(self._get_task_key(task_info.task_id), "retry_count", str(task_info.retry_count + 1))
        pipe.hset(self._get_task_key(task_info.task_id), "last_error", str(error))
        await pipe.execute()
        
        # Re-enqueue after delay
        await asyncio.sleep(retry_delay)
        await self.enqueue(
            task_info.payload["type"],
            {k: v for k, v in task_info.payload.items() if k != "type"},
            queue_name=task_info.queue_name,
        )
        
        logger.info(
            f"Task scheduled for retry: {task_info.task_id} (attempt {task_info.retry_count + 1})"
        )
    
    async def start_worker(
        self,
        queue_name: str = "default",
        concurrency: int = 5,
    ) -> None:
        """
        Start a worker that processes tasks from the queue.
        
        Args:
            queue_name: Queue to process
            concurrency: Number of concurrent tasks
        """
        logger.info(f"Starting task worker for queue: {queue_name}")
        
        async def worker() -> None:
            while True:
                task = await self.dequeue(queue_name)
                if task:
                    await self.process_task(task)
                else:
                    await asyncio.sleep(0.1)
        
        # Start multiple workers
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            logger.info("Task workers stopped")
    
    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get task status."""
        if not self.redis:
            return None
        
        task_key = self._get_task_key(task_id)
        task_data = await self.redis.hgetall(task_key)
        
        if not task_data:
            return None
        
        return TaskInfo(
            task_id=task_data["task_id"],
            queue_name=task_data["queue_name"],
            status=TaskStatus(task_data["status"]),
            payload=json.loads(task_data["payload"]),
            created_at=float(task_data["created_at"]),
            started_at=float(task_data["started_at"]) if task_data.get("started_at") else None,
            completed_at=float(task_data["completed_at"]) if task_data.get("completed_at") else None,
            retry_count=int(task_data.get("retry_count", 0)),
            max_retries=int(task_data.get("max_retries", 3)),
            last_error=task_data.get("last_error"),
            result=json.loads(task_data["result"]) if task_data.get("result") else None,
        )
    
    async def get_queue_stats(self, queue_name: str = "default") -> Dict[str, int]:
        """Get queue statistics."""
        if not self.redis:
            return {"pending": 0, "processing": 0, "completed": 0}
        
        queue_key = self._get_queue_key(queue_name)
        processing_key = self._get_processing_key(queue_name)
        
        pending = await self.redis.zcard(queue_key)
        processing = await self.redis.zcard(processing_key)
        
        return {
            "pending": pending,
            "processing": processing,
            "completed": 0,  # Would need separate tracking
        }
