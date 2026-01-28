# Rate Limiting Service
# ============================================================================
# Redis-based Token Bucket rate limiting algorithm
# Protects bot from spam with configurable limits
# ============================================================================

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as redis

from src.core.config import get_settings
from src.core.logging import logger


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_time: float
    limit: int
    window_seconds: int
    
    @property
    def retry_after(self) -> float:
        """Seconds until next request is allowed."""
        if self.allowed:
            return 0.0
        return max(0.0, self.reset_time - time.time())


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    async def check_rate_limit(self, user_id: int) -> RateLimitResult:
        """Check if user is within rate limit."""
        pass
    
    @abstractmethod
    async def get_current_usage(self, user_id: int) -> int:
        """Get current request count for user."""
        pass


class TokenBucketRateLimiter(RateLimiter):
    """
    Redis-based Token Bucket rate limiter.
    
    Implements the Token Bucket algorithm for smooth rate limiting:
    - Tokens are added to the bucket at a fixed rate
    - Each request consumes one token
    - If bucket is empty, requests are denied
    
    Advantages over Fixed Window:
    - More accurate rate limiting
    - Handles burst traffic gracefully
    - No boundary issues at window edges
    
    Example:
        ```python
        limiter = TokenBucketRateLimiter()
        result = await limiter.check_rate_limit(user_id=12345)
        if not result.allowed:
            print(f"Retry after {result.retry_after:.0f} seconds")
        ```
    """
    
    # Lua script for atomic operations
    LUA_SCRIPT = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local capacity = tonumber(ARGV[2])
    local refill_rate = tonumber(ARGV[3])
    local requested = tonumber(ARGV[4])
    
    -- Get current bucket state
    local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(bucket[1])
    local last_refill = tonumber(bucket[2])
    
    -- Initialize if new
    if tokens == nil then
        tokens = capacity
        last_refill = now
    else
        -- Calculate tokens to add based on time passed
        local time_passed = now - last_refill
        local tokens_to_add = time_passed * refill_rate
        tokens = math.min(capacity, tokens + tokens_to_add)
        last_refill = now
    end
    
    -- Check if we have enough tokens
    local allowed = 0
    if tokens >= requested then
        tokens = tokens - requested
        allowed = 1
    end
    
    -- Update bucket state
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
    redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) + 10)
    
    return {allowed, math.floor(tokens)}
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        capacity: int = 5,
        refill_rate: float = 0.0833,  # 5 requests per minute = 1/12 per second
        window_seconds: int = 60,
    ) -> None:
        """
        Initialize token bucket rate limiter.
        
        Args:
            redis_url: Redis connection URL
            capacity: Maximum tokens in bucket (max requests per window)
            refill_rate: Tokens added per second
            window_seconds: Time window in seconds
        """
        self.redis_url = redis_url
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.window_seconds = window_seconds
        self.redis: Optional[redis.Redis] = None
        self._lock = asyncio.Lock()
        
        logger.info(f"TokenBucketRateLimiter initialized: {capacity} requests per {window_seconds}s")
    
    async def connect(self) -> bool:
        """Connect to Redis."""
        async with self._lock:
            try:
                self.redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                await self.redis.ping()
                logger.info("✅ Rate limiter Redis connection established")
                return True
            except Exception as e:
                logger.error(f"❌ Rate limiter Redis connection failed: {e}")
                return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            logger.info("Rate limiter Redis connection closed")
    
    def _get_bucket_key(self, user_id: int) -> str:
        """Generate Redis key for user bucket."""
        return f"ratelimit:bucket:{user_id}"
    
    async def check_rate_limit(self, user_id: int) -> RateLimitResult:
        """
        Check if user is within rate limit.
        
        Args:
            user_id: User ID to check
            
        Returns:
            RateLimitResult with allow/deny decision
        """
        if not self.redis:
            # If Redis is not available, allow the request
            logger.warning("Redis not available, skipping rate limit")
            return RateLimitResult(
                allowed=True,
                remaining=self.capacity,
                reset_time=time.time() + self.window_seconds,
                limit=self.capacity,
                window_seconds=self.window_seconds,
            )
        
        try:
            now = time.time()
            bucket_key = self._get_bucket_key(user_id)
            
            # Execute Lua script atomically
            result = await self.redis.eval(
                self.LUA_SCRIPT,
                1,
                bucket_key,
                now,
                self.capacity,
                self.refill_rate,
                1,  # requested tokens
            )
            
            allowed = bool(result[0])
            remaining = int(result[1])
            
            # Calculate reset time (when bucket will be full)
            tokens_needed = self.capacity - remaining
            reset_time = now + (tokens_needed / self.refill_rate) if remaining < self.capacity else now
            
            if not allowed:
                logger.info(
                    f"Rate limit exceeded for user {user_id}",
                    extra={"event_type": "rate_limit_exceeded", "user_id": user_id}
                )
            
            return RateLimitResult(
                allowed=allowed,
                remaining=max(0, remaining),
                reset_time=reset_time,
                limit=self.capacity,
                window_seconds=self.window_seconds,
            )
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # On error, allow the request (fail-open for UX)
            return RateLimitResult(
                allowed=True,
                remaining=self.capacity,
                reset_time=time.time() + self.window_seconds,
                limit=self.capacity,
                window_seconds=self.window_seconds,
            )
    
    async def get_current_usage(self, user_id: int) -> int:
        """Get current token count for user."""
        if not self.redis:
            return 0
        
        try:
            bucket_key = self._get_bucket_key(user_id)
            tokens = await self.redis.hget(bucket_key, "tokens")
            return int(tokens) if tokens else self.capacity
        except Exception:
            return 0
    
    async def reset_limit(self, user_id: int) -> bool:
        """
        Reset rate limit for a user.
        
        Args:
            user_id: User ID to reset
            
        Returns:
            True if successful
        """
        if not self.redis:
            return False
        
        try:
            bucket_key = self._get_bucket_key(user_id)
            await self.redis.delete(bucket_key)
            logger.info(f"Rate limit reset for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Rate limit reset failed: {e}")
            return False


class InMemoryRateLimiter(RateLimiter):
    """
    In-memory rate limiter for development/testing.
    
    Uses simple counter-based approach without Redis.
    Not recommended for production.
    """
    
    def __init__(
        self,
        capacity: int = 5,
        window_seconds: int = 60,
    ) -> None:
        """Initialize in-memory rate limiter."""
        self.capacity = capacity
        self.window_seconds = window_seconds
        self.buckets: dict[int, list[float]] = {}
    
    async def check_rate_limit(self, user_id: int) -> RateLimitResult:
        """Check if user is within rate limit."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        if user_id in self.buckets:
            self.buckets[user_id] = [
                t for t in self.buckets[user_id] if t > window_start
            ]
        else:
            self.buckets[user_id] = []
        
        current_count = len(self.buckets[user_id])
        
        if current_count < self.capacity:
            self.buckets[user_id].append(now)
            return RateLimitResult(
                allowed=True,
                remaining=self.capacity - current_count - 1,
                reset_time=now + self.window_seconds,
                limit=self.capacity,
                window_seconds=self.window_seconds,
            )
        else:
            oldest = min(self.buckets[user_id]) if self.buckets[user_id] else now
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=oldest + self.window_seconds,
                limit=self.capacity,
                window_seconds=self.window_seconds,
            )
    
    async def get_current_usage(self, user_id: int) -> int:
        """Get current request count for user."""
        now = time.time()
        window_start = now - self.window_seconds
        
        if user_id in self.buckets:
            return len([t for t in self.buckets[user_id] if t > window_start])
        return 0
