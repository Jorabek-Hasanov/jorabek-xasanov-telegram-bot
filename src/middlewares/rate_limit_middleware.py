# Rate Limit Middleware
# ============================================================================
# Middleware for enforcing rate limits using Token Bucket algorithm
# ============================================================================

import time
from typing import Any, Callable, Dict, Optional

from aiogram import BaseMiddleware
from aiogram.types import Message, Update

from src.core.config import get_settings
from src.core.logging import logger
from src.services.rate_limiter import RateLimitResult, TokenBucketRateLimiter


class RateLimitMiddleware(BaseMiddleware):
    """
    Middleware for enforcing rate limits on messages.
    
    Uses Redis-based Token Bucket algorithm to prevent spam.
    
    Usage:
        ```python
        rate_limiter = TokenBucketRateLimiter()
        await rate_limiter.connect()
        dp.message.middleware.register(RateLimitMiddleware(rate_limiter))
        ```
    """
    
    def __init__(
        self,
        rate_limiter: TokenBucketRateLimiter,
        block_on_exceed: bool = True,
    ) -> None:
        """
        Initialize rate limit middleware.
        
        Args:
            rate_limiter: Rate limiter instance
            block_on_exceed: Whether to block or just warn on limit exceed
        """
        self.rate_limiter = rate_limiter
        self.block_on_exceed = block_on_exceed
        logger.info("Rate Limit Middleware initialized")
    
    async def __call__(
        self,
        handler: Callable[[Update, Dict[str, Any]], Any],
        event: Update,
        data: Dict[str, Any],
    ) -> Any:
        """
        Process update with rate limiting.
        
        Args:
            handler: Next handler in chain
            event: Incoming update
            data: Handler data dictionary
            
        Returns:
            Handler result
        """
        if not event.message or not event.message.from_user:
            return await handler(event, data)
        
        user_id = event.message.from_user.id
        
        # Check rate limit
        result = await self.rate_limiter.check_rate_limit(user_id)
        
        if not result.allowed:
            logger.log_security_event(
                event_type="rate_limit_exceeded",
                user_id=user_id,
                details={
                    "remaining": result.remaining,
                    "retry_after": result.retry_after,
                }
            )
            
            if self.block_on_exceed:
                # Uzbek language support for rate limit errors
                await event.message.answer(
                    "🚦 <b>Ko'p so'rovlar!</b>\n\n"
                    f"Siz juda tez yuboryapsiz! Iltimos, <b>{result.retry_after:.0f} soniya</b> "
                    f"kuting va keyin xabar yuboring.\n\n"
                    f"Limit: {result.limit} ta so'rov {result.window_seconds} soniya ichida"
                )
                return  # Block the request
            
            # Just log warning but allow through
            logger.warning(
                f"Rate limit exceeded for user {user_id}, allowing through"
            )
        
        return await handler(event, data)


class RateLimitBlocker:
    """
    Standalone rate limit checker for custom usage.
    
    Can be used outside of aiogram middleware context.
    """
    
    def __init__(
        self,
        rate_limiter: TokenBucketRateLimiter,
    ) -> None:
        """Initialize blocker."""
        self.rate_limiter = rate_limiter
    
    async def check(self, user_id: int) -> RateLimitResult:
        """
        Check rate limit for user.
        
        Args:
            user_id: User ID to check
            
        Returns:
            RateLimitResult with allow/deny decision
        """
        return await self.rate_limiter.check_rate_limit(user_id)
    
    async def get_remaining(self, user_id: int) -> int:
        """
        Get remaining requests for user.
        
        Args:
            user_id: User ID to check
            
        Returns:
            Number of remaining requests
        """
        result = await self.rate_limiter.check_rate_limit(user_id)
        return result.remaining
