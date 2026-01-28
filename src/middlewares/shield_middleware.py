# Shield Middleware
# ============================================================================
# Security middleware for prompt injection protection
# Filters malicious commands and protects against prompt attacks
# ============================================================================

import re
from typing import Any, Callable, Dict, Optional

from aiogram import BaseMiddleware
from aiogram.types import Message, Update

from src.core.config import get_settings
from src.core.logging import logger
from src.services.security import SecurityService


class ShieldMiddleware(BaseMiddleware):
    """
    Security middleware that filters prompt injection attacks.
    
    Features:
    - Detects and blocks prompt injection attempts
    - Filters dangerous commands
    - Sanitizes user input
    - Logs security events
    
    Usage:
        ```python
        dp.message.middleware.register(ShieldMiddleware())
        ```
    """
    
    def __init__(self) -> None:
        """Initialize shield middleware."""
        self.security = SecurityService()
        self.settings = get_settings().security
        
        # Blocked commands that shouldn't be processed
        self.blocked_commands = [
            re.compile(r'^/exec', re.IGNORECASE),
            re.compile(r'^/run', re.IGNORECASE),
            re.compile(r'^/sudo', re.IGNORECASE),
            re.compile(r'^/bash', re.IGNORECASE),
            re.compile(r'^/shell', re.IGNORECASE),
            re.compile(r'^/system', re.IGNORECASE),
            re.compile(r'^/admin', re.IGNORECASE),
        ]
        
        logger.info("Shield Middleware initialized")
    
    async def __call__(
        self,
        handler: Callable[[Update, Dict[str, Any]], Any],
        event: Update,
        data: Dict[str, Any],
    ) -> Any:
        """
        Process the update through security checks.
        
        Args:
            handler: Next handler in chain
            event: Incoming update
            data: Handler data dictionary
            
        Returns:
            Handler result
        """
        if event.message and event.message.text:
            message = event.message
            
            # Check for blocked commands
            for pattern in self.blocked_commands:
                if pattern.search(message.text):
                    logger.log_security_event(
                        event_type="blocked_command",
                        user_id=message.from_user.id,
                        details={"command": message.text}
                    )
                    await message.answer(
                        "🚫 <b>Access Denied</b>\n\n"
                        "This command is not available. I'm designed to be secure by default."
                    )
                    return
            
            # Check for prompt injection
            injection_result = self.security.check_prompt_injection(message.text)
            
            if not injection_result.safe:
                logger.log_security_event(
                    event_type="prompt_injection_blocked",
                    user_id=message.from_user.id,
                    details={
                        "violations": injection_result.violations,
                        "original": message.text[:100],
                    }
                )
                await message.answer(
                    "🛡️ <b>Security Check Passed</b>\n\n"
                    "I detected a potentially malicious input pattern and sanitized it. "
                    "Your message has been processed safely."
                )
                # Update message text with sanitized version
                message.text = injection_result.sanitized_text
            
            # Sanitize input regardless
            sanitized = self.security.sanitize_input(message.text)
            if sanitized != message.text:
                message.text = sanitized
        
        # Call next handler
        return await handler(event, data)


class PromptInjectionBlocker:
    """
    Standalone prompt injection blocker for custom usage.
    
    Can be used outside of aiogram middleware context.
    """
    
    def __init__(self) -> None:
        """Initialize blocker."""
        self.security = SecurityService()
    
    def check(self, text: str) -> tuple[bool, str, list[str]]:
        """
        Check text for prompt injection.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_safe, sanitized_text, violations)
        """
        result = self.security.check_prompt_injection(text)
        return result.safe, result.sanitized_text, result.violations
    
    def sanitize(self, text: str) -> str:
        """Sanitize text."""
        return self.security.sanitize_input(text)
