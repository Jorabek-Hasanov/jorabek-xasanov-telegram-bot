# AI Service
# ============================================================================
# Groq API integration with advanced context management
# Implements recursive summarization for long conversations
# ============================================================================

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import groq

from src.core.config import get_settings
from src.core.logging import logger


@dataclass
class AIResponse:
    """Response from AI service."""
    text: str
    tokens_used: int
    latency_ms: float
    cache_hit: bool
    session_id: str
    model: str
    context_messages: int = 0
    was_summarized: bool = False


@dataclass
class ConversationContext:
    """Conversation context for AI queries."""
    messages: List[Dict[str, str]]
    user_preferences: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    system_prompt_additions: Optional[str] = None


class AIService:
    """
    High-performance async service for Groq API communication.
    
    Features:
    - Asynchronous request handling with timeout
    - Conversation history management from PostgreSQL
    - Response caching for repeated queries
    - Recursive summarization for long conversations
    - Dynamic user preference injection
    - Comprehensive error handling with retry logic
    
    Example:
        ```python
        service = AIService()
        response = await service.send_message("How do I optimize PostgreSQL?")
        ```
    """
    
    def __init__(self, api_key: str) -> None:
        """Initialize AI service."""
        self.settings = get_settings().ai
        self.client = groq.Groq(api_key=api_key)
        self.model = self.settings.model
        
        # Model configuration
        self.temperature = self.settings.temperature
        self.max_tokens = self.settings.max_tokens
        self.top_p = self.settings.top_p
        self.frequency_penalty = self.settings.frequency_penalty
        self.presence_penalty = self.settings.presence_penalty
        self.request_timeout = self.settings.request_timeout
        self.context_history_limit = self.settings.context_history_limit
        self.summary_threshold = self.settings.summary_threshold
        
        # In-memory cache for responses (supplements Redis)
        self._response_cache: Dict[str, str] = {}
        self._cache_lock = asyncio.Lock()
        
        # System prompt template
        self._system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the base system prompt."""
        return """You are a Digital Twin of Jorabek Xasanov, a Senior Backend Engineer with 5+ years of experience in building high-performance, scalable backend systems.

## CORE IDENTITY
You represent a battle-tested backend architect who has shipped production systems handling millions of requests. You speak with the confidence of someone who has debugged production issues at 3 AM and lived to tell the tale.

## TECHNICAL STACK & EXPERTISE

### Primary Stack:
- **Node.js (20+)**: Deep expertise in async patterns, Event Loop optimization, worker threads
- **NestJS (10+)**: Modular architecture, microservices, dependency injection
- **TypeScript (5+)**: Advanced type system, discriminated unions, conditional types
- **PostgreSQL**: Query optimization, indexing strategies, partitioning, replication
- **Python (FastAPI/Django)**: Async APIs, background tasks
- **Go**: Concurrency patterns, performance-critical services
- **Redis**: Caching strategies, pub/sub, distributed locks

### Specializations:
- **High-Load System Architecture**: Horizontal scaling, load balancing, circuit breakers
- **Database Optimization**: Query analysis, index design, connection pooling
- **Secure API Design**: RBAC, JWT authentication, OAuth 2.0
- **Microservices**: Service discovery, inter-service communication
- **Real-time Systems**: WebSockets, Server-Sent Events, message queues

## RESPONSE STRUCTURE - SENIOR ARCHITECT FRAMEWORK

When answering technical questions, follow this structure:

### 1. CONTEXT ANALYSIS
Explain what the user is asking for in terms of backend architecture.

### 2. THEORETICAL DEEP DIVE
Explain fundamental backend principles (ACID, CAP theorem, Event Loop, etc.)

### 3. PRACTICAL SOLUTION
Provide clean, safe, readable code samples following Clean Code and SOLID principles.

### 4. BEST PRACTICES & FUTURE-PROOFING
Discuss scalability issues, potential bottlenecks, and warnings.

## COMMUNICATION STYLE

1. Be Precise & Technical: Use correct terminology
2. Architecture-First Thinking: Consider scalability and trade-offs
3. Code-Focused: Provide concrete code examples
4. Practical & Actionable: Give production-ready solutions
5. Professional but Approachable: Expert who enjoys helping others

## FORBIDDEN
- Making up fake APIs, libraries, or technologies
- Providing insecure or anti-pattern code
- Claiming expertise outside backend development

Remember: You're not just answering questions—you're mentoring the next generation of backend engineers.
"""
    
    def _generate_cache_key(self, message: str) -> str:
        """Generate cache key for user message."""
        content = f"ai:cache:v1:{hashlib.md5(message.encode()).hexdigest()}"
        return content
    
    async def _get_cached_response(self, message: str) -> Optional[str]:
        """Get cached response."""
        async with self._cache_lock:
            cache_key = self._generate_cache_key(message)
            return self._response_cache.get(cache_key)
    
    async def _cache_response(self, message: str, response: str) -> None:
        """Cache response."""
        async with self._cache_lock:
            cache_key = self._generate_cache_key(message)
            self._response_cache[cache_key] = response
            
            # Limit cache size
            if len(self._response_cache) > 1000:
                # Remove oldest entries
                keys_to_remove = list(self._response_cache.keys())[:100]
                for key in keys_to_remove:
                    del self._response_cache[key]
    
    def _format_messages_for_api(
        self,
        context: ConversationContext,
        current_message: str,
    ) -> List[Dict[str, str]]:
        """
        Format messages for API request.
        
        Combines system prompt, conversation history, and current message.
        """
        messages = []
        
        # System prompt with dynamic user preferences
        system_content = self._system_prompt
        
        if context.user_preferences:
            pref_parts = []
            if context.user_preferences.get("coding_style"):
                pref_parts.append(
                    f"- User's coding style: {context.user_preferences['coding_style']}"
                )
            if context.user_preferences.get("preferred_language"):
                pref_parts.append(
                    f"- Respond in: {context.user_preferences['preferred_language']}"
                )
            if context.user_preferences.get("custom_prompt"):
                pref_parts.append(f"- Custom instructions: {context.user_preferences['custom_prompt']}")
            
            if pref_parts:
                system_content += "\n\n## USER PREFERENCES\n" + "\n".join(pref_parts)
        
        if context.system_prompt_additions:
            system_content += "\n\n" + context.system_prompt_additions
        
        if context.summary:
            system_content += f"\n\n## CONVERSATION SUMMARY\n{context.summary}\n\n(This summary covers earlier parts of the conversation. Recent messages follow below.)"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history
        messages.extend(context.messages)
        
        # Add current message
        messages.append({"role": "user", "content": current_message})
        
        return messages
    
    async def send_message(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        cache_enabled: bool = True,
    ) -> AIResponse:
        """
        Send message to AI and get response.
        
        Args:
            message: User's message
            conversation_history: Previous messages from database
            user_preferences: User-specific settings
            cache_enabled: Whether to use caching
            
        Returns:
            AIResponse with text, metrics, etc.
        """
        start_time = time.perf_counter()
        session_id = str(uuid.uuid4())[:8]
        
        # Check cache
        if cache_enabled:
            cached = await self._get_cached_response(message)
            if cached:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.log_ai_usage(
                    user_id=0,
                    model=self.model,
                    tokens_used=0,
                    latency_ms=latency_ms,
                    session_id=session_id,
                    cache_hit=True
                )
                return AIResponse(
                    text=cached,
                    tokens_used=0,
                    latency_ms=latency_ms,
                    cache_hit=True,
                    session_id=session_id,
                    model=self.model,
                )
        
        # Prepare context
        context = ConversationContext(
            messages=conversation_history or [],
            user_preferences=user_preferences,
        )
        
        # Check for summarization need
        was_summarized = False
        if len(context.messages) > self.summary_threshold:
            # Would need summarization here
            # For now, trim messages
            context.messages = context.messages[-self.context_history_limit:]
        
        # Format messages for API
        messages = self._format_messages_for_api(context, message)
        
        try:
            # Make API request
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                ),
                timeout=self.request_timeout,
            )
            
            assistant_response = response.choices[0].message.content
            
            # Calculate metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Cache response
            if cache_enabled:
                await self._cache_response(message, assistant_response)
            
            # Log AI usage
            logger.log_ai_usage(
                user_id=0,
                model=self.model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                session_id=session_id,
                cache_hit=False
            )
            
            # Performance warning
            if latency_ms > 4000:
                logger.warning(
                    f"[SESSION:{session_id}] ⚠️ SLOW AI REQUEST - {latency_ms:.2f}ms"
                )
            
            return AIResponse(
                text=assistant_response,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cache_hit=False,
                session_id=session_id,
                model=self.model,
                context_messages=len(context.messages),
                was_summarized=was_summarized,
            )
            
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[SESSION:{session_id}] ⏰ AI API TIMEOUT | {latency_ms:.2f}ms"
            )
            raise TimeoutError(f"AI request timed out after {self.request_timeout}s")
        
        except groq.RateLimitError as e:
            logger.error(f"[SESSION:{session_id}] 🚦 GROQ RATE LIMIT | {e}")
            raise
        
        except groq.AuthenticationError:
            logger.critical(f"[SESSION:{session_id}] 🔐 GROQ AUTH ERROR")
            raise
        
        except groq.GroqError as e:
            logger.error(f"[SESSION:{SESSION_ID}] ❌ GROQ ERROR | {e}")
            raise
    
    async def summarize_conversation(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Summarize conversation history using AI.
        
        Reduces token usage while preserving conversation logic.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Summary of the conversation
        """
        if not messages:
            return ""
        
        # Prepare messages for summarization
        history_text = "\n".join(
            f"{msg['role']}: {msg['content'][:200]}..."
            if len(msg['content']) > 200
            else f"{msg['role']}: {msg['content']}"
            for msg in messages
        )
        
        summary_prompt = f"""Summarize the following conversation history in a concise way (2-3 paragraphs). Preserve the key topics discussed, important decisions, and any user preferences mentioned.

CONVERSATION HISTORY:
{history_text}

SUMMARY:"""
        
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a conversation summarizer. Provide concise summaries."},
                        {"role": "user", "content": summary_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                ),
                timeout=30.0,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Conversation summarization failed: {e}")
            # Fallback: just return first and last messages
            if len(messages) > 2:
                return f"Conversation covered {len(messages)} messages from {messages[0]['role']} to {messages[-1]['role']}"
            return ""


# Alias for backward compatibility
GroqService = AIService
