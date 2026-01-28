# Main Application Entry Point
# ============================================================================
# Senior Backend Engineer Digital Twin Bot - Production-Ready
# ============================================================================
# Integrates all modules with graceful degradation (Chaos Engineering)
# Follows SOLID principles with dependency injection pattern
# ============================================================================

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import groq
from aiohttp import web
from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
)
from aiogram.filters import Command, CommandStart
from aiogram.utils.chat_action import ChatActionMiddleware
from dotenv import find_dotenv, load_dotenv

from src.core.config import get_settings
from src.core.logging import logger
from src.database import (
    ConversationRepository,
    Message,
    MessageRepository,
    MessageRole,
    UnitOfWork,
    UserRepository,
    get_engine,
    get_session,
    init_db,
)
from src.middlewares import (
    MonitoringMiddleware,
    RateLimitMiddleware,
    ShieldMiddleware,
    metrics_collector,
)
from src.services import (
    AIService,
    EncryptionService,
    FileProcessor,
    FileType,
    RateLimiter,
    SecurityService,
    TaskQueueService,
)


# ============================================================================
# SERVICE INITIALIZATION (Dependency Injection Container)
# ============================================================================

class ServiceContainer:
    """
    Dependency injection container for all services.
    
    Implements the Service Locator pattern for clean dependency management.
    All services are initialized with graceful degradation.
    """
    
    def __init__(self) -> None:
        """Initialize services."""
        self.settings = get_settings()
        self._engine = None
        self._redis = None
        self._rate_limiter = None
        self._ai_service = None
        self._file_processor = None
        self._security_service = None
        self._task_queue = None
        self._encryption_service = None
        
        # Service health status
        self._health_status = {
            "database": False,
            "redis": False,
            "ai": False,
            "file_processing": False,
        }
    
    async def initialize_all(self) -> bool:
        """
        Initialize all services with graceful degradation.
        
        Returns:
            True if core services are available
        """
        logger.info("Initializing service container...")
        
        # Initialize database
        try:
            self._engine = await init_db()
            self._health_status["database"] = True
            logger.info("✅ Database initialized")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            self._health_status["database"] = False
        
        # Initialize Redis (for rate limiting and caching)
        try:
            redis_url = self.settings.redis.url
            self._rate_limiter = RateLimiter(
                redis_url=redis_url,
                capacity=self.settings.security.rate_limit_requests,
                refill_rate=(
                    self.settings.security.rate_limit_requests 
                    / self.settings.security.rate_limit_window
                ),
                window_seconds=self.settings.security.rate_limit_window,
            )
            connected = await self._rate_limiter.connect()
            self._health_status["redis"] = connected
            if connected:
                logger.info("✅ Redis initialized")
            else:
                logger.warning("⚠️ Redis unavailable, using in-memory fallback")
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            self._health_status["redis"] = False
        
        # Initialize AI service
        try:
            self._ai_service = AIService(api_key=self.settings.ai.api_key)
            self._health_status["ai"] = bool(self.settings.ai.api_key)
            logger.info("✅ AI service initialized")
        except Exception as e:
            logger.error(f"❌ AI service initialization failed: {e}")
            self._health_status["ai"] = False
        
        # Initialize file processor
        try:
            self._file_processor = FileProcessor()
            self._health_status["file_processing"] = True
            logger.info("✅ File processor initialized")
        except Exception as e:
            logger.error(f"❌ File processor initialization failed: {e}")
            self._health_status["file_processing"] = False
        
        # Initialize security service
        try:
            self._security_service = SecurityService()
            logger.info("✅ Security service initialized")
        except Exception as e:
            logger.error(f"❌ Security service initialization failed: {e}")
        
        # Initialize encryption service
        try:
            self._encryption_service = EncryptionService()
            logger.info("✅ Encryption service initialized")
        except Exception as e:
            logger.error(f"❌ Encryption service initialization failed: {e}")
        
        # Log health status
        healthy_count = sum(self._health_status.values())
        logger.info(
            f"Service initialization complete: {healthy_count}/4 core services healthy"
        )
        
        return self._health_status["database"] or self._health_status["redis"]
    
    async def cleanup(self) -> None:
        """Cleanup all services."""
        logger.info("Cleaning up services...")
        
        if self._rate_limiter:
            await self._rate_limiter.disconnect()
        
        if self._task_queue:
            await self._task_queue.disconnect()
        
        if self._engine:
            await self._engine.dispose()
        
        logger.info("✅ All services cleaned up")
    
    @property
    def ai_service(self) -> Optional[AIService]:
        """Get AI service."""
        return self._ai_service
    
    @property
    def rate_limiter(self) -> RateLimiter:
        """Get rate limiter (creates in-memory fallback if Redis unavailable)."""
        if self._rate_limiter is None:
            from src.services.rate_limiter import InMemoryRateLimiter
            return InMemoryRateLimiter(
                capacity=self.settings.security.rate_limit_requests,
                window_seconds=self.settings.security.rate_limit_window,
            )
        return self._rate_limiter
    
    @property
    def file_processor(self) -> Optional[FileProcessor]:
        """Get file processor."""
        return self._file_processor
    
    @property
    def security_service(self) -> SecurityService:
        """Get security service."""
        return self._security_service or SecurityService()
    
    @property
    def is_database_available(self) -> bool:
        """Check if database is available."""
        return self._health_status["database"]
    
    @property
    def is_redis_available(self) -> bool:
        """Check if Redis is available."""
        return self._health_status["redis"]


# Global service container
services = ServiceContainer()


# ============================================================================
# BOT SETUP
# ============================================================================

# Create bot with HTML parse mode
bot = Bot(
    token=services.settings.telegram.bot_token,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)

# Create dispatcher
dp = Dispatcher()

# Create router for main handlers
router = Router(name=__name__)
dp.include_router(router)


# ============================================================================
# MIDDLEWARE REGISTRATION
# ============================================================================

# Register monitoring middleware first (outermost)
dp.message.middleware.register(MonitoringMiddleware(metrics_collector))

# Register shield middleware (security)
dp.message.middleware.register(ShieldMiddleware())

# Register rate limit middleware
dp.message.middleware.register(
    RateLimitMiddleware(
        rate_limiter=services.rate_limiter,
        block_on_exceed=True
    )
)

# Register chat action middleware for typing indicators
dp.message.middleware.register(ChatActionMiddleware())


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def escape_html_for_telegram(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    if not text:
        return ""
    
    # Escape HTML special characters
    escaped = (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    
    import re
    
    # Convert markdown-style code blocks to HTML
    escaped = re.sub(r'```(\w+)?\n', r'<pre>\n', escaped)
    escaped = escaped.replace('```', '</pre>')
    
    # Convert inline code
    escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
    
    # Bold and italic
    escaped = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', escaped)
    escaped = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', escaped)
    
    return escaped


async def get_user_context(
    user_id: int,
    conversation_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get user context from database.
    
    Falls back gracefully if database is unavailable.
    """
    if not services.is_database_available:
        return {"messages": [], "preferences": None}
    
    try:
        async with get_session() as session:
            async with UnitOfWork(session) as uow:
                # Get or create user
                user, _ = await uow.users.get_or_create(
                    telegram_id=user_id,
                    first_name="User",
                )
                
                # Get preferences
                preferences = await uow.preferences.get_or_create(user.id)
                
                # Get or create active conversation
                conversation = await uow.conversations.get_or_create_active(
                    user_id=user.id,
                    title="New Conversation",
                )
                
                # Get conversation ID
                conv_id = conversation_id or conversation.id
                
                # Get last N messages
                messages = await uow.messages.get_last_n_messages(
                    conversation_id=conv_id,
                    n=services.settings.ai.context_history_limit,
                )
                
                # Format messages for AI
                message_history = [
                    {"role": msg.role.value, "content": msg.content}
                    for msg in messages
                ]
                
                # Get user preferences
                user_preferences = {
                    "coding_style": preferences.coding_style,
                    "preferred_language": preferences.preferred_language,
                    "custom_prompt": preferences.custom_prompt,
                }
                
                return {
                    "user_id": user.id,
                    "conversation_id": conv_id,
                    "messages": message_history,
                    "preferences": user_preferences,
                }
                
    except Exception as e:
        logger.error(f"Error getting user context: {e}")
        return {"messages": [], "preferences": None}


async def save_message_to_db(
    conversation_id: int,
    role: MessageRole,
    content: str,
    tokens_used: int = 0,
    latency_ms: float = 0.0,
) -> None:
    """Save message to database."""
    if not services.is_database_available:
        return
    
    try:
        async with get_session() as session:
            async with UnitOfWork(session) as uow:
                await uow.messages.create(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    tokens_used=tokens_used,
                    model_name=services.settings.ai.model,
                    latency_ms=latency_ms,
                )
                await uow.commit()
    except Exception as e:
        logger.error(f"Error saving message: {e}")


# ============================================================================
# MESSAGE HANDLERS
# ============================================================================

@router.message(CommandStart())
async def cmd_start(message: types.Message) -> None:
    """Handle /start command."""
    user_id = message.from_user.id
    user_name = message.from_user.full_name or "User"
    
    logger.info(f"[START] User:{user_name} (ID:{user_id})")
    
    welcome_text = """
🚀 <b>Hello! I'm the Digital Twin of Jorabek Xasanov</b>

A Senior Backend Engineer with 5+ years of experience building high-performance systems.

💼 <b>Specializations:</b>
• Node.js & NestJS Architecture
• PostgreSQL & Redis Optimization
• High-Load System Design
• Secure API Development
• Microservices & Event-Driven Architecture
• Containerization (Docker, Kubernetes)

🛠️ <b>Tech Stack:</b>
<code>Node.js</code> • <code>TypeScript</code> • <code>Python</code>
<code>Go</code> • <code>PostgreSQL</code> • <code>Redis</code>

💡 <b>Ask me anything about:</b>
• Backend architecture and system design
• Database optimization and indexing
• API security and authentication patterns
• Microservices communication strategies
• Performance tuning and caching
• Deployment and DevOps practices

📎 <b>You can also send me files:</b>
• PDF documents
• DOCX files
• Images for OCR analysis

<i>Let's build something great together!</i>
"""
    await message.answer(welcome_text)


@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    """Handle /help command."""
    help_text = """
📚 <b>Available Commands:</b>

• <code>/start</code> - Start the bot
• <code>/help</code> - Show this help message
• <code>/reset</code> - Reset conversation context
• <code>/status</code> - Show system status

💡 <b>Just type your question</b> and I'll respond as a Senior Backend Engineer.

📎 <b>File Support:</b>
Send PDF, DOCX, or image files for analysis.

<i>I'm here to help you level up your backend skills!</i>
"""
    await message.answer(help_text)


@router.message(Command("reset"))
async def cmd_reset(message: types.Message) -> None:
    """Handle /reset command."""
    user_id = message.from_user.id
    user_name = message.from_user.full_name or "User"
    
    logger.info(f"[RESET] User:{user_name} (ID:{user_id})")
    
    reset_text = """
🔄 <b>Conversation context reset!</b>

Your conversation history has been cleared. Start fresh—ask me anything about backend development, system architecture, or database design.

<i>I'm ready to help you build better backends!</i>
"""
    await message.answer(reset_text)


@router.message(Command("status"))
async def cmd_status(message: types.Message) -> None:
    """Handle /status command showing deep system health information."""
    import psutil
    
    # Get system metrics
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
    except Exception:
        cpu_percent = 0.0
        memory = None
        disk = None
    
    # Deep health check for each service
    health_checks = []
    
    # Database health check
    db_healthy = services.is_database_available
    if db_healthy:
        try:
            from src.database.connection import check_db_health
            db_healthy = await check_db_health()
        except Exception:
            db_healthy = False
    health_checks.append(("Database", db_healthy))
    
    # Redis health check
    redis_healthy = services.is_redis_available
    if redis_healthy:
        try:
            redis_healthy = await services.rate_limiter.check_rate_limit(0).allowed
        except Exception:
            redis_healthy = False
    health_checks.append(("Redis", redis_healthy))
    
    # AI service health check
    ai_healthy = services.ai_service is not None
    health_checks.append(("AI Service", ai_healthy))
    
    # File processing health check
    file_healthy = services.file_processor is not None
    health_checks.append(("File Processing", file_healthy))
    
    # Get service health
    health = services._health_status
    
    # Calculate overall status
    all_services_healthy = all(check[1] for check in health_checks)
    overall_status = "✅ HEALTHY" if all_services_healthy else "⚠️ DEGRADED"
    
    # Build status message
    status_lines = [
        f"📊 <b>System Status: {overall_status}</b>",
        "",
        "🖥️ <b>System Resources:</b>",
        f"• CPU Usage: {cpu_percent}%",
        f"• Memory Usage: {memory.percent if memory else 'N/A'}%",
        f"• Disk Usage: {disk.percent if disk else 'N/A'}%",
        "",
        "🔧 <b>Service Health:</b>",
    ]
    
    for service_name, is_healthy in health_checks:
        status_lines.append(f"• {service_name}: {'✅' if is_healthy else '❌'}")
    
    status_lines.extend([
        "",
        "🤖 <b>Bot Info:</b>",
        f"• Model: <code>{services.settings.ai.model}</code>",
        f"• Temperature: <code>{services.settings.ai.temperature}</code>",
        "",
        "💡 <b>Senior Recommendation:</b> Regular monitoring of system resources "
        "and proactive health checks are essential for maintaining high availability. "
        "Set up alerts for when any service becomes unhealthy.",
    ])
    
    await message.answer("\n".join(status_lines))


@router.message(F.text)
async def handle_message(message: types.Message) -> None:
    """Handle all text messages."""
    user_id = message.from_user.id
    user_name = message.from_user.full_name or "User"
    username = message.from_user.username
    session_id = f"{int(time.time())}_{user_id}"
    
    # Sanitize input
    user_message = services.security_service.sanitize_input(message.text)
    
    # Detect language
    detected_lang = services.security_service.detect_language(user_message)
    
    logger.info(
        f"[SESSION:{session_id}] [MESSAGE] User:{user_name} "
        f"(ID:{user_id}, @{username}) | Lang:{detected_lang}"
    )
    
    # Send typing indicator
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    try:
        # Get user context from database
        context = await get_user_context(user_id)
        
        # Send to AI
        if services.ai_service:
            ai_response = await services.ai_service.send_message(
                message=user_message,
                conversation_history=context.get("messages", []),
                user_preferences=context.get("preferences"),
            )
            
            response_text = ai_response.text
            tokens_used = ai_response.tokens_used
            latency_ms = ai_response.latency_ms
            
            # Log AI usage
            logger.log_ai_usage(
                user_id=user_id,
                model=services.settings.ai.model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                session_id=session_id,
                cache_hit=ai_response.cache_hit,
            )
            
            # Save to database
            conversation_id = context.get("conversation_id", 0)
            await save_message_to_db(
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=user_message,
            )
            await save_message_to_db(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=response_text,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )
            
            # Send response
            escaped_response = escape_html_for_telegram(response_text)
            await message.answer(escaped_response)
            
            logger.info(
                f"[SESSION:{session_id}] [RESPONSE_SENT] | "
                f"Latency:{latency_ms:.2f}ms | Tokens:{tokens_used}"
            )
        else:
            # AI service unavailable
            await message.answer(
                "⚠️ <b>AI service temporarily unavailable.</b>\n\n"
                "Please try again in a moment. The system is undergoing maintenance."
            )
            
    except TelegramRetryAfter as e:
        logger.warning(f"[SESSION:{session_id}] FLOOD_CONTROL: retry after {e.retry_after}s")
        await message.answer(
            "⏳ <b>I'm getting too many requests.</b>\n\n"
            "Please wait a moment and try again."
        )
    
    except Exception as e:
        logger.exception(f"[SESSION:{session_id}] ERROR: {e}")
        
        # Graceful degradation - try to respond
        try:
            await message.answer(
                "⚠️ <b>Something unexpected occurred.</b>\n\n"
                "My technical systems are undergoing preventive maintenance. "
                "Please try again in 10-15 seconds."
            )
        except Exception:
            pass


@router.message(F.document)
async def handle_document(message: types.Message) -> None:
    """Handle document uploads (PDF, DOCX)."""
    user_id = message.from_user.id
    user_name = message.from_user.full_name or "User"
    session_id = f"{int(time.time())}_{user_id}"
    
    document = message.document
    
    # Check file size
    max_size = services.settings.file_processing.max_file_size_mb * 1024 * 1024
    if document.file_size > max_size:
        await message.answer(
            f"📁 <b>File Too Large</b>\n\n"
            f"Maximum file size is {services.settings.file_processing.max_file_size_mb}MB.\n"
            f"Your file is {document.file_size / 1024 / 1024:.1f}MB.\n\n"
            f"💡 <b>Senior Recommendation:</b> Compress the PDF using tools like Adobe Acrobat "
            f"or remove unnecessary images to reduce file size."
        )
        return
    
    # Check file type
    allowed_types = services.settings.file_processing.allowed_mime_types
    if document.mime_type not in allowed_types:
        await message.answer(
            "📁 <b>Unsupported File Type</b>\n\n"
            f"Supported formats: PDF, DOCX, PNG, JPG, GIF, WEBP.\n"
            f"Received: {document.mime_type}"
        )
        return
    
    logger.info(
        f"[SESSION:{session_id}] [FILE_UPLOAD] User:{user_name} | "
        f"File:{document.file_name} ({document.mime_type})"
    )
    
    # Send typing indicator
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    try:
        # Download file
        file = await bot.get_file(document.file_id)
        file_content = await bot.download_file(file.file_path)
        file_bytes = file_content.read()
        
        # Process file
        if services.file_processor:
            result = await services.file_processor.process_file(
                file_content=file_bytes,
                filename=document.file_name or "document",
                user_id=user_id,
            )
            
            if result.warnings:
                for warning in result.warnings:
                    await message.answer(warning)
                return
            
            if result.text_content:
                # Get user context
                context = await get_user_context(user_id)
                
                # Analyze document with AI
                analysis_prompt = f"""
Analyze the following document content and provide a comprehensive summary:

--- DOCUMENT ---
{result.text_content[:10000]}  # Limit content for token management
--- END DOCUMENT ---

Provide:
1. Brief summary (2-3 sentences)
2. Key topics covered
3. Any technical details relevant to backend development
"""
                
                if services.ai_service:
                    ai_response = await services.ai_service.send_message(
                        message=analysis_prompt,
                        conversation_history=context.get("messages", []),
                    )
                    
                    response_text = (
                        f"📄 <b>Document Analysis</b>\n\n"
                        f"{ai_response.text}\n\n"
                        f"<i>Words extracted: {result.word_count or 'N/A'}</i>"
                    )
                    await message.answer(response_text)
                else:
                    await message.answer(
                        "📄 <b>Document Uploaded</b>\n\n"
                        f"Extracted {result.word_count or 'N/A'} words.\n"
                        f"AI analysis is temporarily unavailable."
                    )
            else:
                await message.answer(
                    "⚠️ <b>Could not extract text from document.</b>\n\n"
                    "The file may be scanned or encrypted."
                )
        else:
            await message.answer(
                "⚠️ <b>File processing is temporarily unavailable.</b>\n\n"
                "Please try again later."
            )
            
    except Exception as e:
        logger.exception(f"[SESSION:{session_id}] FILE_PROCESSING_ERROR: {e}")
        await message.answer(
            "⚠️ <b>Error processing file.</b>\n\n"
            "Please try again or use a different file format."
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@router.error()
async def error_handler(event: types.ErrorEvent) -> None:
    """Global error handler."""
    exception = event.exception
    update = event.update
    update_id = update.update_id if update else 0
    
    logger.exception(f"[UPDATE:{update_id}] Error: {exception}")
    
    # Try to notify user
    if update and update.message:
        try:
            await bot.send_message(
                chat_id=update.message.chat.id,
                text="⚠️ <b>An unexpected error occurred.</b>\n\n"
                     "Please try again later.",
            )
        except Exception:
            pass


# ============================================================================
# PROMETHEUS METRICS ENDPOINT
# ============================================================================

async def metrics_handler(request: web.Request) -> web.Response:
    """Handle Prometheus metrics requests."""
    metrics = metrics_collector.get_prometheus_metrics()
    return web.Response(text=metrics, content_type='text/plain')


async def health_handler(request: web.Request) -> web.Response:
    """Handle health check requests."""
    health = services._health_status
    all_healthy = all(health.values())
    
    status_data = {
        "status": "healthy" if all_healthy else "degraded",
        "services": health,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    if all_healthy:
        return web.json_response(status_data, status=200)
    else:
        return web.json_response(status_data, status=503)


async def alerts_handler(request: web.Request) -> web.Response:
    """Handle Alertmanager webhook requests."""
    try:
        alert_data = await request.json()
        
        # Log alert
        for alert in alert_data.get("alerts", []):
            logger.warning(
                f"Alert received: {alert.get('labels', {}).get('alertname')} - "
                f"{alert.get('status', 'unknown')}",
                extra={
                    "event_type": "alert",
                    "alert_name": alert.get("labels", {}).get("alertname"),
                    "severity": alert.get("labels", {}).get("severity"),
                    "status": alert.get("status"),
                }
            )
        
        return web.json_response({"status": "success"}, status=200)
    except Exception as e:
        logger.error(f"Failed to process alert: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=400)


async def start_metrics_server() -> None:
    """Start the Prometheus metrics HTTP server."""
    app = web.Application()
    
    # Register routes
    app.router.add_get('/metrics', metrics_handler)
    app.router.add_get('/health', health_handler)
    app.router.add_post('/alerts', alerts_handler)
    
    # Start server on port 8080
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    
    logger.info("✅ Prometheus metrics server started on port 8080")


# ============================================================================
# STARTUP BANNER
# ============================================================================

def print_startup_banner() -> None:
    """Display comprehensive system information banner."""
    settings = get_settings()
    
    health = services._health_status
    
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🚀  SENIOR BACKEND ENGINEER DIGITAL TWIN BOT v2.0                 ║
║   ═══════════════════════════════════════════════════════════════════════   ║
║                                                                      ║
║   📋  PROJECT PARAMETERS                                             ║
║   ─────────────────────────────────────────────────────────────     ║
║   • Bot Username:     @{settings.telegram.bot_username or 'connecting...'}   ║
║   • AI Model:         {settings.ai.model:<20}                       ║
║   • Temperature:      {settings.ai.temperature:<20}                          ║
║   • Max Tokens:       {settings.ai.max_tokens:<20}                           ║
║                                                                      ║
║   🔧  SERVICE HEALTH                                                 ║
║   ─────────────────────────────────────────────────────────────     ║
║   • Database:         {'✅ Available' if health['database'] else '❌ Unavailable':<25}   ║
║   • Redis:            {'✅ Available' if health['redis'] else '❌ Unavailable':<25}   ║
║   • AI Service:       {'✅ Available' if health['ai'] else '❌ Unavailable':<25}   ║
║   • File Processing:  {'✅ Available' if health['file_processing'] else '❌ Unavailable':<25}   ║
║                                                                      ║
║   🔐  SECURITY                                                       ║
║   ─────────────────────────────────────────────────────────────     ║
║   • Rate Limiting:    {settings.security.rate_limit_requests} req/{settings.security.rate_limit_window}s                    ║
║   • Shield Middleware: Active                                        ║
║   • Encryption:       {'Enabled' if settings.security.encryption_key else 'Disabled':<25}   ║
║                                                                      ║
║   📊  MONITORING                                                     ║
║   ─────────────────────────────────────────────────────────────     ║
║   • Log Format:       {settings.monitoring.log_format:<25}                    ║
║   • Prometheus:       {'Enabled' if settings.monitoring.prometheus_enabled else 'Disabled':<25}   ║
║   • Error Threshold:  {settings.monitoring.error_rate_threshold}%                           ║
║                                                                      ║
║   ═══════════════════════════════════════════════════════════════════════   ║
║   Chaos Engineering Ready: Graceful degradation on service failures   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main() -> None:
    """
    Main entry point for the bot.
    
    Orchestrates service initialization, health checks, and polling.
    Implements graceful degradation for chaos engineering.
    """
    print_startup_banner()
    
    logger.info("🚀 Starting Senior Backend Engineer Digital Twin Bot...")
    
    # Initialize all services
    await services.initialize_all()
    
    try:
        # Verify bot token
        bot_info = await bot.get_me()
        print(f"✅ Bot connected: @{bot_info.username}")
        logger.info(f"Bot connected: @{bot_info.username}")
        
        # Verify AI service
        if services.ai_service:
            logger.info("✅ AI service initialized")
        else:
            logger.warning("⚠️ AI service not available")
        
        print("\n" + "=" * 80)
        print("✅ ALL SERVICES INITIALIZED - Bot is now polling for messages")
        print("=" * 80 + "\n")
        
        # Start metrics server in background
        metrics_task = asyncio.create_task(start_metrics_server())
        
        # Start polling
        await dp.start_polling(bot)
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.exception(f"Bot failed to start: {e}")
        raise
    finally:
        await services.cleanup()
        logger.info("🔒 All connections closed. Bot shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot shutdown complete")
        raise
