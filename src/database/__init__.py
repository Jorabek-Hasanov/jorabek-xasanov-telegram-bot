# Database Module
# ============================================================================
# PostgreSQL database layer with SQLAlchemy 2.0
# Implements Users, Conversations, and Messages tables
# ============================================================================

from .connection import (
    AsyncSessionLocal,
    async_session_factory,
    get_engine,
    get_session,
    init_db,
)
from .models import (
    Base,
    Conversation,
    Message,
    MessageRole,
    User,
    UserPreference,
)
from .repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
    UnitOfWork,
)

__all__ = [
    # Connection management
    "get_engine",
    "async_session_factory",
    "AsyncSessionLocal",
    "get_session",
    "init_db",
    # Models
    "Base",
    "User",
    "Conversation",
    "Message",
    "MessageRole",
    "UserPreference",
    # Repositories
    "UserRepository",
    "ConversationRepository",
    "MessageRepository",
    "UnitOfWork",
]
