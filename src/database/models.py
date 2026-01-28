# Database Models
# ============================================================================
# SQLAlchemy 2.0 models for Users, Conversations, and Messages
# Implements proper indexing and relationships for context management
# ============================================================================

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .connection import Base


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class User(Base):
    """
    User model for storing user data.
    
    Attributes:
        id: Primary key
        telegram_id: Telegram user ID (unique)
        username: Telegram username
        first_name: User's first name
        last_name: User's last name
        language_code: User's language preference
        is_premium: Telegram premium status
        created_at: Creation timestamp
        updated_at: Last update timestamp
        is_active: Active status for the bot
    """
    
    __tablename__ = "users"
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Telegram user info
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    first_name: Mapped[str] = mapped_column(String(255), nullable=False)
    last_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    language_code: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    is_premium: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Preferences (stored as JSON-like text for simplicity)
    preferences_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    preferences: Mapped[Optional["UserPreference"]] = relationship(
        "UserPreference",
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False
    )
    
    # B-Tree index on telegram_id for fast user lookups
    __table_args__ = (
        Index("idx_users_telegram_id_btree", "telegram_id", postgresql_using="btree"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<User(id={self.id}, telegram_id={self.telegram_id}, username={self.username})>"


class Conversation(Base):
    """
    Conversation model for tracking chat sessions.
    
    Attributes:
        id: Primary key
        user_id: Foreign key to User
        title: Conversation title
        created_at: Creation timestamp
        updated_at: Last update timestamp
        is_active: Whether the conversation is active
        message_count: Number of messages in the conversation
    """
    
    __tablename__ = "conversations"
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Conversation info
    title: Mapped[str] = mapped_column(String(500), default="New Conversation")
    session_uuid: Mapped[str] = mapped_column(UUID(as_uuid=False), default=lambda: str(uuid4()), unique=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Summary for long conversations
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_summary_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    
    # Index for fast user conversations lookup
    __table_args__ = (
        Index("idx_conversations_user_id_btree", "user_id", postgresql_using="btree"),
        Index("idx_conversations_created_at", "created_at", postgresql_using="btree"),
        Index("idx_conversations_session_uuid", "session_uuid", postgresql_using="btree"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Conversation(id={self.id}, user_id={self.user_id}, title={self.title})>"


class Message(Base):
    """
    Message model for storing conversation history.
    
    Attributes:
        id: Primary key
        conversation_id: Foreign key to Conversation
        role: Message role (user/assistant/system)
        content: Message content
        tokens_used: Number of tokens used for this message
        created_at: Creation timestamp
        is_summarized: Whether this message was included in a summary
    """
    
    __tablename__ = "messages"
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Message content
    role: Mapped[MessageRole] = mapped_column(
        SQLEnum(MessageRole, name="message_role_enum", create_type=False),
        nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # AI metrics
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Metadata
    is_summarized: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
    
    # Indexes for fast message retrieval
    __table_args__ = (
        Index("idx_messages_conversation_id_btree", "conversation_id", postgresql_using="btree"),
        Index("idx_messages_created_at", "created_at", postgresql_using="btree"),
        Index("idx_messages_role", "role", postgresql_using="btree"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Message(id={self.id}, conversation_id={self.conversation_id}, role={self.role})>"


class UserPreference(Base):
    """
    User preference model for storing user-specific settings.
    
    Attributes:
        id: Primary key
        user_id: Foreign key to User (one-to-one)
        coding_style: User's preferred coding style
        preferred_language: User's preferred language for responses
        notification_enabled: Whether to send notifications
        max_context_messages: Max messages to keep in context
        custom_prompt: Custom system prompt additions
    """
    
    __tablename__ = "user_preferences"
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key (one-to-one with User)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )
    
    # Preference fields
    coding_style: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    preferred_language: Mapped[str] = mapped_column(String(10), default="English")
    notification_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    max_context_messages: Mapped[int] = mapped_column(Integer, default=15)
    custom_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="preferences")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<UserPreference(user_id={self.user_id}, language={self.preferred_language})>"
