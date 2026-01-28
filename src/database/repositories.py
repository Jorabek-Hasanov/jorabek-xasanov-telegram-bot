# Repository Layer
# ============================================================================
# Data access layer with repositories for each entity
# Implements Unit of Work pattern for transaction management
# ============================================================================

from datetime import datetime
from typing import List, Optional, Tuple
from uuid import uuid4

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import logger
from .models import Conversation, Message, MessageRole, User, UserPreference


class UserRepository:
    """Repository for User entity operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with session."""
        self.session = session
    
    async def get_by_telegram_id(self, telegram_id: int) -> Optional[User]:
        """Get user by Telegram ID."""
        query = select(User).where(User.telegram_id == telegram_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        query = select(User).where(User.id == user_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def create(
        self,
        telegram_id: int,
        username: Optional[str] = None,
        first_name: str = "",
        last_name: Optional[str] = None,
        language_code: Optional[str] = None,
        is_premium: bool = False,
    ) -> User:
        """Create a new user."""
        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            language_code=language_code,
            is_premium=is_premium,
        )
        self.session.add(user)
        await self.session.flush()
        logger.info(f"Created user: {user.telegram_id}")
        return user
    
    async def update(self, user: User) -> User:
        """Update user."""
        user.updated_at = datetime.now()
        await self.session.flush()
        return user
    
    async def get_or_create(
        self,
        telegram_id: int,
        username: Optional[str] = None,
        first_name: str = "",
        last_name: Optional[str] = None,
        language_code: Optional[str] = None,
        is_premium: bool = False,
    ) -> Tuple[User, bool]:
        """
        Get existing user or create new one.
        
        Returns:
            Tuple of (User, created_bool)
        """
        user = await self.get_by_telegram_id(telegram_id)
        if user:
            return user, False
        user = await self.create(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            language_code=language_code,
            is_premium=is_premium,
        )
        return user, True


class ConversationRepository:
    """Repository for Conversation entity operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with session."""
        self.session = session
    
    async def get_by_id(self, conversation_id: int) -> Optional[Conversation]:
        """Get conversation by ID."""
        query = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_uuid(self, session_uuid: str) -> Optional[Conversation]:
        """Get conversation by UUID."""
        query = select(Conversation).where(Conversation.session_uuid == session_uuid)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_user_conversations(
        self,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
        active_only: bool = True,
    ) -> List[Conversation]:
        """Get user's conversations with pagination."""
        query = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at))
        )
        if active_only:
            query = query.where(Conversation.is_active == True)
        
        query = query.offset(offset).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def create(
        self,
        user_id: int,
        title: Optional[str] = None,
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            user_id=user_id,
            title=title or "New Conversation",
            session_uuid=str(uuid4()),
        )
        self.session.add(conversation)
        await self.session.flush()
        logger.info(f"Created conversation: {conversation.id} for user {user_id}")
        return conversation
    
    async def get_or_create_active(
        self,
        user_id: int,
        title: Optional[str] = None,
    ) -> Conversation:
        """Get user's active conversation or create new one."""
        query = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .where(Conversation.is_active == True)
            .order_by(desc(Conversation.updated_at))
            .limit(1)
        )
        result = await self.session.execute(query)
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            conversation = await self.create(user_id, title)
        
        return conversation
    
    async def update_message_count(self, conversation_id: int) -> None:
        """Update message count for conversation."""
        conversation = await self.get_by_id(conversation_id)
        if conversation:
            conversation.message_count += 1
            conversation.updated_at = datetime.now()
            await self.session.flush()
    
    async def update_summary(
        self,
        conversation_id: int,
        summary: str,
    ) -> None:
        """Update conversation summary."""
        conversation = await self.get_by_id(conversation_id)
        if conversation:
            conversation.summary = summary
            conversation.last_summary_at = datetime.now()
            await self.session.flush()


class MessageRepository:
    """Repository for Message entity operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with session."""
        self.session = session
    
    async def get_by_id(self, message_id: int) -> Optional[Message]:
        """Get message by ID."""
        query = select(Message).where(Message.id == message_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_conversation_messages(
        self,
        conversation_id: int,
        limit: int = 50,
        offset: int = 0,
        exclude_summarized: bool = False,
    ) -> List[Message]:
        """Get messages for a conversation with pagination."""
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        if exclude_summarized:
            query = query.where(Message.is_summarized == False)
        
        query = query.offset(offset).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_last_n_messages(
        self,
        conversation_id: int,
        n: int = 20,
    ) -> List[Message]:
        """Get last N messages from a conversation."""
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(desc(Message.created_at))
            .limit(n)
        )
        result = await self.session.execute(query)
        messages = list(result.scalars().all())
        # Reverse to get chronological order
        return list(reversed(messages))
    
    async def get_conversation_history_for_ai(
        self,
        conversation_id: int,
        max_tokens: int = 8000,
    ) -> List[Message]:
        """
        Get conversation history optimized for AI context.
        
        Fetches messages and respects token limits.
        """
        messages = await self.get_conversation_messages(
            conversation_id=conversation_id,
            limit=100,
        )
        
        # Simple token estimation (avg 4 chars per token)
        total_chars = sum(len(m.content) for m in messages)
        max_chars = max_tokens * 4
        
        # If over limit, take only recent messages
        if total_chars > max_chars:
            cutoff = int(len(messages) * 0.7)  # Keep 70% of recent messages
            return messages[-cutoff:]
        
        return messages
    
    async def create(
        self,
        conversation_id: int,
        role: MessageRole,
        content: str,
        tokens_used: int = 0,
        model_name: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> Message:
        """Create a new message."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens_used=tokens_used,
            model_name=model_name,
            latency_ms=latency_ms,
        )
        self.session.add(message)
        await self.session.flush()
        
        # Update conversation message count
        await self.session.refresh(message)
        conversation_repo = ConversationRepository(self.session)
        await conversation_repo.update_message_count(conversation_id)
        
        return message
    
    async def mark_as_summarized(self, message_ids: List[int]) -> None:
        """Mark messages as summarized."""
        if not message_ids:
            return
        query = (
            select(Message)
            .where(Message.id.in_(message_ids))
        )
        result = await self.session.execute(query)
        for message in result.scalars().all():
            message.is_summarized = True
        await self.session.flush()
    
    async def count_conversation_messages(self, conversation_id: int) -> int:
        """Count total messages in a conversation using SQL COUNT().
        
        This is optimized for performance with millions of records.
        The COUNT query uses B-Tree index on conversation_id for fast lookup.
        """
        query = (
            select(func.count(Message.id))
            .where(Message.conversation_id == conversation_id)
        )
        result = await self.session.execute(query)
        return result.scalar() or 0


class UserPreferenceRepository:
    """Repository for UserPreference entity operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with session."""
        self.session = session
    
    async def get_by_user_id(self, user_id: int) -> Optional[UserPreference]:
        """Get preferences for a user."""
        query = select(UserPreference).where(UserPreference.user_id == user_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def create(
        self,
        user_id: int,
        coding_style: Optional[str] = None,
        preferred_language: str = "English",
        notification_enabled: bool = True,
        max_context_messages: int = 15,
        custom_prompt: Optional[str] = None,
    ) -> UserPreference:
        """Create user preferences."""
        preferences = UserPreference(
            user_id=user_id,
            coding_style=coding_style,
            preferred_language=preferred_language,
            notification_enabled=notification_enabled,
            max_context_messages=max_context_messages,
            custom_prompt=custom_prompt,
        )
        self.session.add(preferences)
        await self.session.flush()
        return preferences
    
    async def get_or_create(self, user_id: int) -> UserPreference:
        """Get existing preferences or create new ones."""
        preferences = await self.get_by_user_id(user_id)
        if not preferences:
            preferences = await self.create(user_id)
        return preferences
    
    async def update(
        self,
        preferences: UserPreference,
        coding_style: Optional[str] = None,
        preferred_language: Optional[str] = None,
        notification_enabled: Optional[bool] = None,
        max_context_messages: Optional[int] = None,
        custom_prompt: Optional[str] = None,
    ) -> UserPreference:
        """Update user preferences."""
        if coding_style is not None:
            preferences.coding_style = coding_style
        if preferred_language is not None:
            preferences.preferred_language = preferred_language
        if notification_enabled is not None:
            preferences.notification_enabled = notification_enabled
        if max_context_messages is not None:
            preferences.max_context_messages = max_context_messages
        if custom_prompt is not None:
            preferences.custom_prompt = custom_prompt
        
        preferences.updated_at = datetime.now()
        await self.session.flush()
        return preferences


class UnitOfWork:
    """
    Unit of Work pattern for transaction management.
    
    Provides atomic operations across multiple repositories.
    
    Example:
        ```python
        async with UnitOfWork() as uow:
            user = await uow.users.create(...)
            conversation = await uow.conversations.create(...)
            await uow.commit()
        ```
    """
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize UoW with session."""
        self.session = session
        self.users = UserRepository(session)
        self.conversations = ConversationRepository(session)
        self.messages = MessageRepository(session)
        self.preferences = UserPreferenceRepository(session)
    
    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()
    
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.session.rollback()
    
    async def __aenter__(self) -> "UnitOfWork":
        """Enter context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        if exc_type:
            await self.rollback()
