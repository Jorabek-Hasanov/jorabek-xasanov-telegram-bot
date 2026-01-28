# ============================================================================
# Initial Database Migration
# ============================================================================
# Creates all tables for the Telegram Bot with proper indexes
# ============================================================================

from typing import Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "000000000001"
down_revision: Union[str, None] = None
branch_labels: tuple[str, ...] = ("bot", "core")
depends_on: tuple[str, ...] = ()


def upgrade() -> None:
    """Create all database tables with B-Tree indexes."""
    
    # Create ENUM types
    op.execute("CREATE TYPE message_role_enum AS ENUM ('user', 'assistant', 'system')")
    
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("telegram_id", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.String(255), nullable=True),
        sa.Column("first_name", sa.String(255), nullable=False),
        sa.Column("last_name", sa.String(255), nullable=True),
        sa.Column("language_code", sa.String(10), nullable=True),
        sa.Column("is_premium", sa.Boolean(), default=False, nullable=False),
        sa.Column("preferences_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    
    # B-Tree index on telegram_id for fast lookups
    op.create_index("idx_users_telegram_id_btree", "users", ["telegram_id"], postgresql_using="btree")
    op.create_index("idx_users_created_at", "users", ["created_at"], postgresql_using="btree")
    
    # Create user_preferences table
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("coding_style", sa.String(500), nullable=True),
        sa.Column("preferred_language", sa.String(10), default="English", nullable=False),
        sa.Column("notification_enabled", sa.Boolean(), default=True, nullable=False),
        sa.Column("max_context_messages", sa.Integer(), default=15, nullable=False),
        sa.Column("custom_prompt", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )
    
    # Create conversations table
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(500), default="New Conversation", nullable=False),
        sa.Column("session_uuid", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.Column("message_count", sa.Integer(), default=0, nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("last_summary_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    
    # B-Tree indexes for conversations
    op.create_index("idx_conversations_user_id_btree", "conversations", ["user_id"], postgresql_using="btree")
    op.create_index("idx_conversations_created_at", "conversations", ["created_at"], postgresql_using="btree")
    op.create_index("idx_conversations_session_uuid", "conversations", ["session_uuid"], postgresql_using="btree")
    op.create_index("idx_conversations_updated_at", "conversations", ["updated_at"], postgresql_using="btree")
    
    # Create messages table
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.Enum("user", "assistant", "system", name="message_role_enum", create_type=False), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tokens_used", sa.Integer(), default=0, nullable=False),
        sa.Column("model_name", sa.String(100), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("is_summarized", sa.Boolean(), default=False, nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    
    # B-Tree indexes for messages
    op.create_index("idx_messages_conversation_id_btree", "messages", ["conversation_id"], postgresql_using="btree")
    op.create_index("idx_messages_created_at", "messages", ["created_at"], postgresql_using="btree")
    op.create_index("idx_messages_role", "messages", ["role"], postgresql_using="btree")
    op.create_index("idx_messages_conversation_created", "messages", ["conversation_id", "created_at"], postgresql_using="btree")


def downgrade() -> None:
    """Drop all tables and indexes."""
    
    # Drop in reverse order due to foreign keys
    op.drop_index("idx_messages_conversation_created", table_name="messages", postgresql_using="btree")
    op.drop_index("idx_messages_role", table_name="messages", postgresql_using="btree")
    op.drop_index("idx_messages_created_at", table_name="messages", postgresql_using="btree")
    op.drop_index("idx_messages_conversation_id_btree", table_name="messages", postgresql_using="btree")
    op.drop_table("messages")
    
    op.drop_index("idx_conversations_updated_at", table_name="conversations", postgresql_using="btree")
    op.drop_index("idx_conversations_session_uuid", table_name="conversations", postgresql_using="btree")
    op.drop_index("idx_conversations_created_at", table_name="conversations", postgresql_using="btree")
    op.drop_index("idx_conversations_user_id_btree", table_name="conversations", postgresql_using="btree")
    op.drop_table("conversations")
    
    op.drop_table("user_preferences")
    
    op.drop_index("idx_users_created_at", table_name="users", postgresql_using="btree")
    op.drop_index("idx_users_telegram_id_btree", table_name="users", postgresql_using="btree")
    op.drop_table("users")
    
    # Drop ENUM type
    op.execute("DROP TYPE IF EXISTS message_role_enum")
