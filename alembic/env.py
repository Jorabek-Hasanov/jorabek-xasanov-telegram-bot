# ============================================================================
# Alembic Migration Environment Configuration
# ============================================================================
# Configures the migration environment for SQLAlchemy 2.0 with asyncpg
# ============================================================================

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.schema import MetaData

from alembic import context

from src.core.config import get_settings
from src.database.connection import Base

# Import all models to ensure they're registered with Base.metadata
from src.database.models import User, Conversation, Message, UserPreference

# Get configuration from context or use defaults
config = context.config

# Override sqlalchemy.url from environment
settings = get_settings()
db_settings = settings.database

# Set the database URL
config.set_main_option("sqlalchemy.url", db_settings.async_url)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Target metadata for migrations
target_metadata = Base.metadata


def get_url():
    """Get database URL from settings."""
    return db_settings.async_url


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here too.  By skipping the Engine creation
    we don't even need a DBAPI to be available.
    
    Calls to context.execute() here emit the given
    string to the script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations(conn: Connection) -> None:
    """
    Run async migrations with async connection.
    
    This is used for SQLAlchemy 2.0 async migrations.
    """
    context.configure(
        connection=conn,
        target_metadata=target_metadata,
        compare_type=True,
        include_schemas=True,
        render_as_batch=True,  # For PostgreSQL ALTER TABLE batching
    )

    async with context.begin_transaction():
        await context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Run migrations with provided connection.
    
    This handles both sync and async connections.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_schemas=True,
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    
    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async def run_async():
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)

    asyncio.run(run_async())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
