# sentinel/db.py
"""
sentinel/db.py — Async database session and engine management (PostgreSQL + SQLite async fallback).
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, StaticPool

from sentinel.config import settings

log = logging.getLogger(__name__)

db_url = str(settings.DATABASE_URL)

# ── Engine ────────────────────────────────────────────────────────────────────
engine_kwargs = {
    "echo": settings.DEBUG,
}

if "sqlite" in db_url:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
    engine_kwargs["poolclass"] = StaticPool
else:
    engine_kwargs["pool_size"] = settings.DATABASE_POOL_SIZE
    engine_kwargs["max_overflow"] = settings.DATABASE_MAX_OVERFLOW
    engine_kwargs["pool_timeout"] = settings.DATABASE_POOL_TIMEOUT
    engine_kwargs["pool_recycle"] = settings.DATABASE_POOL_RECYCLE
    engine_kwargs["pool_pre_ping"] = True

engine = create_async_engine(db_url, **engine_kwargs)

# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Dependency ────────────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency. Yields an async session.
    Commits on success, rolls back on any exception.
    Session is always closed — no connection leak.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
