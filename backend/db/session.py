import os
from contextlib import asynccontextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.db.models import Base

# Sync database setup (SQLite / PostgreSQL fallback)
SYNC_DB_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///output/sentinel.db"
)

# Async database setup
ASYNC_DB_URL = os.getenv(
    "POSTGRES_URL",
    "sqlite+aiosqlite:///output/sentinel.db"
)

# Sync engine
sync_connect_args = {"check_same_thread": False, "timeout": 60.0} if "sqlite" in SYNC_DB_URL else {}
engine = create_engine(
    SYNC_DB_URL,
    connect_args=sync_connect_args,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)

if "sqlite" in SYNC_DB_URL:
    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

# Async engine
async_connect_args = {"check_same_thread": False, "timeout": 60.0} if "sqlite" in ASYNC_DB_URL else {}
try:
    async_engine = create_async_engine(
        ASYNC_DB_URL,
        connect_args=async_connect_args,
        pool_pre_ping=True,
        echo=False,
    )
    AsyncSessionLocal = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
except Exception:
    async_engine = None
    AsyncSessionLocal = None


def init_db(drop_first: bool = False):
    """Create all tables in sync and async database and ensure extensions & columns exist"""
    if drop_first:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    try:
        with engine.connect() as conn:
            if "sqlite" in SYNC_DB_URL:
                try:
                    conn.execute(text("PRAGMA journal_mode=WAL;"))
                    conn.execute(text("PRAGMA synchronous=NORMAL;"))
                    conn.execute(text("PRAGMA cache_size=10000;"))
                    conn.execute(text("PRAGMA busy_timeout=5000;"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_journeys_camera_cluster ON journeys(camera_id, global_person_id, seen_at DESC);"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_review_pending ON reid_review_queue(status, created_at);"))
                    conn.commit()
                except Exception:
                    pass

            if "postgres" in SYNC_DB_URL or "postgresql" in SYNC_DB_URL:
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS btree_gin"))
                    conn.commit()
                except Exception:
                    pass

            audit_cols = [
                ("event_type", "VARCHAR(64)"),
                ("entity_id", "VARCHAR(64)"),
                ("entity_type", "VARCHAR(32)"),
                ("actor_id", "VARCHAR(64)"),
                ("model_version", "VARCHAR(32)"),
                ("payload", "JSONB" if "postgres" in SYNC_DB_URL else "JSON"),
                ("payload_hash", "VARCHAR(64)"),
                ("prev_log_hash", "VARCHAR(64)"),
                ("logged_at", "TIMESTAMPTZ" if "postgres" in SYNC_DB_URL else "TIMESTAMP"),
            ]
            for col_name, col_type in audit_cols:
                try:
                    conn.execute(text(f"ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS {col_name} {col_type}"))
                    conn.commit()
                except Exception:
                    pass

            calib_cols = [
                ("gps_anchor_lat", "FLOAT"),
                ("gps_anchor_lon", "FLOAT"),
                ("bearing_deg", "FLOAT DEFAULT 0.0"),
                ("held_out_error_m", "FLOAT"),
                ("quality_gate", "VARCHAR(16) DEFAULT 'good'"),
                ("notes", "TEXT"),
                # `quality_gate` alone conflates two different claims and a
                # calibration can hold one without the other. Surveyed ground
                # control points give POSITION; cross-camera journey legs give
                # SPEED. Reading a single gate led to rows whose stored value
                # said `degraded` while recomputing it from the error said
                # `rejected` — both correct about different things.
                #
                # These two are explicit so nothing has to be inferred from the
                # error figure, and so a writer that recomputes `quality_gate`
                # cannot silently flip a speed-validated camera off.
                ("positional_error_is_measured", "BOOLEAN DEFAULT 0"),
                ("speed_validated", "BOOLEAN DEFAULT 0"),
                ("calibration_method", "VARCHAR(48)"),
            ]
            for col_name, col_type in calib_cols:
                try:
                    conn.execute(text(f"ALTER TABLE camera_calibrations ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                except Exception:
                    pass
    except Exception:
        pass


@asynccontextmanager
async def get_db_session():
    if AsyncSessionLocal:
        try:
            async with AsyncSessionLocal() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise
                finally:
                    await session.close()
                return
        except Exception:
            pass

    # Fallback to sync session in threadpool
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_db_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
