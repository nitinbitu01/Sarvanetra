# sentinel/config.py
"""
sentinel/config.py — Enterprise configuration management for Sentinel Gujarat (Day 16.2).
"""

import os
import secrets
from functools import lru_cache
from typing import List, Literal, Optional, Union

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Environment ───────────────────────────────────────────────────────────
    ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # ── Database ──────────────────────────────────────────────────────────────
    # Supports postgresql+asyncpg:// or sqlite+aiosqlite://
    DATABASE_URL: str = "sqlite+aiosqlite:///./output/sentinel_async.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 1800  # seconds — recycle before TCP keepalive fails

    # ── JWT Auth ──────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: SecretStr = SecretStr(
        os.environ.get("JWT_SECRET_KEY", "c8d0e7f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9")
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── Security ──────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    BCRYPT_ROUNDS: int = 12
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_AUTH_PER_MINUTE: int = 10

    # ── Feedback ──────────────────────────────────────────────────────────────
    FALSE_ALARM_FLAG_THRESHOLD: int = 3

    # ── HLS Streaming ────────────────────────────────────────────────────────
    HLS_SEGMENT_DURATION: int = 2
    HLS_LIST_SIZE: int = 3
    HLS_DIR: str = "hls"

    # ── Data Retention ────────────────────────────────────────────────────────
    FOOTAGE_RETENTION_DAYS: int = 30
    ALERT_RETENTION_DAYS: int = 365
    AUDIT_LOG_RETENTION_DAYS: int = 2555  # 7 years
    PUSH_LOG_RETENTION_DAYS: int = 90
    EVIDENCE_LOCKED_RETENTION_DAYS: int = 1825  # 5 years

    # ── Analytics ────────────────────────────────────────────────────────────
    ANALYTICS_HOURLY_SNAPSHOT_ENABLED: bool = True
    ANALYTICS_DAILY_SNAPSHOT_ENABLED: bool = True

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"
    EXTERNAL_LOG_SINK_URL: Optional[str] = None

    # ── VAPID (Day 15) ────────────────────────────────────────────────────────
    VAPID_PUBLIC_KEY: str = ""
    VAPID_PRIVATE_KEY: str = ""
    VAPID_CLAIM_EMAIL: str = "admin@sentinel.local"

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: SecretStr) -> SecretStr:
        raw = v.get_secret_value()
        if len(raw) < 64:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 64 characters. "
                "Generate: python -c \"import secrets; print(secrets.token_hex(64))\""
            )
        return v

    @field_validator("ENV")
    @classmethod
    def validate_production_settings(cls, v: str, info) -> str:
        if v == "production":
            data = info.data
            if data.get("DEBUG"):
                raise ValueError("DEBUG must be False in production")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
