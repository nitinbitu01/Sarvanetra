# sentinel/auth/service.py
"""
sentinel/auth/service.py — JWT token management, refresh token rotation, and bcrypt password hashing.
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

import structlog
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from sentinel.config import settings
from sentinel.models import Officer, RefreshToken

log = structlog.get_logger(__name__)

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.BCRYPT_ROUNDS,
)


# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False


# ── Access token ──────────────────────────────────────────────────────────────

def create_access_token(officer_id: int, role: str) -> str:
    """
    Short-lived JWT (15 minutes default).
    Contains: officer_id, role, expiry, jti.
    """
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(officer_id),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "jti": secrets.token_hex(16),
    }
    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY.get_secret_value(),
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict:
    """
    Decode and validate access token.
    Raises JWTError on invalid/expired token.
    """
    return jwt.decode(
        token,
        settings.JWT_SECRET_KEY.get_secret_value(),
        algorithms=[settings.JWT_ALGORITHM],
    )


# ── Refresh token ─────────────────────────────────────────────────────────────

def _hash_token(raw_token: str) -> str:
    """SHA-256 of raw token. Only hash is stored in DB."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


async def create_refresh_token(
    officer_id: int,
    ip_address: Optional[str],
    family_id: Optional[str],
    db: AsyncSession,
) -> str:
    """
    Create a new refresh token.
    Returns raw token — caller sets it as HTTP-only cookie.
    """
    raw_token = secrets.token_urlsafe(64)
    token_hash = _hash_token(raw_token)
    family = family_id or str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    )

    db.add(RefreshToken(
        officer_id=officer_id,
        token_hash=token_hash,
        family_id=family,
        revoked=False,
        expires_at=expires_at,
        ip_address=ip_address,
    ))
    await db.flush()

    log.info("refresh_token.created",
             officer_id=officer_id, family_id=family, ip=ip_address)
    return raw_token


async def rotate_refresh_token(
    raw_token: str,
    ip_address: Optional[str],
    db: AsyncSession,
) -> Tuple[str, int, str]:
    """
    Verify + rotate a refresh token.
    Returns (new_raw_token, officer_id, role_name).

    Token reuse detection:
    If presented token is already revoked, revoke the entire family.
    """
    token_hash = _hash_token(raw_token)
    now = datetime.now(timezone.utc)

    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    token_row = result.scalar_one_or_none()

    if not token_row:
        log.warning("refresh_token.not_found", token_hash=token_hash[:8])
        raise ValueError("Invalid refresh token")

    if token_row.revoked:
        # Token reuse detected — revoke entire family
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.family_id == token_row.family_id)
            .values(revoked=True)
        )
        await db.flush()
        log.critical("refresh_token.reuse_detected",
                     officer_id=token_row.officer_id,
                     family_id=token_row.family_id,
                     ip=ip_address)
        raise ValueError("Token reuse detected — all sessions revoked")

    exp = token_row.expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    if exp < now:
        log.info("refresh_token.expired", officer_id=token_row.officer_id)
        raise ValueError("Refresh token expired")

    # Revoke the current token (rotation)
    token_row.revoked = True

    # Fetch officer for role info
    officer_result = await db.execute(
        select(Officer).options(selectinload(Officer.role)).where(Officer.id == token_row.officer_id)
    )
    officer = officer_result.scalar_one_or_none()
    if not officer or not officer.is_active:
        raise ValueError("Officer not found or suspended")

    await db.flush()

    # Issue new token in same family
    new_raw = await create_refresh_token(
        officer_id=officer.id,
        ip_address=ip_address,
        family_id=token_row.family_id,
        db=db,
    )

    role_name = officer.role.name if officer.role else "officer"
    return new_raw, officer.id, role_name


async def revoke_all_sessions(officer_id: int, db: AsyncSession) -> None:
    """Revoke all refresh tokens for an officer."""
    await db.execute(
        update(RefreshToken)
        .where(
            RefreshToken.officer_id == officer_id,
            RefreshToken.revoked == False,
        )
        .values(revoked=True)
    )
    await db.flush()
    log.info("refresh_token.all_revoked", officer_id=officer_id)
