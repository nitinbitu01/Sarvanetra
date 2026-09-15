# sentinel/auth/router.py
"""
sentinel/auth/router.py — Authentication router for login, token refresh, and logout.
"""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from sentinel.auth.service import (
    create_access_token,
    create_refresh_token,
    rotate_refresh_token,
    revoke_all_sessions,
    verify_password,
    _hash_token,
)
from sentinel.config import settings
from sentinel.db import get_db
from sentinel.models import Officer, RefreshToken

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE_NAME = "sentinel_refresh"
COOKIE_SECURE = settings.ENV == "production"


def _set_refresh_cookie(response: Response, token: str) -> None:
    """
    Set HTTP-only refresh token cookie.
    """
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="strict",
        max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/",
    )


class LoginJSONRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str


@router.post("/login")
async def login(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """
    Officer login. Accepts both OAuth2 form data and JSON.
    Returns access token in response body and sets HTTP-only refresh cookie.
    """
    username = ""
    password = ""

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        username = body.get("username") or body.get("email") or ""
        password = body.get("password") or ""
    else:
        form = await request.form()
        username = form.get("username") or form.get("email") or ""
        password = form.get("password") or ""

    result = await db.execute(
        select(Officer).options(selectinload(Officer.role)).where(
            (Officer.email == username) | (Officer.name == username)
        )
    )
    officer = result.scalar_one_or_none()

    dummy_hash = "$2b$12$dummy_hash_for_timing_resistance_padding_value"
    valid = (
        officer is not None
        and officer.is_active
        and verify_password(password, officer.hashed_password or dummy_hash)
    )

    client_ip = request.client.host if request.client else "unknown"

    if not valid:
        log.warning("auth.login_failed", email=username, ip=client_ip)
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    role_name = officer.role.name if officer.role else "officer"
    access_token = create_access_token(officer.id, role_name)
    refresh_token = await create_refresh_token(
        officer_id=officer.id,
        ip_address=client_ip,
        family_id=None,
        db=db,
    )

    _set_refresh_cookie(response, refresh_token)

    log.info("auth.login_success",
             officer_id=officer.id,
             role=role_name,
             ip=client_ip)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "officer_id": officer.id,
        "officer_name": officer.name,
        "role": role_name,
    }


@router.post("/refresh")
async def refresh(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """
    Rotate refresh token and issue a fresh access token.
    """
    raw_token = request.cookies.get(REFRESH_COOKIE_NAME)
    if not raw_token:
        # Fallback to header or body if cookie not present
        body = await request.json().catch(lambda: {}) if request.headers.get("content-type") == "application/json" else {}
        raw_token = body.get("refresh_token")

    if not raw_token:
        raise HTTPException(status_code=401, detail="No refresh token provided")

    client_ip = request.client.host if request.client else "unknown"

    try:
        new_raw, officer_id, role = await rotate_refresh_token(
            raw_token, client_ip, db
        )
    except ValueError as e:
        response.delete_cookie(REFRESH_COOKIE_NAME, path="/")
        raise HTTPException(status_code=401, detail=str(e))

    access_token = create_access_token(officer_id, role)
    _set_refresh_cookie(response, new_raw)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "role": role,
    }


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Revoke all sessions for the authenticated refresh token."""
    raw_token = request.cookies.get(REFRESH_COOKIE_NAME)
    if raw_token:
        try:
            token_hash = _hash_token(raw_token)
            result = await db.execute(
                select(RefreshToken).where(RefreshToken.token_hash == token_hash)
            )
            token_row = result.scalar_one_or_none()
            if token_row:
                await revoke_all_sessions(token_row.officer_id, db)
        except Exception as e:
            log.warning("auth.logout_revocation_failed", error=str(e))

    response.delete_cookie(REFRESH_COOKIE_NAME, path="/")
    return {"message": "Logged out"}
