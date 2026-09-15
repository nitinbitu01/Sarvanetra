"""
backend/auth/reviewer_auth.py — JWT Authentication & Reviewer Token Verification.

Provides:
  - create_access_token: Encodes HS256 JWT token with roles.
  - get_current_reviewer: FastAPI dependency extracting and verifying ReviewerToken.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

SECRET_KEY = os.getenv("SENTINEL_JWT_SECRET", "sentinel_gujarat_production_secret_key_2026_x99_sec")
ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 8

bearer_scheme = HTTPBearer(auto_error=False)


class ReviewerToken(BaseModel):
    reviewer_id: str
    username: str
    roles: list[str]


def create_access_token(
    reviewer_id: str,
    username: str,
    roles: list[str] | None = None,
) -> str:
    """Generates signed JWT token for a reviewer."""
    payload = {
        "sub": reviewer_id,
        "username": username,
        "roles": roles or ["reviewer"],
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_reviewer(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> ReviewerToken:
    """FastAPI dependency extracting verified reviewer from JWT token."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired reviewer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not creds or not creds.credentials:
        raise credentials_exc

    token = creds.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        reviewer_id: Optional[str] = payload.get("sub")
        username: Optional[str] = payload.get("username")
        roles: list[str] = payload.get("roles", [])
        if reviewer_id is None:
            raise credentials_exc
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.PyJWTError:
        raise credentials_exc

    valid_roles = {"reviewer", "admin", "OPERATOR", "ADMIN", "officer"}
    if not any(r in valid_roles for r in roles):
        raise HTTPException(status_code=403, detail="Insufficient role.")

    return ReviewerToken(reviewer_id=reviewer_id, username=username or reviewer_id, roles=roles)
