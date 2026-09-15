# sentinel/auth/dependencies.py
"""
sentinel/auth/dependencies.py — RBAC decorators, role hierarchy, and token validation dependencies.
"""

from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from sentinel.auth.service import decode_access_token
from sentinel.db import get_db
from sentinel.models import Officer

log = structlog.get_logger(__name__)
bearer = HTTPBearer(auto_error=False)

ROLE_HIERARCHY = {
    "readonly":   0,
    "officer":    1,
    "supervisor": 2,
    "admin":      3,
}


async def get_current_officer(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: AsyncSession = Depends(get_db),
) -> Officer:
    """
    Base auth dependency. Validates access token and checks active status in DB.
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(credentials.credentials)
        officer_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError) as e:
        log.warning("auth.token_invalid", error=str(e), ip=request.client.host if request.client else "unknown")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = await db.execute(
        select(Officer).options(selectinload(Officer.role)).where(Officer.id == officer_id, Officer.is_active == True)
    )
    officer = result.scalar_one_or_none()

    if not officer:
        log.warning("auth.officer_not_found_or_suspended", officer_id=officer_id)
        raise HTTPException(
            status_code=401,
            detail="Officer not found or suspended",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return officer


def require_role(*required_roles: str):
    """
    Decorator factory for role-based endpoint protection.
    """
    async def _check_role(
        officer: Officer = Depends(get_current_officer)
    ) -> Officer:
        user_role = officer.role.name if officer.role else "officer"
        officer_role_level = ROLE_HIERARCHY.get(user_role, -1)
        required_level = min(
            ROLE_HIERARCHY.get(r, 999) for r in required_roles
        )

        if officer_role_level < required_level:
            log.warning("auth.insufficient_role",
                        officer_id=officer.id,
                        officer_role=user_role,
                        required_roles=required_roles)
            raise HTTPException(
                status_code=403,
                detail=f"Requires one of: {list(required_roles)}. Your role: {user_role}",
            )
        return officer

    return _check_role


# ── Type aliases ──────────────────────────────────────────────────────────────
AnyOfficer = Annotated[Officer, Depends(require_role("officer", "supervisor", "admin", "readonly"))]
ActiveOfficer = Annotated[Officer, Depends(require_role("officer", "supervisor", "admin"))]
SupervisorOrAdmin = Annotated[Officer, Depends(require_role("supervisor", "admin"))]
AdminOnly = Annotated[Officer, Depends(require_role("admin"))]
