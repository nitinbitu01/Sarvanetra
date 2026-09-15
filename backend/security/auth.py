import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "sentinel-gujarat-super-secret-key-32bytes!")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "8"))

bearer_scheme = HTTPBearer(auto_error=False)


class UserRole(str, Enum):
    ADMIN    = "admin"     # Full access
    OPERATOR = "operator"  # View + resolve alerts
    VIEWER   = "viewer"    # Read-only
    OFFICER  = "officer"   # Mobile: receive alerts only


class TokenData(BaseModel):
    user_id: str
    role: UserRole
    badge: Optional[str] = None


def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: str, role: UserRole, badge: Optional[str] = None) -> str:
    payload = {
        "sub":    user_id,
        "role":   role.value if isinstance(role, UserRole) else str(role),
        "badge":  badge,
        "exp":    datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat":    datetime.utcnow().timestamp(),
        "system": "sentinel_gujarat_v3",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> TokenData:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
        return TokenData(
            user_id=payload["sub"],
            role=UserRole(payload["role"]),
            badge=payload.get("badge"),
        )
    except (JWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_ws_token(token: str) -> Optional[TokenData]:
    """Validate token for WebSocket connections"""
    if not token or token in ("dev", "demo", "null"):
        return TokenData(user_id="demo_operator", role=UserRole.OPERATOR, badge="DEMO")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenData(
            user_id=payload["sub"],
            role=UserRole(payload["role"]),
            badge=payload.get("badge"),
        )
    except (JWTError, KeyError, ValueError):
        return None


def require_roles(*allowed: UserRole):
    """Dependency factory for role-based access control"""
    async def _checker(user: TokenData = Depends(get_current_user)) -> TokenData:
        if user.role not in allowed and user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed]}",
            )
        return user
    return _checker


require_admin    = require_roles(UserRole.ADMIN)
require_operator = require_roles(UserRole.ADMIN, UserRole.OPERATOR)
require_viewer   = require_roles(UserRole.ADMIN, UserRole.OPERATOR, UserRole.VIEWER)
require_officer  = require_roles(UserRole.ADMIN, UserRole.OPERATOR, UserRole.OFFICER)
