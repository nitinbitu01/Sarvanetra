import logging
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

_log = logging.getLogger(__name__)

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production-min-32-chars-sentinel-key")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "8"))

security = HTTPBearer(auto_error=False)


class UserRole(str, Enum):
    admin = "admin"
    operator = "operator"
    viewer = "viewer"
    officer = "officer"


# Departments own cameras; a user belongs to one. ALL_DEPARTMENTS is the
# state-level scope - only admins and audit roles should hold it, because it
# crosses every department boundary at once.
ALL_DEPARTMENTS = "*"


def create_token(
    user_id: str,
    role: UserRole,
    name: str = "",
    department: str = ALL_DEPARTMENTS,
) -> str:
    """Issue a JWT carrying BOTH role and department.

    The department is part of the signed token rather than looked up per
    request: it is an authorisation fact, so it must be tamper-evident. A
    client that could assert its own department could read another
    department's cameras simply by asking.
    """
    return jwt.encode(
        {
            "sub": user_id,
            "role": role.value if isinstance(role, UserRole) else str(role),
            "name": name,
            "dept": department,
            "exp": datetime.utcnow() + timedelta(hours=EXPIRE_HOURS),
            "iat": datetime.utcnow(),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    if not credentials:
        # SECURITY: this used to return a full admin identity whenever NO
        # credentials were supplied, which is a complete authentication
        # bypass - every endpoint, including camera management and evidence
        # access, was reachable by anyone who simply omitted the
        # Authorization header. Convenient in local dev, catastrophic in a
        # platform whose purpose is controlling access to 26 departments'
        # surveillance feeds.
        #
        # Now opt-in and off by default, and it warns every time so it
        # cannot be left enabled unnoticed.
        if os.getenv("SENTINEL_ALLOW_ANON_ADMIN", "0") == "1":
            _log.warning(
                "SENTINEL_ALLOW_ANON_ADMIN=1: anonymous request granted ADMIN. "
                "Development only - never enable on a real deployment."
            )
            return {"sub": "USER_ADMIN_01", "role": "admin",
                    "name": "System Administrator", "dept": ALL_DEPARTMENTS}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


def require_role(*roles: UserRole):
    """Allow the listed roles; admin always passes.

    SECURITY: the previous condition was

        if user_role not in allowed and "admin" not in allowed:

    The second clause tests the ALLOWED list, not the CALLER. So whenever
    'admin' was among the permitted roles - i.e. on exactly the endpoints
    meant to be most restricted - that clause was False, the whole condition
    was False, and NOBODY was rejected. A viewer passed an admin-only gate.
    Verified against the old code before changing it.

    The caller's own role is what decides.
    """
    async def checker(user: dict = Depends(get_current_user)):
        user_role = user.get("role")
        allowed = {r.value if isinstance(r, UserRole) else str(r) for r in roles}
        if user_role != UserRole.admin.value and user_role not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Requires role: {', '.join(sorted(allowed))}",
            )
        return user

    return checker


def caller_department(user: dict) -> str:
    """Department scope for this caller. ALL_DEPARTMENTS means unrestricted.

    Only 'admin' gets automatic state-wide scope when no department is set.
    Any other role with a missing department is deny-by-default - it was
    previously treated the same as admin, so a non-admin account that simply
    never got a department assigned (e.g. a seeded viewer) silently saw
    every department's cameras instead of none. An unassigned account is a
    data gap, not a grant, matching the same principle applied to unassigned
    cameras (see backend/scripts/normalize_departments.py).
    """
    if user.get("role") == UserRole.admin.value:
        return ALL_DEPARTMENTS
    dept = user.get("dept")
    if not dept:
        return ""
    return str(dept).strip().lower()


def can_access_department(user: dict, department: Optional[str]) -> bool:
    """May this caller see a resource owned by `department`?

    Case-normalized matching prevents 'Police' vs 'police' split.
    Unassigned resources (None/empty) are not leaked to department-scoped callers.
    """
    scope = caller_department(user)
    if scope == ALL_DEPARTMENTS:
        return True
    if not department or not str(department).strip():
        return False
    return str(department).strip().lower() == scope.strip().lower()


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Check a password against its bcrypt hash.

    SECURITY: the previous fallback was

        return plain == hashed or plain == "admin123" or plain == "operator123"

    which meant "admin123" and "operator123" authenticated as ANY user - a
    universal backdoor into every account on the platform, including admin.
    The `plain == hashed` arm additionally let anyone in who could read a
    hash from the database and present it verbatim as the password, turning
    read access into full impersonation.

    Only the bcrypt comparison remains. Seeded demo accounts still work
    because their stored values are real bcrypt hashes of those passwords -
    see backend/db/seed_admin.py - so this removes the backdoor without
    breaking legitimate demo logins.
    """
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # Not a valid bcrypt hash (e.g. a legacy plaintext row). Refuse
        # rather than degrading to a string compare.
        _log.warning("verify_password: stored credential is not a valid "
                     "bcrypt hash - rejecting login")
        return False
