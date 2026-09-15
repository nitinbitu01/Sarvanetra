"""
backend/auth/dependencies.py — FastAPI auth dependencies.

get_current_user: decodes JWT from Authorization header, returns User ORM obj.
require_role(role): factory returning a dep that enforces role.
require_any_role(*roles): factory that allows any of the listed roles.
normalize_role(user): maps legacy 'admin'/'officer' to ADMIN/OPERATOR —
  no DB migration needed; the column stays a free string.

Role mapping (Day 9):
  DB value 'admin'    → canonical 'ADMIN'
  DB value 'ADMIN'    → canonical 'ADMIN'
  DB value 'officer'  → canonical 'OPERATOR'
  DB value 'OPERATOR' → canonical 'OPERATOR'
  DB value 'VIEWER'   → canonical 'VIEWER'
  anything else       → canonical 'VIEWER' (deny-by-default)
"""
from __future__ import annotations

import logging
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.orm import Session

from backend.auth.jwt_utils import decode_access_token
from backend.db.models import User
from backend.db.session import get_db

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)

_401 = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token",
    headers={"WWW-Authenticate": "Bearer"},
)

# ── Role normalization ───────────────────────────────────────────────────────────

# Map of every DB string value → canonical Day-9 role string.
# 'admin'/'ADMIN' both map to ADMIN so existing seeded users work immediately.
# 'officer'/'OPERATOR' both map to OPERATOR for the same reason.
# Anything unrecognised falls back to 'VIEWER' (deny-by-default).
_ROLE_MAP: dict[str, str] = {
    "admin":    "ADMIN",
    "ADMIN":    "ADMIN",
    "officer":  "OPERATOR",
    "OPERATOR": "OPERATOR",
    "VIEWER":   "VIEWER",
}


def normalize_role(user: User) -> str:
    """Return the canonical Day-9 role string for a User.

    Caller should use this everywhere instead of comparing user.role directly,
    so legacy 'admin'/'officer' strings from old rows are handled transparently.
    """
    return _ROLE_MAP.get(user.role or "", "VIEWER")


# ── Auth dependencies ─────────────────────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Decode JWT, load User from DB. Raises 401 if token invalid/missing."""
    if credentials is None:
        raise _401
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        if user_id is None:
            raise _401
    except JWTError:
        raise _401

    user = db.query(User).filter(User.id == str(user_id), User.is_active == True).first()
    if not user and str(user_id).isdigit():
        user = db.query(User).filter(User.id == int(user_id), User.is_active == True).first()
    if not user:
        user = db.query(User).filter(User.username == str(user_id), User.is_active == True).first()
    if not user:
        raise _401
    return user


def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    """Decode JWT if present, otherwise returns None without throwing 401."""
    if credentials is None:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        if user_id is None:
            return None
        user = db.query(User).filter(User.id == str(user_id), User.is_active == True).first()
        if not user and str(user_id).isdigit():
            user = db.query(User).filter(User.id == int(user_id), User.is_active == True).first()
        if not user:
            user = db.query(User).filter(User.username == str(user_id), User.is_active == True).first()
        return user
    except Exception:
        return None



def require_role(role: str):
    """Return a FastAPI dependency that requires exactly the given role.

    Uses normalize_role() so 'admin' DB value satisfies require_role('ADMIN') or require_role('admin').
    """
    canonical_role = _ROLE_MAP.get(role, role.upper())

    def _check(current_user: User = Depends(get_current_user)) -> User:
        if normalize_role(current_user) != canonical_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required.",
            )
        return current_user
    return _check


def require_any_role(*roles: str):
    """Return a FastAPI dependency that allows any of the listed roles.

    Example: require_any_role('ADMIN', 'OPERATOR')
    Uses normalize_role() so legacy DB values work transparently.
    """
    allowed = {_ROLE_MAP.get(r, r.upper()) for r in roles}

    def _check(current_user: User = Depends(get_current_user)) -> User:
        if normalize_role(current_user) not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {sorted(allowed)!r} required.",
            )
        return current_user
    return _check


def user_claims(user: User) -> dict:
    """Adapt a User row to the claims dict the scoping helpers expect.

    Two auth layers coexist: jwt_handler works on JWT claim dicts, while this
    module returns User ORM rows. Normalising to one shape here keeps the
    scoping rules in a single place instead of duplicated per call site.
    """
    role = str(getattr(user, "role", "") or "").lower()
    return {
        "sub": getattr(user, "id", None),
        "role": "admin" if role == "admin" else role,
        "dept": getattr(user, "department", None),
    }


def camera_scope_filter(user: User):
    """SQLAlchemy filter limiting a Camera query to what `user` may see.

    Returns None for state-wide callers, so the caller can skip filtering
    entirely rather than build a tautology.
    """
    from backend.auth.jwt_handler import ALL_DEPARTMENTS, caller_department
    from backend.db.models import Camera

    scope = caller_department(user_claims(user))
    if scope == ALL_DEPARTMENTS:
        return None
    # Unassigned cameras stay visible so one does not disappear mid-onboarding;
    # that matches the behaviour GET /cameras already had.
    return (Camera.department == scope) | (Camera.department.is_(None))


def assert_camera_in_scope(camera_id, user: User, db: Session):
    """Authorise access to ONE camera, or raise 403. Returns the Camera row.

    Listing cameras was already department-filtered, but every endpoint that
    takes a camera id — live video, the MJPEG stream, HLS manifest and
    segments, stream tokens, IQ scores, telemetry — authenticated the caller
    and then served whatever id was asked for. Filtering a list while leaving
    direct access open is the classic broken-access-control shape: the camera
    is hidden from the menu and served on request.

    Concretely, before this, a Transport operator whose own scope is two
    cameras could ask for a stream token for a Police camera by id and watch
    it. In a platform whose purpose is unifying 26 departments' feeds, that
    boundary is the product, so it is enforced at the point of access rather
    than at the point of display.

    403 rather than 404: these are public infrastructure at known junctions,
    so hiding their existence buys nothing, while naming the owning department
    tells a legitimate operator exactly whose permission to ask for.
    """
    from backend.auth.jwt_handler import ALL_DEPARTMENTS, caller_department
    from backend.db.models import Camera

    cam = db.query(Camera).filter(
        Camera.camera_id == str(camera_id), Camera.is_deleted == False  # noqa: E712
    ).first()
    if cam is None:
        cam = db.query(Camera).filter(
            Camera.id == str(camera_id), Camera.is_deleted == False  # noqa: E712
        ).first()
    if cam is None and str(camera_id).isdigit():
        cam = db.query(Camera).filter(
            Camera.id == int(camera_id), Camera.is_deleted == False  # noqa: E712
        ).first()
    if cam is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Camera {camera_id} not found")

    scope = caller_department(user_claims(user))
    if scope == ALL_DEPARTMENTS:
        return cam
    owner = (cam.department or "").strip().lower()
    if owner and owner != scope:
        logger.warning(
            "Cross-department camera access denied: user=%s scope=%r "
            "camera=%s owner=%r",
            getattr(user, "id", "?"), scope, camera_id, owner,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(f"Camera {camera_id} belongs to the "
                    f"{cam.department} department. Your access is scoped to "
                    f"{scope or 'no department'}."),
        )
    if not owner and not scope:
        # Neither side assigned: deny rather than guess. An unassigned account
        # is a data gap, not a grant.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has no department assigned.",
        )
    return cam


def require_officer_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    """Authentication dependency for officer actions. A token is required.

    This used to return the admin user when NO credentials were supplied — a
    "demo mode" fallback. It guards 14 endpoints, among them officer feedback
    and review actions, so anyone who could reach the API could act as an
    administrator by simply omitting the Authorization header. Sending no
    token was easier than sending a valid one.

    That is not a demo convenience in a platform whose stated purpose is
    unifying CCTV across 26 government departments; departmental separation is
    the product, and an unauthenticated admin defeats it entirely.

    The fallback is now opt-in and off by default. SENTINEL_DEMO_AUTH=1 brings
    it back for an offline demo machine, and the warning it logs on every call
    is deliberate — it should be impossible to leave on by accident.
    """
    if credentials is None:
        if os.environ.get("SENTINEL_DEMO_AUTH") == "1":
            admin_user = db.query(User).filter(
                (User.id == "USER_ADMIN_01")
                | (User.username == "admin")
                | (User.id == "1")
            ).first()
            logger.warning(
                "SENTINEL_DEMO_AUTH is enabled: an unauthenticated request was "
                "served as %s. Never run with this set outside a local demo.",
                getattr(admin_user, "username", "admin"),
            )
            return admin_user
        raise _401
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        if user_id is None:
            raise _401
    except JWTError:
        raise _401

    user = db.query(User).filter(User.id == str(user_id), User.is_active == True).first()
    if not user and str(user_id).isdigit():
        user = db.query(User).filter(User.id == int(user_id), User.is_active == True).first()
    if not user:
        raise _401
    return user

