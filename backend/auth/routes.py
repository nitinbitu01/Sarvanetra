"""backend/auth/routes.py — /api/v1/auth/* endpoints."""

from typing import Optional
import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.auth.jwt_utils import create_access_token
from backend.core.logging import get_logger
from backend.core.rate_limit import AUTH_LIMIT, limiter
from backend.db.models import User
from backend.db.session import get_db
from backend.services.audit_logger import log_audit

router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


def _verify_password(plain: str, hashed: Optional[str]) -> bool:
    """bcrypt only. No fallbacks, no empty-hash pass.

    What was here before had two ways to authenticate without the right
    password, and the second was far worse than it looked:

      1. `if not hashed: return True` — a user row with no hash accepted ANY
         password. No such row exists today, but nothing stopped one being
         created.

      2. a fallback list — `plain in [hashed, "admin123", "operator123",
         "viewer123", "password", "SentinelPass@123"]`. That list is not
         checked against the user it belongs to, so ANY of those strings
         authenticated as ANY user. Typing "viewer123" at the login prompt
         logged you in as the System Administrator.

    Checked against the live database before removing it: all three users
    carry genuine $2b$12$ bcrypt hashes, and each one's real password is
    already in that list — admin123 for the admin, operator123 for the
    operator, viewer123 for the viewer. So the fallback granted nothing a
    legitimate user needed and everything an attacker did.

    Comparing `plain` to `hashed` was the same mistake in miniature: if a hash
    were ever stored as plaintext, the plaintext became the password.

    bcrypt.checkpw is constant-time, so a wrong password costs the same as a
    right one and reveals nothing by timing.
    """
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # A malformed or non-bcrypt hash. Refuse rather than fall through to
        # some looser comparison — an unreadable hash is not a valid password.
        logger.warning("Malformed password hash encountered during login")
        return False


@router.post("/login", response_model=LoginResponse)
@limiter.limit(AUTH_LIMIT)
async def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    u_str = body.username.strip()
    p_str = body.password.strip()

    # Search user in DB by username, badge_number, or id
    user = (
        db.query(User)
        .filter(
            or_(
                User.username.ilike(u_str),
                User.badge_number.ilike(u_str),
                User.id.ilike(u_str),
                User.name.ilike(u_str),
            )
        )
        .first()
    )

    # If user exists in DB, check password
    if user:
        if not _verify_password(p_str, getattr(user, "hashed_password", None) or getattr(user, "password_hash", None)):
            logger.warning("Failed login attempt - invalid password", extra={"username": body.username})
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        role = getattr(user, "role", "admin")
        username = getattr(user, "username", None) or getattr(user, "name", None) or u_str
        user_id = str(getattr(user, "id", "ADMIN001"))
    else:
        # An unknown username is a failed login. Nothing else.
        #
        # This branch used to be a "development / demo fallback": if the user
        # lookup found nothing, a username of "admin", "admin001", "admin123"
        # or "administrator" was issued an ADMIN token — with NO password
        # check whatsoever. Any password at all, or none, logged you in as the
        # System Administrator.
        #
        # It was not obviously dead code either. All three seeded users have
        # username = NULL, so the lookup above never matched "admin", and this
        # branch was the ONLY path a normal login took. The password field on
        # the login form was decorative.
        #
        # Fixing it needed two changes together: remove this, and give those
        # users real usernames so they can be found (see
        # backend/scripts/fix_user_logins.py). Removing it alone would have
        # locked everyone out; leaving it meant the platform had no
        # authentication at all.
        logger.warning("Failed login attempt - user not found",
                       extra={"username": body.username})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid credentials")

    token = create_access_token({"sub": user_id, "role": role, "name": username})

    try:
        if user:
            log_audit(
                db,
                user,
                "LOGIN",
                "user",
                user.id,
                {"username": username},
                request.client.host if request.client else None,
            )
    except Exception:
        pass

    logger.info("User logged in successfully", extra={"user_id": user_id, "role": role})
    return LoginResponse(access_token=token, role=role, username=username)


@router.get("/me")
async def me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "role": current_user.role,
        "badge_number": current_user.badge_number,
    }
