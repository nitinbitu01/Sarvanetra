"""
backend/auth/jwt_utils.py — JWT encode/decode helpers.

Uses python-jose with HS256. Secret comes from settings.JWT_SECRET.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from backend.core.config import settings


def create_access_token(data: dict[str, Any]) -> str:
    """Create a signed JWT with an expiry from settings.JWT_EXPIRE_MINUTES."""
    payload = dict(data)
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload["exp"] = expire
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT. Raises JWTError on invalid/expired token."""
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])


# Minutes a stream token stays valid. Short on purpose: it travels in a URL
# query string, so it can end up in proxy logs, browser history and Referer
# headers in a way an Authorization header never does.
STREAM_TOKEN_MINUTES = 10


def create_stream_token(user_id: str, camera_id: str) -> str:
    """A short-lived token that authorises viewing ONE camera's video.

    Video is delivered to an <img> tag — MJPEG for the live stream, JPEG for
    the snapshot — and an <img> cannot send an Authorization header. That is
    why those two endpoints were left unauthenticated: the obvious fix does
    not work, so no fix was applied and anyone on the network could watch
    thirty government cameras.

    The standard answer is a token in the query string, made safe by being
    narrow and brief rather than by being secret:

      scope   "stream" only, so it cannot be replayed against the ordinary
              API even though it is signed with the same key
      cam     one camera, so a leaked URL does not open the whole fleet
      exp     ten minutes

    The frontend asks for one of these with its real bearer token and puts it
    in the image URL.
    """
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "scope": "stream",
        "cam": str(camera_id),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=STREAM_TOKEN_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET,
                      algorithm=settings.JWT_ALGORITHM)


def verify_stream_token(token: str, camera_id: str) -> bool:
    """True when `token` authorises video for exactly `camera_id`."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET,
                             algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        return False
    # Both checks matter: scope stops an ordinary access token being used as a
    # stream token (and vice versa), cam stops one camera's token opening
    # another's.
    return (payload.get("scope") == "stream"
            and str(payload.get("cam")) == str(camera_id))
