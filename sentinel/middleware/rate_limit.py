# sentinel/middleware/rate_limit.py
"""
sentinel/middleware/rate_limit.py — SlowAPI rate limiter and HTTP 429 JSON response handler.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from sentinel.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"],
    storage_uri="memory://",
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    retry_after = getattr(exc, "retry_after", 60)
    limit = getattr(exc, "limit", f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "retry_after": str(retry_after),
            "limit": str(limit),
        },
        headers={"Retry-After": str(retry_after)},
    )
