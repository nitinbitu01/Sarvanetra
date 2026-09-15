"""
backend/core/rate_limit.py — slowapi limiter configuration.

Usage:
    from backend.core.rate_limit import limiter, GENERAL_LIMIT, AUTH_LIMIT
    @app.get("/some-route")
    @limiter.limit(GENERAL_LIMIT)
    async def some_route(request: Request): ...
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.core.config import settings

# One global Limiter instance — mounted on the FastAPI app in main.py
limiter = Limiter(key_func=get_remote_address)

# Limit strings derived from settings so they're config-driven
GENERAL_LIMIT: str = f"{settings.RATE_LIMIT_PER_MINUTE}/minute"
AUTH_LIMIT: str = f"{settings.AUTH_RATE_LIMIT_PER_MINUTE}/minute"
