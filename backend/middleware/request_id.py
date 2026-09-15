"""
backend/middleware/request_id.py — Injects a UUID request_id per HTTP request.

The request_id is:
  1. Written into a contextvars.ContextVar so SentinelJsonFormatter picks
     it up and adds it to every log line emitted during that request.
  2. Returned in the X-Request-ID response header for client-side correlation.

This makes it possible to trace a single request end-to-end through logs
without a full distributed tracing stack.
"""
from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.core.logging import request_id_var


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns a UUID request_id to every incoming HTTP request."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        req_id = str(uuid.uuid4())
        token = request_id_var.set(req_id)
        try:
            response: Response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            request_id_var.reset(token)
