"""
backend/core/logging.py — JSON-structured logging for Sentinel Gujarat.

Every log line is a valid JSON object with:
  timestamp, level, module, message, request_id (if in context), extra fields.

Usage:
    from backend.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Camera added", extra={"camera_id": "CAM-05", "user_id": 1})

request_id is injected by RequestIDMiddleware into a contextvars.ContextVar
and automatically included by SentinelJsonFormatter in every log record.
"""
from __future__ import annotations

import contextvars
import logging
from typing import Any

from pythonjsonlogger.json import JsonFormatter

from backend.core.config import settings

# ── Context variable — set per-request by RequestIDMiddleware ────────────────
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


class SentinelJsonFormatter(JsonFormatter):
    """Extend JsonFormatter to inject request_id and normalize field names."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        # Inject request_id from context (empty string if not in a request)
        log_record["request_id"] = request_id_var.get("")

        # Normalize field names to match structured log convention
        log_record.setdefault("level", record.levelname)
        log_record.setdefault("module", record.name)

        # Remove redundant keys added by pythonjsonlogger
        for key in ("levelname", "name"):
            log_record.pop(key, None)

    def format(self, record: logging.LogRecord) -> str:
        try:
            return super().format(record)
        except Exception:
            try:
                record.message = str(record.msg)
                return super().format(record)
            except Exception:
                return str(record.msg)


def configure_logging() -> None:
    """Configure root logger with JSON output.

    Call this once at application startup (before any logger is used).
    Level is controlled by settings.LOG_LEVEL from .env.
    """
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = SentinelJsonFormatter(
        fmt="%(timestamp)s %(level)s %(module)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        rename_fields={"asctime": "timestamp"},
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)

    # Replace any existing handlers (e.g., basicConfig from Day 1–4 code)
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (convenience wrapper).

    Args:
        name: Usually __name__ of the calling module.
    """
    return logging.getLogger(name)
