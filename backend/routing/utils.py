"""
backend/routing/utils.py — shared distance + timestamp helpers (Day 14).

Import these everywhere routing needs a distance or a timestamp. Do not
reimplement inline: the SQLite-vs-Python timezone handling below is the exact
class of bug that has already cost this project twice (the abandoned-object
wall-clock/video_time mismatch, and evidence_capture's naive .timestamp()).

TIMEZONE CONTRACT
  SQLite's datetime('now') returns UTC in the form '2024-01-15 14:23:01' —
  space-separated, no offset, no 'Z'. Python's datetime.fromisoformat() parses
  that into a NAIVE datetime. Doing arithmetic against datetime.now() (local)
  or calling .timestamp() on it (which assumes local time) silently shifts
  every result by the host's UTC offset — 5.5 hours on this deployment's IST.
  Every helper here attaches tzinfo=UTC explicitly so that cannot happen.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import atan2, cos, radians, sin, sqrt


# ── Haversine ────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km. No external libraries."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def haversine_display_km(lat1: float, lng1: float, lat2: float, lng2: float) -> str:
    """Human-readable distance for UI display ONLY.

    Never sort on this — it returns a string, and '< 50 m' has no ordering
    relationship to '1.20 km'. Sort with haversine_km.
    """
    d = haversine_km(lat1, lng1, lat2, lng2)
    if d < 0.05:
        return "< 50 m"
    return f"{d:.2f} km"


# ── Datetime helpers ─────────────────────────────────────────────────────────

def parse_sqlite_utc(dt_value: str | datetime) -> datetime:
    """Parse a SQLite datetime('now') value into a UTC-aware datetime.

    Accepts a str (raw SQLite text) or a datetime (SQLAlchemy may already
    have coerced the column). A naive datetime from this database always
    MEANS UTC, so it is labelled rather than converted.
    """
    if isinstance(dt_value, datetime):
        return dt_value if dt_value.tzinfo else dt_value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(dt_value)).replace(tzinfo=timezone.utc)


def to_iso8601(dt_value: str | datetime | None) -> str | None:
    """SQLite datetime → ISO-8601 with a Z suffix, for WebSocket payloads.

    '2024-01-15 14:23:01'  →  '2024-01-15T14:23:01Z'

    Safari's Date() constructor rejects the space-separated form, so a raw
    SQLite string reaching the browser renders as "Invalid Date" on iOS and
    macOS Safari while working fine in Chrome — a bug that only shows up on
    the reviewer's laptop.
    """
    if dt_value is None:
        return None
    return parse_sqlite_utc(dt_value).strftime("%Y-%m-%dT%H:%M:%SZ")


def seconds_elapsed_since(assigned_at: str | datetime | None) -> int:
    """Whole seconds since `assigned_at`. Never negative, never None."""
    if assigned_at is None:
        return 0
    assigned = parse_sqlite_utc(assigned_at)
    now = datetime.now(timezone.utc)
    return max(0, int((now - assigned).total_seconds()))
