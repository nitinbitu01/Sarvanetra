"""
backend/db.py — SQLite data-access layer for Sentinel Gujarat.

All other modules call functions from this file. No raw SQL should appear
outside this module — that would scatter schema knowledge across the codebase
and make future migrations a nightmare.

Connection strategy:
  - WAL (Write-Ahead Log) mode is enabled on every new connection, which
    allows concurrent readers without blocking writers.
  - A threading.Lock() guards all write operations so the Day 2 pipeline
    thread and FastAPI's async event loop can both write without "database
    is locked" errors.
  - conn.row_factory = sqlite3.Row makes query results dict-like — every
    caller gets `row["column_name"]` instead of positional `row[0]`.
  - get_connection() is a simple factory — one connection per call site.
    For a hackathon prototype this is fine; a production system would use
    a proper connection pool (e.g., SQLAlchemy).

Schema design notes:
  - tracks.global_id is NULL until Day 6's ReID engine assigns a cross-camera
    identity. Don't read it before Day 6.
  - alerts.metadata is a JSON string — use it for alert-type-specific fields
    instead of altering the schema for every new alert type. See README.
  - watchlist_persons stores InsightFace FACE embeddings only.
    OSNet BODY embeddings go into tracks.body_embedding. Do NOT mix them.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sqlite3
import threading
import time
import uuid as _uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ── Module-level write lock — shared across all callers in the same process ──
_write_lock = threading.Lock()

# ── Retry config for SQLite locking ──────────────────────────────────
_MAX_RETRIES   = 5
_RETRY_DELAY_S = 0.05  # 50 ms between retries


def _retry_on_locked(func: F) -> F:
    """Decorator: retry a DB write function up to _MAX_RETRIES times on lock.

    SQLite in WAL mode should rarely hit OperationalError('database is locked'),
    but under high write contention (pipeline + OCR worker + FastAPI all writing
    simultaneously) it can occur. The module-level _write_lock serialises writes
    at the Python level, but the retry adds a second safety layer for any edge
    cases that slip through (e.g., external writer, WAL checkpoint contention).
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower():
                    last_exc = exc
                    logger.warning(
                        "DB locked on %s() attempt %d/%d — retrying in %.0fms",
                        func.__name__, attempt, _MAX_RETRIES, _RETRY_DELAY_S * 1000,
                    )
                    time.sleep(_RETRY_DELAY_S * attempt)  # exponential-ish backoff
                else:
                    raise  # non-lock OperationalError: propagate immediately
        raise RuntimeError(
            f"{func.__name__}() failed after {_MAX_RETRIES} retries: {last_exc}"
        ) from last_exc

    return wrapper  # type: ignore[return-value]

# ── Schema DDL ───────────────────────────────────────────────────────────────
_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS cameras (
    camera_id   TEXT PRIMARY KEY,
    name        TEXT,
    zone        TEXT,
    gps_lat     REAL,
    gps_lon     REAL,
    status      TEXT DEFAULT 'unknown',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tracks (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id            TEXT NOT NULL,
    track_id             INTEGER NOT NULL,
    global_id            TEXT,
    first_seen_frame     INTEGER,
    first_seen_timestamp REAL,
    last_seen_frame      INTEGER,
    last_seen_timestamp  REAL,
    best_crop_path       TEXT,
    best_confidence      REAL,
    body_embedding       BLOB,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(camera_id, track_id)
);

CREATE TABLE IF NOT EXISTS journeys (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id       TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    track_id        INTEGER NOT NULL,
    entry_timestamp REAL,
    exit_timestamp  REAL,
    sequence_order  INTEGER
);

CREATE TABLE IF NOT EXISTS watchlist_persons (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT NOT NULL,
    reason               TEXT,
    face_embedding       BLOB NOT NULL,
    embedding_dim        INTEGER NOT NULL,
    reference_photo_path TEXT,
    active               BOOLEAN DEFAULT 1,
    added_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist_vehicles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT UNIQUE NOT NULL,
    reason       TEXT,
    active       BOOLEAN DEFAULT 1,
    added_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type    TEXT NOT NULL,
    camera_id     TEXT,
    track_id      INTEGER,
    global_id     TEXT,
    score         REAL,
    severity      TEXT,
    status        TEXT DEFAULT 'new',
    evidence_path TEXT,
    metadata      TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at   TIMESTAMP
);
"""


# ────────────────────────────────────────────────────────────────────────────
# Connection factory
# ────────────────────────────────────────────────────────────────────────────

def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode, Row factory, and busy timeout.

    busy_timeout=5000ms: SQLite-level fallback if a write lock is held by
    another *process*. The module-level threading.Lock handles same-process
    write serialisation; busy_timeout handles cross-process contention.

    Args:
        db_path: Path to the SQLite file (created if absent).

    Returns:
        An open sqlite3.Connection ready to use.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")  # 5s SQLite-level wait before SQLITE_BUSY
    return conn


def init_db(db_path: str) -> None:
    """Create all tables if they don't exist. Safe to call multiple times.

    Args:
        db_path: SQLite file path.
    """
    conn = get_connection(db_path)
    with _write_lock:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    conn.close()
    logger.info("Database initialised (WAL mode) at %s", db_path)


# ────────────────────────────────────────────────────────────────────────────
# cameras table
# ────────────────────────────────────────────────────────────────────────────

@_retry_on_locked
def insert_camera(
    db_path: str,
    camera_id: str,
    name: str,
    zone: str,
    gps_lat: float,
    gps_lon: float,
    status: str = "unknown",
) -> None:
    """Insert or replace a camera record.

    Uses INSERT OR REPLACE — safe to call on each startup to refresh status.

    Args:
        db_path: SQLite file path.
        camera_id: Unique camera identifier string.
        name: Human-readable camera name.
        zone: Geographic zone label.
        gps_lat: GPS latitude.
        gps_lon: GPS longitude.
        status: 'online' | 'offline' | 'unknown'.
    """
    conn = get_connection(db_path)
    with _write_lock:
        conn.execute(
            """INSERT OR REPLACE INTO cameras
               (camera_id, name, zone, gps_lat, gps_lon, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (camera_id, name, zone, gps_lat, gps_lon, status),
        )
        conn.commit()
    conn.close()
    logger.debug("Camera upserted: %s (%s)", camera_id, name)


def get_all_cameras(db_path: str) -> list[sqlite3.Row]:
    """Return all camera records.

    Returns:
        List of sqlite3.Row objects (dict-like).
    """
    conn = get_connection(db_path)
    rows = conn.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall()
    conn.close()
    return rows


# ────────────────────────────────────────────────────────────────────────────
# tracks table
# ────────────────────────────────────────────────────────────────────────────

@_retry_on_locked
def upsert_track(
    db_path: str,
    camera_id: str,
    track_id: int,
    frame_number: int,
    timestamp: float,
    crop_path: str | None = None,
    confidence: float | None = None,
    body_embedding: bytes | None = None,
) -> None:
    """Insert a new track or update an existing one (camera_id, track_id) pair.

    On first appearance: inserts with first_seen_* = last_seen_* = current.
    On subsequent calls: updates last_seen_*, best_crop_path/confidence if
    the new confidence is better than the stored best.

    Args:
        db_path: SQLite file path.
        camera_id: Camera where this track was seen.
        track_id: Local (per-camera) track ID from Day 1 pipeline.
        frame_number: Current frame index.
        timestamp: Monotonic video timestamp (seconds).
        crop_path: Path to the person crop image, if available.
        confidence: Detection confidence for this frame.
        body_embedding: Serialized OSNet embedding (bytes), populated Day 6+.
    """
    conn = get_connection(db_path)
    with _write_lock:
        conn.execute(
            """INSERT INTO tracks
               (camera_id, track_id, first_seen_frame, first_seen_timestamp,
                last_seen_frame, last_seen_timestamp, best_crop_path, best_confidence,
                body_embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(camera_id, track_id) DO UPDATE SET
                 last_seen_frame     = excluded.last_seen_frame,
                 last_seen_timestamp = excluded.last_seen_timestamp,
                 best_crop_path      = CASE
                     WHEN excluded.best_confidence > tracks.best_confidence
                       OR tracks.best_confidence IS NULL
                     THEN excluded.best_crop_path
                     ELSE tracks.best_crop_path
                   END,
                 best_confidence     = MAX(
                     COALESCE(tracks.best_confidence, 0),
                     COALESCE(excluded.best_confidence, 0)
                 ),
                 body_embedding      = COALESCE(excluded.body_embedding, tracks.body_embedding),
                 updated_at          = CURRENT_TIMESTAMP""",
            (camera_id, track_id, frame_number, timestamp,
             frame_number, timestamp, crop_path, confidence, body_embedding),
        )
        conn.commit()
    conn.close()
    logger.debug("Track upserted: camera=%s track_id=%d frame=%d", camera_id, track_id, frame_number)


@_retry_on_locked
def set_track_global_id(db_path: str, camera_id: str, track_id: int, global_id: str) -> None:
    """Assign a cross-camera global_id to a confirmed track. Called by Day 6 ReID engine.

    Args:
        db_path: SQLite file path.
        camera_id: Camera identifier.
        track_id: Local track ID.
        global_id: UUID string assigned by the ReID engine.
    """
    conn = get_connection(db_path)
    with _write_lock:
        conn.execute(
            "UPDATE tracks SET global_id=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE camera_id=? AND track_id=?",
            (global_id, camera_id, track_id),
        )
        conn.commit()
    conn.close()
    logger.debug("global_id=%s assigned to camera=%s track_id=%d", global_id, camera_id, track_id)


def get_track(db_path: str, camera_id: str, track_id: int) -> sqlite3.Row | None:
    """Fetch a single track record by (camera_id, track_id).

    Returns:
        sqlite3.Row or None if not found.
    """
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM tracks WHERE camera_id=? AND track_id=?",
        (camera_id, track_id),
    ).fetchone()
    conn.close()
    return row


# ────────────────────────────────────────────────────────────────────────────
# journeys table
# ────────────────────────────────────────────────────────────────────────────

@_retry_on_locked
def insert_journey_entry(
    db_path: str,
    global_id: str,
    camera_id: str,
    track_id: int,
    entry_timestamp: float,
    exit_timestamp: float | None,
    sequence_order: int,
) -> int:
    """Add one step in a person's cross-camera journey.

    Args:
        db_path: SQLite file path.
        global_id: Cross-camera identity UUID from ReID engine.
        camera_id: Camera where this sighting occurred.
        track_id: Local track ID at this camera.
        entry_timestamp: When the person entered this camera's view.
        exit_timestamp: When they left (None if still visible).
        sequence_order: Position in the journey sequence (1 = first sighting).

    Returns:
        Row ID of the inserted record.
    """
    conn = get_connection(db_path)
    with _write_lock:
        cur = conn.execute(
            """INSERT INTO journeys
               (global_id, camera_id, track_id, entry_timestamp, exit_timestamp, sequence_order)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (global_id, camera_id, track_id, entry_timestamp, exit_timestamp, sequence_order),
        )
        conn.commit()
        row_id = cur.lastrowid
    conn.close()
    return row_id


def get_journey(db_path: str, global_id: str) -> list[sqlite3.Row]:
    """Retrieve the full journey for a global identity, ordered by sequence.

    Args:
        db_path: SQLite file path.
        global_id: Cross-camera identity UUID.

    Returns:
        List of sqlite3.Row objects, ordered by sequence_order ASC.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM journeys WHERE global_id=? ORDER BY sequence_order ASC",
        (global_id,),
    ).fetchall()
    conn.close()
    return rows


# ────────────────────────────────────────────────────────────────────────────
# watchlist tables
# ────────────────────────────────────────────────────────────────────────────

@_retry_on_locked
def insert_watchlist_person(
    db_path: str,
    name: str,
    reason: str,
    face_embedding: bytes,
    embedding_dim: int,
    reference_photo_path: str | None = None,
) -> int:
    """Add a person to the watchlist.

    IMPORTANT: face_embedding must be an InsightFace FACE embedding.
    Do NOT put OSNet body embeddings here — they belong in tracks.body_embedding.

    Args:
        db_path: SQLite file path.
        name: Person's name.
        reason: 'wanted' | 'suspect' etc.
        face_embedding: Bytes from encode_embedding() — InsightFace only.
        embedding_dim: Vector length (used for decode validation).
        reference_photo_path: Optional path to reference photo.

    Returns:
        Row ID of the inserted record.
    """
    conn = get_connection(db_path)
    with _write_lock:
        cur = conn.execute(
            """INSERT INTO watchlist_persons
               (name, reason, face_embedding, embedding_dim, reference_photo_path)
               VALUES (?, ?, ?, ?, ?)""",
            (name, reason, face_embedding, embedding_dim, reference_photo_path),
        )
        conn.commit()
        row_id = cur.lastrowid
    conn.close()
    logger.info("Watchlist person added: %s (reason: %s)", name, reason)
    return row_id


@_retry_on_locked
def insert_watchlist_vehicle(
    db_path: str,
    plate_number: str,
    reason: str,
) -> int:
    """Add a vehicle plate to the watchlist.

    Args:
        db_path: SQLite file path.
        plate_number: Plate string (e.g. 'GJ05AB1234').
        reason: 'stolen' | 'wanted' | 'violation'.

    Returns:
        Row ID of the inserted record.
    """
    conn = get_connection(db_path)
    with _write_lock:
        cur = conn.execute(
            "INSERT INTO watchlist_vehicles (plate_number, reason) VALUES (?, ?)",
            (plate_number.strip().upper(), reason),
        )
        conn.commit()
        row_id = cur.lastrowid
    conn.close()
    logger.info("Watchlist vehicle added: plate=%s (reason: %s)", plate_number, reason)
    return row_id


def get_active_watchlist_persons(db_path: str) -> list[sqlite3.Row]:
    """Return all active watchlist persons with their face embeddings.

    Called by Day 7's face-matching engine on each alert check.

    Returns:
        List of sqlite3.Row — each has face_embedding BLOB and embedding_dim.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM watchlist_persons WHERE active=1"
    ).fetchall()
    conn.close()
    return rows


def get_active_watchlist_vehicles(db_path: str) -> list[sqlite3.Row]:
    """Return all active watchlist vehicle plates.

    Called by Day 4's ANPR engine.

    Returns:
        List of sqlite3.Row objects.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM watchlist_vehicles WHERE active=1"
    ).fetchall()
    conn.close()
    return rows


def is_plate_on_watchlist(db_path: str, plate_number: str) -> sqlite3.Row | None:
    """Check if a plate number is on the active watchlist.

    Args:
        db_path: SQLite file path.
        plate_number: Plate string to look up (case-insensitive).

    Returns:
        sqlite3.Row if found and active, else None.
    """
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM watchlist_vehicles WHERE plate_number=? AND active=1",
        (plate_number.strip().upper(),),
    ).fetchone()
    conn.close()
    return row


# ────────────────────────────────────────────────────────────────────────────
# alerts table
# ────────────────────────────────────────────────────────────────────────────

@_retry_on_locked
def insert_alert(
    db_path: str,
    alert_type: str,
    camera_id: str | None = None,
    track_id: int | None = None,
    global_id: str | None = None,
    score: float | None = None,
    severity: str | None = None,
    evidence_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Insert a new alert record.

    The `metadata` dict (alert-type-specific fields) is serialised to JSON.
    Later days read it back with json.loads(row["metadata"]).

    Valid alert_types:
        'watchlist_person' | 'watchlist_vehicle' | 'loitering' |
        'crowd_anomaly' | 'abandoned_object' | 'anpr_uncertain' | 'review_queue'

    Valid severity:  'critical' | 'high' | 'medium'

    Args:
        db_path: SQLite file path.
        alert_type: One of the strings above.
        camera_id: Source camera ID.
        track_id: Local track ID that triggered the alert.
        global_id: Cross-camera identity UUID, if known.
        score: Numeric danger/confidence score (0.0–1.0).
        severity: 'critical' | 'high' | 'medium'.
        evidence_path: Path to saved evidence clip/image.
        metadata: Dict of alert-type-specific fields — serialized to JSON TEXT.

    Returns:
        Row ID of the inserted alert.
    """
    conn = get_connection(db_path)
    metadata_json = json.dumps(metadata) if metadata else None

    # Build the INSERT from the columns the table ACTUALLY has.
    #
    # The CREATE TABLE above is not what ships: output/sentinel.db is built
    # by the Alembic migrations, whose alerts table has no `track_id` and
    # stores metadata in `meta_json`, not `metadata`. Because the table
    # already exists, this module's CREATE TABLE IF NOT EXISTS never runs and
    # the divergence stays invisible until an INSERT fails with
    #     sqlite3.OperationalError: table alerts has no column named track_id
    # at which point it takes down the whole detection pipeline thread.
    #
    # Introspecting keeps this working across both schemas instead of
    # guessing which one is present.
    # PRAGMA gives (cid, name, type, notnull, default, pk)
    info = {r[1]: {"type": (r[2] or "").upper(), "notnull": bool(r[3]),
                   "default": r[4], "pk": bool(r[5])}
            for r in conn.execute("PRAGMA table_info(alerts)")}
    available = set(info)

    values: dict[str, Any] = {}
    for name, value in (
        ("alert_type", alert_type),
        ("camera_id", camera_id),
        ("track_id", track_id),
        ("global_id", global_id),
        ("score", score),
        ("severity", severity),
        ("evidence_path", evidence_path),
        # metadata lives under different names in the two schemas
        ("metadata" if "metadata" in available else "meta_json", metadata_json),
    ):
        if name in available:
            values[name] = value
        elif value is not None:
            logger.debug("alerts table has no column %r - dropping value", name)

    # Satisfy NOT NULL columns this legacy signature knows nothing about.
    #
    # The shipped schema is the ORM/Alembic one, which differs sharply from
    # the CREATE TABLE above: `id` is a VARCHAR(36) UUID rather than an
    # autoincrement integer, and camera_id / severity / danger_score /
    # timestamp are all NOT NULL. Inserting without them raises
    #     sqlite3.IntegrityError: NOT NULL constraint failed: alerts.id
    # which - like the missing-column failure before it - killed the whole
    # detection pipeline thread rather than just skipping one alert.
    for name, meta in info.items():
        if not meta["notnull"] or meta["default"] is not None:
            continue
        if values.get(name) is not None:
            continue
        t = meta["type"]
        if meta["pk"] and t.startswith(("VARCHAR", "TEXT", "CHAR")):
            values[name] = str(_uuid.uuid4())
        elif meta["pk"]:
            continue                       # integer rowid: let SQLite assign
        elif name == "danger_score":
            values[name] = float(score) if score is not None else 0.0
        elif name == "timestamp":
            values[name] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        elif t.startswith(("VARCHAR", "TEXT", "CHAR")):
            values[name] = "unknown"
        elif t.startswith(("FLOAT", "REAL", "DOUBLE")):
            values[name] = 0.0
        elif t.startswith(("INT", "BOOL")):
            values[name] = 0

    cols = [c for c in values if values[c] is not None or not info[c]["notnull"]]
    sql = (f"INSERT INTO alerts ({', '.join(cols)}) "
           f"VALUES ({', '.join('?' * len(cols))})")
    with _write_lock:
        cur = conn.execute(sql, tuple(values[c] for c in cols))
        conn.commit()
        row_id = values.get("id", cur.lastrowid)
    conn.close()
    logger.info(
        "Alert inserted: id=%d type=%s severity=%s camera=%s score=%s",
        row_id, alert_type, severity, camera_id, score,
    )
    return row_id


def get_alerts(
    db_path: str,
    status: str | None = None,
    limit: int = 100,
) -> list[sqlite3.Row]:
    """Fetch recent alerts, optionally filtered by status.

    Args:
        db_path: SQLite file path.
        status: 'new' | 'acknowledged' | 'dispatched' | 'resolved' | None (all).
        limit: Max rows to return.

    Returns:
        List of sqlite3.Row, newest first.
    """
    conn = get_connection(db_path)
    if status:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE status=? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return rows


@_retry_on_locked
def resolve_alert(db_path: str, alert_id: int) -> None:
    """Mark an alert as resolved.

    Args:
        db_path: SQLite file path.
        alert_id: Primary key of the alert to resolve.
    """
    conn = get_connection(db_path)
    with _write_lock:
        conn.execute(
            "UPDATE alerts SET status='resolved', resolved_at=CURRENT_TIMESTAMP WHERE id=?",
            (alert_id,),
        )
        conn.commit()
    conn.close()


# ────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────────────────────

def get_table_counts(db_path: str) -> dict[str, int]:
    """Return row counts for every table. Used by seed_data.py for summary logging.

    Returns:
        Dict mapping table name → row count.
    """
    tables = ["cameras", "tracks", "journeys",
              "watchlist_persons", "watchlist_vehicles", "alerts"]
    conn = get_connection(db_path)
    counts = {}
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) as n FROM {table}").fetchone()
        counts[table] = row["n"]
    conn.close()
    return counts


def verify_wal_mode(db_path: str) -> bool:
    """Confirm WAL journal mode is active. Returns True if WAL is enabled."""
    conn = get_connection(db_path)
    row = conn.execute("PRAGMA journal_mode").fetchone()
    conn.close()
    mode = row[0] if row else ""
    return mode.lower() == "wal"
