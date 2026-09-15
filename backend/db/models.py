"""
All 9 database models for Sentinel Gujarat.
Using SQLAlchemy with async session and SQLite/Postgres dual-engine compatibility.
"""

import enum
import uuid
from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, Enum as SAEnum, Float,
    ForeignKey, Index, Integer, JSON, LargeBinary, SmallInteger, String, Text,
    UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

import json
from sqlalchemy.types import TypeDecorator

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None


class EmbeddingVectorType(TypeDecorator):
    """
    Dual-compatibility vector type:
    Uses pgvector Vector(512) on PostgreSQL, and JSON / Text on SQLite.
    """
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and Vector is not None:
            return dialect.type_descriptor(Vector(512))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return json.dumps(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="ignore")
            except Exception:
                return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return None
        if isinstance(value, list):
            return value
        return value


Base = declarative_base()


class CrimeLevel(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"


class AlertSeverity(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class UserRole(str, enum.Enum):
    admin = "admin"
    operator = "operator"
    viewer = "viewer"
    officer = "officer"


# ── Model 1: Camera ────────────────────────────────────────────────────────────
class Camera(Base):
    __tablename__ = "cameras"

    # Integer surrogate PK. `camera_id` below is the human-facing string key
    # ("AHM-TRF-014"); `id` is what every camera_db_id foreign key points at —
    # EvidenceRecord.camera_db_id, Journey.camera_db_id and
    # ConnectionManager(camera_db_id=...) are all declared Integer, and
    # GET /iq/cameras/{camera_id} declares `camera_id: int`.
    # A UUID-string PK here makes that route return 422 for every camera.
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), nullable=True, index=True)
    name = Column(String(200), nullable=False)
    url = Column(Text, nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    district = Column(String(100), nullable=True)
    zone = Column(String(100), nullable=True)
    crime_level = Column(String(32), default="medium")
    is_restricted = Column(Boolean, default=False)
    department = Column(String(100), nullable=True)
    status = Column(String(32), default="ONLINE")
    is_online = Column(Boolean, default=True)
    is_deleted = Column(Boolean, default=False)
    ip_address = Column(String(64), nullable=True)
    protocol = Column(String(32), default="rtsp")
    risk_level = Column(String(32), default="medium")
    # Named per migration f3a1c9d27b44 (Day 7). services/camera_heartbeat.py
    # assigns camera.last_heartbeat_at directly — under any other name that
    # assignment creates a stray Python attribute that is never persisted, so
    # every camera reads as "never seen" and the offline sweep misfires.
    last_heartbeat_at = Column(DateTime, nullable=True)
    total_detections_today = Column(Integer, default=0)
    consecutive_failures = Column(Integer, default=0)
    deleted_at = Column(DateTime, nullable=True)
    location_label = Column(String(100), nullable=True)
    stream_url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Migration d16_camera_registry_enrichment (Model 1 gap-closure):
    # hardware type for the GIS map's "camera type" layer — free-text on
    # purpose (Fixed/PTZ/Dome/Bullet/... vary by department, a fixed enum
    # would reject a real department's own vocabulary at onboarding time).
    camera_type = Column(String(32), nullable=True, default="Fixed")
    # When the physical unit was installed — distinct from created_at (when
    # its REGISTRY ROW was created, which can be months after installation
    # for backfilled onboarding). Powers gap-analysis's ageing-infrastructure
    # section. Nullable: unknown install date is itself a real, reportable
    # gap, not an error.
    installed_at = Column(DateTime, nullable=True)
    # Manually-set operational state, independent of camera_heartbeat.py's
    # automatic ONLINE/OFFLINE — a camera pulled for repair is not "offline"
    # in the sense that phrase means elsewhere (unreachable/broken); it is
    # known-down for a known reason. camera_heartbeat.py's _apply_result
    # skips status transitions entirely while this is set, so the automatic
    # health check cannot fight a deliberate maintenance flag.
    maintenance_mode = Column(Boolean, default=False)
    maintenance_note = Column(Text, nullable=True)
    maintenance_since = Column(DateTime, nullable=True)


# ── Model 2: Detection ─────────────────────────────────────────────────────────
class Detection(Base):
    __tablename__ = "detections"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(64), nullable=False)
    track_id = Column(Integer, nullable=True)
    reid_id = Column(String(100), nullable=True)
    object_type = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_det_camera_time", "camera_id", "timestamp"),
        Index("idx_det_reid", "reid_id"),
    )


# ── Model 3: Alert ─────────────────────────────────────────────────────────────
class Alert(Base):
    __tablename__ = "alerts"

    # The LIVE database declares this VARCHAR(36) NOT NULL with no default and
    # holds 332 rows of UUIDs. An Integer autoincrement declaration cannot work
    # against that column: SQLite only autoincrements a genuine INTEGER PRIMARY
    # KEY, so every ORM insert bound NULL and failed with
    #     IntegrityError: NOT NULL constraint failed: alerts.id
    # That is precisely why loitering, crowd and every other detector writing
    # through SQLAlchemy produced nothing - each computed its alert correctly,
    # then could not persist it, and the failure surfaced only as a logged
    # traceback nobody was reading.
    #
    # KNOWN DEBT, deliberately left alone here: routed_alerts.alert_id,
    # push_delivery_log.alert_id, alert_feedback.alert_id and
    # Officer.current_alert_id are INTEGER columns referencing this UUID key,
    # so those joins cannot match. All are empty except routed_alerts (3 rows,
    # already unmatched before this change). Reconciling them means rebuilding
    # the alerts table and remapping 332 live rows - a migration worth doing
    # deliberately, not as a side effect of unblocking the behaviour engine.
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # The live table declares this VARCHAR(64) NOT NULL, and every one of the
    # 500+ existing rows holds a STRING camera key. The Integer/nullable
    # declaration that used to sit here disagreed on both type and nullability,
    # so a caller passing None - which the API path does whenever it has no
    # Camera row - failed with
    #     IntegrityError: NOT NULL constraint failed: alerts.camera_id
    # That surfaced the moment the behaviour engine started firing on real
    # video: loitering was correctly detected on two tracks and neither alert
    # could be written.
    camera_id = Column(String(64), nullable=False)
    alert_type = Column(String(100), nullable=False)
    # Present in the authoritative CREATE TABLE (backend/db.py:148) and written
    # by the face-watchlist and behaviour dispatch paths. Without it every
    # Alert(...) construction in those paths raises TypeError.
    track_id = Column(Integer, nullable=True)
    # The live table declares severity VARCHAR(32) NOT NULL with no DB default,
    # so a None here fails the insert outright. Detectors that predate the
    # severity system (loitering, crowd) never set it, which blocked every
    # behavioural alert from persisting even after the PK was fixed.
    # "medium" is the neutral placeholder; danger_score carries the real signal
    # and Day 13's severity pass overwrites this on the primary alert.
    severity = Column(String(32), nullable=False, default="medium")
    score = Column(Float, nullable=True)
    danger_score = Column(Float, nullable=False, default=5.0)
    score_breakdown = Column(JSON, nullable=True)
    description = Column(Text, nullable=True)
    reid_id = Column(String(100), nullable=True)
    plate_text = Column(String(32), nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    zone = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    department = Column(String(100), nullable=True)
    assigned_officer_id = Column(Integer, nullable=True)   # -> Officer.id
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    evidence_path = Column(Text, nullable=True)
    evidence_hash = Column(String(64), nullable=True)
    is_simulated = Column(Boolean, default=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Added columns for Day 5–18 features:
    subject_label = Column(String(200), nullable=True)
    confidence = Column(Float, nullable=True)
    snapshot_path = Column(Text, nullable=True)
    status = Column(String(32), default="new")
    false_positive_reason = Column(Text, nullable=True)
    lifecycle_status = Column(String(32), default="OPEN")
    reviewed_at = Column(DateTime, nullable=True)
    feedback = Column(String(32), nullable=True)
    iq_contribution = Column(Float, nullable=True)
    iq_breakdown_json = Column(Text, nullable=True)
    # Points at another Alert.id, so it must share that column's Integer type.
    # As String it silently never matches, and Day 11 dedup stops collapsing
    # duplicate alerts — the feed fills with repeats and nothing errors.
    merged_into_alert_id = Column(Integer, nullable=True)
    meta_json = Column(Text, nullable=True)
    global_id = Column(String(64), nullable=True)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_alert_severity", "severity"),
        Index("idx_alert_camera_time", "camera_id", "timestamp"),
        Index("idx_alert_resolved", "is_resolved"),
    )


# ── Model 4: JourneyEvent ──────────────────────────────────────────────────────
class JourneyEvent(Base):
    __tablename__ = "journey_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reid_id = Column(String(100), nullable=False, index=True)
    camera_id = Column(String(64), nullable=False)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    zone = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    is_cross_zone = Column(Boolean, default=False)
    object_class = Column(String(32), default="person")  # "vehicle" | "person"
    color = Column(String(32), nullable=True)             # "White", "Black", etc.
    subtype = Column(String(64), nullable=True)           # "SUV", "Sedan", etc.
    plate_text = Column(String(32), nullable=True)
    visual_score = Column(Float, default=0.0)
    final_score = Column(Float, default=0.0)
    match_type = Column(String(32), default="visual_reid") # "plate_confirmed" | "visual_reid"
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)



# ── Model 5: ANPREvent ─────────────────────────────────────────────────────────
class ANPREvent(Base):
    __tablename__ = "anpr_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(64), nullable=True)
    plate_text = Column(String(32), nullable=False, index=True)
    confidence = Column(Float, nullable=True)
    is_stolen = Column(Boolean, default=False)
    is_wanted = Column(Boolean, default=False)
    vehicle_type = Column(String(50), nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)


# ── Model 6: Officer ───────────────────────────────────────────────────────────
class Officer(Base):
    __tablename__ = "officers"

    # Integer PK: RoutedAlert.officer_id, AckEvent.officer_id and
    # PushSubscription.officer_id are all declared
    # Column(Integer, ForeignKey("officers.id")). A UUID-string PK here breaks
    # every one of those joins, so nearest-officer dispatch assigns nobody.
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    badge_number = Column(String(50), unique=True, nullable=True)
    phone = Column(String(32), nullable=True)
    zone = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    current_lat = Column(Float, nullable=True)
    current_lon = Column(Float, nullable=True)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    status = Column(String(32), default="AVAILABLE")
    is_available = Column(Boolean, default=True)
    current_alert_id = Column(Integer, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    last_gps_update = Column(DateTime, nullable=True)


# ── Model 7: WatchlistPlate ────────────────────────────────────────────────────
class WatchlistPlate(Base):
    __tablename__ = "watchlist_plates"

    plate = Column(String(32), primary_key=True)
    plate_number = Column(String(32), nullable=True)
    reason = Column(String(200), nullable=True)
    category = Column(String(50), default="stolen")
    added_at = Column(DateTime, default=datetime.utcnow)
    added_by = Column(String(100), nullable=True)
    # Migration d17_watchlist_plate_active: deactivate (case resolved,
    # entry retired) without deleting — a live plate-watchlist hit's Alert
    # row references this plate by text, and history should not vanish
    # along with a row someone removes. Mirrors WatchlistPerson.active.
    active = Column(Boolean, default=True)


# ── Model 8: WatchlistPerson ───────────────────────────────────────────────────
class WatchlistPerson(Base):
    __tablename__ = "watchlist_persons"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reid_id = Column(String(100), unique=True, index=True, nullable=True)
    name = Column(String(200), nullable=False)
    embedding = Column(Text, nullable=True)
    face_embedding = Column(LargeBinary if False else Text, nullable=True)
    reason = Column(String(200), nullable=True)
    danger_level = Column(String(32), default="high")
    photo_path = Column(Text, nullable=True)
    reference_photo_path = Column(Text, nullable=True)
    active = Column(Boolean, default=True)
    embedding_dim = Column(Integer, default=512)
    added_at = Column(DateTime, default=datetime.utcnow)


# ── Model 9: User (System users) ──────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=True)
    username = Column(String(64), nullable=True, unique=True)
    badge_number = Column(String(50), unique=True, nullable=True)
    role = Column(String(32), default="viewer")
    password_hash = Column(String(200), nullable=True)
    hashed_password = Column(String(200), nullable=True)
    zone = Column(String(100), nullable=True)
    # Which of the 26 departments this user belongs to. NULL means
    # state-level (unscoped) and should be reserved for admin/audit roles.
    # Camera.department and Alert.department already existed, but nothing
    # linked a USER to a department, so the boundary could not be enforced -
    # any authenticated account could enumerate every department's cameras.
    department = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Model 10: AuditLog ────────────────────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), nullable=True)
    action = Column(String(64), nullable=True)
    event_type = Column(String(64), nullable=True, index=True)
    resource_type = Column(String(64), nullable=True)
    resource_id = Column(String(64), nullable=True)
    entity_id = Column(String(64), nullable=True, index=True)
    entity_type = Column(String(32), nullable=True)
    actor_id = Column(String(64), nullable=True)
    model_version = Column(String(32), nullable=True)
    details = Column(Text, nullable=True)
    payload = Column(JSON, nullable=True)
    payload_hash = Column(String(64), nullable=True)
    prev_log_hash = Column(String(64), nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    logged_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# ── Model 11: GlobalPerson & Review ───────────────────────────────────────────
#
# The five models below mirror migration a1b2c3d4e5f6 (Day 6 ReID tables),
# 23d3352229b6 (Day 9 journey query log), and the legacy `journeys` CREATE
# TABLE in backend/db.py. Columns present in those schemas but absent here mean
# every ORM write to the table raises TypeError at construction time — which is
# how Day 9 and Day 12 ReID review stopped working while detection carried on
# looking healthy.

class GlobalPerson(Base):
    __tablename__ = "global_persons"
    id = Column(Integer, primary_key=True, autoincrement=True)
    reid_id = Column(String(100), unique=True, index=True)
    label = Column(String(100), nullable=True)
    # ── migration a1b2c3d4e5f6 ──
    representative_embedding = Column(LargeBinary, nullable=True)
    faiss_index_position = Column(Integer, nullable=True, unique=True)
    first_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    total_sightings = Column(Integer, default=1, nullable=False)
    # DPDP retention fields — a person record is purged when
    # retention_expires_at passes UNLESS retention_hold is set (court order).
    retention_expires_at = Column(DateTime, nullable=True)
    retention_hold = Column(Boolean, default=False, nullable=False)
    legal_basis = Column(String(256), nullable=False,
                         default="routine_public_safety_monitoring")
    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ReIDReviewItem(Base):
    __tablename__ = "reid_review_items"
    id = Column(Integer, primary_key=True, autoincrement=True)
    local_track_id = Column(Integer, ForeignKey("tracks.id"), nullable=True)
    candidate_global_person_id = Column(
        Integer, ForeignKey("global_persons.id"), nullable=True)
    similarity_score = Column(Float, nullable=True)
    # The two images an officer compares side by side in the review queue.
    # crop_image_path is fed from Track.best_crop_path — if either is NULL the
    # comparison UI degrades to two placeholders.
    crop_image_path = Column(String(512), nullable=True)
    candidate_reference_image_path = Column(String(512), nullable=True)
    status = Column(String(16), default="PENDING", nullable=False, index=True)
    # NOTE: no ForeignKey to users.id here. The migration declares this Integer
    # against users.id, but User.id in this model file is String(50) — emitting
    # the constraint would create a type-mismatched FK. Column only.
    reviewed_by_user_id = Column(Integer, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    time_gap_seconds = Column(Integer, nullable=True)
    calibration_log_id = Column(
        Integer, ForeignKey("reid_calibration_log.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ReIDCalibrationLog(Base):
    __tablename__ = "reid_calibration_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    similarity_score = Column(Float, nullable=True)
    decision = Column(String(32), nullable=True)
    was_correct = Column(Boolean, nullable=True)   # NULL until an officer rules
    time_gap_seconds = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class JourneyQueryLog(Base):
    """Day 9 §2 — per-query RBAC audit trail for journey lookups."""
    __tablename__ = "journey_query_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    reid_id = Column(String(100), nullable=True)
    user_id = Column(Integer, nullable=True)
    role = Column(String(16), nullable=True)
    global_id_queried = Column(Integer, nullable=True)
    query_time = Column(DateTime, default=datetime.utcnow)
    source_ip = Column(String(64), nullable=True)
    # DENIED rows are the point of this table — it records refused lookups,
    # not just successful ones.
    outcome = Column(String(16), nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Journey(Base):
    __tablename__ = "journeys"
    id = Column(Integer, primary_key=True, autoincrement=True)
    # ── legacy columns (backend/db.py:115-123), kept exactly as-is ──
    global_id = Column(String(64), nullable=True)
    camera_id = Column(String(64), nullable=True)
    track_id = Column(Integer, nullable=True)
    entry_timestamp = Column(Float, nullable=True)
    exit_timestamp = Column(Float, nullable=True)
    sequence_order = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    # ── migration a1b2c3d4e5f6 ──
    global_person_id = Column(Integer, nullable=True, index=True)
    camera_db_id = Column(Integer, nullable=True)
    local_track_id = Column(Integer, nullable=True)
    seen_at = Column(DateTime, nullable=True)
    confidence = Column(Float, nullable=True)
    # Shown verbatim next to any match below the validated threshold.
    confidence_caveat = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Track(Base):
    __tablename__ = "tracks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), nullable=False)
    track_id = Column(Integer, nullable=False)
    global_id = Column(String(64), nullable=True)
    first_seen_frame = Column(Integer, nullable=True)
    first_seen_timestamp = Column(Float, nullable=True)
    last_seen_frame = Column(Integer, nullable=True)
    last_seen_timestamp = Column(Float, nullable=True)
    total_frames = Column(Integer, default=1)

    # The three columns below, created_at/updated_at, and the UNIQUE constraint
    # all exist in the authoritative CREATE TABLE in backend/db.py (lines
    # 98-113). This ORM model must mirror that table, not a subset of it.
    #
    # When they were absent, Track(...) in connection_manager raised
    # "'best_confidence' is an invalid keyword argument for Track" on EVERY
    # track, and _ensure_track_row swallowed it as a warning — so detection
    # kept running while ReID and face-watchlist dispatch silently stopped
    # firing. Nothing failed loudly; the features just went quiet.
    best_crop_path = Column(Text, nullable=True)      # crop shown in review queue
    best_confidence = Column(Float, nullable=True)
    body_embedding = Column(LargeBinary, nullable=True)  # OSNet body vector
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Load-bearing, not decorative: _find_or_create_track_row relies on this
    # constraint raising IntegrityError to resolve the concurrent-create race
    # between two frames of the same track. Without it in the ORM metadata,
    # create_all() builds a table that accepts duplicates and the race handler
    # never runs.
    __table_args__ = (
        UniqueConstraint("camera_id", "track_id", name="uq_tracks_camera_track"),
    )


class Evidence(Base):
    __tablename__ = "evidence"
    id = Column(Integer, primary_key=True, autoincrement=True)
    # References Alert.id (String UUID or Integer).
    alert_id = Column(String(36), nullable=True, index=True)
    status = Column(String(32), default="COMPLETE")
    failure_reason = Column(String(200), nullable=True)
    camera_str_id = Column(String(64), nullable=True)
    camera_db_id = Column(Integer, nullable=True)
    sha256 = Column(String(64), nullable=True)
    clip_path = Column(String(256), nullable=True)
    pdf_path = Column(String(256), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    codec = Column(String(32), nullable=True)
    frame_count = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    event_time = Column(DateTime, default=datetime.utcnow)
    clip_start_time = Column(DateTime, nullable=True)
    clip_end_time = Column(DateTime, nullable=True)
    requested_pre_seconds = Column(Float, default=5.0)
    requested_post_seconds = Column(Float, default=5.0)
    actual_pre_seconds = Column(Float, default=5.0)
    actual_post_seconds = Column(Float, default=5.0)
    actual_duration_seconds = Column(Float, default=10.0)
    partial_pre_roll = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class ZoneIncident(Base):
    __tablename__ = "zone_incidents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    zone = Column(String(100), nullable=True)
    center_lat = Column(Float, nullable=True)
    center_lon = Column(Float, nullable=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    alert_ids = Column(Text, nullable=True)


class CameraCalibration(Base):
    __tablename__ = "camera_calibration"
    camera_id = Column(String(64), primary_key=True)
    px_per_meter = Column(Float, default=30.0)
    calibration_method = Column(String(64), default="default")
    calibrated_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)


class RoutedAlert(Base):
    __tablename__ = "routed_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False)
    assigned_officer = Column(Integer, ForeignKey("officers.id"), nullable=True)
    status = Column(String(24), nullable=False, default="PENDING")
    assigned_at = Column(DateTime, nullable=True)
    ack_at = Column(DateTime, nullable=True)
    escalated_at = Column(DateTime, nullable=True)
    escalation_note = Column(Text, nullable=True)
    escalation_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_routed_alerts_alert_id", "alert_id"),
        Index("ix_routed_alerts_status_esc", "status", "escalation_count"),
    )


class PushSubscription(Base):
    __tablename__ = "push_subscriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    officer_id = Column(Integer, ForeignKey("officers.id"), nullable=False)
    endpoint = Column(Text, nullable=False, unique=True)
    p256dh = Column(Text, nullable=False)
    auth = Column(Text, nullable=False)
    device_label = Column(String(128), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_push_sub_officer", "officer_id"),
    )


class PushDeliveryLog(Base):
    __tablename__ = "push_delivery_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False)
    subscription_id = Column(Integer, nullable=True)
    officer_id = Column(Integer, ForeignKey("officers.id"), nullable=True)
    status = Column(String(16), nullable=False)
    status_code = Column(Integer, nullable=True)
    error_detail = Column(Text, nullable=True)
    sent_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_push_log_alert", "alert_id"),
        Index("idx_push_log_officer", "officer_id"),
    )


class AlertFeedback(Base):
    __tablename__ = "alert_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False, unique=True)
    officer_id = Column(Integer, ForeignKey("officers.id"), nullable=True)
    verdict = Column(String(32), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_feedback_verdict_date", "verdict", "created_at"),
    )


class FeedbackFlagLog(Base):
    __tablename__ = "feedback_flag_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_name = Column(String(100), nullable=False)
    count = Column(Integer, nullable=False)
    flagged_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    resolved = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("idx_flag_log_zone_date", "zone_name", "flagged_at", "resolved"),
    )


# ── Model 25: VaultEntry (Active Learning & HITL Hard Negative / Positive Vault) ──
class VaultEntry(Base):
    __tablename__ = "vault_entries"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    compartment = Column(String(32), nullable=False, default="probationary", index=True)
    trust_level = Column(String(32), nullable=False, default="probationary")
    entity_type = Column(String(32), nullable=False, default="vehicle")

    alert_id = Column(String(36), ForeignKey("alerts.id"), nullable=True)
    camera_id = Column(String(64), nullable=False, index=True)
    captured_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    stored_at = Column(DateTime, default=datetime.utcnow)

    crop_minio_path = Column(Text, nullable=True)
    embedding_vector = Column(EmbeddingVectorType, nullable=True)
    embedding_hash = Column(String(64), nullable=True, index=True)

    camera_zone = Column(String(128), nullable=True)
    lighting_condition = Column(String(32), nullable=True, index=True)
    vehicle_color_hsv = Column(JSON, nullable=True)
    vehicle_type = Column(String(32), nullable=True)
    plate_text = Column(String(32), nullable=True)

    ai_confidence = Column(Float, nullable=True)
    ai_cosine_distance = Column(Float, nullable=True)
    model_version = Column(String(32), nullable=True)

    # Uncertainty sampling: |ai_cosine_distance - active threshold|
    boundary_distance = Column(Float, nullable=True)
    anchor_group_id = Column(String(64), nullable=True, index=True)

    training_eligible = Column(Boolean, default=False, nullable=False)
    used_in_training = Column(Boolean, default=False, nullable=False)
    last_used_job_id = Column(String(36), nullable=True)

    crop_expires_at = Column(DateTime, nullable=True)
    embedding_expires_at = Column(DateTime, nullable=True)
    is_gold_retained = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    reviews = relationship("OfficerReview", back_populates="vault_entry", lazy="select")


# ── Model 26: OfficerReview (Server-Authoritative Time Gating & Blind Reviews) ──
class OfficerReview(Base):
    __tablename__ = "officer_reviews"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    review_session_id = Column(String(36), unique=True, nullable=True, index=True)
    vault_entry_id = Column(String(36), ForeignKey("vault_entries.id", ondelete="SET NULL"), nullable=True)
    alert_id = Column(String(36), ForeignKey("alerts.id"), nullable=True)

    officer_id = Column(String(64), nullable=False, index=True)
    officer_role = Column(String(32), nullable=False)
    officer_zone = Column(String(128), nullable=True)

    decision = Column(String(32), nullable=False)
    review_duration_sec = Column(Float, nullable=False)
    client_reported_duration_sec = Column(Float, nullable=True)
    duration_discrepancy_flag = Column(Boolean, nullable=False, default=False)

    confidence_shown = Column(Boolean, default=False)
    ai_confidence_at_review = Column(Float, nullable=True)

    agreement_round = Column(Integer, default=1)
    agreement_partner_id = Column(String(64), nullable=True)
    agreed = Column(Boolean, nullable=True)

    officer_notes = Column(Text, nullable=True)
    review_started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    review_completed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    vault_entry = relationship("VaultEntry", back_populates="reviews")


# ── Model 27: ReviewPair (Blind Double-Review Orchestration) ──────────────────
class ReviewPair(Base):
    __tablename__ = "review_pairs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    alert_id = Column(String(36), ForeignKey("alerts.id", ondelete="CASCADE"), nullable=True)
    first_review_id = Column(String(36), ForeignKey("officer_reviews.id", ondelete="CASCADE"), nullable=True)
    second_review_id = Column(String(36), ForeignKey("officer_reviews.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(32), nullable=False, default="awaiting_second")
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Model 28: CameraCalibrationProfile (Per-Weather Dynamic Profiles) ─────────
class CameraCalibrationProfile(Base):
    __tablename__ = "camera_calibration_profiles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(64), nullable=False, index=True)
    profile_version = Column(Integer, default=1)
    calibrated_at = Column(DateTime, default=datetime.utcnow)
    calibration_job_id = Column(String(36), nullable=True)

    hsv_profiles = Column(JSON, nullable=False)
    reid_thresholds = Column(JSON, nullable=False)

    prev_vehicle_day_threshold = Column(Float, nullable=True)
    prev_vehicle_night_threshold = Column(Float, nullable=True)
    delta_vehicle_day = Column(Float, nullable=True)
    delta_vehicle_night = Column(Float, nullable=True)

    sample_count = Column(Integer, nullable=True)
    calibration_score = Column(Float, nullable=True)
    approved = Column(Boolean, default=False)
    approved_by = Column(String(64), nullable=True)
    is_active = Column(Boolean, default=False)

    pending_supervisor_review = Column(Boolean, nullable=False, default=False)
    anchor_eval_report = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Model 29: AuditCheckpoint (Checkpointed Hash-Chain Verification) ──────────
class AuditCheckpoint(Base):
    __tablename__ = "audit_checkpoint"

    id = Column(SmallInteger, primary_key=True, default=1)
    last_verified_id = Column(Integer, nullable=False, default=0)
    last_verified_at = Column(DateTime, nullable=True)
    last_status = Column(String(16), nullable=False, default="unknown")
    last_broken_links = Column(JSON, nullable=True)


# ── Model 30: TrafficViolation (Wrong-Way & Traffic Intelligence) ───────────
class TrafficViolation(Base):
    __tablename__ = "traffic_violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    violation_uuid = Column(String(64), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    violation_type = Column(String(32), nullable=False)  # WRONG_WAY, TRIPLE_RIDING, NO_HELMET
    camera_id = Column(String(32), nullable=False, index=True)
    track_id = Column(Integer, nullable=False)
    vehicle_class = Column(String(32), nullable=False)

    # Triple Riding & Ridership
    rider_count = Column(Integer, nullable=False, default=1)
    majority_ratio = Column(Float, nullable=False, default=1.0)
    counting_method = Column(String(64), nullable=False, default="direct")

    # Flow & Speed
    flow_angle_deg = Column(Float, nullable=True, default=0.0)
    speed_kmh = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False, default=0.9)

    # ANPR
    license_plate = Column(String(32), nullable=True, index=True)
    plate_confidence = Column(Float, nullable=True)
    plate_format_valid = Column(Boolean, nullable=False, default=False)

    # Helmet & Compounded Fines
    helmet_results = Column(JSON, nullable=True)
    violation_breakdown = Column(JSON, nullable=True)
    challan_amount_inr = Column(Integer, nullable=False, default=1500)

    # Evidence
    crop_path = Column(String(512), nullable=False)
    plate_crop_path = Column(String(512), nullable=True)
    evidence_clip_path = Column(String(512), nullable=True)
    evidence_clip_hash = Column(String(64), nullable=True)
    composite_evidence_path = Column(String(512), nullable=True)
    evidence_panel_hashes = Column(JSON, nullable=True)
    composite_hash = Column(String(64), nullable=True)

    # Environmental Context
    weather_regime = Column(String(16), nullable=True)
    camera_angle_deg = Column(Float, nullable=True)
    michelson_contrast = Column(Float, nullable=True)

    # Review workflow
    challan_status = Column(String(16), nullable=False, default="pending_review", index=True)
    reviewed_by = Column(String(64), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    dismiss_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# ── Model 31: ReviewAuditLog (Append-Only Legal Chain of Custody) ───────────
class ReviewAuditLog(Base):
    __tablename__ = "review_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    violation_id = Column(Integer, ForeignKey("traffic_violations.id"), nullable=False, index=True)
    action = Column(String(16), nullable=False)  # confirmed | dismissed | escalated
    reviewer_id = Column(String(64), nullable=False, index=True)
    reviewer_ip = Column(String(45), nullable=True)
    reason = Column(Text, nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)


# ── Model 32: CameraHomographyCalibration (Homography & Ground Control) ─────
class CameraHomographyCalibration(Base):
    __tablename__ = "camera_calibrations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(32), nullable=False, index=True)
    homography_matrix = Column(Text, nullable=False)  # JSON 9-float array
    reprojection_error_m = Column(Float, nullable=False)
    camera_angle_deg = Column(Float, nullable=False, default=0.0)
    gps_anchor_lat = Column(Float, nullable=True)
    gps_anchor_lon = Column(Float, nullable=True)
    bearing_deg = Column(Float, nullable=True, default=0.0)
    held_out_error_m = Column(Float, nullable=True)
    quality_gate = Column(String(16), nullable=False, default="good")  # good | degraded | rejected
    # `quality_gate` describes POSITIONAL accuracy against surveyed points, and
    # cannot express a calibration that is validated for speed but not for
    # position. These two say it directly rather than leaving it to be inferred
    # from `reprojection_error_m` — which produced a row whose stored gate said
    # `degraded` while recomputing it from the error said `rejected`.
    positional_error_is_measured = Column(Boolean, nullable=False, default=False)
    speed_validated = Column(Boolean, nullable=False, default=False)
    calibration_method = Column(String(48), nullable=True)
    notes = Column(Text, nullable=True)
    calibrated_by = Column(String(64), nullable=False)
    calibrated_on = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# ── Model 33: VehicleTrack (Track-Level Macro Measurement) ───────────────────
class VehicleTrack(Base):
    __tablename__ = "vehicle_track"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), nullable=False, index=True)
    track_id = Column(Integer, nullable=False)
    vehicle_class = Column(String(32), nullable=True)
    first_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    entry_lat = Column(Float, nullable=True)
    entry_lon = Column(Float, nullable=True)
    exit_lat = Column(Float, nullable=True)
    exit_lon = Column(Float, nullable=True)
    heading_deg = Column(Float, nullable=True)
    speed_kmh = Column(Float, nullable=True)       # NULL if uncalibrated or rejected
    # Image-plane speed, which needs no homography and so is available on
    # every camera. Not convertible to km/h and not comparable between
    # cameras — it exists so a camera can be compared against its own history,
    # which is all a congestion anomaly ever asks.
    speed_px_s = Column(Float, nullable=True)

    # What the ANPR engine read for this vehicle, and how sure it was.
    #
    # The pipeline already reads a plate per track and set it on an in-memory
    # dict that was dropped when the track flushed. 179,569 tracks were
    # persisted with no plate recorded anywhere in the database — so live
    # read rate could not be computed at all, per camera or otherwise, and
    # there was no way to tell a plate the detector never found from one it
    # found and misread. Those are different failures with different fixes.
    plate_text = Column(String, nullable=True)
    plate_confidence = Column(Float, nullable=True)
    plate_state = Column(String, nullable=True)
    plate_votes = Column(Integer, nullable=True)      # frames that agreed
    plate_locked = Column(Boolean, nullable=True)     # voter reached consensus
    # Whether a plate region was located at all, independent of whether it
    # could be read. Capture rate and read rate are separate metrics and
    # blending them hides which stage is failing.
    plate_detected = Column(Boolean, nullable=True)
    speed_ci_kmh = Column(Float, nullable=True)    # 95% CI half-width
    path_length_m = Column(Float, nullable=True)
    n_frames = Column(Integer, nullable=False, default=1)
    detector_conf = Column(Float, nullable=True)
    calibration_err_m = Column(Float, nullable=True)
    quality = Column(String(32), nullable=False, default="good")  # good | degraded | speed_unavailable

    __table_args__ = (
        Index("idx_vt_cam_time", "camera_id", "first_seen"),
    )


# ── Model 33b: TrackAppearance ─────────────────────────────────────────────
class TrackAppearance(Base):
    """One appearance vector per completed vehicle track.

    WHY THIS TABLE EXISTS
      The pipeline already computes an OSNet-IBN embedding for every detection
      — it has to, because BoT-SORT associates on appearance — and then threw
      all of them away except a 12% sample harvested into the training vault.
      That meant cross-camera re-identification could only ever run offline,
      over clips, re-extracting what the live system had already computed
      seconds earlier and discarded.

      Persisting one vector per track is what lets a journey be continued
      across a camera where the PLATE could not be read: the vehicle's
      appearance at the cameras that did read it is matched against every
      track the intermediate cameras saw. That is the difference between a
      trajectory that stops at the first unreadable plate and one that does
      not.

      Stored as a mean over the track's frames, L2-normalised, which is
      steadier than any single frame and is the same reduction the offline
      re-ID scripts use — so a vector written here is directly comparable with
      one computed from a clip.

    RETENTION
      `expires_at` is set on write. Appearance of a vehicle is personal data;
      it is kept for the investigation window and then removed, in the same
      spirit as the vault's own 180-day expiry.
    """
    __tablename__ = "track_appearance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), nullable=False, index=True)
    track_id = Column(Integer, nullable=False)
    # Namespaced id, because BoT-SORT restarts numbering on every camera.
    global_track_key = Column(String(96), nullable=False, index=True)
    vehicle_class = Column(String(32), nullable=True)
    plate_text = Column(String(32), nullable=True, index=True)
    first_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen = Column(DateTime, nullable=False, default=datetime.utcnow,
                       index=True)
    n_frames = Column(Integer, nullable=False, default=1)
    # 512-d, L2-normalised. JSON on SQLite, pgvector on PostgreSQL.
    embedding = Column(EmbeddingVectorType, nullable=True)
    expires_at = Column(DateTime, nullable=True, index=True)

    # NOT unique on (camera_id, track_id). BoT-SORT numbers tracks from 1 on
    # every pipeline start, so a restarted pipeline reuses ids already stored.
    # A unique constraint here failed the insert — and because the row shares
    # a transaction with the track itself, it rolled back the track, its plate
    # and its speed too. Measured on the first restart after it shipped.
    __table_args__ = (
        Index("idx_ta_cam_track", "camera_id", "track_id"),
        Index("idx_ta_cam_lastseen", "camera_id", "last_seen"),
    )


# ── Model 34: CameraMetrics1M (1-Minute Rollup for Dashboard) ───────────────
class CameraMetrics1M(Base):
    __tablename__ = "camera_metrics_1m"

    bucket_start = Column(DateTime, primary_key=True, nullable=False)
    camera_id = Column(String(64), primary_key=True, nullable=False)
    vehicle_count = Column(Integer, nullable=False, default=0)
    count_by_class = Column(JSON, nullable=True)
    median_speed_kmh = Column(Float, nullable=True)
    p15_speed_kmh = Column(Float, nullable=True)
    p85_speed_kmh = Column(Float, nullable=True)
    # Image-plane equivalent, populated on every camera. Congestion is
    # detected as a drop against this camera's own baseline, so the unit only
    # has to be consistent — not metric. `speed_basis` records which of the
    # two the anomaly engine actually used, so a reader is never left to guess
    # whether a figure is metric.
    median_speed_px_s = Column(Float, nullable=True)
    speed_basis = Column(String, nullable=True)   # "kmh" | "px_s"
    speed_samples = Column(Integer, nullable=False, default=0)
    confidence = Column(Float, nullable=True)
    congestion_index = Column(Float, nullable=True)  # NULL until free_flow_speed bootstraps
    health_status = Column(String(32), nullable=False, default="ONLINE")

    __table_args__ = (
        Index("idx_cm1m_cam_bucket", "camera_id", "bucket_start"),
    )


# ── Model 35: ShadowViolation (Pre-Live Validation) ─────────────────────────
class ShadowViolation(Base):
    __tablename__ = "shadow_violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    violation_uuid = Column(String(64), unique=True, nullable=False)
    camera_id = Column(String(32), nullable=False)
    track_id = Column(Integer, nullable=False)
    rider_count = Column(Integer, nullable=False)
    majority_ratio = Column(Float, nullable=False)
    speed_kmh = Column(Float, nullable=False)
    weather_regime = Column(String(16), nullable=False)
    ground_truth = Column(String(16), nullable=False, default="pending")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# ── Model 34: RetrainingItem (Active Learning Feedback Queue) ───────────────
class RetrainingItem(Base):
    __tablename__ = "retraining_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    violation_id = Column(Integer, nullable=False)
    crop_path = Column(String(512), nullable=False)
    label = Column(String(32), nullable=False)
    rider_count = Column(Integer, nullable=False)
    majority_ratio = Column(Float, nullable=False)
    camera_id = Column(String(32), nullable=False)
    weather_regime = Column(String(16), nullable=False)
    used_in_training = Column(Boolean, nullable=False, default=False)
    added_at = Column(DateTime, nullable=False, default=datetime.utcnow)












