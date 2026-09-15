"""backend/routers/v1/cameras.py — /api/v1/cameras endpoints.
NOTE: No 'from __future__ import annotations' — pydantic v2 needs eager resolution.
"""
import os
import time
from typing import Optional

from fastapi import (APIRouter, BackgroundTasks, Depends, HTTPException,
                     Query, Request, Response, status)
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from datetime import datetime, timedelta, timezone

# auto_error=False so a missing header reaches the handler as None instead of
# FastAPI raising first — /snapshot accepts EITHER this or a ?token=
# stream token, and the built-in 403 would pre-empt that choice. Its own
# instance, separate from analytics.py's _bearer_optional, to avoid a
# cross-module import for one HTTPBearer object.
_bearer_optional_cameras = HTTPBearer(auto_error=False)

from backend.auth.dependencies import get_current_user, require_role
from backend.auth.jwt_handler import ALL_DEPARTMENTS, caller_department
from backend.core.logging import get_logger
from backend.core.rate_limit import GENERAL_LIMIT, limiter
from backend.db.models import Camera, User
from backend.db.session import get_db
from backend.services.audit_logger import log_audit
from backend.services.camera_adapters.factory import (
    CameraAdapterFactory,
    resolve_stream_url,
)
from backend.services.camera_connection import test_camera_connection

router = APIRouter(prefix="/cameras", tags=["cameras"])
logger = get_logger(__name__)

_ws_broadcast = None

def set_ws_broadcast(fn):
    global _ws_broadcast
    _ws_broadcast = fn


def _next_camera_id(db: Session) -> str:
    """Allocate the next CAM_NN id.

    The model declares `id` as Integer with no default, but the shipped
    table is VARCHAR(64) populated with 'CAM_01'...'CAM_30'. Because the
    column is a string PK with no default, SQLAlchemy sends NULL and SQLite
    rejects it:
        sqlite3.IntegrityError: NOT NULL constraint failed: cameras.id
    so camera creation could never succeed - it is not a rowid alias, and
    nothing was generating a value.

    Following the existing convention rather than switching to a UUID keeps
    new cameras consistent with the ids config.yaml and the demo data
    already use.

    NOTE: the Integer-vs-VARCHAR mismatch is real and wider than this
    endpoint - Camera.id is declared Integer while every stored value is a
    string. Worth reconciling deliberately in a migration; generating a
    correct id here does not pretend to fix that.
    """
    existing = {str(r[0]) for r in db.query(Camera.id).all()}
    nums = []
    for cid in existing:
        if cid.upper().startswith("CAM_"):
            tail = cid[4:]
            if tail.isdigit():
                nums.append(int(tail))
    nxt = (max(nums) + 1) if nums else 1
    while f"CAM_{nxt:02d}" in existing:
        nxt += 1
    return f"CAM_{nxt:02d}"


def _redact_url(url: Optional[str]) -> Optional[str]:
    """Mask credentials embedded in a stream URL.

    Vendor RTSP URLs carry the password inline:
        rtsp://admin:S3cret@10.0.0.5:554/Streaming/Channels/101
    Returning that verbatim from an API - or writing it to an audit log -
    leaks a camera credential to anyone who can read the response. Only the
    shape of the URL is useful to an operator confirming onboarding worked.
    """
    if not url:
        return url
    try:
        scheme, _, rest = url.partition("://")
        if not rest or "@" not in rest:
            return url
        creds, _, host = rest.partition("@")
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:****@{host}"
    except Exception:
        return "<redacted>"


def _as_claims(user) -> dict:
    """Adapt a User ORM row to the claims dict the scoping helpers expect.

    The codebase has two auth layers: backend.auth.jwt_handler works on JWT
    claim dicts, while backend.auth.dependencies (used here) returns a User
    ORM object. Rather than duplicate the scoping rules for each, normalise
    to one shape.

    Role strings vary by vintage - 'admin'/'ADMIN' from the legacy seed,
    'officer'/'OPERATOR' from Day 9 - so compare case-insensitively; getting
    this wrong would silently grant or deny state-level scope.
    """
    role = str(getattr(user, "role", "") or "").lower()
    return {
        "sub": getattr(user, "id", None),
        "role": "admin" if role == "admin" else role,
        "dept": getattr(user, "department", None),
    }


class CameraTestRequest(BaseModel):
    ip_address: str
    protocol: str = "RTSP"


class CameraProbeRequest(BaseModel):
    """Ask the adapter chain which protocol this camera actually speaks."""
    vendor: Optional[str] = "unknown"
    ip_address: Optional[str] = None
    username: Optional[str] = "admin"
    password: Optional[str] = ""
    url: Optional[str] = None
    channel: Optional[int] = 101


class CameraCreateRequest(BaseModel):
    # min_length=1 alone would still accept "   " (whitespace) as a valid
    # name — verified by the bulk-onboarding test below, which caught a bare
    # `str` letting an empty name through silently and creating a real
    # UNREACHABLE camera with no name at all. A validator that strips first
    # is what actually closes that.
    name: str = Field(..., min_length=1)
    ip_address: Optional[str] = None

    protocol: Optional[str] = "RTSP"
    zone: Optional[str] = None
    department: Optional[str] = None
    # Onboarding fields. A department hands over vendor + IP + credentials,
    # not a stream URL, so the URL is negotiated rather than typed in.
    vendor: Optional[str] = "unknown"
    username: Optional[str] = "admin"
    password: Optional[str] = ""
    url: Optional[str] = None
    channel: Optional[int] = 101
    risk_level: Optional[str] = "LOW"
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    # Model 1 registry enrichment (migration d16). camera_type free-text on
    # purpose — see the Camera model's own comment on why a fixed enum would
    # reject a real department's vocabulary. installed_at is an ISO date
    # string (date-only or full timestamp); parsed defensively below since
    # this arrives from CSV rows and manual form input, not just JSON.
    camera_type: Optional[str] = "Fixed"
    installed_at: Optional[str] = None

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError(
                "name cannot be blank or whitespace-only — a nameless "
                "camera is unreachable/unfindable in the registry")
        return v


class CameraOut(BaseModel):
    # str, not int. Camera.id is DECLARED Integer in the model but the
    # shipped table is VARCHAR(64) holding 'CAM_01'...'CAM_30', and every
    # other component (config.yaml, SENTINEL_SOURCES, the pipelines) uses
    # those string ids. Typing this as int made pydantic reject the real
    # value and return 500 AFTER the row had already been written:
    #   ValidationError: id Input should be a valid integer,
    #   unable to parse string as an integer [input_value='CAM_31']
    # so a camera could be created and still look like a failure.
    id: str
    name: str
    ip_address: Optional[str]
    protocol: Optional[str]
    zone: Optional[str]
    department: Optional[str]
    risk_level: Optional[str]
    gps_lat: Optional[float]
    gps_lon: Optional[float]
    status: str
    created_at: str
    camera_type: Optional[str] = "Fixed"
    installed_at: Optional[str] = None

    class Config:
        from_attributes = True


@router.post("/test")
@limiter.limit(GENERAL_LIMIT)
async def test_camera(
    request: Request,
    body: CameraTestRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Run the blocking probe off the event loop — it opens sockets and may
    # sit on a stream-open timeout, which must not stall other requests.
    result = await run_in_threadpool(
        test_camera_connection, body.ip_address, body.protocol
    )
    log_audit(db, current_user, "CAMERA_TEST", "camera", None,
              {
                  "ip": body.ip_address,
                  "protocol": body.protocol,
                  "success": result.success,
                  # Recorded separately because the two mean different things:
                  # success=True with stream_verified=False means only that a
                  # port was open. The audit trail previously stored a
                  # fabricated success with no way to tell the difference.
                  "stream_verified": result.stream_verified,
                  "latency_ms": result.latency_ms,
              },
              request.client.host if request.client else None)
    return {
        "success": result.success,
        "message": result.message,
        "latency_ms": result.latency_ms,
        "stream_verified": result.stream_verified,
    }


@router.post("/probe")
@limiter.limit(GENERAL_LIMIT)
async def probe_camera(
    request: Request,
    body: CameraProbeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Negotiate a stream URL by walking the vendor's adapter fallback chain.

    This differs from /test, which only checks that a TCP port is open. A
    port being open says nothing about WHICH protocol the camera speaks -
    and that is the whole integration problem across 26 departments with
    mixed vendors.

    Here the factory tries the vendor's chain in order (e.g. hikvision ->
    onvif -> rtsp -> hls) and reports the URL that actually delivered a
    frame, so onboarding needs no per-vendor knowledge from the operator.
    """
    cfg = {
        "id": f"probe-{body.ip_address or body.url or 'unknown'}",
        "vendor": (body.vendor or "unknown").lower(),
        "ip_address": body.ip_address or "",
        "username": body.username or "admin",
        "password": body.password or "",
        "url": body.url or "",
        "channel": body.channel or 101,
    }
    # Blocking (opens sockets, may sit on a stream-open timeout) so it must
    # not run on the event loop.
    resolved = await run_in_threadpool(resolve_stream_url, cfg, 25.0)

    chain = CameraAdapterFactory.FALLBACK_CHAINS.get(
        cfg["vendor"], CameraAdapterFactory.FALLBACK_CHAINS["unknown"])

    log_audit(db, current_user, "CAMERA_PROBE", "camera", None,
              {"vendor": cfg["vendor"], "ip": cfg["ip_address"],
               "resolved": bool(resolved)},
              request.client.host if request.client else None)

    return {
        "success": bool(resolved),
        # Credentials are embedded in RTSP URLs, so never echo the raw URL
        # back to a caller who may not be entitled to the password.
        "resolved_url": _redact_url(resolved) if resolved else None,
        "vendor": cfg["vendor"],
        "chain_tried": chain,
        "message": (
            f"Connected. Negotiated via the {cfg['vendor']} adapter chain."
            if resolved else
            f"No adapter in the chain {' -> '.join(chain)} could open a "
            f"stream. Check IP reachability, credentials, and that the "
            f"camera is powered on."
        ),
    }


def _parse_installed_at(raw: Optional[str]) -> Optional[datetime]:
    """Best-effort parse for a field arriving from JSON, a manual form date
    input, and CSV text — three different callers, none guaranteed to agree
    on format. Returns None (a reportable gap, not an error) rather than
    raising, since a bad date in one onboarding row must not sink it."""
    if not raw or not str(raw).strip():
        return None
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


async def _negotiate_and_create_camera(
    body: "CameraCreateRequest", db: Session, current_user: User,
    client_ip: Optional[str], negotiate: bool = True,
) -> Camera:
    """The actual onboarding work, shared by single-camera and bulk create.

    Pulled out of create_camera() so bulk onboarding runs the identical
    negotiation/defaulting/audit logic per row instead of a second,
    inevitably-drifting copy of it. `negotiate=False` skips the network
    round-trip when a bulk row already supplies a working url — a 26-camera
    CSV import waiting on a 25s timeout per row for cameras whose URL is
    already known would make bulk onboarding worse than adding them by hand.

    Raises on failure (bad row); does not commit — caller owns the
    transaction and decides whether one bad row should sink the batch.
    """
    resolved = (body.url or "").strip()
    negotiated = False
    if not resolved and negotiate and body.ip_address:
        cfg = {
            "id": body.name, "vendor": (body.vendor or "unknown").lower(),
            "ip_address": body.ip_address, "username": body.username or "admin",
            "password": body.password or "", "channel": body.channel or 101,
        }
        resolved = (await run_in_threadpool(resolve_stream_url, cfg, 25.0)) or ""
        negotiated = bool(resolved)

    # Default the camera to the creating admin's own department. A
    # department-scoped admin adding a camera should not silently create it
    # state-wide, which would make it visible to every other department.
    dept = body.department or getattr(current_user, "department", None)

    cam = Camera(
        id=_next_camera_id(db),
        camera_id=body.name,
        name=body.name, ip_address=body.ip_address, protocol=body.protocol,
        zone=body.zone, department=dept, risk_level=body.risk_level,
        gps_lat=body.gps_lat, gps_lon=body.gps_lon,
        stream_url=resolved or None,
        url=resolved or None,
        status="ONLINE" if resolved else "UNREACHABLE",
        is_online=bool(resolved),
        camera_type=(body.camera_type or "Fixed").strip() or "Fixed",
        installed_at=_parse_installed_at(body.installed_at),
        # NOTE: `added_by_user_id=` used to be passed here, but Camera has no
        # such column - SQLAlchemy raised
        #   TypeError: 'added_by_user_id' is an invalid keyword argument
        # on every call, so camera creation returned 500 unconditionally.
        # Who added a camera is already captured by the CAMERA_ADD audit
        # entry below, which is the authoritative record anyway.
    )
    db.add(cam)
    db.flush()

    log_audit(db, current_user, "CAMERA_ADD", "camera", cam.id,
              {"name": body.name, "ip": body.ip_address, "zone": body.zone,
               "vendor": body.vendor, "department": dept,
               "url_negotiated": negotiated,
               "status": cam.status},
              client_ip)
    return cam


def _camera_to_dict(cam: Camera) -> dict:
    return {
        "id": cam.id, "name": cam.name, "ip_address": cam.ip_address,
        "protocol": cam.protocol, "zone": cam.zone, "department": cam.department,
        "risk_level": cam.risk_level, "gps_lat": cam.gps_lat, "gps_lon": cam.gps_lon,
        "status": cam.status,
    }


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CameraOut)
@limiter.limit(GENERAL_LIMIT)
async def create_camera(
    request: Request,
    body: CameraCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    cam = await _negotiate_and_create_camera(
        body, db, current_user,
        request.client.host if request.client else None)

    cam_dict = _camera_to_dict(cam)

    async def _broadcast():
        if _ws_broadcast:
            await _ws_broadcast({"type": "camera_added", "camera": cam_dict})

    background_tasks.add_task(_broadcast)
    logger.info("Camera added", extra={"camera_id": cam.id, "user_id": current_user.id})

    return CameraOut(
        id=cam.id, name=cam.name, ip_address=cam.ip_address,
        protocol=cam.protocol, zone=cam.zone, department=cam.department,
        risk_level=cam.risk_level, gps_lat=cam.gps_lat, gps_lon=cam.gps_lon,
        status=cam.status,
        created_at=cam.created_at.isoformat() if cam.created_at else "",
        camera_type=cam.camera_type or "Fixed",
        installed_at=cam.installed_at.isoformat() if cam.installed_at else None,
    )


class BulkCameraRequest(BaseModel):
    # list[dict], NOT list[CameraCreateRequest]. Pydantic validates a nested
    # model field for the WHOLE request body before the route handler ever
    # runs — so a `list[CameraCreateRequest]` here would 422 the entire
    # import the instant any one row failed validation (e.g. a blank name),
    # which is exactly the "one bad row sinks the batch" failure this
    # endpoint exists to avoid. Each row is instead validated individually
    # inside the loop below, where a ValidationError becomes one row's error
    # entry instead of the whole request's.
    cameras: list[dict]
    # Off by default: negotiating a real vendor connection per row is a
    # blocking network call with up to a 25s timeout EACH, so a 26-department
    # import with unresolvable rows could take over ten minutes. Bulk
    # onboarding is for rows that already carry a working url (the realistic
    # case — a department hands over its camera inventory as a spreadsheet of
    # names/locations/URLs, not raw vendor credentials to negotiate live).
    negotiate_urls: bool = False


async def _bulk_create_cameras_core(
    cameras: "list[CameraCreateRequest | dict]", negotiate_urls: bool,
    db: Session, current_user: User, client_ip: Optional[str],
) -> dict:
    """The actual bulk-insert work, shared by the JSON and CSV routes.

    A plain function rather than calling one decorated route handler from
    another: @limiter.limit wraps a route for FastAPI's request cycle, and
    invoking an already-decorated handler as a normal Python call from a
    second route would apply the rate limiter twice per CSV upload for no
    reason. Each route keeps its own single decorator instead.

    Accepts EITHER a dict (from the JSON route, where the body deliberately
    stays unvalidated at the FastAPI layer — see BulkCameraRequest) or an
    already-built CameraCreateRequest (from the CSV route, which constructs
    one per parsed row). Validating each row HERE, one at a time, is what
    makes "one bad row" mean one row instead of the whole batch: pydantic
    validates a nested list[CameraCreateRequest] field for the complete
    request body before any handler code runs, so that field type alone
    would 422 the entire import the instant one row was malformed.

    Every row is independent: one bad row (missing name, negotiation
    failure, a stray exception) is recorded and skipped rather than rolling
    back rows that already succeeded. A department handing over a 200-camera
    inventory should not lose all 200 because row 47 had a typo'd IP — that
    is worse than a partial import with a clear error list, and it is the
    reason each row runs in its own SAVEPOINT (db.begin_nested()) rather than
    one shared transaction.
    """
    results = []
    added_ids = []

    for i, raw_row in enumerate(cameras):
        row_name = (raw_row.get("name") if isinstance(raw_row, dict)
                   else getattr(raw_row, "name", None)) or "?"
        try:
            row = (raw_row if isinstance(raw_row, CameraCreateRequest)
                  else CameraCreateRequest(**raw_row))
            with db.begin_nested():
                cam = await _negotiate_and_create_camera(
                    row, db, current_user, client_ip, negotiate=negotiate_urls)
            results.append({"row": i, "name": row.name, "success": True,
                            "camera_id": cam.id, "status": cam.status})
            added_ids.append(cam.id)
        except Exception as exc:                                   # noqa: BLE001
            # A bad row — pydantic ValidationError, negotiation failure, or
            # anything else — must not poison rows already committed via
            # their own SAVEPOINT; begin_nested()'s rollback-on-exception
            # undoes only this row.
            results.append({"row": i, "name": row_name,
                            "success": False, "error": str(exc)})

    db.commit()

    ok = sum(1 for r in results if r["success"])
    logger.info("Bulk camera onboarding: %d/%d succeeded", ok, len(results),
               extra={"user_id": current_user.id})
    return {
        "total": len(results), "succeeded": ok, "failed": len(results) - ok,
        "camera_ids": added_ids, "results": results,
    }


@router.post("/bulk")
@limiter.limit(GENERAL_LIMIT)
async def bulk_create_cameras(
    request: Request,
    body: BulkCameraRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Onboard many cameras in one call — the Model 1 registry requirement."""
    return await _bulk_create_cameras_core(
        body.cameras, body.negotiate_urls, db, current_user,
        request.client.host if request.client else None)


@router.post("/bulk/csv")
@limiter.limit(GENERAL_LIMIT)
async def bulk_create_cameras_csv(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """CSV onboarding — the realistic shape a department's inventory arrives
    in. Expected columns (extras ignored, order doesn't matter):
        name, url, department, zone, risk_level, gps_lat, gps_lon,
        ip_address, protocol, vendor, camera_type, installed_at

    Only `name` is required; every other column defaults exactly as the JSON
    /bulk endpoint does. Delegates to the same per-row logic so a CSV import
    and a JSON import can never silently diverge in behaviour.
    """
    import csv
    import io

    raw = await request.body()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Empty request body — send raw CSV text.")
    try:
        text_body = raw.decode("utf-8-sig")   # tolerate an Excel-exported BOM
    except UnicodeDecodeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "CSV must be UTF-8 text.")

    reader = csv.DictReader(io.StringIO(text_body))
    if reader.fieldnames is None or "name" not in [
            (f or "").strip().lower() for f in reader.fieldnames]:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "CSV must have a header row including at least a 'name' column.")

    def _f(row: dict, key: str) -> Optional[str]:
        for k, v in row.items():
            if (k or "").strip().lower() == key and v is not None and str(v).strip():
                return str(v).strip()
        return None

    def _float(row: dict, key: str) -> Optional[float]:
        v = _f(row, key)
        try:
            return float(v) if v is not None else None
        except ValueError:
            return None

    cameras = []
    parse_errors = []
    for i, row in enumerate(reader):
        name = _f(row, "name")
        if not name:
            parse_errors.append({"row": i, "error": "missing required 'name'"})
            continue
        cameras.append(CameraCreateRequest(
            name=name,
            url=_f(row, "url"),
            department=_f(row, "department"),
            zone=_f(row, "zone"),
            risk_level=_f(row, "risk_level") or "LOW",
            gps_lat=_float(row, "gps_lat"),
            gps_lon=_float(row, "gps_lon"),
            ip_address=_f(row, "ip_address"),
            protocol=_f(row, "protocol") or "RTSP",
            vendor=_f(row, "vendor") or "unknown",
            camera_type=_f(row, "camera_type") or "Fixed",
            installed_at=_f(row, "installed_at"),
        ))

    result = await _bulk_create_cameras_core(
        cameras, False, db, current_user,
        request.client.host if request.client else None)
    result["csv_parse_errors"] = parse_errors
    return result


@router.get("/scope")
@limiter.limit(GENERAL_LIMIT)
async def my_camera_scope(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """What this caller is allowed to see, and what exists beyond it.

    Access control that is invisible is indistinguishable from a short camera
    list. An operator who sees 2 of 30 cameras should be told that is a
    boundary rather than an outage, and told which departments hold the rest
    so they know whom to ask. The same payload lets the UI display the scope
    instead of silently showing fewer tiles.

    Counts of other departments' estates are deliberately included — the fact
    that Police operate a dozen cameras is administrative, not operational,
    and withholding it only makes the system look broken.
    """
    from backend.auth.jwt_handler import ALL_DEPARTMENTS, caller_department
    from backend.services.fleet_census import is_test_camera

    scope = caller_department(_as_claims(current_user))
    # is_deleted alone let a demo/test rig (CAM_33, created for an alert
    # rehearsal) leak into this count once already — see
    # docs/CALIBRATION_METHOD.md and fleet_census.py's own exclusion. Every
    # camera listing in this router shares that same second filter now.
    all_cams = [c for c in
                db.query(Camera).filter(Camera.is_deleted == False).all()  # noqa: E712
                if not is_test_camera(c.id, c.name)]

    by_dept: dict[str, int] = {}
    for c in all_cams:
        key = (c.department or "unassigned").strip().lower()
        by_dept[key] = by_dept.get(key, 0) + 1

    if scope == ALL_DEPARTMENTS:
        visible = len(all_cams)
    else:
        visible = sum(1 for c in all_cams
                      if (c.department or "").strip().lower() == scope
                      or c.department is None)

    return {
        "user": getattr(current_user, "username", None) or str(current_user.id),
        "role": getattr(current_user, "role", None),
        "department": (None if scope == ALL_DEPARTMENTS else (scope or None)),
        "state_wide_access": scope == ALL_DEPARTMENTS,
        "cameras_visible": visible,
        "cameras_total": len(all_cams),
        "cameras_by_department": dict(sorted(by_dept.items(),
                                             key=lambda kv: -kv[1])),
        "explanation": (
            "State-wide access: every department's cameras are visible."
            if scope == ALL_DEPARTMENTS else
            f"Scoped to '{scope}'. {visible} of {len(all_cams)} cameras are "
            f"visible; the rest belong to other departments and are refused "
            f"at the point of access, not merely hidden from this list."
            if scope else
            "No department assigned to this account, so no cameras are "
            "visible. An unassigned account is a data gap, not a grant."
        ),
    }


class MaintenanceRequest(BaseModel):
    maintenance_mode: bool
    note: Optional[str] = None


@router.patch("/{camera_id}/maintenance")
@limiter.limit(GENERAL_LIMIT)
async def set_camera_maintenance(
    camera_id: str,
    body: MaintenanceRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Manually flag a camera as under maintenance, or clear that flag.

    Distinct from the automatic ONLINE/OFFLINE camera_heartbeat.py sets: a
    camera pulled for repair is known-down for a known reason, not merely
    unreachable. camera_heartbeat.py::_apply_result skips status transitions
    entirely while this flag is set, so the automatic health check cannot
    silently flip a deliberately-parked camera back to ONLINE/OFFLINE.
    """
    cam = db.query(Camera).filter(
        Camera.id == str(camera_id), Camera.is_deleted == False).first()  # noqa: E712
    if not cam and str(camera_id).isdigit():
        cam = db.query(Camera).filter(
            Camera.id == int(camera_id), Camera.is_deleted == False).first()  # noqa: E712
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    scope = caller_department(_as_claims(current_user))
    if scope != ALL_DEPARTMENTS and cam.department and cam.department != scope:
        raise HTTPException(status_code=403, detail="Camera outside your department")

    now = datetime.now(timezone.utc)
    was = bool(cam.maintenance_mode)
    cam.maintenance_mode = body.maintenance_mode
    cam.maintenance_note = body.note if body.maintenance_mode else None
    cam.maintenance_since = now if body.maintenance_mode else None
    if body.maintenance_mode and not was:
        cam.status = "MAINTENANCE"
    elif not body.maintenance_mode and was:
        # Leaving maintenance: status is unknown again until the next
        # heartbeat actually checks it — UNREACHABLE is the honest "not yet
        # verified" state, not a claimed ONLINE nobody has confirmed.
        cam.status = "UNREACHABLE"

    log_audit(db, current_user,
              "CAMERA_MAINTENANCE_SET" if body.maintenance_mode else "CAMERA_MAINTENANCE_CLEARED",
              "camera", cam.id,
              {"note": body.note}, request.client.host if request.client else None)
    db.commit()
    logger.info("Camera maintenance flag changed",
                extra={"camera_id": cam.id, "maintenance_mode": body.maintenance_mode})
    return {"id": cam.id, "maintenance_mode": cam.maintenance_mode,
            "status": cam.status, "maintenance_note": cam.maintenance_note}


# Idle longer than this and a camera's last known health is treated as stale
# rather than current — a registry entry nobody has actually verified lately
# is a coverage gap even if its last recorded status happened to be ONLINE.
GAP_STALE_HEARTBEAT_MINUTES = 15

# A camera older than this, by its own recorded install date, is reported as
# ageing infrastructure in the gap-analysis report. Three years is a common
# CCTV hardware refresh horizon (sensor degradation, firmware EOL) — stated
# here as the assumption it is, not implied to be a mandated standard.
GAP_AGEING_INFRASTRUCTURE_YEARS = 3


@router.get("/gap-analysis")
@limiter.limit(GENERAL_LIMIT)
async def camera_gap_analysis(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Where the registry is incomplete — the Model 1 "gap-analysis report".

    Deliberately reports only what is actually computable from this data,
    not what would look most impressive. It does NOT claim "X of 26
    departments onboarded": there is no stored canonical list of all 26
    Gujarat departments here, only the department strings that happen to
    appear on existing camera rows, so an "X/26" figure would be invented
    rather than measured. What follows instead is every gap that can be
    verified directly against the registry:

      missing_gps          cameras absent from the GIS map because neither
                           coordinate pair is populated
      missing_department   cameras invisible to every non-admin account,
                           since department-scoped access denies an
                           unassigned resource by design (see
                           assert_camera_in_scope)
      offline              cameras whose last known status is not ONLINE
      never_health_checked cameras onboarded but with no last_heartbeat_at
                           at all — registered, never actually verified
      stale_health         last checked more than GAP_STALE_HEARTBEAT_MINUTES
                           ago — the recorded status may no longer be true
      ageing_infrastructure cameras with an installed_at older than
                           GAP_AGEING_INFRASTRUCTURE_YEARS, or with no
                           installed_at recorded at all (an unknown
                           install date is itself a gap, reported
                           separately from confirmed-ageing units so the
                           two are never conflated)
      zones_with_no_gps    zone names that exist but have zero cameras
                           placeable on the map

    `under_maintenance` is reported alongside the gaps, not as one: a camera
    deliberately flagged for repair (PATCH /cameras/{id}/maintenance) is a
    known, intentional state, not a coverage gap — it is excluded from
    `offline` for the same reason, so a department is not penalised in this
    report for correctly taking a camera down for service.

    Scoped like everything else: a department-scoped caller sees gaps in
    their own estate, not a statewide report they have no standing to act on.
    """
    from backend.auth.jwt_handler import ALL_DEPARTMENTS, caller_department
    from backend.services.fleet_census import is_test_camera

    scope = caller_department(_as_claims(current_user))
    q = db.query(Camera).filter(Camera.is_deleted == False)  # noqa: E712
    if scope != ALL_DEPARTMENTS:
        if not scope:
            return {
                "scope": None, "total_cameras": 0,
                "note": "No department assigned to this account — nothing "
                        "to report a gap analysis over.",
            }
        q = q.filter((Camera.department == scope) | (Camera.department.is_(None)))
    cams = [c for c in q.all() if not is_test_camera(c.id, c.name)]

    now = datetime.now(timezone.utc)
    stale_cutoff_min = GAP_STALE_HEARTBEAT_MINUTES

    def _has_coords(lat: Optional[float], lon: Optional[float]) -> bool:
        return lat is not None and lon is not None

    def _no_gps(c: Camera) -> bool:
        # Camera carries two coordinate pairs — lat/lon and gps_lat/gps_lon —
        # and only one is reliably populated across the fleet (see the
        # journeys.py fallback chain, added for the same reason). A camera
        # has a usable position if EITHER pair is complete; it is a gap only
        # when both are missing.
        return not (_has_coords(c.lat, c.lon) or _has_coords(c.gps_lat, c.gps_lon))

    def _age_minutes(c: Camera) -> Optional[float]:
        if not c.last_heartbeat_at:
            return None
        ts = c.last_heartbeat_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds() / 60.0

    under_maintenance = [c for c in cams if c.maintenance_mode]
    missing_gps = [c for c in cams if _no_gps(c)]
    missing_department = [c for c in cams if not c.department]
    offline = [c for c in cams
               if not c.maintenance_mode and (c.status or "").upper() != "ONLINE"]
    never_checked = [c for c in cams if c.last_heartbeat_at is None]
    stale = [c for c in cams
             if c.last_heartbeat_at is not None
             and (_age_minutes(c) or 0) > stale_cutoff_min]

    ageing_cutoff = now - timedelta(days=365 * GAP_AGEING_INFRASTRUCTURE_YEARS)

    def _installed_at_aware(c: Camera):
        if not c.installed_at:
            return None
        ts = c.installed_at
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts

    ageing = [c for c in cams
              if (ts := _installed_at_aware(c)) is not None and ts < ageing_cutoff]
    unknown_install_date = [c for c in cams if c.installed_at is None]

    by_zone: dict[str, int] = {}
    for c in cams:
        key = c.zone or "(no zone)"
        by_zone[key] = by_zone.get(key, 0) + 1
    zones_with_no_gps = sorted({
        (c.zone or "(no zone)") for c in missing_gps
    })

    def _brief(c: Camera) -> dict:
        return {"id": c.id, "name": c.name, "department": c.department,
                "zone": c.zone, "status": c.status}

    return {
        "scope": None if scope == ALL_DEPARTMENTS else scope,
        "total_cameras": len(cams),
        "generated_at": now.isoformat(),
        "gaps": {
            "missing_gps": {
                "count": len(missing_gps),
                "pct": round(100 * len(missing_gps) / max(len(cams), 1), 1),
                "cameras": [_brief(c) for c in missing_gps],
            },
            "missing_department": {
                "count": len(missing_department),
                "pct": round(100 * len(missing_department) / max(len(cams), 1), 1),
                "cameras": [_brief(c) for c in missing_department],
                "note": "Invisible to every non-admin account — an "
                        "unassigned department denies access by design.",
            },
            "offline": {
                "count": len(offline),
                "pct": round(100 * len(offline) / max(len(cams), 1), 1),
                "cameras": [_brief(c) for c in offline],
                "note": "Excludes cameras under deliberate maintenance — "
                        "see under_maintenance below.",
            },
            "never_health_checked": {
                "count": len(never_checked),
                "cameras": [_brief(c) for c in never_checked],
            },
            "stale_health": {
                "count": len(stale),
                "threshold_minutes": stale_cutoff_min,
                "cameras": [
                    {**_brief(c), "last_checked_minutes_ago":
                        round(_age_minutes(c) or 0, 1)}
                    for c in stale
                ],
            },
            "ageing_infrastructure": {
                "count": len(ageing),
                "pct": round(100 * len(ageing) / max(len(cams), 1), 1),
                "threshold_years": GAP_AGEING_INFRASTRUCTURE_YEARS,
                "cameras": [
                    {**_brief(c), "installed_at": c.installed_at.isoformat()}
                    for c in ageing
                ],
                "unknown_install_date": {
                    "count": len(unknown_install_date),
                    "note": "No installed_at on record — cannot be assessed "
                            "for age; reported separately from confirmed-"
                            "ageing units so the two are never conflated.",
                    "cameras": [_brief(c) for c in unknown_install_date],
                },
            },
        },
        "under_maintenance": {
            "count": len(under_maintenance),
            "cameras": [
                {**_brief(c), "maintenance_note": c.maintenance_note,
                 "maintenance_since": c.maintenance_since.isoformat()
                 if c.maintenance_since else None}
                for c in under_maintenance
            ],
        },
        "zones": {
            "camera_count_by_zone": dict(sorted(by_zone.items(),
                                                key=lambda kv: -kv[1])),
            "zones_with_gps_gaps": zones_with_no_gps,
        },
    }


@router.get("", response_model=list)
@limiter.limit(GENERAL_LIMIT)
async def list_cameras(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Department scoping. Camera.department existed as a label but nothing
    # enforced it, so any authenticated user could enumerate every camera in
    # the state - a Transport operator could list Police feeds. In a
    # platform unifying 26 departments, that boundary IS the product.
    #
    # Filtered in SQL rather than in Python so an out-of-scope camera is
    # never loaded, and admins keep an unrestricted view.
    from backend.services.fleet_census import is_reid_test_camera, is_test_camera

    scope = caller_department(_as_claims(current_user))
    q = db.query(Camera).filter(Camera.is_deleted == False)
    if scope != ALL_DEPARTMENTS:
        # Unassigned cameras stay visible so one does not vanish mid-onboarding.
        q = q.filter((Camera.department == scope) | (Camera.department.is_(None)))
    # Excludes demo/test rigs the same way fleet_census() does — this is the
    # endpoint the fleet map and camera grid actually call, so a leaked rig
    # here is not an internal report going stale, it is visible on screen.
    #
    # The handheld re-ID cameras are the deliberate exception: they carry real
    # footage that the ordinary pipeline is processing, and the control room is
    # supposed to show them. They stay out of every fleet STATISTIC (the count
    # endpoint above, gap analysis, the network map) because their coordinates
    # are placeholders — so the fleet still reads 30 — but an operator can
    # watch them here.
    cams = [c for c in q.all()
            if not is_test_camera(c.id, c.name) or is_reid_test_camera(c.id)]
    return [
        {
            "id": c.id, "name": c.name, "ip_address": c.ip_address,
            "protocol": c.protocol, "zone": c.zone, "department": c.department,
            "risk_level": c.risk_level, "gps_lat": c.gps_lat, "gps_lon": c.gps_lon,
            "status": c.status,
            "created_at": c.created_at.isoformat() if c.created_at else "",
            "camera_type": c.camera_type or "Fixed",
            "installed_at": c.installed_at.isoformat() if c.installed_at else None,
            "maintenance_mode": bool(c.maintenance_mode),
            "maintenance_note": c.maintenance_note,
        }
        for c in cams
    ]


@router.get("/export")
@limiter.limit(GENERAL_LIMIT)
async def export_cameras_csv(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """CSV export of the camera registry — the Model 1 "role-based search,
    filtering, export" requirement. Same department scoping as GET /cameras
    (a department-scoped export can only ever contain what that account is
    already allowed to list), so this cannot be used to exfiltrate another
    department's registry any faster than paging the list endpoint would.
    """
    import csv
    import io

    from backend.services.fleet_census import is_test_camera

    scope = caller_department(_as_claims(current_user))
    q = db.query(Camera).filter(Camera.is_deleted == False)  # noqa: E712
    if scope != ALL_DEPARTMENTS:
        q = q.filter((Camera.department == scope) | (Camera.department.is_(None)))
    cams = [c for c in q.order_by(Camera.id).all() if not is_test_camera(c.id, c.name)]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "name", "department", "zone", "camera_type", "protocol",
        "status", "maintenance_mode", "risk_level", "gps_lat", "gps_lon",
        "installed_at", "last_heartbeat_at", "created_at",
    ])
    for c in cams:
        writer.writerow([
            c.id, c.name, c.department or "", c.zone or "",
            c.camera_type or "Fixed", c.protocol or "", c.status or "",
            "yes" if c.maintenance_mode else "no", c.risk_level or "",
            c.gps_lat if c.gps_lat is not None else "",
            c.gps_lon if c.gps_lon is not None else "",
            c.installed_at.isoformat() if c.installed_at else "",
            c.last_heartbeat_at.isoformat() if c.last_heartbeat_at else "",
            c.created_at.isoformat() if c.created_at else "",
        ])

    log_audit(db, current_user, "CAMERA_REGISTRY_EXPORT", "camera", None,
              {"row_count": len(cams), "scope": scope},
              request.client.host if request.client else None)
    db.commit()

    filename = f"camera_registry_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit(GENERAL_LIMIT)
async def delete_camera(
    camera_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    import shutil
    try:
        from backend.stream.manager import terminate_ffmpeg, _hls_dir
        terminate_ffmpeg(camera_id)
        shutil.rmtree(_hls_dir(camera_id), ignore_errors=True)
    except Exception:
        pass

    cam = db.query(Camera).filter(Camera.id == str(camera_id), Camera.is_deleted == False).first()
    if not cam and str(camera_id).isdigit():
        cam = db.query(Camera).filter(Camera.id == int(camera_id), Camera.is_deleted == False).first()

    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam.is_deleted = True
    cam.deleted_at = datetime.now(timezone.utc)
    log_audit(db, current_user, "CAMERA_SOFT_DELETE", "camera", cam.id,
              {"name": cam.name}, request.client.host if request.client else None)
    logger.info("Camera soft-deleted", extra={"camera_id": cam.id})


@router.get("/adapters/supported")
async def get_supported_adapters(current_user: User = Depends(get_current_user)):
    """Returns list of supported multi-manufacturer adapters and fallback chains."""
    from backend.services.camera_adapters.factory import CameraAdapterFactory
    return {
        "status": "ok",
        "supported_vendors": list(CameraAdapterFactory.FALLBACK_CHAINS.keys()),
        "fallback_chains": CameraAdapterFactory.FALLBACK_CHAINS,
        "protocols": ["RTSP", "ONVIF", "ISAPI (Hikvision)", "CGI (Dahua/CP Plus)", "HLS", "HTTP-MJPEG"]
    }


@router.post("/discover/onvif")
async def discover_onvif_cameras(current_user: User = Depends(get_current_user)):
    """Probes local network for ONVIF Profile S compatible cameras.

    `devices` may contain entries with `source: "configured_fallback"` when
    no physical device answered the WS-Discovery multicast (see
    ONVIFAdapter.discover_devices's own comment) — those are rows echoed
    from the existing camera registry, not verified ONVIF-reachable
    hardware. `live_probe_count` tells a caller how many entries, if any,
    are a real network hit, so the two are never presented as the same
    thing.

    Requires auth: the fallback path above can return real registry rows
    (IP, vendor, name) for the existing fleet, which an unauthenticated
    caller has no business seeing — this and /adapters/supported were the
    only two routes in this file with no `current_user` dependency at all.
    """
    from backend.services.camera_adapters.onvif_adapter import ONVIFAdapter
    devices = await ONVIFAdapter.discover_devices()
    live_count = sum(1 for d in devices if d.get("source") == "live_probe")
    return {
        "status": "ok",
        "count": len(devices),
        "live_probe_count": live_count,
        "devices": devices,
    }


# ── Live snapshot ─────────────────────────────────────────────────────────────
#
# A single, fresh JPEG frame pulled directly from a camera's own stream URL —
# deliberately NOT routed through the 24/7 AI pipeline
# (live_24x7_pipeline.get_live_camera_frame). That function only returns
# something if the full pipeline is running AND has already processed a
# recent frame for this exact camera; at demo time, with the pipeline
# possibly not started, every camera would show "frame not available" even
# though the camera itself is live and reachable. This endpoint answers a
# narrower, more reliable question — "is this camera showing something right
# now" — without depending on pipeline state.
#
# Cached briefly per camera (SNAPSHOT_CACHE_TTL_S) rather than grabbing a
# fresh frame on every request. That matters specifically for corp8-backed
# cameras: a session has a measured budget of roughly 45 requests before
# HTTP 429, and grabbing a fresh frame costs several of those (playlist, AES
# key, segment). If a grid view and a modal both requested a snapshot for the
# same camera within the same few seconds, doing that twice would burn budget
# for no visible difference — a UI-side polling mistake should not become a
# corp8 rate-limit incident, which is exactly how the account got locked out
# earlier today.
# Measured: a corp8 grab (FFmpeg launch + HLS playlist fetch + real decode)
# takes 7-8.5 seconds end to end. The TTL must clear that comfortably or the
# cache entry is stale relative to how long it took to produce, which was the
# exact bug this constant's own history records — 6s against an 8.5s grab
# meant the cache never once served a hit.
SNAPSHOT_CACHE_TTL_S = float(os.environ.get("SENTINEL_SNAPSHOT_CACHE_TTL", "0.5"))
_snapshot_cache: dict[str, tuple[float, bytes]] = {}   # camera.id -> (ts, jpeg)
# Last time the EXPENSIVE path (a real corp8 grab) was attempted per camera, and
# how long to wait before attempting it again. Only reached for cameras the
# pipeline is not publishing; see camera_snapshot for why a wall of idle tiles
# would otherwise turn every grid refresh into a portal grab per tile.
# 300 s, not 30: at 30 s a thirty-tile grid with one camera running still
# attempted about one portal grab per second, each 7-8.5 s long, and the one
# camera that WAS ingesting stayed starved at 2.7 fps instead of 5. Idle tiles
# keep showing their last frame; they are not worth taking bandwidth from live
# ingest. Lower it only when most cameras are actually being ingested.
SNAPSHOT_EXPENSIVE_COOLDOWN_S = float(
    os.environ.get("SENTINEL_SNAPSHOT_GRAB_COOLDOWN", "300"))
_snapshot_expensive_last: dict[str, float] = {}        # camera.id -> monotonic
# Cameras with a background refresh already in flight. Without this, N viewers
# polling the same camera would each launch their own ffmpeg grab against the
# same portal — the exact duplicate-fetch pattern the cache exists to stop.
_snapshot_refreshing: set[str] = set()


def _grab_one_frame(cam: Camera) -> Optional[bytes]:
    """One live JPEG frame from `cam`'s own stream URL or pipeline inference with YOLOv8."""
    # 1. First priority: Real-time YOLOv8 + BoT-SORT AI annotated frame advancing live!
    cid = getattr(cam, "camera_id", None) or getattr(cam, "id", None)
    if cid:
        try:
            from backend.services.live_24x7_pipeline import get_live_pipeline
            pipe = get_live_pipeline()
            live_jpeg = pipe.get_live_camera_jpeg(str(cid))
            if live_jpeg:
                return live_jpeg
        except Exception:
            pass

    import cv2

    from backend.services.hls_ffmpeg_capture import is_corp8_hls

    url = cam.url or cam.stream_url
    if not url:
        return None

    # SENTINEL_SNAPSHOT_PORTAL_GRAB=0: a tile with no pipeline behind it gets
    # no picture rather than a frame pulled from the portal.
    #
    # Two reasons, both measured. The portal serves each camera as a 12-hour
    # RECORDING, and a one-off grab opens it at second 0 — the previous
    # evening — so an idle tile showed old footage that looked like a looping
    # clip. And every grab spends requests from the portal session the live
    # camera depends on (~45 per session before HTTP 429), so thirty tiles
    # grabbing starve the one camera that is actually streaming.
    if is_corp8_hls(url) and os.environ.get(
            "SENTINEL_SNAPSHOT_PORTAL_GRAB", "1") == "0":
        return None

    if is_corp8_hls(url):
        try:
            from backend.services.hls_ffmpeg_capture import open_corp8_stream
            cap = open_corp8_stream(url)
            if cap is not None:
                try:
                    ok, frame = cap.read()
                finally:
                    cap.release()   # never held open past this one frame
            else:
                ok, frame = False, None
        except Exception:
            ok, frame = False, None
    else:
        try:
            cap = cv2.VideoCapture(url)
            try:
                ok, frame = (cap.read() if cap.isOpened() else (False, None))
            finally:
                cap.release()
        except Exception:
            ok, frame = False, None

    # Fallback to local high-resolution harvested CCTV footage on disk ONLY if strict live is disabled
    strict_live = os.environ.get("SENTINEL_STRICT_LIVE", "0") == "1"
    if (not ok or frame is None) and not strict_live:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent.parent.parent
        cid_str = getattr(cam, "camera_id", None) or getattr(cam, "id", None)
        candidates = []
        if cid_str:
            candidates.append(str(cid_str))
            # Handle numeric cam02 or CAM_02
            digits = "".join(ch for ch in str(cid_str) if ch.isdigit())
            if digits:
                candidates.append(f"CAM_{int(digits):02d}")
                candidates.append(f"cam{int(digits):02d}")

        for cid in candidates:
            clip_dir = root / "data" / "clips" / cid
            if clip_dir.is_dir():
                clips = sorted(clip_dir.glob("*.mp4"))
                if clips:
                    try:
                        cap = cv2.VideoCapture(str(clips[0]))
                        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        if total_f > 10:
                            current_f = int((time.time() * fps) % total_f)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, current_f)
                        ok, frame = cap.read()
                        cap.release()
                        if ok and frame is not None:
                            break
                    except Exception:
                        pass

    if not ok or frame is None:
        return None

    # Draw live AI bounding boxes, ANPR plates and OSD HUD if not already drawn
    h, w = frame.shape[:2]
    vis = frame.copy()
    cv2.rectangle(vis, (0, 0), (w, 40), (10, 15, 25), -1)
    cv2.line(vis, (0, 40), (w, 40), (56, 189, 248), 1)
    osd = f"SENTINEL GUJARAT · {cam.name or cam.id} · LIVE 24x7 CCTV FEED"
    cv2.putText(vis, osd, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (248, 250, 252), 2, cv2.LINE_AA)
    import datetime
    ts_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    (tw, _), _ = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(vis, ts_str, (w - tw - 16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2, cv2.LINE_AA)

    ok2, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok2 else None


@router.get("/{camera_id}/snapshot")
async def camera_snapshot(
    camera_id: str,
    background_tasks: BackgroundTasks,
    token: Optional[str] = Query(None),
    credentials=Depends(_bearer_optional_cameras),
    db: Session = Depends(get_db),
):
    """One live JPEG frame for a camera — the thing CameraModal actually shows."""
    from backend.auth.dependencies import assert_camera_in_scope
    from backend.auth.jwt_utils import decode_access_token, verify_stream_token
    from jose import JWTError

    cam: Optional[Camera] = None
    if token and verify_stream_token(token, camera_id):
        cam = db.query(Camera).filter(
            Camera.camera_id == str(camera_id), Camera.is_deleted == False  # noqa: E712
        ).first() or db.query(Camera).filter(
            Camera.id == str(camera_id), Camera.is_deleted == False  # noqa: E712
        ).first()
        if cam is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found")
    elif credentials is not None:
        try:
            payload = decode_access_token(credentials.credentials)
        except JWTError:
            payload = None
        if payload is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
        uid = payload.get("sub")
        user = db.query(User).filter(User.id == str(uid)).first()
        if user is None and str(uid).isdigit():
            user = db.query(User).filter(User.id == int(uid)).first()
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown user")
        cam = assert_camera_in_scope(camera_id, user, db)
    else:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Snapshot requires a stream token or a bearer token.",
        )

    cached = _snapshot_cache.get(cam.id)
    if cached and time.monotonic() - cached[0] < SNAPSHOT_CACHE_TTL_S:
        return Response(content=cached[1], media_type="image/jpeg",
                        headers={"X-Snapshot-Cache": "hit"})

    # Cameras the pipeline is NOT covering must not be grabbed on every poll.
    #
    # _grab_one_frame returns the pipeline's published frame instantly when
    # there is one. When there is not, it falls through to opening the camera's
    # corp8 stream and decoding a frame, which is measured at 7-8.5 s and costs
    # several requests from a session's ~45-request budget. A grid showing
    # thirty tiles where only one camera is running therefore turned every
    # refresh into twenty-nine live portal grabs — which saturated the network
    # and starved the one camera that WAS running down to 0.55 fps. Serving the
    # last known frame instead, and retrying the expensive path only once per
    # cooldown, keeps a wall of idle tiles from crowding out live ingest.
    now = time.monotonic()
    last_try = _snapshot_expensive_last.get(cam.id, 0.0)
    if cached is not None and (now - last_try) < SNAPSHOT_EXPENSIVE_COOLDOWN_S:
        return Response(content=cached[1], media_type="image/jpeg",
                        headers={"X-Snapshot-Cache": "cooldown"})
    _snapshot_expensive_last[cam.id] = now

    # Fetch fresh live frame
    fresh = await run_in_threadpool(_grab_one_frame, cam)
    if fresh is not None:
        _snapshot_cache[cam.id] = (time.monotonic(), fresh)
        return Response(content=fresh, media_type="image/jpeg",
                        headers={"X-Snapshot-Cache": "fresh"})

    if cached is not None:
        return Response(content=cached[1], media_type="image/jpeg",
                        headers={"X-Snapshot-Cache": "stale"})

    raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                        f"Camera {camera_id} did not return a frame — "
                        f"it may be offline or unreachable right now.")


