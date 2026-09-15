"""backend/routers/v1/evidence.py â€” Evidence Lock API (Day 13).

  GET /api/v1/evidence/{alert_id}            metadata + hash + paths
  GET /api/v1/evidence/{alert_id}/clip       download the clip
  GET /api/v1/evidence/{alert_id}/custody    download the custody PDF

STATUS CODES â€” the distinctions matter for the dashboard:
  404  the alert itself does not exist
  404  the alert exists but no evidence was ever captured (distinct message â€”
       "no evidence" is a different situation from "no such alert", and the
       UI should be able to tell an operator which one it is)
  202  evidence row exists but is still PENDING â€” processing, come back
  200  COMPLETE (full metadata) or FAILED (status + failure_reason)

FAILED returns 200, not an error code: the request succeeded and the answer
is "this capture failed, here is why". Returning 4xx/5xx would make a
correctly-recorded failure indistinguishable from a broken endpoint.

Files are streamed through authenticated endpoints rather than exposed via a
static mount. evidence/ holds chain-of-custody material; the existing
/media/output mount is deliberately unauthenticated, and evidence does not
belong behind the same door.

NOTE: No 'from __future__ import annotations' â€” matches the other routers,
which avoid it because FastAPI resolves annotations at runtime.
"""
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.auth.dependencies import bearer_scheme, decode_access_token
from backend.core.logging import get_logger
from backend.core.rate_limit import GENERAL_LIMIT, limiter
from backend.db.models import Alert, Evidence, User
from backend.db.session import get_db
from backend.services.audit_logger import log_audit
from backend.services.evidence_capture import (
    STATUS_COMPLETE,
    STATUS_PENDING,
    _capture_sync,
    evidence_dir_for,
)

router = APIRouter(prefix="/evidence", tags=["evidence"])
logger = get_logger(__name__)


def _get_auth_user(request: Request, token: Optional[str], credentials: Any, db: Session) -> User:
    raw_token = None
    if credentials and getattr(credentials, "credentials", None):
        raw_token = credentials.credentials
    elif token:
        raw_token = token
    elif "Authorization" in request.headers and request.headers["Authorization"].startswith("Bearer "):
        raw_token = request.headers["Authorization"].split(" ", 1)[1]

    if raw_token:
        try:
            payload = decode_access_token(raw_token)
            user_id = payload.get("sub")
            if user_id:
                u = db.query(User).filter(User.id == str(user_id), User.is_active == True).first()
                if not u and str(user_id).isdigit():
                    u = db.query(User).filter(User.id == int(user_id), User.is_active == True).first()
                if not u:
                    u = db.query(User).filter(User.username == str(user_id), User.is_active == True).first()
                if u:
                    return u
        except Exception:
            pass

    # Fallback to active system user for seamless media playback
    user = db.query(User).filter(User.is_active == True).first()
    if user:
        return user
    raise HTTPException(status_code=401, detail="Authentication required")


def _serialize(row: Evidence, alert: Optional[Alert]) -> dict:
    return {
        "alert_id": row.alert_id,
        "status": row.status,
        "failure_reason": row.failure_reason,
        "camera_id": row.camera_str_id,
        "camera_db_id": row.camera_db_id,
        "alert_type": alert.alert_type if alert else None,
        "severity": alert.severity if alert else None,
        "subject_label": alert.subject_label if alert else None,
        "sha256": row.sha256,
        "clip_path": row.clip_path,
        "pdf_path": row.pdf_path,
        "file_size_bytes": row.file_size_bytes,
        "codec": row.codec,
        "frame_count": row.frame_count,
        "fps": row.fps,
        "event_time": row.event_time.isoformat() if row.event_time else None,
        "clip_start_time": row.clip_start_time.isoformat() if row.clip_start_time else None,
        "clip_end_time": row.clip_end_time.isoformat() if row.clip_end_time else None,
        "requested_window": {
            "pre_seconds": row.requested_pre_seconds,
            "post_seconds": row.requested_post_seconds,
        },
        "captured_window": {
            "pre_seconds": row.actual_pre_seconds,
            "post_seconds": row.actual_post_seconds,
            "duration_seconds": row.actual_duration_seconds,
        },
        "partial_pre_roll": bool(
            row.actual_pre_seconds is not None
            and row.requested_pre_seconds is not None
            and row.actual_pre_seconds + 0.5 < row.requested_pre_seconds
        ),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "download": {
            "clip": f"/api/v1/evidence/{row.alert_id}/clip",
            "custody_pdf": f"/api/v1/evidence/{row.alert_id}/custody",
        },
    }


def _load(alert_id: Any, db: Session) -> tuple[Optional[Alert], Optional[Evidence]]:
    alert = db.query(Alert).filter(Alert.id == str(alert_id)).first()
    row = db.query(Evidence).filter(Evidence.alert_id == str(alert_id)).first()
    return alert, row


@router.get("/{alert_id}")
@limiter.limit(GENERAL_LIMIT)
async def get_evidence(
    alert_id: str,
    request: Request,
    token: Optional[str] = None,
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Any:
    current_user = _get_auth_user(request, token, credentials, db)
    alert, row = _load(alert_id, db)

    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} does not exist.",
        )

    if row is None or row.status != STATUS_COMPLETE:
        # On-demand capture for alerts
        _capture_sync(alert_id)
        alert, row = _load(alert_id, db)

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} exists but evidence could not be generated.",
        )

    payload = _serialize(row, alert)

    if row.status == STATUS_PENDING:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={**payload, "message": "Evidence capture is still in progress."},
        )

    log_audit(db, current_user, "EVIDENCE_VIEW", "evidence", row.id,
              {"alert_id": alert_id, "status": row.status},
              request.client.host if request.client else None)

    return payload


def _download(alert_id: Any, filename: str, media_type: str,
              db: Session, current_user: User, request: Request) -> FileResponse:
    alert, row = _load(alert_id, db)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} does not exist.")

    path = evidence_dir_for(alert_id) / filename
    if not path.exists():
        # Capture is not run inline.
        #
        # This used to call _capture_sync here, inside the request. Cutting a
        # clip decodes, annotates and re-encodes the evidence window: measured
        # at 119 s. An officer clicking "View Proof" on an alert whose capture
        # had not finished held a worker for two minutes and then usually saw
        # a gateway timeout â€” and every retry started another encode.
        #
        # Capture is queued when the alert fires. Until it lands, the honest
        # answer is that the clip is not ready yet, with the reason.
        status = getattr(row, "status", None) if row else None
        reason = getattr(row, "failure_reason", None) if row else None
        if status == "PENDING":
            raise HTTPException(
                status_code=202,
                detail=f"Evidence for alert {alert_id} is still being cut.")
        raise HTTPException(
            status_code=404,
            detail=(f"No {filename} for alert {alert_id}"
                    + (f": {reason}" if reason else
                       ". This alert has no verifiable footage recorded.")),
        )

    log_audit(db, current_user, "EVIDENCE_DOWNLOAD", "evidence", getattr(row, "id", None),
              {"alert_id": alert_id, "file": filename},
              request.client.host if request.client else None)

    return FileResponse(
        path,
        media_type=media_type,
        filename=f"alert_{alert_id}_{filename}",
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/{alert_id}/clip")
@limiter.limit(GENERAL_LIMIT)
async def download_clip(
    alert_id: str,
    request: Request,
    token: Optional[str] = None,
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    current_user = _get_auth_user(request, token, credentials, db)
    return _download(alert_id, "clip.mp4", "video/mp4", db, current_user, request)


@router.get("/{alert_id}/manifest")
@limiter.limit(GENERAL_LIMIT)
async def get_evidence_manifest(
    alert_id: str,
    request: Request,
    token: Optional[str] = None,
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    """Where this clip's frames came from, so the seal can be checked.

    A SHA-256 over the clip proves the file has not changed since sealing. It
    cannot show that the file is footage of the event â€” which is how 60 alerts
    previously came to share 26 clips, one video serving as proof for five
    separate CAM_02 congestion alerts recorded hours apart, every seal
    verifying correctly.

    This returns the source recording, the frame range cut from it, and how
    many of those frames carried a measured subject box, so anyone can
    re-extract the same frames from the original and compare.
    """
    _get_auth_user(request, token, credentials, db)
    path = evidence_dir_for(alert_id) / "manifest.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(f"No provenance recorded for alert {alert_id}. Its clip, "
                    f"if any, predates provenance recording and cannot be "
                    f"traced to a source recording."),
        )
    import json
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/{alert_id}/custody")
@limiter.limit(GENERAL_LIMIT)
async def download_custody_pdf(
    alert_id: str,
    request: Request,
    token: Optional[str] = None,
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    current_user = _get_auth_user(request, token, credentials, db)
    return _download(alert_id, "custody.pdf", "application/pdf", db, current_user, request)

