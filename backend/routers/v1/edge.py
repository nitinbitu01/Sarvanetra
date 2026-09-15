"""Ingest for edge nodes — the central half of the edge-inference split.

WHY THIS EXISTS
  Sending video to a central GPU does not scale. Measured on this project's own
  files: one camera's video is 2.20 Mbit/s, while the metadata that same camera
  produces is 2.62 kbit/s — **839x smaller**. At 3,000 cameras that is the
  difference between 6.60 Gbit/s (impossible) and 7.87 Mbit/s (a home
  broadband line).

  So an edge node near the camera runs stages 2-4 of the pipeline — detect,
  track, read the plate — and posts only the answer here. Stages 5-9 (identity,
  trajectory, analytics, alerts, dashboard) stay central and are unchanged: the
  rows written below are ordinary VehicleTrack rows, so everything downstream,
  including the dashboard, treats an edge sighting exactly like a locally
  processed one.

  The cut lands between stage 4 and stage 5 because that is where the pipeline
  stops needing pixels. See docs/UNIFIED_ARCHITECTURE.md.

AUTHENTICATION
  A field device cannot hold a user's password, so it presents a device key in
  X-Edge-Key. Set SENTINEL_EDGE_KEY on both the server and the node. Requests
  without it are refused — this endpoint writes to the evidence tables.
"""
from __future__ import annotations

import hmac
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.db.models import VehicleTrack
from backend.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/edge", tags=["edge"])

# Dev default so the node runs out of the box on a laptop. In the field this is
# per-device and provisioned with the device.
EDGE_KEY = os.environ.get("SENTINEL_EDGE_KEY", "sentinel-edge-dev-key")

# Per-node counters, kept in memory. They exist to make the bandwidth argument
# checkable live rather than only in a report: the node reports how many bytes
# of video it decoded, the server records how many bytes of metadata arrived,
# and the ratio between them is the whole case for edge inference.
_nodes: Dict[str, Dict[str, Any]] = {}
_nodes_lock = threading.Lock()

# The last few events as they arrived, kept so the dashboard can show WHAT was
# received and not only how much. A ratio on its own is indistinguishable from a
# hardcoded number; a plate that also appears on the camera the viewer is
# watching, seconds after the vehicle passed, is not.
_recent: List[Dict[str, Any]] = []
RECENT_MAX = 12


class EdgeEvent(BaseModel):
    """One vehicle, as an edge node sees it. No pixels."""
    cam: str = Field(..., description="camera id, e.g. CAM_09")
    track: int = Field(..., description="track id local to that node")
    t: Optional[str] = Field(None, description="ISO timestamp of last sighting")
    t_first: Optional[str] = None
    plate: Optional[str] = None
    conf: Optional[float] = None
    state: Optional[str] = None
    votes: Optional[int] = None
    cls: Optional[str] = None
    kmh: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    hdg: Optional[float] = None
    frames: int = 1


class EdgeBatch(BaseModel):
    node_id: str = Field(..., description="which edge node is reporting")
    events: List[EdgeEvent]
    # What the node had to chew through to produce these events. Reported by the
    # node because only the node knows it — the server never sees the video.
    video_bytes: int = 0
    window_sec: float = 0.0


def _require_key(x_edge_key: Optional[str] = Header(None)) -> str:
    if not x_edge_key or not hmac.compare_digest(x_edge_key, EDGE_KEY):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Edge ingest requires a valid X-Edge-Key.",
        )
    return x_edge_key


def _parse_ts(v: Optional[str]) -> datetime:
    if not v:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(v.replace("Z", ""))
    except ValueError:
        return datetime.utcnow()


@router.post("/events", status_code=status.HTTP_202_ACCEPTED)
async def ingest_edge_events(
    batch: EdgeBatch,
    request: Request,
    _key: str = Depends(_require_key),
    db: Session = Depends(get_db),
):
    """Accept a batch of detections from an edge node and store them.

    Returns the byte accounting, so the node — and a demonstration — can show
    the compression against the video it never sent.
    """
    meta_bytes = int(request.headers.get("content-length") or 0)

    written = 0
    for e in batch.events:
        try:
            db.add(VehicleTrack(
                camera_id=e.cam,
                track_id=int(e.track),
                vehicle_class=e.cls,
                first_seen=_parse_ts(e.t_first or e.t),
                last_seen=_parse_ts(e.t),
                entry_lat=e.lat,
                entry_lon=e.lon,
                heading_deg=e.hdg,
                speed_kmh=e.kmh,
                plate_text=e.plate,
                plate_confidence=e.conf,
                plate_state=e.state,
                plate_votes=e.votes,
                plate_locked=bool(e.votes and e.votes >= 3),
                plate_detected=bool(e.plate),
                n_frames=max(1, int(e.frames)),
            ))
            written += 1
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("edge event rejected from %s: %s", batch.node_id, exc)

    try:
        db.commit()
    except Exception as exc:                                       # noqa: BLE001
        db.rollback()
        logger.error("edge batch commit failed for %s: %s", batch.node_id, exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Could not store edge events")

    with _nodes_lock:
        n = _nodes.setdefault(batch.node_id, {
            "node_id": batch.node_id, "events": 0, "meta_bytes": 0,
            "video_bytes": 0, "window_sec": 0.0, "batches": 0,
            "first_seen": time.time(),
        })
        n["events"] += written
        n["meta_bytes"] += meta_bytes
        n["video_bytes"] += int(batch.video_bytes)
        n["window_sec"] += float(batch.window_sec)
        n["batches"] += 1
        n["last_seen"] = time.time()
        n["cameras"] = sorted({e.cam for e in batch.events} |
                              set(n.get("cameras") or []))
        ratio = (n["video_bytes"] / n["meta_bytes"]) if n["meta_bytes"] else 0.0

        for e in batch.events:
            _recent.insert(0, {
                "node": batch.node_id, "cam": e.cam, "track": e.track,
                "plate": e.plate, "conf": e.conf, "kmh": e.kmh,
                "cls": e.cls, "frames": e.frames,
                "seen_at": e.t,
                "received_at": time.time(),
                "batch_bytes": meta_bytes,
            })
        del _recent[RECENT_MAX:]

    logger.info("[EDGE] %s: +%d events, %d bytes metadata (%.0fx smaller than "
                "the %d bytes of video it replaced)",
                batch.node_id, written, meta_bytes, ratio, batch.video_bytes)

    return {
        "accepted": written,
        "metadata_bytes": meta_bytes,
        "video_bytes_avoided": batch.video_bytes,
        "compression_ratio": round(ratio, 1),
    }


# ── Demo control: start and stop a local node from the dashboard ─────────────
#
# An endpoint that starts a process is a serious thing, so this one is
# deliberately narrow: it takes NO command from the caller. The argument list
# below is fixed in code, the camera id must match one of the cameras this
# machine has a prefetched buffer for, and it requires a logged-in user. There
# is no path by which a request can influence what is executed.
#
# It exists because the start/stop is the proof. Numbers on a page could be
# hardcoded; a panel that goes quiet when you stop the node and resumes within
# one batch when you start it cannot be.

_PROC_LOCK = threading.Lock()
_procs: Dict[str, Any] = {}          # camera_id -> Popen

# .../sentinel_gujarat_day8/sentinel gujarat/backend/routers/v1/edge.py
#   parents[3] = "sentinel gujarat"          (the project)
#   parents[4] = "sentinel_gujarat_day8"     (the repo root, where edge/ lives)
_PROJ = Path(__file__).resolve().parents[3]
_REPO = _PROJ.parent


def _buffer_for(cam: str) -> Optional[Path]:
    p = _PROJ / "output" / "hls_buffer" / cam / "index.m3u8"
    return p if p.is_file() else None


@router.get("/control/status")
def edge_control_status():
    """Which locally-launched nodes are alive, and which cameras could run one."""
    with _PROC_LOCK:
        running = {c: (p.poll() is None) for c, p in _procs.items()}
    buf_dir = _PROJ / "output" / "hls_buffer"
    available = sorted(d.name for d in buf_dir.iterdir()
                       if d.is_dir() and (d / "index.m3u8").is_file()
                       ) if buf_dir.is_dir() else []
    return {"running": [c for c, alive in running.items() if alive],
            "available_cameras": available}


@router.post("/control/start")
def edge_control_start(
    camera: str = "CAM_09",
    user=Depends(get_current_user),
):
    """Launch the edge node for one camera. Idempotent."""
    cam = camera.upper()
    if not _buffer_for(cam):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"No prefetched buffer for {cam}. Build one first with "
            f"prefetch_live_buffer.py.")

    with _PROC_LOCK:
        p = _procs.get(cam)
        if p is not None and p.poll() is None:
            return {"status": "already_running", "camera": cam, "pid": p.pid}

        # Fixed argv. Nothing here comes from the request except `cam`, which
        # was just checked against the buffers that actually exist on disk.
        argv = [
            sys.executable, "-m", "edge.agent",
            "--camera", cam,
            "--source", str(_buffer_for(cam)),
            "--calibration", str(_PROJ / "edge" / "config" / f"{cam}.json"),
            "--server", os.environ.get("SENTINEL_EDGE_SERVER",
                                       "http://127.0.0.1:8000"),
            "--fps", "5", "--batch-seconds", "6",
        ]
        log_path = _PROJ / "output" / f"edge_{cam}.log"
        try:
            log_f = open(log_path, "ab", buffering=0)
            proc = subprocess.Popen(argv, cwd=str(_REPO),
                                    stdout=log_f, stderr=log_f)
        except Exception as exc:                                   # noqa: BLE001
            logger.error("could not launch edge node for %s: %s", cam, exc)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                                f"Could not start edge node: {exc}")
        _procs[cam] = proc

    logger.info("[EDGE] launched node for %s (pid %s) by %s",
                cam, proc.pid, getattr(user, "username", "?"))
    return {"status": "started", "camera": cam, "pid": proc.pid,
            "note": "Models take ~20 s to load before the first batch."}


@router.post("/control/stop")
def edge_control_stop(
    camera: str = "CAM_09",
    user=Depends(get_current_user),
):
    """Stop the node. The panel should then go quiet — that is the point."""
    cam = camera.upper()
    with _PROC_LOCK:
        p = _procs.get(cam)
        if p is None or p.poll() is not None:
            return {"status": "not_running", "camera": cam}
        p.terminate()
        try:
            p.wait(timeout=8)
        except Exception:                                          # noqa: BLE001
            p.kill()
        _procs.pop(cam, None)
    logger.info("[EDGE] stopped node for %s by %s",
                cam, getattr(user, "username", "?"))
    return {"status": "stopped", "camera": cam}


@router.get("/nodes")
def list_edge_nodes():
    """What each node has sent, and the bandwidth it saved doing so.

    This is the scalability claim made checkable: `bytes_saved_ratio` is
    measured from real traffic, not estimated.
    """
    now = time.time()
    out = []
    with _nodes_lock:
        for n in _nodes.values():
            meta, vid = n["meta_bytes"], n["video_bytes"]
            secs = n["window_sec"] or 1.0
            out.append({
                "node_id": n["node_id"],
                "cameras": n.get("cameras") or [],
                "events": n["events"],
                "batches": n["batches"],
                "seconds_since_last_report": round(now - n["last_seen"], 1),
                "metadata_bytes": meta,
                "video_bytes_avoided": vid,
                "bytes_saved_ratio": round(vid / meta, 1) if meta else None,
                "metadata_kbit_s": round(meta * 8 / secs / 1e3, 2),
                "video_mbit_s": round(vid * 8 / secs / 1e6, 2),
            })
        recent = [dict(r, age_sec=round(now - r["received_at"], 1))
                  for r in _recent]
    # server_time lets the client prove the page is live rather than trusting a
    # rendered "seconds ago": the value moves on every poll.
    return {"nodes": out, "count": len(out),
            "recent_events": recent, "server_time": now}
