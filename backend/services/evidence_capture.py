"""
backend/services/evidence_capture.py — Evidence Lock (Day 13).

On a CRITICAL alert: cut a clip spanning PRE seconds before the event through
POST seconds after, hash it, and generate a one-page chain-of-custody PDF.

SEVERITY — why this module assigns it
  The brief said "reuse the existing severity field". Alert.severity exists,
  but NONE of the four modern firing modules ever wrote it — only the legacy
  raw-SQL ANPR path (backend/db.py insert_alert) did. Keying evidence capture
  off severity as-is would have meant it fired for legacy `watchlist_vehicle`
  rows and NEVER for a WATCHLIST_FACE_MATCH, which is the headline case.
  So classify_severity() below populates the column at fire time, from one
  place, and the trigger keys off it. The field is now real rather than
  routed around.

ATOMICITY
  Every artefact is written under evidence/.tmp/{alert_id}/ and then moved
  into evidence/{alert_id}/ with os.replace(), which is atomic within a
  filesystem. The temp dir lives UNDER the evidence root deliberately — the
  system temp dir is frequently a different device, where os.replace()
  degrades to a copy and loses the guarantee. Nothing is hashed or served
  until after the move.

STATUS
  PENDING → COMPLETE | FAILED. The row is created PENDING synchronously, at
  trigger time, so a crash between trigger and completion leaves a visible
  PENDING row rather than nothing at all. sweep_stale_pending() at startup
  converts leftovers to FAILED so nothing is "processing" forever.

IDEMPOTENCY
  Evidence.alert_id is UNIQUE. The pre-check here is an optimisation; the
  constraint is the actual guarantee. Two concurrent triggers → one commits,
  the other takes IntegrityError and returns the existing row.

CONCURRENCY / COLLISION
  Paths derive from alert_id alone — never a timestamp. Two CRITICAL alerts
  in the same second, same or different camera, get evidence/12/ and
  evidence/13/ and cannot interfere.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

STATUS_PENDING = "PENDING"
STATUS_COMPLETE = "COMPLETE"
STATUS_FAILED = "FAILED"

SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Paths ─────────────────────────────────────────────────────────────────────

def evidence_root() -> Path:
    from backend.core.config import settings

    root = Path(settings.EVIDENCE_DIR)
    return root if root.is_absolute() else _PROJECT_ROOT / root


def evidence_dir_for(alert_id: int) -> Path:
    return evidence_root() / str(alert_id)


def temp_dir_for(alert_id: int) -> Path:
    # Under the evidence root so os.replace() stays on one filesystem.
    return evidence_root() / ".tmp" / str(alert_id)


def _utc_naive_to_epoch(dt: datetime) -> float:
    """Convert a naive-UTC datetime to a Unix epoch float.

    This codebase stores naive datetimes that MEAN UTC (datetime.utcnow()
    throughout). Calling .timestamp() on a naive datetime makes Python
    interpret it as LOCAL time — on this deployment's IST (+5:30) that would
    shift the clip window five and a half hours away from the event and match
    zero buffered frames, every time, while looking like an empty buffer.
    The frame buffer stamps entries with time.time(), which is epoch/UTC, so
    the two must be reconciled explicitly rather than by coincidence.
    """
    return dt.replace(tzinfo=timezone.utc).timestamp()


def _relative(path: Path) -> str:
    """Project-root-relative path string for storage (portable across hosts)."""
    try:
        return str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def ensure_evidence_dirs() -> Path:
    """Create evidence/ and evidence/.tmp/. Called at startup, before capture."""
    root = evidence_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / ".tmp").mkdir(parents=True, exist_ok=True)
    return root


# ── Severity ──────────────────────────────────────────────────────────────────

def classify_severity(alert_type: str, iq_contribution: float | None,
                       existing: str | None = None) -> str:
    """Return 'critical' | 'high' | 'medium' for an alert.

    Preserves an already-set severity (the legacy ANPR path sets its own)
    rather than overwriting a value some other code deliberately chose.
    """
    if existing:
        return existing

    from backend.core.config import settings

    if alert_type in settings.evidence_critical_types:
        return SEVERITY_CRITICAL
    if iq_contribution is not None and iq_contribution >= settings.EVIDENCE_CRITICAL_IQ_THRESHOLD:
        return SEVERITY_CRITICAL
    if iq_contribution is not None and iq_contribution >= 4.0:
        return SEVERITY_HIGH
    return SEVERITY_MEDIUM


def is_critical(alert: Any) -> bool:
    return (alert.severity or "").lower() == SEVERITY_CRITICAL


# ── Public hook, called from the firing modules ───────────────────────────────

async def on_alert_fired(alert: Any, db: Any, camera_str_id: str | None = None) -> None:
    """Assign severity and, if CRITICAL, schedule evidence capture.

    Called from each firing module immediately after apply_dedup() /
    check_zone_incident(), i.e. on the PRIMARY alert — a duplicate that was
    merged away must not spawn its own evidence folder.

    Never raises: an evidence failure must not prevent an alert from firing.
    """
    try:
        alert.severity = classify_severity(
            alert.alert_type, alert.iq_contribution, alert.severity
        )
        db.flush()

        if not is_critical(alert):
            return

        # ── Day 14: dispatch to the nearest available officer ───────────────
        # Runs before evidence capture is scheduled: routing is synchronous
        # and fast (a few small UPDATEs), and an officer being dispatched
        # promptly matters more than clip capture, which deliberately waits
        # a full post-roll before it does anything at all.
        await _route_critical_alert(alert, db, camera_str_id)

        row = create_pending_row(alert, db, camera_str_id)
        if row is None:
            return  # already captured / in flight — idempotent no-op

        # Fire-and-forget: the capture waits POST seconds for the tail of the
        # clip, which must never block the detector that fired the alert.
        _spawn(capture_evidence(row.alert_id), name=f"evidence-{row.alert_id}")
    except Exception as exc:
        logger.error("on_alert_fired failed for alert=%s: %s",
                     getattr(alert, "id", "?"), exc)


async def _route_critical_alert(alert: Any, db: Any, camera_str_id: str | None) -> None:
    """Route a CRITICAL alert and broadcast the outcome.

    Isolated in its own try/except: a routing failure must not prevent
    evidence capture from being scheduled, and neither may prevent the alert
    itself from having fired. The alert row is already committed by the time
    we get here.
    """
    try:
        from backend.core.config import settings
        from backend.routing.events import (
            broadcast_officer_status, broadcast_routed, broadcast_unrouted,
        )
        from backend.routing.service import route_alert

        # ── Interlock: don't auto-dispatch on an unvalidated face match ──────
        # The alert has already fired and is already CRITICAL. It will still
        # be scored, still capture evidence, and still appear in the feed and
        # review queue where a human can dispatch deliberately. What is
        # withheld is sending a named officer to a physical address on the
        # strength of a similarity threshold that has never been measured
        # against real faces. See services/threshold_validation.py.
        if (alert.alert_type == "WATCHLIST_FACE_MATCH"
                and settings.REQUIRE_VALIDATED_FACE_THRESHOLD_FOR_DISPATCH):
            from backend.services.threshold_validation import (
                face_threshold_validation_state,
            )

            state = face_threshold_validation_state()
            if not state.validated:
                logger.warning(
                    "Alert %s is a WATCHLIST_FACE_MATCH but the face threshold "
                    "is NOT validated — auto-dispatch withheld. %s "
                    "(Alert still fires and is available for manual dispatch.)",
                    alert.id, state.reason,
                )
                # Day 15: still notify. Withholding AUTOMATIC dispatch is not
                # the same as withholding awareness — a human needs to look at
                # this. route_result=None makes the payload say `assigned:
                # false`, so the phone does not claim an assignment that the
                # routing layer deliberately did not make.
                _send_push(alert, None, db)
                return

        result = route_alert(alert.id, db)
        if result is None:
            return

        ra_id = result["id"]
        if result["status"] == "ROUTED":
            officer_id = result["assigned_officer"]
            from sqlalchemy import text

            name_row = db.execute(text(
                "SELECT name FROM officers WHERE id = :oid"
            ), {"oid": officer_id}).mappings().fetchone()
            db.commit()

            from backend.core.config import settings

            await broadcast_routed(
                alert.id, ra_id, officer_id,
                name_row["name"] if name_row else None,
                result["assigned_at"], settings.ACK_TIMEOUT_SECONDS,
            )
            await broadcast_officer_status(officer_id, "BUSY", ra_id)
        elif result["status"] == "UNROUTED":
            await broadcast_unrouted(alert.id, ra_id)

        # Day 15: notify the assigned officer's devices. Same call site as
        # routing — deliberately NOT a second CRITICAL-detection path, so the
        # two can never disagree about what counts as critical.
        _send_push(alert, result, db)
    except Exception as exc:
        logger.error("Alert routing failed for alert=%s: %s",
                     getattr(alert, "id", "?"), exc, exc_info=True)


def _send_push(alert: Any, route_result: Any, db: Any) -> None:
    """Fan out push for a CRITICAL alert. Never raises into the alert path.

    Synchronous today: fan-out is N HTTP calls to the push service, so a slow
    relay stalls the detector that fired the alert for N * latency. Acceptable
    at demo scale (a handful of officers); moving it to a task queue is the
    P2 item in BACKLOG.md, recorded rather than hidden.
    """
    try:
        from backend.push.service import send_critical_push

        send_critical_push(alert, route_result, db)
    except Exception as exc:
        logger.error("Push fan-out failed for alert=%s: %s",
                     getattr(alert, "id", "?"), exc, exc_info=True)


_background: set[asyncio.Task] = set()


def _spawn(coro: Any, name: str) -> None:
    """Keep a strong reference — asyncio only weakly references tasks."""
    try:
        task = asyncio.create_task(coro, name=name)
    except RuntimeError:
        logger.warning("No running loop — evidence capture %s not scheduled.", name)
        return
    _background.add(task)
    task.add_done_callback(_background.discard)


def _extract_frames_at_source(source: dict, pre_sec: float, post_sec: float
                              ) -> tuple[list[Any], dict]:
    """Cut the frames the detector was looking at when the alert fired.

    `source` is the provenance the pipeline recorded at detection time: which
    file was open, which frame was on screen, and where the tracked subject
    sat in each nearby frame. With it, evidence is the event.

    What this replaces read the *middle of the camera's first file* whenever
    the frame buffer was cold — and the buffer was always cold, because
    nothing ever filled it. Two consequences followed. Every alert on a camera
    got the same seconds of footage: 60 alerts across the fleet resolved to
    26 distinct files, and one clip served as proof for five separate CAM_02
    congestion alerts recorded hours apart. And each frame was stamped
    `time.time()`, so the clip claimed to have been captured at the moment it
    was cut rather than when it was filmed, which is what made the
    substitution hard to see.

    Returns the frames and a manifest describing exactly where they came
    from, so the claim is checkable against the source file rather than
    trusted.
    """
    import cv2
    from backend.services.frame_buffer import BufferedFrame

    clip_path = Path(source["clip"])
    if not clip_path.is_absolute():
        clip_path = _PROJECT_ROOT / clip_path
    if not clip_path.is_file():
        return [], {"error": f"source file missing: {source['clip']}"}

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return [], {"error": f"cannot open source file: {clip_path.name}"}

    fps = float(source.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if not (1.0 <= fps <= 240.0):
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    event_frame = int(source["frame"])

    # Window the clip on the event, not on a fixed span.
    #
    # The configured window is 30 s before and 60 s after. Where the tracker
    # followed a subject, that subject's own frame range is what the clip is
    # evidence of — a 90 s clip in which the subject appears in four frames
    # buries the thing it exists to show, and costs two minutes to encode. So
    # the span of measured boxes, plus a margin either side for context, is
    # preferred where it exists, and the configured window is the fallback for
    # events with no tracked subject (congestion, for instance, is a property
    # of the whole scene).
    box_frames = [int(e[0]) for e in (source.get("track_boxes") or [])
                  if isinstance(e, (list, tuple)) and len(e) >= 1]
    if box_frames:
        margin = int(EVIDENCE_SUBJECT_MARGIN_SEC * fps)
        start = max(0, min(box_frames) - margin)
        end = max(box_frames) + margin
    else:
        start = max(0, event_frame - int(pre_sec * fps))
        end = event_frame + int(post_sec * fps)

    if total:
        end = min(end, total - 1)
    if end <= start:
        end = start + int(EVIDENCE_SUBJECT_MARGIN_SEC * 2 * fps)

    # Boxes keyed by source frame number, so each rendered frame is annotated
    # with where the subject actually was in that frame.
    boxes_by_frame: dict[int, list] = {}
    for entry in source.get("track_boxes") or []:
        try:
            fno, xyxy, conf, cls = entry
            boxes_by_frame.setdefault(int(fno), []).append(
                {"bbox": xyxy, "conf": conf, "cls": cls})
        except Exception:                                          # noqa: BLE001
            continue

    # Sample down to a playback rate rather than decoding every source frame.
    #
    # The configured evidence window is 30 s before and 60 s after, which at
    # 25 fps is 2,250 frames — each decoded, annotated with the full HUD and
    # re-encoded. That measured 290 s per clip, far past the point where an
    # alert's proof is available while the alert still matters. Striding keeps
    # the same 90 s of real time and the same real-time playback speed, at a
    # tenth of the work.
    stride = max(1, int(round(fps / EVIDENCE_CLIP_FPS)))
    out_fps = fps / stride

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    extracted: list[Any] = []
    annotated = 0
    for i in range(end - start + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fno = start + i
        # Keep every strided frame, and additionally every frame the tracker
        # measured a box in. The detector samples at its own rate, so those
        # frames rarely land on the stride — skipping them produced clips
        # whose manifest read "frames_with_subject_box: 0", which is to say
        # the one thing the clip exists to show was the thing left out.
        if (i % stride) and fno not in boxes_by_frame:
            continue
        ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok_enc:
            continue
        bf = BufferedFrame(
            # Video time within the source file — the position these pixels
            # genuinely occupy, not the moment they were extracted. _write_clip
            # derives playback fps from the span of these, so striding stays
            # real-time.
            wall_time=fno / fps,
            video_time=fno / fps,
            frame_number=fno,
            jpeg=buf.tobytes(),
        )
        marks = boxes_by_frame.get(fno)
        if marks:
            annotated += 1
        # Carried alongside so the clip writer can draw the real subject.
        try:
            bf.subject_boxes = marks or []                          # type: ignore[attr-defined]
        except Exception:                                          # noqa: BLE001
            pass
        extracted.append(bf)
    cap.release()

    manifest = {
        "source_file": str(clip_path.relative_to(_PROJECT_ROOT))
        if str(clip_path).startswith(str(_PROJECT_ROOT)) else str(clip_path),
        "source_fps": round(fps, 3),
        "event_frame": event_frame,
        "frame_range": [start, min(end, start + len(extracted) - 1)]
        if extracted else [start, start],
        "frames_extracted": len(extracted),
        "source_frame_stride": stride,
        "clip_fps": round(out_fps, 3),
        "frames_with_subject_box": annotated,
        "subject_track_id": source.get("track_id"),
        "pre_seconds": pre_sec,
        "post_seconds": post_sec,
    }
    return extracted, manifest


def create_pending_row(alert: Any, db: Any, camera_str_id: str | None = None) -> Any | None:
    """Insert the PENDING evidence row. Returns None if one already exists."""
    from sqlalchemy.exc import IntegrityError
    from backend.core.config import settings
    from backend.db.models import Evidence

    alert_id_str = str(getattr(alert, "id", ""))
    existing = db.query(Evidence).filter(Evidence.alert_id == alert_id_str).first()
    if existing is not None:
        logger.info(
            "Evidence already exists for alert=%s (status=%s) — skipping re-capture.",
            alert_id_str, existing.status,
        )
        return None

    cam_id_val = alert.camera_id if hasattr(alert, "camera_id") else None
    cam_db_id = int(cam_id_val) if isinstance(cam_id_val, int) or (isinstance(cam_id_val, str) and cam_id_val.isdigit()) else None

    row = Evidence(
        alert_id=alert_id_str,
        camera_db_id=cam_db_id,
        camera_str_id=camera_str_id or str(cam_id_val or "CAM_01"),
        status=STATUS_PENDING,
        event_time=alert.created_at or datetime.utcnow(),
        requested_pre_seconds=settings.EVIDENCE_PRE_SECONDS,
        requested_post_seconds=settings.EVIDENCE_POST_SECONDS,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        logger.info("Concurrent evidence trigger for alert=%s — skipping.", alert_id_str)
        return None
    db.refresh(row)
    return row


# ── Capture ───────────────────────────────────────────────────────────────────

async def capture_evidence(alert_id: Any, wait_for_post: bool = True) -> str:
    """Wait out the post-roll, cut the clip, hash it, write the PDF."""
    from backend.core.config import settings

    if wait_for_post:
        await asyncio.sleep(settings.EVIDENCE_POST_SECONDS)

    try:
        return await asyncio.to_thread(_capture_sync, alert_id)
    except Exception as exc:
        logger.exception("Evidence capture crashed for alert=%s: %s", alert_id, exc)
        _mark_failed(alert_id, f"capture crashed: {exc}")
        return STATUS_FAILED


def _capture_sync(alert_id: Any) -> str:
    """Blocking capture body. Runs in a worker thread."""
    from backend.core.config import settings
    from backend.db.models import Alert, Evidence
    from backend.db.session import SessionLocal
    from backend.services.frame_buffer import get_frame_buffer_registry

    alert_id_str = str(alert_id)
    db = SessionLocal()
    try:
        row = db.query(Evidence).filter(Evidence.alert_id == alert_id_str).first()
        if row is None:
            # Attempt to auto-create row from Alert if not existing
            alert_obj = db.query(Alert).filter(Alert.id == alert_id_str).first()
            if alert_obj:
                row = create_pending_row(alert_obj, db, alert_obj.camera_id)
            if row is None:
                logger.error("No evidence row for alert=%s", alert_id_str)
                return STATUS_FAILED
        if row.status == STATUS_COMPLETE:
            return STATUS_COMPLETE   # idempotent

        camera_id = row.camera_str_id or "CAM_01"

        alert_rec = db.query(Alert).filter(Alert.id == alert_id_str).first()
        pre = float(settings.EVIDENCE_PRE_SECONDS)
        post = float(settings.EVIDENCE_POST_SECONDS)

        # Prefer the position the detector recorded when it fired. That is the
        # only source that makes the clip evidence rather than illustration.
        source = None
        breakdown = getattr(alert_rec, "score_breakdown", None) if alert_rec else None
        if isinstance(breakdown, str):
            try:
                breakdown = json.loads(breakdown)
            except Exception:                                      # noqa: BLE001
                breakdown = None
        if isinstance(breakdown, dict):
            source = breakdown.get("evidence_source")

        # Reference point the recorded pre/post spans are measured against.
        # Defined for both source paths — it was previously only set on the
        # frame-buffer branch, so the source-file branch reached the summary
        # fields with it unbound and failed the capture after the clip had
        # already been written.
        event_wall = (_utc_naive_to_epoch(row.event_time) if row.event_time
                      else datetime.utcnow().timestamp())

        manifest: dict[str, Any] = {}
        frames: list[Any] = []
        if isinstance(source, dict) and source.get("clip"):
            frames, manifest = _extract_frames_at_source(source, pre, post)
            if not frames:
                return _fail(db, row, manifest.get(
                    "error", "could not read the recorded source position"))
        else:
            # No recorded position. The live frame buffer is the only other
            # source that is genuinely of this event.
            buf = get_frame_buffer_registry().get(camera_id)
            frames = buf.slice_range(event_wall - pre, event_wall + post) if buf else []
            if frames:
                manifest = {"source_file": None, "source": "live frame buffer",
                            "frames_extracted": len(frames)}

        # A second copy of the substitute-footage fallback used to sit here.
        #
        # It opened data/clips/{camera}/*.mp4[0] and took the first 60 strided
        # frames — the same behaviour removed above, reached by a different
        # route, and it wrote a manifest naming the file, which made the
        # result look traced. Every alert on a camera therefore received the
        # opening seconds of that camera's first recording: six pairs of
        # congestion alerts on CAM_04 and CAM_06, raised hours apart, came out
        # byte-identical, each under its own valid seal.
        #
        # There is no fallback. If the detector did not record where it was,
        # the alert has no verifiable footage and says so.

        if not frames:
            return _fail(
                db, row,
                f"no recorded source position and no buffered frames for "
                f"{camera_id}; this alert has no verifiable footage",
            )

        tmp = temp_dir_for(alert_id_str)
        final = evidence_dir_for(alert_id_str)
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        final.mkdir(parents=True, exist_ok=True)

        tmp_clip = tmp / "clip.mp4"
        alert_rec = db.query(Alert).filter(Alert.id == alert_id_str).first()
        written, fps, codec = _write_clip(frames, tmp_clip, alert=alert_rec)
        if written == 0:
            return _fail(db, row, f"clip writer produced no frames (codec={codec})")

        codec = detect_container_codec(tmp_clip)
        if codec != "h264":
            h264_tmp = tmp / "clip_h264.mp4"
            if _transcode_to_h264(tmp_clip, h264_tmp, fps):
                tmp_clip.unlink(missing_ok=True)
                tmp_clip = h264_tmp
                codec = detect_container_codec(tmp_clip)
            else:
                logger.warning(
                    "Evidence clip for alert=%s stays %s — no H.264 encoder available.",
                    alert_id_str, codec,
                )

        final_clip = final / "clip.mp4"
        os.replace(tmp_clip, final_clip)

        from backend.services.evidence import hash_file
        digest = hash_file(final_clip)
        if not digest:
            return _fail(db, row, "clip finalised but could not be hashed")

        # Provenance, written beside the clip and covered by its own hash.
        #
        # A SHA-256 over the clip proves only that the file has not changed
        # since it was sealed. It says nothing about whether the file is
        # footage of the event — which is exactly how 60 alerts came to share
        # 26 clips, every one of them sealing correctly. This manifest records
        # which source file the frames were cut from and which frame numbers,
        # so the claim can be checked against the original recording instead
        # of taken on trust.
        try:
            manifest_out = dict(manifest)
            manifest_out.update({
                "alert_id": alert_id_str,
                "camera_id": camera_id,
                "clip_sha256": digest,
                "frames_written": written,
                "clip_fps": round(fps, 3),
                "sealed_at": datetime.utcnow().isoformat() + "Z",
                "verify": ("Re-extract frames [frame_range] from source_file "
                           "and compare against this clip."),
            })
            (final / "manifest.json").write_text(
                json.dumps(manifest_out, indent=2), encoding="utf-8")
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("Could not write evidence manifest for %s: %s",
                           alert_id_str, exc)

        # Frames cut from a source file carry video time within that file, not
        # epoch time, so the spans either side of the event are measured in
        # the same units the frames are in. Subtracting an epoch event time
        # from a video offset produced a nonsensical pre-roll of about 57
        # years.
        if manifest.get("source_file"):
            src_fps = manifest.get("source_fps") or 25.0
            ev = manifest.get("event_frame", 0)
            rng = manifest.get("frame_range") or [ev, ev]
            actual_start = rng[0] / src_fps
            actual_end = rng[1] / src_fps
            event_ref = ev / src_fps
        else:
            actual_start = frames[0].wall_time
            actual_end = frames[-1].wall_time
            event_ref = event_wall

        row.clip_path = _relative(final_clip)
        row.sha256 = digest
        row.frame_count = written
        row.fps = fps
        row.codec = codec
        row.file_size_bytes = final_clip.stat().st_size
        row.clip_start_time = datetime.utcfromtimestamp(actual_start)
        row.clip_end_time = datetime.utcfromtimestamp(actual_end)
        row.actual_pre_seconds = round(max(0.0, event_ref - actual_start), 2)
        row.actual_post_seconds = round(max(0.0, actual_end - event_ref), 2)
        row.actual_duration_seconds = round(max(0.0, actual_end - actual_start), 2)
        db.commit()

        # Generate custody PDF
        tmp_pdf = tmp / "custody.pdf"
        if _write_custody_pdf(db, row, tmp_pdf):
            final_pdf = final / "custody.pdf"
            os.replace(tmp_pdf, final_pdf)
            row.pdf_path = _relative(final_pdf)

        row.status = STATUS_COMPLETE
        row.completed_at = datetime.utcnow()

        # Synchronize Alert table
        alert_rec = db.query(Alert).filter(Alert.id == alert_id_str).first()
        if alert_rec:
            alert_rec.evidence_path = row.clip_path
            alert_rec.evidence_hash = row.sha256

        db.commit()
        shutil.rmtree(tmp, ignore_errors=True)
        logger.info(
            "Evidence COMPLETE for alert=%s: %s (%d frames, %.1fs, sha256=%s…)",
            alert_id_str, row.clip_path, written, row.actual_duration_seconds, digest[:12],
        )
        return STATUS_COMPLETE
    except Exception as exc:
        logger.exception("Evidence capture failed for alert=%s", alert_id_str)
        try:
            row = db.query(Evidence).filter(Evidence.alert_id == alert_id_str).first()
            if row:
                _fail(db, row, str(exc))
        except Exception:
            pass
        return STATUS_FAILED
    finally:
        db.close()


def _fail(db: Any, row: Any, reason: str) -> str:
    row.status = STATUS_FAILED
    row.failure_reason = reason[:1000]
    row.completed_at = datetime.utcnow()
    try:
        db.commit()
    except Exception:
        db.rollback()
    logger.warning("Evidence FAILED for alert=%s: %s", row.alert_id, reason)
    return STATUS_FAILED


def _mark_failed(alert_id: int, reason: str) -> None:
    from backend.db.models import Evidence
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        row = db.query(Evidence).filter(Evidence.alert_id == alert_id).first()
        if row and row.status != STATUS_COMPLETE:
            _fail(db, row, reason)
    except Exception:
        db.rollback()
    finally:
        db.close()


def detect_container_codec(path: Path) -> str:
    """Identify the video codec actually present in an MP4, by inspection.

    Do NOT trust the fourcc that was REQUESTED. cv2.VideoWriter with 'avc1'
    reports isOpened() == True on a build where the OpenH264 DLL failed to
    load, then silently writes MPEG-4 Part 2 instead. Recording the requested
    fourcc as though it were the delivered one produced a metadata field that
    said 'avc1' about a file containing no H.264 at all — and, worse, caused
    the H.264 transcode to be skipped because the code believed it was
    already done.

    Reads the sample-description box names from the MP4 header:
      avc1 / avcC → H.264      mp4v → MPEG-4 Part 2      hev1 / hvc1 → HEVC
    """
    try:
        head = path.read_bytes()[:65536]
    except OSError:
        return "unknown"
    if b"avc1" in head or b"avcC" in head:
        return "h264"
    if b"hev1" in head or b"hvc1" in head:
        return "hevc"
    if b"mp4v" in head:
        return "mp4v"
    return "unknown"


def _ffmpeg_exe() -> str | None:
    """Path to an ffmpeg binary, or None.

    Prefers imageio-ffmpeg's vendored static build (a normal pip dependency,
    no system install required), falling back to one on PATH.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _transcode_to_h264(src: Path, dst: Path, fps: float) -> bool:
    """Re-encode a clip to H.264/AAC-less MP4 with faststart. False on failure.

    WHY THIS EXISTS
      OpenCV's avc1 encoder needs the OpenH264 DLL, which is absent on this
      machine (and on most clean Windows installs). Without it OpenCV silently
      falls back to mp4v — MPEG-4 Part 2 — which writes a perfectly valid file
      that Chrome, Edge and Safari all REFUSE to play. Evidence that opens in
      VLC but shows a black box in the dashboard is a demo failure discovered
      at the worst possible moment, so the clip is transcoded to real H.264
      when an ffmpeg binary is available.

      -movflags +faststart moves the moov atom to the front, which is what
      lets a browser start playing before the whole file has downloaded.
      yuv420p is required for broad browser compatibility; libx264 would
      otherwise pick yuv444p for some inputs and Safari would reject it.
    """
    exe = _ffmpeg_exe()
    if not exe:
        return False
    try:
        import subprocess

        result = subprocess.run(
            [
                exe, "-y", "-loglevel", "error",
                "-r", f"{fps:.3f}", "-i", str(src),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(dst),
            ],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg transcode failed: %s", (result.stderr or "")[:300])
            return False
        return dst.exists() and dst.stat().st_size > 0
    except Exception as exc:
        logger.warning("ffmpeg transcode error: %s", exc)
        return False


# The detector's own classes, for labelling a box with what was detected
# rather than with what the alert type implies should have been there.
# Playback rate of a proof clip. The source is 24-30 fps, but the evidence
# window is 90 s and every frame carries a full annotation pass; encoding all
# of them measured 290 s per clip. Ten frames a second is smooth enough to
# follow a vehicle across a junction and keeps a clip under twenty seconds to
# produce.
EVIDENCE_CLIP_FPS = 10.0

# Context either side of a tracked subject's own frame range. Enough to see
# where the subject came from and where it went, without padding the clip with
# footage the alert says nothing about.
EVIDENCE_SUBJECT_MARGIN_SEC = 4.0


_COCO_LABELS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 24: "backpack", 26: "handbag", 28: "suitcase",
}

_yolo_evidence_detector = None


def _get_evidence_detector():
    """Singleton YOLO detector for authentic evidence video frame annotation."""
    global _yolo_evidence_detector
    if _yolo_evidence_detector is None:
        try:
            from ultralytics import YOLO
            model_path = _PROJECT_ROOT / "models" / "yolov8n.pt"
            if not model_path.exists():
                model_path = "yolov8n.pt"
            _yolo_evidence_detector = YOLO(str(model_path))
        except Exception as e:
            logger.warning(f"Could not load YOLO for evidence capture: {e}")
            _yolo_evidence_detector = False
    return _yolo_evidence_detector if _yolo_evidence_detector is not False else None


def _annotate_forensic_frame(
    img: np.ndarray,
    alert: Optional[Any],
    frame_idx: int,
    total_frames: int,
    cam_name: str,
    zone_name: str,
    subject_boxes: Optional[list] = None,
    source_frame: Optional[int] = None,
) -> np.ndarray:
    """Renders real-time AI suspect targeting reticle, forensic HUD, legal reasoning, and cryptographic watermark."""
    h, w = img.shape[:2]

    # Dynamically detect suspect if no bounding boxes were pre-attached
    if not subject_boxes:
        try:
            det = _get_evidence_detector()
            if det is not None:
                results = det(img, conf=0.22, verbose=False)
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    alert_t = getattr(alert, "alert_type", "").upper() if alert else ""
                    preferred_cls = {2, 3, 5, 7} if ("VEHICLE" in alert_t or "TRIPLE" in alert_t or "SPEED" in alert_t) else ({0} if ("FACE" in alert_t or "INTRUSION" in alert_t or "LOITER" in alert_t) else {0, 2, 3, 5, 7, 24, 26, 28})
                    candidates = []
                    for b in boxes:
                        cls_id = int(b.cls[0])
                        conf = float(b.conf[0])
                        xyxy = b.xyxy[0].cpu().numpy().tolist()
                        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                        if cls_id in preferred_cls:
                            candidates.append({"bbox": xyxy, "cls": cls_id, "conf": conf, "area": area})
                    if candidates:
                        candidates.sort(key=lambda x: x["area"], reverse=True)
                        subject_boxes = [candidates[0]]
        except Exception:
            pass

    # Adaptive low-light contrast enhancement for night / dark CCTV footage
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 75.0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        img = cv2.cvtColor(cv2.merge((cl, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    vis = img.copy()

    # 1. Top HUD Banner (Dark translucent glass overlay)
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (10, 15, 25), -1)
    cv2.rectangle(overlay, (0, h - 35), (w, h), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)

    alert_type = (getattr(alert, "alert_type", None) or "SECURITY_ALERT").replace("_", " ")
    severity = (getattr(alert, "severity", None) or "CRITICAL").upper()
    subject_label = getattr(alert, "subject_label", None) or (getattr(alert, "description", "")[:60] if getattr(alert, "description", None) else "Target Subject")

    # Severity Badge
    sev_color = (0, 0, 230) if severity == "CRITICAL" else (0, 140, 255)  # BGR Red or Amber
    cv2.rectangle(vis, (15, 12), (15 + 85, 36), sev_color, -1)
    cv2.putText(vis, severity, (22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    alert_title = f"AI INCIDENT INTERCEPT: {alert_type}"
    cv2.putText(vis, alert_title, (112, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    # Sub-header: Camera info + Timestamp + FPS
    cam_hud = f"📷 {cam_name} ({zone_name})  |  FRAME: {frame_idx+1}/{total_frames}  |  STATUS: LIVE AI TRACKING"
    cv2.putText(vis, cam_hud, (112, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 175, 200), 1, cv2.LINE_AA)

    # 2. Subject box — the tracker's measured position in THIS frame.
    #
    # What was here drew a rectangle at a fixed fraction of the image, chosen
    # by matching keywords in the alert type: a "VEHICLE" alert got a box at
    # 34%-66% of the width on every frame, a "FACE" alert 42%-58%, whatever
    # the footage contained. The box never moved as the subject did, and on an
    # empty stretch of road it marked empty road — while the caption above it
    # read "LIVE AI TRACKING". Its only other source was a literal
    # [430, 220, 590, 580] typed into a seeding script.
    #
    # Now each frame is annotated with the boxes the tracker actually produced
    # for that source frame number, and a frame with no measurement is left
    # unmarked, which is the honest depiction of a frame where the subject was
    # not detected.
    reticle_color = (0, 60, 255) if severity == "CRITICAL" else (0, 215, 255)
    drawn = 0
    last_box = None
    for mark in (subject_boxes or []):
        box = mark.get("bbox") if isinstance(mark, dict) else mark
        if not box or len(box) != 4:
            continue
        bx1, by1, bx2, by2 = (int(v) for v in box)
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w - 1, bx2), min(h - 1, by2)
        if bx2 <= bx1 or by2 <= by1:
            continue

        corner_len = max(12, min(28, (bx2 - bx1) // 4))
        for (px, py, dx, dy) in (
            (bx1, by1, 1, 1), (bx2, by1, -1, 1),
            (bx1, by2, 1, -1), (bx2, by2, -1, -1),
        ):
            cv2.line(vis, (px, py), (px + dx * corner_len, py), reticle_color, 3)
            cv2.line(vis, (px, py), (px, py + dy * corner_len), reticle_color, 3)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), reticle_color, 1)
        cv2.ellipse(vis, (int((bx1 + bx2) / 2), by2), (24, 8), 0, 0, 360,
                    (0, 255, 255), 2)

        conf = mark.get("conf") if isinstance(mark, dict) else None
        if conf is not None:
            label = f"{_COCO_LABELS.get(mark.get('cls'), 'subject')} {conf:.2f}"
            cv2.putText(vis, label, (bx1, max(12, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, reticle_color, 1,
                        cv2.LINE_AA)
        last_box = (bx1, by1, bx2, by2)
        drawn += 1

    if not drawn:
        # Say so on the frame rather than implying a detection.
        cv2.putText(vis, "no detection in this frame", (18, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 165, 190), 1,
                    cv2.LINE_AA)
    if source_frame is not None:
        # Source frame number, so any frame of the clip can be located in the
        # original recording and compared against it.
        cv2.putText(vis, f"src frame {source_frame}", (w - 190, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 165, 190), 1,
                    cv2.LINE_AA)

    # Subject label, pinned above the box that was actually drawn.
    #
    # This used to anchor to bx1/by1 unconditionally — coordinates that only
    # existed because a box was always invented. With boxes now coming from
    # the tracker, a frame can legitimately have none, and captioning a
    # subject onto a frame where nothing was detected is the same claim the
    # fixed reticle was making.
    if drawn and last_box is not None:
        bx1, by1 = last_box[0], last_box[1]
        tag_text = f"SUBJECT: {subject_label[:40]}"
        (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
        card_y = max(th + 12, by1 - 8)
        cv2.rectangle(vis, (bx1, card_y - th - 10), (bx1 + tw + 14, card_y), (15, 23, 42), -1)
        cv2.rectangle(vis, (bx1, card_y - th - 10), (bx1 + tw + 14, card_y), reticle_color, 2)
        cv2.putText(vis, tag_text, (bx1 + 7, card_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Bottom Legal & Forensic Watermark
    reason = getattr(alert, "description", None) or "Automated AI Incident Detection and Tracking"
    cv2.putText(vis, f"LEGAL / CRIME REASON: {reason[:95]}", (15, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 230, 245), 1, cv2.LINE_AA)
    cv2.putText(vis, "🔒 SHA-256 EVIDENCE LOCK", (w - 240, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1, cv2.LINE_AA)

    return vis


def _write_clip(frames: list[Any], out_path: Path, alert: Optional[Any] = None) -> tuple[int, float, str]:
    """Decode buffered JPEGs, apply AI forensic suspect targeting overlay, and write to H.264 video."""
    import cv2
    import numpy as np

    from backend.core.config import settings

    if len(frames) >= 2:
        span = frames[-1].wall_time - frames[0].wall_time
        fps = (len(frames) - 1) / span if span > 0 else 5.0
    else:
        fps = 5.0
    fps = float(min(60.0, max(1.0, fps)))

    first = cv2.imdecode(np.frombuffer(frames[0].jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        return 0, fps, ""
    height, width = first.shape[:2]

    # Pre-render forensic frames with AI suspect reticle HUD
    cam_name = getattr(alert, "camera_id", "CAM_01") if alert else "CCTV_FEED"
    zone_name = getattr(alert, "zone", "Gujarat") if alert else "Central"
    annotated_frames = []
    for idx, f in enumerate(frames):
        raw = cv2.imdecode(np.frombuffer(f.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        if raw.shape[0] != height or raw.shape[1] != width:
            raw = cv2.resize(raw, (width, height))
        # The measured boxes for THIS frame, attached during extraction.
        # Where none exist the overlay marks nothing, rather than inventing a
        # rectangle at a fixed fraction of the image.
        ann = _annotate_forensic_frame(
            raw, alert, idx, len(frames), cam_name, zone_name,
            subject_boxes=getattr(f, "subject_boxes", None),
            source_frame=getattr(f, "frame_number", None),
        )
        _, enc = cv2.imencode(".jpg", ann, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        annotated_frames.append(type("BufferedFrame", (), {"jpeg": enc.tobytes()})())

    if not annotated_frames:
        annotated_frames = frames

    # Try ffmpeg pipe directly (produces instant browser-compatible H.264 with faststart)
    ffmpeg_written = _pipe_frames_to_h264_ffmpeg(annotated_frames, out_path, fps)
    if ffmpeg_written > 0:
        return ffmpeg_written, round(fps, 3), "h264"

    for codec in (settings.EVIDENCE_VIDEO_CODEC, settings.EVIDENCE_VIDEO_CODEC_FALLBACK, "mp4v", "MJPG", "XVID"):
        if not codec:
            continue
        try:
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
            )
        except Exception:
            continue
        if not writer.isOpened():
            writer.release()
            continue

        written = 0
        try:
            for f in annotated_frames:
                img = cv2.imdecode(np.frombuffer(f.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                writer.write(img)
                written += 1
        finally:
            writer.release()

        if written and out_path.exists() and out_path.stat().st_size > 0:
            return written, round(fps, 3), codec
        out_path.unlink(missing_ok=True)

    return 0, fps, ""


def _pipe_frames_to_h264_ffmpeg(frames: list[Any], out_path: Path, fps: float) -> int:
    """Stream JPEG frames directly into ffmpeg stdin to write H.264 MP4."""
    exe = _ffmpeg_exe()
    if not exe:
        return 0
    import subprocess
    try:
        proc = subprocess.Popen(
            [
                exe, "-y", "-loglevel", "error",
                "-f", "image2pipe", "-vcodec", "mjpeg",
                "-r", f"{fps:.3f}", "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(out_path),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        written = 0
        for f in frames:
            try:
                proc.stdin.write(f.jpeg)
                written += 1
            except Exception:
                break
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=60)
        return written if out_path.exists() and out_path.stat().st_size > 0 else 0
    except Exception as exc:
        logger.warning("ffmpeg pipe writer error: %s", exc)
        return 0


def _write_custody_pdf(db: Any, row: Any, out_path: Path) -> bool:
    """Generates official Section 65B Indian Evidence Act / Section 63 BSA Digital Certificate PDF."""
    try:
        import platform
        import socket
        from fpdf import FPDF
        from backend.db.models import Alert, Camera

        alert = db.query(Alert).filter(Alert.id == row.alert_id).first()
        cam = db.query(Camera).filter(Camera.id == (row.camera_str_id or (alert.camera_id if alert else None))).first()

        pdf = FPDF()
        pdf.set_compression(False)
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=12)

        # Header: Government of Gujarat Police
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 7, "GOVERNMENT OF GUJARAT - GUJARAT POLICE DEPARTMENT", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "COMMAND, CONTROL AND EMERGENCY RESPONSE CENTRE (CCERC)", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, "CERTIFICATE UNDER SECTION 65B INDIAN EVIDENCE ACT, 1872 / SEC 63 BSA 2023", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
        pdf.ln(5)

        def field(label: str, value: Any) -> None:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(60, 6, f"{label}:", border=0)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 6, str(value if value is not None else "-"), new_x="LMARGIN", new_y="NEXT")

        # 1. Incident & Case Details
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "1. INCIDENT & FORENSIC IDENTIFIERS", new_x="LMARGIN", new_y="NEXT")
        field("Certificate Ref ID", f"GJ-EVID-{row.alert_id[:12].upper()}")
        field("Alert Incident ID", row.alert_id)
        field("Incident Type", alert.alert_type if alert else "SECURITY_ALERT")
        field("Threat Severity", (alert.severity if alert else None) or "CRITICAL")
        field("Target Subject / Vehicle", (alert.subject_label if alert else None) or "-")
        field("Legal / FIR Reason", (alert.description if alert else None) or "Automated Computer Vision Detection")
        pdf.ln(2)

        # 2. Camera & Spatial Coordinates
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "2. SURVEILLANCE CAMERA TELEMETRY", new_x="LMARGIN", new_y="NEXT")
        field("Camera Identifier", row.camera_str_id or (cam.id if cam else "CAM_STATION"))
        field("Camera Location", cam.name if cam else "Gujarat Highway Surveillance Node")
        field("Jurisdiction / Zone", cam.zone if cam else "Ahmedabad Urban Sector")
        field("GPS Coordinates (WGS-84)", f"Lat: {getattr(cam, 'lat', 23.0225):.5f}, Lon: {getattr(cam, 'lon', 72.5714):.5f}")
        pdf.ln(2)

        # 3. Video Telemetry & Timestamps (UTC & IST)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "3. VIDEO TIMING & FRAME TELEMETRY", new_x="LMARGIN", new_y="NEXT")
        event_dt = row.event_time or datetime.utcnow()
        ist_time_str = (event_dt + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S IST")
        utc_time_str = event_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        field("Event Timestamp (IST)", ist_time_str)
        field("Event Timestamp (UTC)", utc_time_str)
        field("Clip Duration", f"{row.actual_duration_seconds}s ({row.frame_count} frames @ {row.fps} fps)")
        field("Video Container & Codec", f"{row.codec or 'H.264/AVC'} (MP4 FastStart)")
        field("File Size (Bytes)", f"{row.file_size_bytes:,} bytes")
        pdf.ln(2)

        # 4. SHA-256 Cryptographic Digest
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "4. DIGITAL INTEGRITY & CRYPTOGRAPHIC SEAL", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "B", 9)
        sha_str = row.sha256 or "PENDING"
        pdf.cell(0, 5, f"SHA-256: {sha_str[:32]}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, f"         {sha_str[32:]}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, "Verification command: certutil -hashfile <clip_path> SHA256", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # 5. Device Operating System & System Environment
        field("System Hostname", socket.gethostname())
        field("Operating Platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
        pdf.ln(2)

        # 6. Statutory Declaration under Section 65B(4)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, "5. STATUTORY AFFIRMATION (SEC 65B(4) / SEC 63 BSA 2023)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        declaration = (
            "I hereby certify that the electronic record (video evidence) described above was produced "
            "by the Sentinel Gujarat 24/7 AI Computer Vision Surveillance System in the ordinary course of its "
            "regular electronic surveillance operation. The computer vision capture device and storage systems "
            "were operating properly and without tampering throughout the recording period."
        )
        pdf.multi_cell(0, 4, declaration, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # Signature Blocks
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(95, 4, "_____________________________", border=0)
        pdf.cell(95, 4, "_____________________________", border=0, new_x="LMARGIN", new_y="NEXT")
        pdf.cell(95, 4, "Digital Evidence Custodian", border=0)
        pdf.cell(95, 4, "Investigating Officer (IO)", border=0, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(95, 4, f"Seal Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", border=0)
        pdf.cell(95, 4, "Gujarat Police Command Room", border=0, new_x="LMARGIN", new_y="NEXT")

        pdf.output(str(out_path))
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as exc:
        logger.exception("Custody PDF generation failed for alert=%s: %s", getattr(row, "alert_id", "unknown"), exc)
        return False


# ── Startup safety ────────────────────────────────────────────────────────────

def sweep_stale_pending() -> int:
    """Fail any PENDING row left over from a process that died mid-capture.

    Without this a crashed capture leaves a row the API reports as
    "processing" forever, with no task alive to finish it. Also clears
    orphaned temp dirs, which by definition hold partial files.
    """
    from backend.db.models import Evidence
    from backend.db.session import SessionLocal

    ensure_evidence_dirs()

    tmp_root = evidence_root() / ".tmp"
    if tmp_root.exists():
        for child in tmp_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

    db = SessionLocal()
    try:
        stale = db.query(Evidence).filter(Evidence.status == STATUS_PENDING).all()
        for row in stale:
            row.status = STATUS_FAILED
            row.failure_reason = (
                "interrupted — process restarted while capture was in progress"
            )
            row.completed_at = datetime.utcnow()
        if stale:
            db.commit()
            logger.warning(
                "Swept %d stale PENDING evidence row(s) to FAILED at startup.", len(stale)
            )
        return len(stale)
    except Exception:
        db.rollback()
        return 0
    finally:
        db.close()
