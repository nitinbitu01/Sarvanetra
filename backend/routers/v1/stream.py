# backend/routers/v1/stream.py

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

try:
    from backend.auth.dependencies import require_officer_auth
    from backend.db.session import get_db
    from backend.stream.manager import ffmpeg_processes, spawn_ffmpeg, _hls_dir
except ImportError:
    from sentinel.auth import require_officer_auth
    from sentinel.db import get_db
    from sentinel.stream.manager import ffmpeg_processes, spawn_ffmpeg, _hls_dir

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stream", tags=["stream"])


async def _wait_for_manifest(manifest_path: Path, timeout_seconds: float = 10.0) -> bool:
    """
    Poll until manifest exists and has content.
    """
    elapsed = 0.0
    while elapsed < timeout_seconds:
        if manifest_path.exists() and manifest_path.stat().st_size > 0:
            return True
        await asyncio.sleep(0.5)
        elapsed += 0.5
    return False


@router.get("/{camera_id}/output.m3u8")
async def get_manifest(
    camera_id: Any,
    db: Session = Depends(get_db),
    officer=Depends(require_officer_auth),
):
    """
    Serve the HLS manifest for a camera.
    Auth required: live footage must not be publicly accessible.
    Department scope required too: authenticated is not the same as entitled,
    and this serves another department's live footage on request otherwise.
    """
    from backend.auth.dependencies import assert_camera_in_scope
    assert_camera_in_scope(camera_id, officer, db)

    camera = db.execute(
        text("SELECT id, stream_url FROM cameras WHERE id = :id"),
        {"id": str(camera_id)}
    ).fetchone()
    if not camera and str(camera_id).isdigit():
        camera = db.execute(
            text("SELECT id, stream_url FROM cameras WHERE id = :id"),
            {"id": int(camera_id)}
        ).fetchone()

    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Camera not found"}
        )

    if not camera.stream_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error":     "No stream configured for this camera",
                "camera_id": camera_id,
                "hint":      "Set cameras.stream_url to an RTSP URL or file path",
            }
        )

    manifest_path = _hls_dir(camera_id) / "output.m3u8"

    # Lazy spawn for cold-start (pre-warmed cameras skip this block)
    proc = ffmpeg_processes.get(camera_id)
    if not proc or proc.poll() is not None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            spawn_ffmpeg,
            camera_id,
            camera.stream_url,
        )

        ready = await _wait_for_manifest(manifest_path, timeout_seconds=10.0)
        if not ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error":     "Stream not ready yet — retry in a few seconds",
                    "camera_id": camera_id,
                    "hint":      "Cold-start requires ~6s for initial segments",
                }
            )

    return FileResponse(
        manifest_path,
        media_type="application/vnd.apple.mpegurl",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Pragma":        "no-cache",
        }
    )


@router.get("/{camera_id}/{segment}.ts")
async def get_segment(
    camera_id: Any,
    segment: str,
    db: Session = Depends(get_db),
    officer=Depends(require_officer_auth),
):
    """
    Serve a .ts segment file.

    Scoped as well as authenticated. Guarding only the manifest would be
    theatre: the segments carry the actual video, their names are predictable
    (seg00001.ts), and a caller who wanted another department's footage would
    simply skip the manifest and ask for them directly.
    """
    from backend.auth.dependencies import assert_camera_in_scope
    assert_camera_in_scope(camera_id, officer, db)

    if ".." in segment or "/" in segment or "\\" in segment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid segment name"}
        )

    segment_path = _hls_dir(camera_id) / f"{segment}.ts"
    if not segment_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Segment not found or expired"}
        )

    return FileResponse(
        segment_path,
        media_type="video/MP2T",
        headers={
            "Cache-Control": "max-age=60",
        }
    )
