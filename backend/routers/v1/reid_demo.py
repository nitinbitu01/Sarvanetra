"""Serve the cross-camera Re-ID demonstration: four clips, one identity.

One motorcycle was filmed passing four points on four phones. Each clip is
annotated with the identity the matcher assigned, and the same global id appears
on all four — which is the entire claim of cross-camera Re-ID, shown rather than
asserted.

The link is only drawn when a clip's similarity to the first clip beats the best
score ANY of 500 vehicles from this project's own CCTV achieved against that
same query. The bar is therefore not chosen: it is whatever the hardest wrong
answer scored. A clip that fails it is labelled NO MATCH in its own video, so
the demonstration is capable of failing.

Built by backend/scripts/reid_demo_build.py. If nothing has been built yet these
endpoints say so rather than inventing a result.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reid-demo", tags=["reid-demo"])

_PROJ = Path(__file__).resolve().parents[3]
DEMO_DIR = _PROJ / "output" / "reid_demo"
MANIFEST = DEMO_DIR / "manifest.json"


@router.get("/manifest")
def reid_demo_manifest():
    """What was matched, to what, and by how much."""
    if not MANIFEST.is_file():
        return {
            "available": False,
            "reason": ("No demonstration has been built yet. Run "
                       "backend/scripts/reid_demo_build.py"),
            "clips": [],
        }
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:                                       # noqa: BLE001
        logger.error("re-id demo manifest unreadable: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            "Demonstration manifest is unreadable")
    data["available"] = True
    for c in data.get("clips", []):
        c["playable"] = (DEMO_DIR / f"{c['camera']}.mp4").is_file()
    return data


@router.get("/video/{camera}")
def reid_demo_video(camera: str):
    """One annotated clip. Named cameras only — this serves no arbitrary path."""
    cam = camera.upper()
    if cam not in {"CAM_M1", "CAM_M2", "CAM_M3", "CAM_M4"}:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown demo clip")
    p = DEMO_DIR / f"{cam}.mp4"
    if not p.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"{cam} has not been built yet")
    return FileResponse(str(p), media_type="video/mp4", filename=f"{cam}.mp4")
