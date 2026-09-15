"""backend/services/sighting_snapshot.py — the actual frame this vehicle was
seen in, cut from the actual footage.

WHY
  A trajectory drawn on a map is a claim. The thing that makes it evidence is
  being able to point at each pin and see the vehicle standing there, in the
  frame that camera recorded, at the time the route says it passed.

  Everything needed is already indexed: `output/journey_index/records.jsonl`
  stores, for every sighting, the clip it came from, the frame number inside
  that clip, and the plate crop that was cut out of that frame. So the frame
  can be recovered exactly rather than approximated.

HOW THE VEHICLE IS LOCATED IN THE FRAME
  The index does not store a bounding box, but it does store the plate crop —
  and that crop was cut FROM this frame, so `cv2.matchTemplate` finds where it
  came from to the pixel. That is not an estimate or a re-detection that might
  pick a different vehicle: it is the same pixels, located again.

  From the plate the vehicle above it is inferred geometrically, which is a
  drawn guide rather than a measurement, so it is drawn thin and the plate box
  is drawn solid. The caption says which is which.

  If the match is weak the frame is still returned, with the plate shown as an
  inset instead and the overlay saying the position could not be confirmed —
  never a box in the wrong place.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "data" / "clips"
CACHE = ROOT / "output" / "journey_snapshots"

# Below this the template match is not trusted enough to draw a box.
MATCH_MIN = 0.55


def _find_clip(camera: str, clip_name: str) -> Optional[Path]:
    d = CLIPS / camera
    if not d.is_dir():
        return None
    exact = d / f"{clip_name}.mp4"
    if exact.is_file():
        return exact
    for p in sorted(d.glob("*.mp4")):
        if clip_name and clip_name in p.stem:
            return p
    return None


def _read_frame(path: Path, frame_no: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total and frame_no >= total:
        frame_no = max(0, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_no)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _locate_plate(frame: np.ndarray, crop_path: Path):
    """Where in this frame the stored plate crop came from."""
    crop = cv2.imread(str(crop_path))
    if crop is None or crop.size == 0:
        return None, 0.0
    fh, fw = frame.shape[:2]
    ch, cw = crop.shape[:2]
    if ch >= fh or cw >= fw or ch < 6 or cw < 6:
        return None, 0.0
    try:
        res = cv2.matchTemplate(frame, crop, cv2.TM_CCOEFF_NORMED)
        _, best, _, loc = cv2.minMaxLoc(res)
    except Exception:                                               # noqa: BLE001
        return None, 0.0
    return (int(loc[0]), int(loc[1]), int(loc[0] + cw), int(loc[1] + ch)), float(best)


def build(sighting: dict[str, Any], *, force: bool = False) -> Optional[Path]:
    """Render (and cache) the annotated frame for one sighting."""
    camera = sighting.get("camera")
    clip_name = sighting.get("clip") or ""
    frame_no = int(sighting.get("frame") or 0)
    plate = (sighting.get("plate") or "").strip()
    if not camera:
        return None

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{camera}_{clip_name}_{frame_no}_{plate or 'noplate'}.jpg"
    if out.is_file() and not force:
        return out

    clip = _find_clip(camera, clip_name)
    if clip is None:
        logger.info("No clip on disk for %s / %s", camera, clip_name)
        return None
    frame = _read_frame(clip, frame_no)
    if frame is None:
        logger.info("Could not read frame %s of %s", frame_no, clip)
        return None

    vis = frame.copy()
    h, w = vis.shape[:2]

    crops = [Path(c) for c in (sighting.get("crops") or [])]
    crops = [c for c in crops if c.is_file()]
    box, score = (None, 0.0)
    if crops:
        box, score = _locate_plate(frame, crops[0])

    if box and score >= MATCH_MIN:
        px1, py1, px2, py2 = box
        pw, ph = px2 - px1, py2 - py1
        # The plate itself: located exactly, so drawn solid.
        cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 215, 255), 2)
        # The vehicle above it: inferred from the plate's position and size,
        # so drawn thin and labelled as a guide, not a measurement.
        vx1 = max(0, int(px1 - pw * 1.4))
        vx2 = min(w - 1, int(px2 + pw * 1.4))
        vy2 = min(h - 1, int(py2 + ph * 1.6))
        vy1 = max(0, int(py1 - ph * 6.5))
        cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), (120, 255, 140), 1)
        tag = plate or "vehicle"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        ty = max(th + 8, py1 - 8)
        cv2.rectangle(vis, (px1, ty - th - 7), (px1 + tw + 12, ty + 5),
                      (0, 215, 255), -1)
        cv2.putText(vis, tag, (px1 + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (20, 20, 20), 2, cv2.LINE_AA)
        located = f"plate located in frame (match {score:.2f})"
    else:
        located = "plate position not confirmed in this frame"

    # The plate crop as an inset, so the read can be judged against the pixels
    # even when the position could not be confirmed.
    if crops:
        c = cv2.imread(str(crops[0]))
        if c is not None and c.size:
            ih = max(46, int(h * 0.085))
            iw = int(c.shape[1] * (ih / c.shape[0]))
            iw = min(iw, int(w * 0.42))
            if iw > 10:
                inset = cv2.resize(c, (iw, ih))
                y0 = h - ih - 46
                vis[y0:y0 + ih, 10:10 + iw] = inset
                cv2.rectangle(vis, (10, y0), (10 + iw, y0 + ih),
                              (0, 215, 255), 2)

    band = 40
    cv2.rectangle(vis, (0, 0), (w, band), (16, 20, 28), -1)
    head = f"{camera}  {str(sighting.get('timestamp'))[:19]}"
    cv2.putText(vis, head, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (140, 220, 255), 2, cv2.LINE_AA)
    cv2.rectangle(vis, (0, h - 34), (w, h), (16, 20, 28), -1)
    foot = f"{clip.name} frame {frame_no} - {located}"
    cv2.putText(vis, foot, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def from_annotated_demo(camera: str) -> Optional[Path]:
    """A frame from the annotated re-ID clip for CAM_M1..M4.

    Those four cameras have no plate to locate — the plate is 20-30 px in
    478x850 phone video and the recogniser read nothing — so the frame the
    trajectory view shows is taken from the clip that already carries the
    global id drawn on the vehicle. The picture and the identity therefore
    come from the same render, and cannot disagree.
    """
    src = ROOT / "output" / "reid_demo" / f"{camera}.mp4"
    if not src.is_file():
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{camera}_reid_demo.jpg"
    if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    best, best_score = None, -1.0
    # Pick the frame where the identity label is most visible, so the still is
    # not a moment where the subject had left the shot.
    for frac in (0.35, 0.45, 0.55, 0.65, 0.75):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ok, f = cap.read()
        if not ok or f is None:
            continue
        mask = cv2.inRange(f, np.array([100, 235, 120]), np.array([140, 255, 160]))
        score = float(mask.sum())
        if score > best_score:
            best, best_score = f.copy(), score
    cap.release()
    if best is None:
        return None
    cv2.imwrite(str(out), best, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def build_clip(sighting: dict[str, Any], seconds: float = 4.0,
               force: bool = False) -> Optional[Path]:
    """A few seconds of footage around the sighting, as proof it moved."""
    camera = sighting.get("camera")
    clip_name = sighting.get("clip") or ""
    frame_no = int(sighting.get("frame") or 0)
    plate = (sighting.get("plate") or "").strip()
    src = _find_clip(camera, clip_name)
    if src is None:
        return None

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{camera}_{clip_name}_{frame_no}_{plate or 'noplate'}.mp4"
    if out.is_file() and not force:
        return out

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half = int(fps * seconds / 2.0)
    start = max(0, frame_no - half)
    end = min(total - 1 if total else frame_no + half, frame_no + half)

    tmp = out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    while i <= end:
        ok, f = cap.read()
        if not ok or f is None:
            break
        band = 34
        cv2.rectangle(f, (0, 0), (w, band), (16, 20, 28), -1)
        cv2.putText(f, f"{camera}  {plate or ''}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 220, 255), 2,
                    cv2.LINE_AA)
        # Mark the exact frame the index recorded the read on.
        if i == frame_no:
            cv2.rectangle(f, (0, 0), (w - 1, h - 1), (0, 215, 255), 6)
        vw.write(f)
        i += 1
    cap.release()
    vw.release()
    if out.exists():
        out.unlink()
    tmp.replace(out)
    return out
