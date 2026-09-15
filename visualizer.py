"""
visualizer.py — Annotated frame rendering for Sentinel Gujarat Day 1.

Draws bounding boxes, track IDs, and confidence scores on frames.
Only operates on ConfirmedTrack objects — unconfirmed tracks are never drawn.

This module is intentionally thin — all visual state (colors, fonts) is
contained here so future UI changes don't touch pipeline logic.
"""

from __future__ import annotations

import cv2
import numpy as np

from tracker import ConfirmedTrack

# ── Visual constants ─────────────────────────────────────────────────────────
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2
_LABEL_PAD = 4          # pixels of padding around label text
_ALPHA = 0.35           # fill alpha for label background


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR colour for a given track ID.

    Uses a fixed set of visually distinct colours cycled by track_id.
    This keeps the same person's box the same colour across frames.
    """
    palette = [
        (0, 200, 255),   # amber
        (0, 255, 128),   # spring green
        (255, 100, 0),   # blue-ish
        (0, 100, 255),   # orange
        (200, 0, 255),   # magenta
        (0, 255, 200),   # cyan-green
        (128, 0, 255),   # violet
        (255, 200, 0),   # sky blue
    ]
    return palette[track_id % len(palette)]


def draw_confirmed_tracks(
    frame: np.ndarray,
    tracks: list[ConfirmedTrack],
) -> np.ndarray:
    """Draw bounding boxes and labels for confirmed tracks on the frame.

    Args:
        frame: BGR image (will be modified in-place).
        tracks: Confirmed tracks for this frame.

    Returns:
        The annotated frame.
    """
    for track in tracks:
        color = _track_color(track.track_id)
        x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)

        # ── Bounding box ──────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, _BOX_THICKNESS)

        # ── Label: "ID:5  93%" ─────────────────────────────────────────────
        label = f"ID:{track.track_id}  {track.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)

        lx1 = x1
        ly1 = max(y1 - th - baseline - _LABEL_PAD * 2, 0)
        lx2 = x1 + tw + _LABEL_PAD * 2
        ly2 = max(y1, th + baseline + _LABEL_PAD * 2)

        # Semi-transparent filled background for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, cv2.FILLED)
        cv2.addWeighted(overlay, _ALPHA, frame, 1 - _ALPHA, 0, frame)

        # Label text in white over the coloured background
        cv2.putText(
            frame,
            label,
            (lx1 + _LABEL_PAD, ly2 - _LABEL_PAD - baseline),
            _FONT,
            _FONT_SCALE,
            (255, 255, 255),
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    # ── HUD: frame info in top-right corner ──────────────────────────────
    if tracks:
        hud = f"Tracked: {len(tracks)}"
        hx, hy = frame.shape[1] - 10, 24
        (hw, hh), _ = cv2.getTextSize(hud, _FONT, 0.5, 1)
        cv2.putText(frame, hud, (hx - hw, hy), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
