"""
backend/services/anpr_track_aggregator_v2.py

Confidence × quality weighted temporal plate voter.
Replaces simple _vote_characters() from old anpr_engine.py.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional, Tuple, Dict, List
import cv2
import numpy as np


def split_stacked_plate(crop: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Splits a 2:1 double-line/stacked plate crop (e.g. Indian two-wheelers)
    horizontally into Line 1 (state + RTO) and Line 2 (series + number).

    Uses horizontal edge energy projection to detect the gutter between lines.
    Returns (top_strip, bottom_strip) or (None, None) if crop is too small.
    """
    if crop is None or crop.shape[0] < 16 or crop.shape[1] < 20:
        return None, None

    h, w = crop.shape[:2]
    gray = (
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if crop.ndim == 3 and crop.shape[2] >= 3
        else crop.reshape(crop.shape[0], crop.shape[1])
    )

    # Compute horizontal projection of vertical gradients to find horizontal line gap
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    abs_sobel = np.abs(sobel_y)
    row_energy = np.mean(abs_sobel, axis=1)

    y_min, y_max = int(h * 0.35), int(h * 0.65)
    if y_max > y_min:
        seam_y = y_min + int(np.argmin(row_energy[y_min:y_max]))
    else:
        seam_y = int(h * 0.5)

    top_strip = crop[:seam_y, :]
    bot_strip = crop[seam_y:, :]
    return top_strip, bot_strip


class TrackletPlateBuffer:
    """
    Accumulates plate crops per vehicle track ID in a ring buffer.
    Determines layout (single-line vs double-line) and provides
    crops for multi-frame sub-pixel fusion.
    """

    def __init__(self, max_crops: int = 8):
        self.max_crops = max_crops
        self.buffers: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.layouts: Dict[int, str] = {}

    def add_crop(self, track_id: int, crop: np.ndarray) -> None:
        if crop is None or crop.size == 0:
            return
        buf = self.buffers[track_id]
        buf.append(crop)
        if len(buf) > self.max_crops:
            # Keep highest sharpness crops
            def _sharpness(c):
                g = (
                    cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
                    if c.ndim == 3 and c.shape[2] >= 3
                    else c
                )
                return cv2.Laplacian(g, cv2.CV_64F).var()

            buf.sort(key=_sharpness, reverse=True)
            del buf[self.max_crops:]

        if track_id not in self.layouts and len(buf) >= 1:
            h, w = crop.shape[:2]
            ar = w / max(h, 1)
            if ar >= 3.0:
                self.layouts[track_id] = "single_line"
            elif ar <= 2.5:
                self.layouts[track_id] = "double_line"
            else:
                self.layouts[track_id] = "unknown"

    def get_crops(self, track_id: int) -> List[np.ndarray]:
        return self.buffers.get(track_id, [])

    def get_layout(self, track_id: int) -> str:
        return self.layouts.get(track_id, "unknown")

    def remove(self, track_id: int) -> None:
        self.buffers.pop(track_id, None)
        self.layouts.pop(track_id, None)


class TemporalPlateVoter:
    """
    Accumulates plate reads across a vehicle track lifetime.
    Uses confidence × quality weighted character-position voting.
    """

    def __init__(self, min_reads: int = 3, min_confidence: float = 0.45):
        self.min_reads       = min_reads
        self.min_confidence  = min_confidence
        self.tracks: dict[int, list[dict]] = defaultdict(list)
        self.finalized: dict[int, dict]    = {}

    def add_reading(
        self,
        track_id:      int,
        plate_text:    str,
        confidence:    float,
        quality_score: float,
    ) -> None:
        if track_id in self.finalized:
            return
        weight = float(confidence) * (float(quality_score) / 100.0)
        self.tracks[track_id].append({
            "text":    plate_text,
            "conf":    float(confidence),
            "quality": float(quality_score),
            "weight":  weight,
        })

    def vote(self, track_id: int) -> tuple[Optional[str], float]:
        readings = self.tracks.get(track_id, [])
        if len(readings) < self.min_reads:
            return None, 0.0

        texts      = [r["text"] for r in readings]
        lengths    = [len(t) for t in texts]
        target_len = max(set(lengths), key=lengths.count)
        valid      = [r for r in readings if len(r["text"]) == target_len]

        if not valid:
            return None, 0.0

        result           = []
        total_confidence = 0.0

        for pos in range(target_len):
            char_weights: dict[str, float] = defaultdict(float)
            for r in valid:
                char_weights[r["text"][pos]] += r["weight"]

            best_char = max(char_weights, key=char_weights.get)
            best_w    = char_weights[best_char]
            total_w   = sum(char_weights.values())

            result.append(best_char)
            total_confidence += best_w / max(total_w, 1e-9)

        return "".join(result), total_confidence / target_len

    def finalize(self, track_id: int) -> Optional[dict]:
        plate, conf = self.vote(track_id)
        self.tracks.pop(track_id, None)
        if plate and conf >= self.min_confidence:
            result = {
                "plate":      plate,
                "confidence": round(conf, 3),
                "num_reads":  len(self.tracks.get(track_id, [])),
            }
            self.finalized[track_id] = result
            return result
        return None
