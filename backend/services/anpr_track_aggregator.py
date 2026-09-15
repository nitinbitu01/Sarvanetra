"""
backend/services/anpr_track_aggregator_v2.py

Replaces the simple _vote_characters() function in old anpr_engine.py.

Changes:
  • Per-track TemporalPlateVoter object (not a global function)
  • Weight = OCR confidence × (quality_score / 100) — blurry frames get less vote
  • Character-position weighted majority voting
  • finalize() locks the plate and cleans up the track
  • min_reads=3 prevents single-frame false positives
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional


class TemporalPlateVoter:
    """
    Accumulates plate reads across a vehicle's track lifetime.
    Produces one consensus plate via confidence×quality weighted voting.
    """

    def __init__(self, min_reads: int = 3, min_confidence: float = 0.45):
        self.min_reads      = min_reads
        self.min_confidence = min_confidence
        self.tracks: dict[int, list[dict]] = defaultdict(list)
        self.finalized: dict[int, dict]    = {}

    # ── Accumulation ──────────────────────────────────────────────────────────

    def add_reading(
        self,
        track_id:     int,
        plate_text:   str,
        confidence:   float,
        quality_score: float,
    ) -> None:
        """Called each time CRNN produces a new read for this vehicle."""
        if track_id in self.finalized:
            return   # Already decided — ignore new reads

        weight = float(confidence) * (float(quality_score) / 100.0)
        self.tracks[track_id].append({
            "text":    plate_text,
            "conf":    float(confidence),
            "quality": float(quality_score),
            "weight":  weight,
        })

    # ── Consensus ─────────────────────────────────────────────────────────────

    def vote(self, track_id: int) -> tuple[Optional[str], float]:
        """
        Character-position weighted majority vote.
        Returns (plate_text, avg_confidence) or (None, 0.0).
        """
        readings = self.tracks.get(track_id, [])
        if len(readings) < self.min_reads:
            return None, 0.0

        # Use most common plate length
        texts       = [r["text"] for r in readings]
        lengths     = [len(t) for t in texts]
        target_len  = max(set(lengths), key=lengths.count)

        valid = [r for r in readings if len(r["text"]) == target_len]
        if not valid:
            return None, 0.0

        result          = []
        total_confidence = 0.0

        for pos in range(target_len):
            char_weights: dict[str, float] = defaultdict(float)
            for r in valid:
                char_weights[r["text"][pos]] += r["weight"]

            best_char    = max(char_weights, key=char_weights.get)
            best_w       = char_weights[best_char]
            total_w      = sum(char_weights.values())

            result.append(best_char)
            total_confidence += best_w / max(total_w, 1e-9)

        return "".join(result), total_confidence / target_len

    # ── Finalisation ──────────────────────────────────────────────────────────

    def finalize(self, track_id: int) -> Optional[dict]:
        """
        Lock the plate. Called when vehicle leaves the frame.
        Returns result dict or None if below threshold.
        """
        plate, conf = self.vote(track_id)
        if plate and conf >= self.min_confidence:
            result = {
                "plate":      plate,
                "confidence": round(conf, 3),
                "num_reads":  len(self.tracks.get(track_id, [])),
            }
            self.finalized[track_id] = result
            self.tracks.pop(track_id, None)
            return result

        self.tracks.pop(track_id, None)
        return None