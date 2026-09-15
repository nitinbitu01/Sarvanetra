"""
backend/anpr_track_aggregator.py — Per-track OCR reading scheduler, aggregator,
finalization engine, and alert writer for Sentinel Gujarat ANPR.

Responsibilities:
  1. Per-track read scheduling — track how many processed frames each vehicle
     track has appeared in; enqueue OCR via AnprWorker every
     `anpr.read_interval_frames` track-frames (NOT global frame number modulo).
  2. Collect OCR results delivered by the worker thread callback.
  3. Finalization — decide when to stop and what the final plate reading is.
  4. Alert writing — call db.insert_alert() based on watchlist match + confidence.

Thread safety:
  - Detection thread calls update_tracks() and schedule_ocr_if_due().
  - Worker thread calls on_ocr_result() via the callback.
  - Both paths are protected by self._lock (threading.Lock).

Finalization triggers (whichever happens first):
  - Track accumulates `min_readings_before_finalize` valid (format_valid=True) readings.
  - Track reaches `max_readings_per_track` total OCR attempts (cap).
  - Track expires (no update for `track_expiry_seconds`) — caller signals this
    by calling expire_track(track_id).

Zero-OCR-attempt tracks:
  If a track expires with zero OCR readings attempted (vehicle too fast / glimpse
  too short), NO alert or DB row is created. Only a DEBUG log is emitted.
  An alert with no plate information is noise, not signal.

Finalization logic:
  - Among readings where format_valid=True, majority vote on normalized text.
  - Tie → highest-confidence valid reading wins.
  - Zero valid readings → status = 'anpr_uncertain'.

Alert decisions:
  - Watchlist exact match (active=1) → alert_type='watchlist_vehicle', severity='critical'
  - No match, conf >= uncertain_threshold, format_valid → log only, no alert
  - Conf < uncertain_threshold OR format_valid=False (but >=1 attempt) → 'anpr_uncertain'
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

from backend import db
from backend.plate_utils import is_valid_plate, normalize_plate_text, plates_are_equal
from backend.scripts.indian_plate_grammar import decode_plate

if TYPE_CHECKING:
    from backend.anpr_worker import AnprWorker

logger = logging.getLogger(__name__)


@dataclass
class _Reading:
    """One OCR reading for a vehicle track."""
    text: str | None
    confidence: float
    source: str
    agreement: bool
    format_valid: bool
    frame_number: int
    timestamp: float
    sharpness: float = 1.0


@dataclass
class _TrackAnprState:
    """Per-vehicle-track ANPR state."""
    track_id: str                     # e.g. "V-42"
    camera_id: str
    db_path: str
    cfg: dict

    frames_since_last_read: int = 0   # Per-track counter — NOT global frame modulo
    total_jobs_enqueued: int = 0
    readings: list[_Reading] = field(default_factory=list)
    finalized: bool = False
    first_seen_time: float = field(default_factory=time.monotonic)
    last_seen_time: float = field(default_factory=time.monotonic)

    @property
    def max_readings(self) -> int:
        return self.cfg.get("anpr", {}).get("max_readings_per_track", 5)

    @property
    def min_before_finalize(self) -> int:
        return self.cfg.get("anpr", {}).get("min_readings_before_finalize", 3)

    @property
    def read_interval(self) -> int:
        return self.cfg.get("anpr", {}).get("read_interval_frames", 5)

    @property
    def uncertain_threshold(self) -> float:
        return self.cfg.get("anpr", {}).get("uncertain_confidence_threshold", 0.70)

    @property
    def plate_regex(self) -> str:
        return self.cfg.get("anpr", {}).get("plate_regex", r"^GJ\d{2}[A-Z]{1,2}\d{4}$")

    @property
    def valid_readings(self) -> list[_Reading]:
        return [r for r in self.readings if r.format_valid and r.text is not None]

    @property
    def should_finalize(self) -> bool:
        """Return True when any finalization condition is satisfied."""
        if self.finalized:
            return False
        # Hit the max cap
        if self.total_jobs_enqueued >= self.max_readings:
            return True
        # Enough valid readings accumulated
        if len(self.valid_readings) >= self.min_before_finalize:
            return True
        return False

    def due_for_ocr(self) -> bool:
        """Return True if this track is due for another OCR job."""
        if self.finalized:
            return False
        if self.total_jobs_enqueued >= self.max_readings:
            return False
        self.frames_since_last_read += 1
        if self.frames_since_last_read >= self.read_interval:
            self.frames_since_last_read = 0
            return True
        return False


class AnprTrackAggregator:
    """Manages OCR scheduling and finalization across all active vehicle tracks.

    Args:
        cfg: Full config dict.
        db_path: SQLite database path (from config).
        ocr_worker: AnprWorker instance for submitting jobs.
        camera_id: Camera identifier for alert metadata.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        db_path: str,
        ocr_worker: "AnprWorker",
        camera_id: str,
    ) -> None:
        self._cfg = cfg
        self._db_path = db_path
        self._worker = ocr_worker
        self._camera_id = camera_id
        self._lock = threading.Lock()
        self._tracks: dict[str, _TrackAnprState] = {}

    # ── Called from detection thread ──────────────────────────────────────────

    def on_confirmed_vehicle(
        self,
        track_id: str,
        crop: np.ndarray,
        frame_number: int,
        timestamp: float,
    ) -> None:
        """Called each time a confirmed vehicle track is seen in a processed frame.

        Updates per-track frame counter and enqueues an OCR job if due.
        Never blocks — OCR is always dispatched to the worker thread.

        Args:
            track_id: Vehicle track ID string (e.g. "V-42").
            crop: BGR bounding-box crop of the vehicle.
            frame_number: Current frame index.
            timestamp: Monotonic video timestamp.
        """
        with self._lock:
            if track_id not in self._tracks:
                self._tracks[track_id] = _TrackAnprState(
                    track_id=track_id,
                    camera_id=self._camera_id,
                    db_path=self._db_path,
                    cfg=self._cfg,
                )
                logger.debug("ANPR aggregator: new vehicle track %s", track_id)

            state = self._tracks[track_id]
            if state.finalized:
                return

            state.last_seen_time = time.monotonic()

            if state.due_for_ocr():
                state.total_jobs_enqueued += 1
                logger.debug(
                    "ANPR: enqueuing OCR job for track %s (job #%d)",
                    track_id, state.total_jobs_enqueued,
                )
                # Release lock before calling worker — worker never calls back
                # synchronously, so this is safe
        # Enqueue outside the lock so we don't hold it during queue operations
        self._worker.enqueue_job(track_id, crop, frame_number, timestamp)

        with self._lock:
            state = self._tracks.get(track_id)
            if state and state.should_finalize:
                self._finalize(state)

    def expire_track(self, track_id: str) -> None:
        """Signal that a vehicle track has expired (no update for expiry window).

        Triggers finalization if readings have accumulated; otherwise logs
        at DEBUG level only (zero-attempt tracks produce no alert).

        Args:
            track_id: Vehicle track ID string.
        """
        with self._lock:
            state = self._tracks.get(track_id)
            if state is None or state.finalized:
                return
            logger.debug(
                "ANPR: track %s expired — total jobs=%d readings=%d",
                track_id, state.total_jobs_enqueued, len(state.readings),
            )
            if state.total_jobs_enqueued == 0:
                # Zero OCR attempts — no alert, no DB row
                logger.debug(
                    "ANPR: track %s had 0 OCR attempts (vehicle too brief) — no alert created.",
                    track_id,
                )
                state.finalized = True
                return
            self._finalize(state)

    # ── Called from OCR worker thread ─────────────────────────────────────────

    def on_ocr_result(
        self,
        track_id: str,
        frame_number: int,
        timestamp: float,
        result: dict[str, Any],
    ) -> None:
        """Receive an OCR result from the worker thread and store it.

        This is called from the worker thread — protected by self._lock.

        Args:
            track_id: Vehicle track ID string.
            frame_number: Frame the OCR job was for.
            timestamp: Timestamp of that frame.
            result: Dict from read_plate() — {text, confidence, source, agreement}.
        """
        with self._lock:
            state = self._tracks.get(track_id)
            if state is None or state.finalized:
                return

            text = result.get("text")
            conf = result.get("confidence", 0.0)
            sharpness = float(result.get("sharpness", 1.0))
            format_valid = is_valid_plate(text, state.plate_regex)

            reading = _Reading(
                text=text,
                confidence=conf,
                source=result.get("source", "unknown"),
                agreement=result.get("agreement", False),
                format_valid=format_valid,
                frame_number=frame_number,
                timestamp=timestamp,
                sharpness=sharpness,
            )
            state.readings.append(reading)

            logger.info(
                "ANPR reading: track=%s text=%r conf=%.3f sharpness=%.1f format_valid=%s source=%s",
                track_id, text, conf, sharpness, format_valid, result.get("source"),
            )

            if state.should_finalize:
                self._finalize(state)

    # ── Finalization ──────────────────────────────────────────────────────────

    def _finalize(self, state: _TrackAnprState) -> None:
        """Decide final plate reading via Weighted Multi-Frame Consensus and write alert."""
        state.finalized = True
        valid = state.valid_readings

        logger.info(
            "ANPR finalizing track=%s | total_readings=%d valid=%d",
            state.track_id, len(state.readings), len(valid),
        )

        if not valid:
            # No valid readings — write anpr_uncertain if any attempts were made
            self._write_uncertain_alert(
                state,
                final_text=None,
                final_conf=0.0,
                reason="zero_valid_readings",
            )
            return

        # ── Weighted Multi-Frame Sharpness & Confidence Consensus ─────────────
        score_by_text = Counter()
        conf_by_text = defaultdict(list)

        for r in valid:
            s = max(r.sharpness, 1.0)
            # Weight = conf * agreement_multiplier * log(1 + Laplacian_sharpness)
            agree_mult = 1.35 if r.agreement else 1.0
            w = r.confidence * agree_mult * float(np.log1p(s))
            score_by_text[r.text] += w
            conf_by_text[r.text].append(r.confidence)

        best_raw_text, best_score = score_by_text.most_common(1)[0]
        avg_conf = float(np.mean(conf_by_text[best_raw_text]))

        # Grammar check & syntax repair (O->0, I->1, B->8 where appropriate)
        grammar_decoded = decode_plate(best_raw_text)
        final_text = grammar_decoded["plate"] or best_raw_text

        # Multi-frame consensus precision boost
        num_agreeing_frames = len(conf_by_text[best_raw_text])
        if num_agreeing_frames >= 2 and grammar_decoded.get("valid"):
            final_conf = min(avg_conf * 1.06, 0.99)
        else:
            final_conf = avg_conf

        agree_count = sum(1 for r in state.readings if r.agreement)
        agreement_rate = agree_count / len(state.readings) if state.readings else 0.0

        logger.info(
            "ANPR final (Multi-Frame Consensus): track=%s plate=%r conf=%.3f (score=%.2f, frames=%d/%d, agree_rate=%.2f)",
            state.track_id, final_text, final_conf, best_score, num_agreeing_frames, len(state.readings), agreement_rate,
        )

        # ── Watchlist lookup ──────────────────────────────────────────────────
        wl_rows = db.get_active_watchlist_vehicles(state.db_path)
        watchlist_match = None
        for row in wl_rows:
            if plates_are_equal(final_text, row["plate_number"]):
                watchlist_match = row
                break

        if watchlist_match:
            self._write_watchlist_alert(state, final_text, final_conf,
                                        watchlist_match, len(state.readings), agreement_rate)
            return

        # ── No watchlist match ────────────────────────────────────────────────
        if is_valid_plate(final_text, state.plate_regex) and final_conf >= state.uncertain_threshold:
            logger.info(
                "ANPR: track=%s plate=%r conf=%.3f — ordinary vehicle, no alert.",
                state.track_id, final_text, final_conf,
            )
            return

        # Low confidence or invalid format — write uncertain alert
        self._write_uncertain_alert(state, final_text, final_conf,
                                     reason="low_confidence_or_invalid_format")

    def _write_watchlist_alert(
        self,
        state: _TrackAnprState,
        final_text: str,
        final_conf: float,
        wl_row: Any,
        readings_count: int,
        agreement_rate: float,
    ) -> None:
        """Insert a watchlist_vehicle alert via db.insert_alert()."""
        alert_id = db.insert_alert(
            db_path=state.db_path,
            alert_type="watchlist_vehicle",
            camera_id=state.camera_id,
            track_id=int(state.track_id.lstrip("V-")) if state.track_id.startswith("V-") else None,
            score=7.0,
            severity="critical",
            metadata={
                "plate_text": final_text,
                "confidence": round(final_conf, 4),
                "reason": wl_row["reason"],
                "readings_count": readings_count,
                "agreement_rate": round(agreement_rate, 4),
            },
        )
        logger.info(
            "ALERT watchlist_vehicle: id=%d track=%s plate=%r reason=%s conf=%.3f",
            alert_id, state.track_id, final_text, wl_row["reason"], final_conf,
        )

    def _write_uncertain_alert(
        self,
        state: _TrackAnprState,
        final_text: str | None,
        final_conf: float,
        reason: str,
    ) -> None:
        """Insert an anpr_uncertain alert via db.insert_alert()."""
        alert_id = db.insert_alert(
            db_path=state.db_path,
            alert_type="anpr_uncertain",
            camera_id=state.camera_id,
            track_id=int(state.track_id.lstrip("V-")) if state.track_id.startswith("V-") else None,
            score=final_conf,
            severity=None,
            metadata={
                "plate_text": final_text,
                "confidence": round(final_conf, 4),
                "reason": reason,
                "readings_count": len(state.readings),
                "valid_readings_count": len(state.valid_readings),
            },
        )
        logger.info(
            "ALERT anpr_uncertain: id=%d track=%s plate=%r reason=%s",
            alert_id, state.track_id, final_text, reason,
        )
