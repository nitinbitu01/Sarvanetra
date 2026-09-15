"""
backend/anpr_worker.py — OCR worker thread for Sentinel Gujarat ANPR.

This module owns the OCR singleton lifecycle and the worker thread loop.

Why a separate thread?
  EasyOCR and PaddleOCR calls take 100–300ms each. Running them synchronously
  inside the main detection loop (which also drives person tracking and the
  WebSocket broadcast) would stall everything downstream of the detection
  thread — person tracks would stop flowing to the dashboard every time a
  vehicle is on screen. The worker thread fully decouples OCR latency from
  detection throughput.

Singleton pattern:
  Both readers are constructed ONCE in _load_easyocr_singleton() /
  _load_paddleocr_singleton() at worker thread startup. Model load time
  (which can be several seconds) is paid only once per process lifetime and
  is logged at INFO level with a wall-clock duration so it's visible in
  startup logs.

Backpressure:
  The job queue has a fixed maxsize (from config). If the detection loop
  produces OCR jobs faster than the worker can consume them (unlikely at 5
  FPS but possible if OCR is slow or many vehicles are in frame), the oldest
  pending job for a track is dropped and a WARNING is logged.

Thread safety:
  result_callback is called from the worker thread. The callback must be
  thread-safe. In practice it calls anpr_track_aggregator.on_ocr_result()
  which uses a threading.Lock internally.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend.anpr import read_plate

logger = logging.getLogger(__name__)


# ── Singleton loader functions ────────────────────────────────────────────────

def _load_easyocr_singleton() -> Any:
    """Load EasyOCR reader for English. Logs wall-clock load time.

    Returns:
        easyocr.Reader instance (pre-warmed with CPU inference).
    """
    import easyocr  # deferred — heavy import, only needed in worker thread

    logger.info("EasyOCR: loading model... (this may take a few seconds)")
    t0 = time.time()
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    elapsed = time.time() - t0
    logger.info("EasyOCR: model loaded in %.2fs", elapsed)
    return reader


def _load_paddleocr_singleton() -> Any:
    """Load PaddleOCR reader (English) if available.

    Returns:
        PaddleOCR instance ready for inference, or None if not installed.
    """
    try:
        from paddleocr import PaddleOCR  # deferred — optional import
        logger.info("PaddleOCR: loading model... (this may take a few seconds)")
        t0 = time.time()
        reader = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
        elapsed = time.time() - t0
        logger.info("PaddleOCR: model loaded in %.2fs", elapsed)
        return reader
    except Exception as exc:
        logger.info("PaddleOCR not installed or unavailable (%s) — using EasyOCR singleton.", exc)
        return None


def _load_crnn_singleton() -> tuple["Any", str]:
    """Load the domain-specific CRNN plate recogniser for beam-search decoding.

    Returns:
        Tuple of (model, device_string). Model is in eval mode with gradients
        disabled. Returns (None, "cpu") if loading fails for any reason.
    """
    try:
        import torch
        from backend.scripts.train_plate_recognizer import CHARS, CRNN

        ckpt_path = "models/plate_recognizer/finetuned.pt"
        if not Path(ckpt_path).exists():
            ckpt_path = "models/plate_recognizer/best.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("CRNN plate recogniser: loading %s on %s...", ckpt_path, device)
        t0 = time.time()

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = CRNN(len(CHARS) + 1).to(device)

        # Support both checkpoint formats: {"model": state_dict} or raw state_dict
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        # Disable gradients globally for inference
        for p in model.parameters():
            p.requires_grad_(False)

        elapsed = time.time() - t0
        logger.info("CRNN plate recogniser: loaded in %.2fs (%s)", elapsed, device)
        return model, device

    except Exception as exc:
        logger.warning(
            "CRNN plate recogniser not available (%s) — "
            "falling back to 2-engine ensemble (EasyOCR + PaddleOCR).",
            exc,
        )
        return None, "cpu"


# ── OCR Worker ────────────────────────────────────────────────────────────────

class AnprWorker:
    """Manages the OCR background thread.

    Args:
        cfg: Full config dict. Reads anpr.ocr_worker_queue_maxsize.
        result_callback: Called from the worker thread with each OCR result.
            Signature: callback(track_id: str, frame_number: int,
                                timestamp: float, result: dict)
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        result_callback: Callable[[str, int, float, dict], None],
    ) -> None:
        maxsize = cfg.get("anpr", {}).get("ocr_worker_queue_maxsize", 50)
        self._job_queue: queue.Queue[dict] = queue.Queue(maxsize=maxsize)
        self._result_callback = result_callback
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # Max times the worker will restart after an unexpected crash before giving up
    MAX_RESTART_ATTEMPTS = 3

    def start(self) -> None:
        """Start the OCR worker thread. Returns immediately."""
        if self._thread and self._thread.is_alive():
            logger.warning("AnprWorker.start() called but thread is already running.")
            return
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="anpr-ocr-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("ANPR OCR worker thread started.")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for it to drain."""
        logger.info("ANPR OCR worker: stopping...")
        self._stop_event.set()
        # Unblock the thread if it's waiting on an empty queue
        try:
            self._job_queue.put_nowait(None)  # sentinel value
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("ANPR OCR worker: stopped.")

    def enqueue_job(
        self,
        track_id: str,
        crop: "import numpy; numpy.ndarray",  # type: ignore[name-defined]
        frame_number: int,
        timestamp: float,
    ) -> None:
        """Enqueue an OCR job from the detection thread.

        This MUST NOT block. If the queue is full, applies backpressure:
        drops the oldest pending job for this track (to avoid stale results
        piling up) and logs a WARNING.

        Args:
            track_id: Vehicle track ID string (e.g. "V-42").
            crop: BGR numpy array of the vehicle bounding box region.
            frame_number: Current frame index.
            timestamp: Monotonic video timestamp (seconds).
        """
        job = {
            "track_id": track_id,
            "crop": crop.copy(),   # copy — detection thread may mutate frame
            "frame_number": frame_number,
            "timestamp": timestamp,
        }
        try:
            self._job_queue.put_nowait(job)
        except queue.Full:
            # Queue is at capacity — drop oldest job
            # (Simple drain-and-replace: discard the front item then insert ours)
            logger.warning(
                "ANPR OCR job queue full (%d items). Dropping oldest job — "
                "consider increasing anpr.read_interval_frames or "
                "anpr.ocr_worker_queue_maxsize.",
                self._job_queue.maxsize,
            )
            try:
                self._job_queue.get_nowait()   # discard oldest
            except queue.Empty:
                pass
            try:
                self._job_queue.put_nowait(job)
            except queue.Full:
                pass  # already dropped oldest; this is safe to silently skip

    @property
    def is_running(self) -> bool:
        """True if the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ── Worker loop (runs in background thread) ──────────────────────────────

    def _worker_loop(self) -> None:
        """Outer restart wrapper: re-launches the inner loop on unexpected crashes.

        Restarts up to MAX_RESTART_ATTEMPTS times with exponential backoff.
        Each restart reloads the OCR models from scratch (the previous state
        is gone if the worker crashed, so we can't safely reuse it).
        """
        for attempt in range(1, self.MAX_RESTART_ATTEMPTS + 1):
            if self._stop_event.is_set():
                break
            if attempt > 1:
                backoff = 2 ** (attempt - 1)  # 2s, 4s
                logger.warning(
                    "ANPR OCR worker restarting (attempt %d/%d) after %.0fs backoff...",
                    attempt, self.MAX_RESTART_ATTEMPTS, backoff,
                )
                time.sleep(backoff)

            try:
                self._worker_inner()
            except Exception as exc:
                logger.error(
                    "ANPR OCR worker crashed unexpectedly (attempt %d/%d): %s",
                    attempt, self.MAX_RESTART_ATTEMPTS, exc,
                )

            if self._stop_event.is_set():
                break  # clean shutdown — don't restart

        logger.error(
            "ANPR OCR worker: exhausted %d restart attempts — ANPR is disabled for this session.",
            self.MAX_RESTART_ATTEMPTS,
        )

    def _worker_inner(self) -> None:
        """Load OCR singletons ONCE then drain the job queue until stopped.

        This is the real worker body. _worker_loop() wraps it with restart logic.
        """
        # ── Load all models at thread startup — ONE TIME ONLY per restart ──
        try:
            easyocr_reader   = _load_easyocr_singleton()
            paddleocr_reader = _load_paddleocr_singleton()
        except Exception as exc:
            logger.error(
                "ANPR OCR worker failed to load models: %s — this restart attempt aborted.",
                exc,
            )
            return  # triggers outer restart loop

        # CRNN is optional — if it fails, the 2-engine ensemble still works
        crnn_model, crnn_device = _load_crnn_singleton()

        logger.info(
            "ANPR OCR worker: models loaded (easyocr=✓ paddle=%s crnn=%s), ready for jobs.",
            "✓" if paddleocr_reader else "✗",
            "✓" if crnn_model else "✗",
        )

        while not self._stop_event.is_set():
            try:
                job = self._job_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job is None:
                # Sentinel — stop requested
                break

            track_id     = job["track_id"]
            crop         = job["crop"]
            frame_number = job["frame_number"]
            timestamp    = job["timestamp"]

            logger.debug(
                "OCR worker: processing job track_id=%s frame=%d",
                track_id, frame_number,
            )

            try:
                result = read_plate(crop, easyocr_reader, paddleocr_reader,
                                    crnn_model=crnn_model, crnn_device=crnn_device)
            except Exception as exc:
                logger.warning(
                    "read_plate exception for track_id=%s frame=%d: %s — treating as no-result.",
                    track_id, frame_number, exc,
                )
                result = {"text": None, "confidence": 0.0, "source": "error", "agreement": False}

            try:
                self._result_callback(track_id, frame_number, timestamp, result)
            except Exception as exc:
                logger.error(
                    "result_callback raised for track_id=%s: %s", track_id, exc
                )

        logger.info("ANPR OCR worker inner loop exiting.")
