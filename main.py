"""
main.py — CLI entrypoint and detection pipeline for Sentinel Gujarat.

Day 2 change: The core detection loop is now split into two parts:
  1. `run_detection_pipeline(cfg, source, event_queue)` — importable function
     that runs the full loop in any thread. Called by backend/pipeline_bridge.py.
  2. CLI entrypoint at the bottom — unchanged from Day 1 for standalone testing.

When `event_queue` is None (CLI mode), events go to JSONL only.
When `event_queue` is a queue.Queue (server mode), events go to BOTH
JSONL and the queue, where the FastAPI bridge picks them up.

Loop-video & timestamp monotonicity:
  When camera.loop_video = true and the video ends, the capture is reopened
  from frame 0. A `loop_count` integer tracks how many times this has happened.
  Timestamps are computed as:
      timestamp = (loop_count * video_duration_seconds) + (frame_number / source_fps)
  This guarantees strictly increasing timestamps across loop boundaries.
  `frame_number` in the emitted event is the cumulative frame index
  (i.e., it does NOT reset to 0 on loop restart), so track IDs are not
  restarted either — BoT-SORT state carries across the loop boundary via
  the same model.track(persist=True) call.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

# MUST be set before cv2 is imported - OpenCV reads it when the FFmpeg
# backend initialises, so a later assignment is silently ignored.
#
# Without it FFmpeg uses a ~30s timeout. That is fine on a healthy link and
# useless on a degraded one: the government feeds were measured delivering
# 0.07-0.13 MB/s, where simply OPENING a stream took 175-280s. Every open
# aborted at 30s with "moov atom not found", and the pipeline then reported
# threads alive / frames 0 - healthy-looking and completely dead, which is
# the exact failure preflight_check.py exists to catch.
#
# 120s is a deliberate middle ground: long enough to ride out a slow server,
# short enough that a genuinely dead camera still fails rather than hanging
# a worker forever. Override with SENTINEL_FFMPEG_TIMEOUT_S if needed.
_ff_timeout_us = int(float(os.environ.get("SENTINEL_FFMPEG_TIMEOUT_S", "120")) * 1_000_000)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    f"timeout;{_ff_timeout_us}|stimeout;{_ff_timeout_us}|rw_timeout;{_ff_timeout_us}",
)

import cv2
import numpy as np
import yaml

from detector import OBJECT_CLASS_NAMES, PersonDetector, split_by_class
from event_emitter import EventEmitter
from tracker import ConfirmedTrack, TrackStateManager
from visualizer import draw_confirmed_tracks

# Backend imports (Day 3+) — optional path, graceful if backend not on sys.path
try:
    from backend.startup_checks import run_startup_checks as _run_startup_checks
    _STARTUP_CHECKS_AVAILABLE = True
except ImportError:
    _STARTUP_CHECKS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict[str, Any]:
    """Load and return config from a YAML file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Parsed config dict.

    Raises:
        SystemExit: If the file does not exist or cannot be parsed.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"[ERROR] Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        print(f"[ERROR] Failed to parse config.yaml: {exc}", file=sys.stderr)
        sys.exit(1)
    return cfg


def setup_logging(cfg: dict[str, Any]) -> None:
    """Configure the root logger from config."""
    level_str = cfg.get("logging", {}).get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ────────────────────────────────────────────────────────────────────────────
# Video I/O helpers with Universal Camera Adapter Support
# ────────────────────────────────────────────────────────────────────────────

def open_video_source(source: str, vendor: str = "unknown", camera_id: str = "CAM-01") -> cv2.VideoCapture:
    """Open a video file, webcam stream, or multi-vendor CCTV feed via CameraAdapterFactory.

    Args:
        source: File path string, RTSP/HLS/HTTP stream URL, or webcam index.
        vendor: Camera vendor ('hikvision', 'dahua', 'cpplus', 'onvif', 'rtsp', 'unknown').
        camera_id: Camera identifier string.

    Returns:
        Opened cv2.VideoCapture object.

    Raises:
        RuntimeError: If the source cannot be opened after all fallback adapters.
    """
    src: str | int = int(source) if str(source).isdigit() else source

    is_network = isinstance(src, str) and any(src.startswith(p) for p in ("rtsp://", "http://", "https://", "rtmp://"))
    if isinstance(src, str) and not is_network and not Path(src).exists():
        raise RuntimeError(f"Video file not found: {src}")

    # For network streams, resolve via CameraAdapterFactory fallback chain
    if is_network:
        try:
            from backend.services.camera_adapters.factory import resolve_stream_url
            cam_cfg = {"id": camera_id, "vendor": vendor, "stream_url": src, "rtsp_url": src}
            negotiated_url = resolve_stream_url(cam_cfg)
            if negotiated_url:
                logger.info(f"CameraAdapterFactory successfully negotiated stream: {negotiated_url}")
                src = negotiated_url
        except Exception as e:
            logger.warning(f"CameraAdapterFactory fallback bypass: {e}. Using direct stream URL.")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened() and not is_network:
        raise RuntimeError(f"Could not open video source: {source}")

    return cap


def create_video_writer(
    cfg: dict[str, Any],
    frame_width: int,
    frame_height: int,
    source_fps: float,
) -> cv2.VideoWriter | None:
    """Create an annotated video writer if --save-video was requested."""
    out_path = Path(cfg["output"]["annotated_video"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path), fourcc, source_fps or 25.0, (frame_width, frame_height)
    )
    if not writer.isOpened():
        logger.warning(
            "Failed to create VideoWriter at %s — annotated video will not be saved.", out_path
        )
        return None
    logger.info("Annotated video → %s", out_path)
    return writer


# ────────────────────────────────────────────────────────────────────────────
# Core detection pipeline — importable for server mode (Day 2+)
# ────────────────────────────────────────────────────────────────────────────

def run_detection_pipeline(
    cfg: dict[str, Any],
    source: str,
    event_queue: "queue.Queue[dict] | None" = None,
    stop_event: "threading.Event | None" = None,
    display: bool = False,
    save_video: bool = False,
) -> None:
    """Run the detection + tracking loop. Designed to be called from any thread.

    In CLI mode (event_queue=None): events go to JSONL only.
    In server mode (event_queue provided): events go to BOTH JSONL and the queue.

    Args:
        cfg: Parsed configuration dict.
        source: Path to video file or webcam index string (e.g. "0").
        event_queue: Thread-safe queue for inter-thread event delivery.
                     Pass None to disable (JSONL-only / CLI mode).
        stop_event: Optional threading.Event; when set, the loop exits cleanly.
                    Pass None for CLI mode (runs until EOF or user quit).
        display: Show live cv2.imshow window (CLI mode only).
        save_video: Write annotated output video.
    """
    camera_id: str = cfg.get("camera", {}).get("id", "CAM-01")
    # cfg["camera"]["_no_loop"] is set by the CLI's --no-loop flag. Kept in the
    # cfg dict rather than added as a parameter so every existing caller of
    # run_pipeline (backend server, tests) is unaffected.
    loop_video: bool = (cfg.get("camera", {}).get("loop_video", False)
                        and not cfg.get("camera", {}).get("_no_loop", False))

    # ── Pre-flight checks (Day 4 hardening) ────────────────────────────────
    if _STARTUP_CHECKS_AVAILABLE:
        ok = _run_startup_checks(cfg, source=source, fail_fast=False)
        if not ok:
            logger.error(
                "Startup pre-flight checks failed — see errors above. "
                "Pipeline will not start."
            )
            return
    else:
        # Minimal check when backend not available: ensure video source exists
        if source and not source.isdigit() and not Path(source).exists():
            logger.error("Video source not found: '%s'", source)
            return

    # ── Open video source ────────────────────────────────────────────────
    try:
        cap = open_video_source(source)
    except RuntimeError as exc:
        logger.error("Pipeline startup failed: %s", exc)
        return

    source_fps: float = cap.get(cv2.CAP_PROP_FPS)
    total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if source_fps <= 0:
        source_fps = 30.0
        logger.warning("Could not read source FPS — defaulting to %.1f", source_fps)

    # Video duration is used for monotonic timestamp offset across loops
    video_duration_seconds: float = (
        total_raw_frames / source_fps if total_raw_frames > 0 else 0.0
    )

    target_fps: int = cfg["processing"]["fps"]

    # Frame skipping only makes sense for a FILE, where frames are delivered
    # as fast as they can be decoded and you must throttle down to target_fps.
    # A live network stream is already rate-limited by the network itself, so
    # skipping on top of that multiplies the two limits together.
    #
    # Measured on the corp8 feeds: the stream reports ~25 fps in its header
    # but actually delivers 2.3-3.3 fps over the wire. With target_fps=5 the
    # old formula gave frame_interval=5, i.e. ~0.46 processed fps - roughly
    # 2.2 SECONDS between processed frames. BoT-SORT cannot hold an object
    # association across a gap that long in moving traffic, so tracks never
    # reached tracking.min_confirmation_frames (3) and the pipeline produced
    # zero tracks and zero alerts from perfectly good footage, with no error.
    #
    # For a live source, take every frame the network gives us.
    _is_live = isinstance(source, str) and source.startswith(
        ("rtsp://", "http://", "https://", "rtmp://")
    )
    if _is_live:
        frame_interval = 1
        logger.info(
            "Live network source - processing every delivered frame "
            "(header claims %.1f fps; the network, not frame_interval, is "
            "the real rate limit).", source_fps,
        )
    else:
        frame_interval = max(1, round(source_fps / target_fps))

    logger.info("Source: %s | FPS: %.2f | Frames: %d | Resolution: %dx%d",
                source, source_fps, total_raw_frames, frame_width, frame_height)
    logger.info("Processing every %d-th frame (target: %d fps) | camera_id: %s",
                frame_interval, target_fps, camera_id)

    # ── Initialise components ────────────────────────────────────────────
    try:
        detector = PersonDetector(cfg)
    except RuntimeError as exc:
        logger.error("Model load failed: %s", exc)
        cap.release()
        return

    # Person tracker (Day 1 — unchanged)
    person_state_manager = TrackStateManager(
        min_confirmation_frames=cfg["tracking"]["min_confirmation_frames"]
    )
    # Vehicle tracker (Day 4 — same class, separate instance)
    vehicle_conf_frames = cfg.get("vehicle_detection", {}).get("min_confirmation_frames", 3)
    vehicle_state_manager = TrackStateManager(min_confirmation_frames=vehicle_conf_frames)

    # Object tracker (Day 10 — bags/suitcases, feeds the abandoned-object detector)
    object_cfg = cfg.get("abandoned_object", {})
    object_conf_frames = object_cfg.get("min_confirmation_frames", 3)
    object_state_manager = TrackStateManager(min_confirmation_frames=object_conf_frames)
    warmup_sec: float = float(object_cfg.get("warmup_sec", 30.0))
    # track_id -> is_baseline, decided ONCE per track at first confirmation
    # and never revisited — see config.yaml's abandoned_object.warmup_sec
    # for what this heuristic does and does not claim to know.
    object_baseline: dict[int, bool] = {}

    # Class split config from config.yaml
    person_classes  = cfg["detection"].get("person_classes",  [0])
    vehicle_classes = cfg["detection"].get("vehicle_classes", [2, 3, 5, 7])
    object_classes  = cfg["detection"].get("object_classes",  [24, 26, 28])

    emitter = EventEmitter(cfg, event_queue=event_queue, camera_id=camera_id)

    # ── Day 13: evidence frame buffer ────────────────────────────────────
    # Imported defensively, exactly like the ANPR block below: this module
    # also runs in CLI mode where the backend package's dependencies may not
    # be importable, and evidence capture is a server-mode feature. Resolved
    # once here rather than per-frame.
    try:
        from backend.services.frame_buffer import buffer_frame as _buffer_frame
        logger.info("Evidence frame buffer enabled for camera %s.", camera_id)
    except Exception as exc:
        _buffer_frame = None
        logger.info("Evidence frame buffer unavailable (%s) — clips disabled.", exc)

    # ── ANPR worker thread (Day 4) ───────────────────────────────────────
    # GATED BY MEASURED PLATE YIELD.
    #   Plate-pixel yield was surveyed across all 29 cameras
    #   (backend/scripts/plate_camera_survey.py -> output/
    #   plate_camera_survey.json). Most cameras cannot resolve a plate at all:
    #   CAM_01's median vehicle yields a ~21px plate against a ~100px vendor
    #   floor, and the cameras are PTZ, so many spend long stretches aimed
    #   across traffic where no plate is ever presented.
    #
    #   Running OCR there is not merely wasteful, it is actively harmful:
    #   the worker saturated its 100-item queue and logged "Dropping oldest
    #   job" continuously, dragging the whole pipeline to roughly a tenth of
    #   real time, while contributing 400+ `anpr_uncertain` rows that say only
    #   "a vehicle passed and its plate was unreadable".
    #
    #   Cameras absent from the allow-list still detect, track and run the
    #   behaviour engine - they simply skip plate OCR. Set
    #   `anpr.cameras: []` (or omit it) to restore the previous
    #   run-everywhere behaviour.
    db_path: str | None = None
    anpr_aggregator = None
    anpr_worker_inst = None
    # Ids appear in both spellings across this project: config.camera.id
    # defaults to "CAM-01" (hyphen) while the clip corpus and the plate survey
    # use "CAM_01" (underscore). Comparing them raw would silently disable
    # ANPR on every camera - a gate that looks deliberate but is really a typo.
    def _norm_cam(x: str) -> str:
        return x.strip().upper().replace("-", "_")

    _anpr_cams = {_norm_cam(c) for c in
                  ((cfg.get("anpr", {}) or {}).get("cameras") or [])}
    if _anpr_cams and _norm_cam(camera_id) not in _anpr_cams:
        logger.info(
            "ANPR disabled for %s — not in anpr.cameras allow-list (measured "
            "plate yield too low). Detection and behaviour engine unaffected.",
            camera_id,
        )
        _anpr_enabled = False
    else:
        _anpr_enabled = True
    try:
        if not _anpr_enabled:
            raise RuntimeError("camera not in anpr.cameras allow-list")
        from pathlib import Path as _Path
        from backend.config import load_config as _load_backend_cfg
        from backend.anpr_worker import AnprWorker
        from backend.anpr_track_aggregator import AnprTrackAggregator

        _bcfg = _load_backend_cfg()
        db_path = str(_Path(_bcfg["database"]["path"]))

        anpr_worker_inst = AnprWorker(cfg, result_callback=lambda *a: None)  # placeholder cb
        anpr_aggregator = AnprTrackAggregator(
            cfg=cfg,
            db_path=db_path,
            ocr_worker=anpr_worker_inst,
            camera_id=camera_id,
        )
        # Now set real callback (aggregator exists at this point)
        anpr_worker_inst._result_callback = anpr_aggregator.on_ocr_result
        anpr_worker_inst.start()
        logger.info("ANPR worker started (Day 4)")
    except Exception as exc:
        if _anpr_enabled:
            logger.warning("ANPR worker could not start — vehicle OCR disabled: %s", exc)
        anpr_worker_inst = None
        anpr_aggregator = None

    # ── Wrong-way detection (opt-in, per camera) ─────────────────────────
    # Returns None unless this camera has a validated homography + lane
    # zones on disk. That is deliberate: without a metre-scale ground plane
    # the engine would emit confident but fabricated violations. Absent
    # calibration the whole feature stays silently inert.
    wrong_way = None
    try:
        from backend.services.traffic_registry import load_detector as _load_ww
        wrong_way = _load_ww(camera_id, cfg.get("ai_models", {}).get("traffic_intelligence"))
        if wrong_way is None:
            logger.debug(
                "Wrong-way detection inactive for %s (no calibration in "
                "config/traffic/%s.json).", camera_id, camera_id,
            )
    except Exception as exc:
        logger.warning("Wrong-way detector unavailable for %s: %s", camera_id, exc)

    # ── Behaviour engine (loitering + crowd) ────────────────────────────────
    # Loaded lazily and tolerated as absent: these detectors pull in the
    # backend package, SQLAlchemy and the behaviour state store, and a bare
    # `python main.py --source clip.mp4` on a machine without those must still
    # process video rather than refuse to start.
    behaviour_feed = None
    try:
        from backend.services.behaviour_bridge import feed_persons as _feed
        from backend.services.behaviour_bridge import get_bridge
        get_bridge()                       # start the loop thread up front
        behaviour_feed = _feed
        logger.info("Behaviour engine active for %s (loitering + crowd).",
                    camera_id)
    except Exception as exc:               # noqa: BLE001
        logger.warning("Behaviour engine unavailable for %s: %s", camera_id, exc)

    writer: cv2.VideoWriter | None = None
    if save_video:
        writer = create_video_writer(cfg, frame_width, frame_height, source_fps)

    # ── Motion gate ──────────────────────────────────────────────────────
    # These are FIXED cameras: when nothing in the scene moves, re-running
    # detection re-derives the same answer at full GPU cost. Measured on the
    # harvest runs, several cameras (CAM_24/27/28/29) rejected 80-96% of
    # sampled frames as near-duplicates, so a large share of the fleet's
    # compute is spent confirming that nothing changed.
    #
    # Safe here specifically because TrackStateManager.update() is a pure
    # function of the calls it receives (see tracker.py): confirmed tracks
    # are PRESERVED when absent, and an unconfirmed track's streak only
    # resets on a call where it is missing. Skipping the call entirely
    # freezes state rather than corrupting it - which is the correct
    # behaviour for a frozen scene. Timestamps derive from frame_number,
    # not from a call counter, so loitering durations stay accurate across
    # a gated stretch.
    _mcfg = cfg.get("motion_gate", {}) or {}
    motion_gate_on: bool = bool(_mcfg.get("enabled", False))
    motion_min_diff: float = float(_mcfg.get("min_diff", 2.0))
    # Heartbeat: never stay blind for more than this many consecutive
    # frames, whatever the pixels say. Without it a person standing
    # perfectly still - i.e. exactly the loitering case - could hold the
    # gate shut indefinitely.
    motion_max_gated: int = int(_mcfg.get("max_gated_frames", 15))
    _gate_ref: "np.ndarray | None" = None   # thumbnail of last PROCESSED frame
    _gated_run: int = 0                     # consecutive frames gated so far
    gated_count: int = 0                    # total, for the efficiency report
    last_confirmed: list[ConfirmedTrack] = []

    if motion_gate_on:
        logger.info(
            "Motion gate ON (min_diff=%.2f, heartbeat every %d frames) | camera_id: %s",
            motion_min_diff, motion_max_gated, camera_id,
        )

    # ── Frame loop ───────────────────────────────────────────────────────
    frame_number: int = 0      # Cumulative — never resets across loops
    loop_count: int = 0
    processed_count: int = 0

    try:
        while True:
            # Check stop signal (server mode)
            if stop_event is not None and stop_event.is_set():
                logger.info("Pipeline received stop signal — exiting.")
                break

            ret, frame = cap.read()

            if not ret:
                # End of stream
                if loop_video and total_raw_frames > 0:
                    loop_count += 1
                    logger.info(
                        "Video ended. Loop %d starting (cumulative frame: %d, "
                        "timestamp offset: %.2fs).",
                        loop_count, frame_number, loop_count * video_duration_seconds,
                    )
                    cap.release()
                    cap = open_video_source(source)
                    # BoT-SORT state persists through the loop because we're reusing
                    # the same detector model instance with persist=True.
                    continue
                else:
                    logger.info("End of video stream at cumulative frame %d.", frame_number)
                    break

            if frame is None or frame.size == 0:
                logger.warning("Frame %d: corrupted/empty — skipping.", frame_number)
                frame_number += 1
                continue

            # ── Frame sampling gate ──────────────────────────────────────
            # Default for frames that are sampled out, and for gated frames
            # that overwrite it below. Set before the sampling check so the
            # shared writer/display tail at the bottom of the loop always has
            # something valid to work with.
            annotated = frame
            run_detection = False

            if frame_number % frame_interval == 0:
                processed_count += 1

                # Monotonic timestamp: offset by how many complete loops have passed
                timestamp_offset = loop_count * video_duration_seconds
                # Current frame's position within the current loop pass
                loop_frame = frame_number - (loop_count * total_raw_frames if total_raw_frames > 0 else 0)
                frame_timestamp = timestamp_offset + (loop_frame / source_fps)

                # ── Day 13: feed the evidence frame buffer ───────────────
                # Every PROCESSED frame is buffered, before detection runs,
                # so a clip's pre-roll exists regardless of whether anything
                # was detected in those frames — an empty-looking approach is
                # exactly the footage an investigator wants leading up to an
                # event. Best-effort: buffering must never break detection.
                if _buffer_frame is not None:
                    try:
                        _buffer_frame(camera_id, frame, frame_timestamp, frame_number)
                    except Exception as exc:
                        logger.debug("Frame buffering failed at frame %d: %s",
                                     frame_number, exc)

                # ── Motion gate ──────────────────────────────────────────
                # Compared against the last PROCESSED frame, not the
                # immediately previous one: otherwise a scene drifting
                # slowly (dusk falling, a queue inching forward) would
                # never accumulate enough per-frame difference to reopen
                # the gate. Buffering above deliberately still runs on
                # gated frames - evidence pre-roll must not develop holes.
                run_detection = True
                if motion_gate_on:
                    gate_thumb = cv2.cvtColor(
                        cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY
                    ).astype(np.int16)
                    if _gate_ref is not None and _gated_run < motion_max_gated:
                        if float(np.mean(np.abs(gate_thumb - _gate_ref))) < motion_min_diff:
                            run_detection = False
                    if run_detection:
                        _gate_ref = gate_thumb
                        _gated_run = 0
                    else:
                        _gated_run += 1
                        gated_count += 1

                if not run_detection:
                    # Scene unchanged since the last real inference, so the
                    # previous boxes still describe it. Redrawing them keeps
                    # the annotated stream stable instead of flickering the
                    # overlay off on every gated frame.
                    if last_confirmed:
                        annotated = draw_confirmed_tracks(frame.copy(), last_confirmed)

            # Deliberately at loop level, not nested inside the sampling
            # check: a gated frame must still fall through to the shared
            # writer/display tail below, which owns the imshow and the
            # quit-key handling. An early `continue` here would freeze the
            # CLI preview window and stop 'q' responding for as long as the
            # scene stayed still.
            if run_detection:
                raw_tracks = detector.track_frame(frame, frame_number)  # ONE pass

                # ── Split into person, vehicle, and object streams ───────
                person_raw, vehicle_raw, object_raw = split_by_class(
                    raw_tracks, person_classes, vehicle_classes, object_classes
                )
                logger.debug(
                    "Frame %d: %d persons + %d vehicles + %d objects from single inference pass",
                    frame_number, len(person_raw), len(vehicle_raw), len(object_raw),
                )

                # ── Person pipeline (unchanged from Day 1) ───────────────
                confirmed: list[ConfirmedTrack] = person_state_manager.update(
                    person_raw, frame_number, source_fps
                )
                emitter.emit(confirmed, frame, override_timestamp=frame_timestamp)

                # ── Behaviour engine (loitering + crowd) ─────────────────
                # These detectors existed but were only reachable from the
                # async API server, which sees WebSocket events rather than
                # frames — so they had never run against video. Submitted
                # fire-and-forget: a behaviour tick must never stall frame
                # processing, since a dropped sample costs one point in a
                # 60s window while a stalled pipeline drops frames outright.
                if behaviour_feed is not None:
                    behaviour_feed(camera_id, None, confirmed,
                                   frame_number / max(source_fps, 1e-6))

                # ── Vehicle pipeline (Day 4) ─────────────────────────────
                confirmed_vehicles: list[ConfirmedTrack] = vehicle_state_manager.update(
                    vehicle_raw, frame_number, source_fps
                )

                if anpr_aggregator is not None and confirmed_vehicles:
                    for vt in confirmed_vehicles:
                        vid = "V-%d" % vt.track_id
                        # Extract bounding box crop for this vehicle
                        x1, y1, x2, y2 = (
                            int(vt.bbox[0]), int(vt.bbox[1]),
                            int(vt.bbox[2]), int(vt.bbox[3]),
                        )
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2].copy()
                            anpr_aggregator.on_confirmed_vehicle(
                                vid, crop, frame_number, frame_timestamp
                            )

                # ── Wrong-way driving (traffic intelligence) ─────────────
                # Only runs for cameras with a validated homography; see
                # traffic_registry.load_detector. Uncalibrated cameras get
                # no detector at all rather than one working in pixel space,
                # because speed and heading are meaningless without a
                # metre-scale ground plane.
                if wrong_way is not None and confirmed_vehicles:
                    try:
                        violations = wrong_way.process_detections(
                            [
                                {
                                    "track_id": vt.track_id,
                                    "bbox": vt.bbox,
                                    "vehicle_class": getattr(vt, "class_name", "car"),
                                }
                                for vt in confirmed_vehicles
                            ],
                            frame,
                            now=frame_timestamp,
                        )
                        for v in violations:
                            logger.warning(
                                "WRONG-WAY: camera=%s track=%d lane=%s %.1f km/h "
                                "angle=%.1f deg plate=%s",
                                v.camera_id, v.track_id, v.zone_id, v.speed_kmh,
                                v.discrepancy_angle_deg, v.plate.text or "UNREAD",
                            )
                    except Exception as exc:
                        # Never let traffic analytics take down detection.
                        logger.error("Wrong-way detector failed at frame %d: %s",
                                     frame_number, exc)

                # ── Object pipeline (Day 10 — abandoned-object detector) ─
                confirmed_objects: list[ConfirmedTrack] = object_state_manager.update(
                    object_raw, frame_number, source_fps
                )
                if confirmed_objects:
                    # is_baseline is decided ONCE per track_id, at its first
                    # confirmation, from that moment's video_time — not
                    # recomputed on later ticks (an object doesn't retroactively
                    # stop being a warm-up fixture just because more time has
                    # passed since pipeline start).
                    for obj in confirmed_objects:
                        if obj.track_id not in object_baseline:
                            object_baseline[obj.track_id] = frame_timestamp <= warmup_sec
                    # class_id per track_id, from this frame's raw detections —
                    # a track's class doesn't change frame to frame, so this
                    # frame's mapping is sufficient for every confirmed object
                    # in it (a confirmed object is by definition present in
                    # object_raw this frame too).
                    class_id_by_track = {r.track_id: r.class_id for r in object_raw}
                    emitter.emit_objects(
                        confirmed_objects,
                        object_classes={
                            o.track_id: OBJECT_CLASS_NAMES.get(
                                class_id_by_track.get(o.track_id, -1), "object"
                            )
                            for o in confirmed_objects
                        },
                        object_baseline=object_baseline,
                        override_timestamp=frame_timestamp,
                    )

                # Remembered so a gated frame can redraw the last known boxes
                # instead of showing an un-annotated frame.
                last_confirmed = confirmed
                annotated = draw_confirmed_tracks(frame.copy(), confirmed)

            if writer is not None:
                writer.write(annotated)

            if display:
                cv2.imshow("Sentinel Gujarat — Day 1", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("User requested quit.")
                    break

            frame_number += 1

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        emitter.close()
        # Stop ANPR worker cleanly
        if anpr_worker_inst is not None:
            anpr_worker_inst.stop(timeout=3.0)

    logger.info(
        "Pipeline done. Cumulative frames: %d | Processed: %d | Loops: %d",
        frame_number, processed_count, loop_count,
    )
    if motion_gate_on and processed_count:
        logger.info(
            "Motion gate: %d/%d sampled frames skipped (%.1f%%) — inference "
            "ran on %d | camera_id: %s",
            gated_count, processed_count, gated_count / processed_count * 100,
            processed_count - gated_count, camera_id,
        )


# ────────────────────────────────────────────────────────────────────────────
# CLI entrypoint (unchanged from Day 1)
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sentinel Gujarat — Person Detection + Tracking (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --source video.mp4 --config config.yaml
  python main.py --source video.mp4 --config config.yaml --save-video --display
  python main.py --source 0 --config config.yaml --display   # webcam
        """,
    )
    parser.add_argument("--source", required=True,
                        help="Path to input video file, or webcam index (e.g. '0').")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: config.yaml in CWD).")
    parser.add_argument("--save-video", action="store_true",
                        help="Write annotated output to output/annotated.mp4.")
    parser.add_argument("--camera", default=None,
                        help="Override camera.id from config. Zones, "
                             "calibration and the ANPR allow-list are all "
                             "keyed on this, so replaying CAM_04 footage while "
                             "config still says CAM-01 silently applies the "
                             "wrong camera's zones — or none at all.")
    parser.add_argument("--no-loop", action="store_true",
                        help="Process the source once and stop, overriding "
                             "camera.loop_video. Use this for MEASUREMENT: "
                             "with looping on, a stationary person is "
                             "re-detected on every pass under a fresh track "
                             "id, so a single real event was counted 15 times "
                             "and the alert total said nothing about how much "
                             "the system actually found.")
    parser.add_argument("--display", action="store_true",
                        help="Show live cv2.imshow window (press Q or ESC to quit).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.no_loop:
        cfg.setdefault("camera", {})["_no_loop"] = True
    if args.camera:
        cfg.setdefault("camera", {})["id"] = args.camera
    setup_logging(cfg)
    run_detection_pipeline(
        cfg=cfg,
        source=args.source,
        event_queue=None,       # CLI mode: JSONL only
        stop_event=None,
        display=args.display,
        save_video=args.save_video,
    )
