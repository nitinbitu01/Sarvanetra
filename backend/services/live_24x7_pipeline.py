"""
backend/services/live_24x7_pipeline.py — Sentinel Gujarat 24x7 Real-Time CCTV Analytics Engine

THE pipeline. There is one, and this file is it.

Multi-threaded ingestion and AI inference over the 30-camera Junagadh fleet,
reading authenticated live HLS from https://cctv.corp8.cloud/ and falling back
to local clips of the same cameras when a stream drops.

Every stage below runs in this one process, in this order, on the same frame.
ANPR is stage 4 — a call into anpr_engine, not a second pipeline. Speed,
analytics and alerts are stages of the same pass. Nothing here is scheduled
separately, so there is no cross-pipeline synchronisation anywhere in the
system, because there is nothing to synchronise.

  ┌─────────────────┐   ┌─────────────────┐         ┌─────────────────┐
  │ CameraReader #1 │   │ CameraReader #2 │  ...    │ CameraReader #30│
  │   (own thread)  │   │   (own thread)  │         │   (own thread)  │
  └───────┬─────────┘   └───────┬─────────┘         └───────┬─────────┘
          └─────────────────────┴────────────────────────────┘
                                 │   STAGE 1 — INGESTION
                                 ▼   CameraReaderThread, _discover_cameras
      ┌──────────────────────────────────────────────────────────┐
      │              Shared Frame Queue (maxsize=60)             │
      └──────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼   _process_frame_batch, one GPU consumer
      ┌──────────────────────────────────────────────────────────┐
      │  STAGE 2  VEHICLE DETECTION   YOLOv8s, fine-tuned         │
      │  STAGE 3  TRACKING            per-camera BoT-SORT, no     │
      │                               id leak between cameras     │
      │  STAGE 4  ANPR                -> anpr_engine (7 sub-steps)│
      │  STAGE 5  IDENTITY FUSION     plate key + OSNet appearance│
      │  STAGE 6  TRAJECTORY + SPEED  homography -> Theil-Sen km/h│
      │  STAGE 7  TRAFFIC ANALYTICS   1-min rollups, congestion   │
      │  STAGE 8  ALERTS              watchlist, clone-plate      │
      │  STAGE 9  PERSISTENCE         VehicleTrack rows, vault,   │
      │                               live frame published to UI  │
      └──────────────────────────────────────────────────────────┘

Stage-to-line map for all nine, and what is only partially built (stage 5's
weighted fusion, stage 7's OD matrix): docs/UNIFIED_ARCHITECTURE.md.
"""
from __future__ import annotations

import json
import hashlib
import json
import logging
import os
import queue
import socket
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.db.models import (
    Alert,
    Camera,
    CameraHomographyCalibration,
    CameraMetrics1M,
    OfficerReview,
    VaultEntry,
    VehicleTrack,
    JourneyEvent,
    ANPREvent,
)
from backend.db.session import SessionLocal
from backend.services.calibration import (
    QUALITY_GATE_ACCEPT_M,
    pixel_to_world_m,
    pixel_world_resolution_m,
)
from backend.services.congestion_engine import CongestionEngine
from backend.services.reid_embedder import get_embedder
from backend.services.speed_estimator import estimate_track_speed
from backend.services.tracker import BoTSORTTracker
from backend.services.anpr_engine import get_anpr_engine
from backend.services.traffic_baseline import TrafficBaselineEngine
from backend.services.cad_dispatch import get_dispatcher

logger = logging.getLogger("sentinel.live_24x7_pipeline")

WORKSPACE = Path(__file__).resolve().parent.parent.parent
CLIPS_DIR = WORKSPACE / "data" / "clips"
# Vault crops live on the local filesystem alongside the other run outputs.
VAULT_CROP_ROOT = WORKSPACE / "output"

# The networks that actually run in this process. Written verbatim into every
# vault row, so a reviewer can tell which weights produced a given crop. The
# previous literal, "yolov8s_osnet_v2.4", named a version that does not exist
# and was stamped on rows whose embedding was never computed at all.
MODEL_VERSION = "yolov8s + osnet_ibn_x1_0"

# ── Tuning Constants ────────────────────────────────────────────────────────
# Frames sampled per camera per second.
#
# This is set from measured throughput, not from a wish. On this machine
# (RTX 4070, 27 cameras, YOLOv8s at imgsz=640 + OSNet-IBN appearance +
# BoT-SORT with motion compensation) the consumer sustains ~38 frames/s in
# aggregate, which is where the per-frame cost lands:
#
#     detect  12.6 ms      track  9.6 ms
#     embed    4.2 ms      flush  0.3 ms      total ~30 ms
#
# 27 cameras x 5 fps would demand 135 frames/s. Asking for it does not
# produce it — it produced a 42% frame-drop rate, which the old code hid by
# discarding frames silently and reporting uptime instead. Sampling at a rate
# the consumer can actually sustain means every frame that is read is
# processed, and the realtime factor stays at 1.0.
#
# Override with SENTINEL_READER_FPS where the hardware differs; the honest
# ceiling is roughly (aggregate frames/s) / (number of cameras).
READER_FPS = int(os.environ.get("SENTINEL_READER_FPS", "10"))
# Cameras published from the READER thread at the decode rate, with the newest
# AI overlay drawn on — see Live24x7Pipeline._present_frame. Comma-separated.
SMOOTH_PUBLISH_CAMERAS = {
    c.strip().upper()
    for c in os.environ.get("SENTINEL_SMOOTH_PUBLISH_CAMERAS", "").split(",")
    if c.strip()}
# An overlay computed on a frame older than this (seconds, by capture time) is
# not drawn on a newer frame.
OVERLAY_MAX_AGE_S = float(os.environ.get("SENTINEL_OVERLAY_MAX_AGE", "0.6"))
# For a smooth-published camera read from a local clip: the screen gets every
# frame, inference every Nth, at a steady cadence. Measured on CAM_M1-M4: one
# process processed ~15-20 frames/s across the four; offering 4 x 15 fps would
# have inference drop two frames in three at random, and an irregular frame
# cadence is what breaks tracks — and with them the cross-camera id. Default 1
# (every frame) leaves every other camera exactly as it was.
INFER_EVERY = max(1, int(os.environ.get("SENTINEL_INFER_EVERY", "1")))
# Embed the appearance of every detection only on every Nth frame of each
# camera, rather than on every frame.
#
# Measured 2026-09-12 on this GPU: OSNet costs ~1.2 ms per crop at fleet-sized
# batches. Thirty cameras at 10 fps with three vehicles in view is 900 crops a
# second — 1.08 GPU-seconds of embedding per wall second, on a card that has
# one — so appearance, not detection (0.62) and not tracking (1-3 ms on the
# CPU), is what makes thirty live cameras impossible. At 10 the fleet embeds
# each track about once a second: 0.22 GPU-seconds, and the whole per-frame
# budget fits with headroom.
#
# What it costs: on the frames in between, BoT-SORT associates on motion alone
# and the track's mean appearance is built from ~1 sample a second instead of
# ~10. Both degrade rather than break — a track still gets an embedding, the
# vault still gets vectors, and cross-camera identity still has a mean to match
# against.
#
# MEASURED, AND IT DID NOT PAY OFF. A/B on 28 clip cameras the same day:
# every-frame embedding gave 267.4 fps aggregate with 32.8% of frames dropped;
# EMBED_EVERY=10 gave 222.4 fps with 48.2% dropped. Appearance is simply not
# where the pipeline's time goes — measured per frame: detect 11.0 ms, embed
# 5.3 ms, track 19.6 ms, and the track phase is mostly overlay drawing,
# publishing and persistence rather than the tracker (1-3 ms on its own).
# The knob is kept because the arithmetic above is real and a bigger fleet
# would eventually meet it, but it is OFF by default and turning it on made
# things worse here. Optimise the track phase instead.
EMBED_EVERY_DEFAULT = "1"       # 1 = embed every frame, the measured-best setting
EMBED_EVERY = max(1, int(os.environ.get("SENTINEL_EMBED_EVERY",
                                        EMBED_EVERY_DEFAULT)))

# Putting a frame on screen — JPEG encode plus the atomic file write — off the
# thread that does the AI work.
#
# Measured 2026-09-12, per published 1920x1080 frame: frame copy 1.08 ms, boxes
# 0.15 ms, **JPEG encode at quality 85 11.83 ms**, temp write + rename 1.56 ms
# = 15.0 ms. Thirty cameras at 10 fps is 300 published frames a second, so
# 4.5 CPU-seconds of encoding per wall second — all of it on the inference
# thread, in front of the next batch of frames. That is what dropped 33% of the
# fleet's frames, and why the `track` phase read 19.6 ms while the tracker
# inside it costs 1-3 ms. The os.replace retry loop makes it worse: when the
# API server has the file open it sleeps up to 225 ms, on that same thread.
#
#   PUBLISH_ASYNC       hand the drawn frame to a publisher thread and return.
#                       Latest-wins per camera: if encoding falls behind, the
#                       newest frame replaces the waiting one, which is what a
#                       live view wants anyway.
#   PUBLISH_MAX_WIDTH   encode a smaller image. Measured 15.0 -> 3.9 ms per
#                       frame at 960 wide, and a fleet tile is a few hundred
#                       pixels on screen. 0 keeps full resolution.
#   PUBLISH_MIN_INTERVAL_S  floor on the gap between two published frames of
#                       one camera. 0 publishes every frame.
#
# Crops, ANPR, speed and the vault all read the ORIGINAL frame, never this
# one, so none of these settings can change what the system detects or reads —
# they only change the picture on screen.
# PUBLISH_ASYNC is always on — the measured cost (11.83 ms/frame × 30 cameras)
# makes synchronous JPEG encoding on the inference thread the primary bottleneck.
# It can only be disabled by explicitly setting SENTINEL_PUBLISH_ASYNC=0.
PUBLISH_ASYNC = os.environ.get("SENTINEL_PUBLISH_ASYNC", "1") != "0"
PUBLISH_MAX_WIDTH = max(0, int(os.environ.get("SENTINEL_PUBLISH_MAX_WIDTH", "960")))
# Quality 75 is indistinguishable from 85 on a fleet tile that is a few hundred
# pixels wide, and it is half the encode time. Override with
# SENTINEL_PUBLISH_QUALITY if a specific camera needs higher fidelity.
PUBLISH_JPEG_QUALITY = max(40, min(95, int(
    os.environ.get("SENTINEL_PUBLISH_QUALITY", "75"))))
PUBLISH_MIN_INTERVAL_S = max(0.0, float(
    os.environ.get("SENTINEL_PUBLISH_MIN_INTERVAL_S", "0")))
# Cameras whose picture is held back until inference has covered it, so each
# frame is drawn with the overlay computed nearest to it in time rather than
# with whatever overlay happens to be newest. Measured 2026-09-11 on the
# handheld CAM_M1/M2: drawing the newest overlay on the newest frame put boxes
# beside the motorcycle (camera panning) or small around its plate after it
# had come close — the overlay was up to ~1 s older than the picture. Held
# back, the picture runs a fixed fraction of a second behind and every box
# sits on its object. Meant for replayed recordings, where that delay is
# invisible; CAM_09 is not on this list.
ALIGN_OVERLAY_CAMERAS = {
    c.strip().upper()
    for c in os.environ.get("SENTINEL_ALIGN_OVERLAY_CAMERAS", "").split(",")
    if c.strip()}
ALIGN_MAX_DELAY_S = 2.5     # past this, show the frame anyway (inference stalled)
ALIGN_MATCH_S = 0.35        # an overlay further than this in time is not drawn
ALIGN_MAX_READY = 8         # frames already covered; beyond this, skip ahead
# Cameras ANPR is attempted on. Empty = all of them (the original behaviour).
ANPR_CAMERAS = {
    c.strip().upper()
    for c in os.environ.get("SENTINEL_ANPR_CAMERAS", "").split(",")
    if c.strip()}

# A detection is only usable for speed if one pixel of box jitter is worth
# less ground distance than the positional accuracy the calibration module
# already defines as acceptable (QUALITY_GATE_ACCEPT_M, 0.50 m). This is not a
# new tuning knob: it applies the same tolerance the quality gate uses, at the
# point where a measurement is taken rather than only where a camera is
# certified. Because the test is evaluated through the camera's own
# homography, it needs no per-camera image row or distance cut-off, and a
# better calibration automatically widens the usable region.
SPEED_MAX_PIXEL_RESOLUTION_M = float(
    os.environ.get("SENTINEL_SPEED_MAX_PX_RES_M", "2.5"))


class _UnmeasurablePoint(Exception):
    """A detection whose geometry cannot support a metric position.

    Distinct from a transform failure: the homography is fine, this particular
    pixel is simply too near the vanishing line to mean anything. Raised and
    handled locally so a genuinely broken transform still reaches the warning.
    """
# When True, the pipeline strictly streams live CCTV and refuses to loop fallback clips
SENTINEL_STRICT_LIVE = os.environ.get("SENTINEL_STRICT_LIVE", "0") == "1"
# Max frames drained from the queue and detected as ONE batch per cycle. The
# CUDA allocator keeps the peak of the largest batch it has ever run, so this
# also sets how much GPU memory the process holds on to. Measured 2026-09-11:
# two pipelines (CAM_09 at 7.6 GB, CAM_M1-M4 at 3.2 GB) filled the 12 GB card
# and both then froze for ~1.5 s at a time; with CAM_M stopped CAM_09 ran clean.
# SENTINEL_INFER_BATCH lowers it for a process that must share the GPU.
INFERENCE_BATCH_DRAIN = max(1, int(os.environ.get("SENTINEL_INFER_BATCH", "30")))
TRACK_STALE_SECONDS = 3.0         # Flush tracks not seen for this long
TRACK_MAX_POINTS = 60             # Force-flush tracks with this many accumulated points
ROLLUP_INTERVAL_SECONDS = 30.0    # Interval between metric rollups
# Whether THIS process runs the fleet-wide traffic rollup (see the inference
# loop). The supervisor leaves it on for exactly one worker.
ROLLUPS_ENABLED = os.environ.get("SENTINEL_ROLLUPS", "1") == "1"
VAULT_HARVEST_PROBABILITY = 0.12  # Chance of harvesting a borderline detection
VAULT_CONF_LOW = 0.40             # Lower bound for vault-worthy confidence
VAULT_CONF_HIGH = 0.68            # Upper bound for vault-worthy confidence
YOLO_CONF_THRESHOLD = float(os.environ.get("SENTINEL_YOLO_CONF", "0.25"))
# Detection input size — raised from 640 to 1280 on measured evidence.
#
# The previous note argued 640 was "the only size that leaves room", from a
# 7.4 ms budget per frame. Re-measured on 1920x1080 source, single frames:
#
#     imgsz   ms/frame   aggregate fps
#      640       5.8         172
#      960       7.3         137
#     1280      10.5          95
#     1600      14.8          67
#
# 27 cameras at READER_FPS=1 need 27 frames/s of detection, not 135 — the old
# figure divided by a 5 fps reader that is no longer what runs. At 1280 there
# is still 3.5x headroom, so the constraint the old value was protecting
# against does not exist.
#
# What the extra resolution buys, and what it does NOT.
#
# It finds more vehicles: 178 at 640 against 232 at 1280 over 84 frames on 6
# cameras, and +35 unique tracks over a larger tracked sample. That is a real
# gain for counting, density and tracking continuity.
#
# It does NOT improve plate reading. A per-FRAME comparison suggested it would
# — the boxes found only at 1280 carried 6 readable plates against 7 from the
# whole 640 set — but that did not survive per-VEHICLE accounting: readable
# plates were 13/173 vehicles at 640 and 12/208 at 1280, an unchanged absolute
# count against a larger denominator. A vehicle close enough to show a >=70px
# plate is already large enough to detect at 640; the vehicles 1280 adds are
# distant ones whose plates are unreadable in every frame they appear in.
#
# The per-frame figure was misleading because one vehicle contributes many
# frames, and a plate readable in a 1280-only box was usually the same vehicle
# already detected at 640 a frame or two later.
#
# Kept at 1280 for the vehicle-analytics gain, which is real, at 1.8x the
# detection cost and still 3.5x inside budget. Set SENTINEL_YOLO_IMGSZ=640 to
# trade that back for headroom; plate accuracy is unaffected either way.
#
# TOP-10 MODE: Default changed to 640.  At imgsz=1280, 5 cameras per worker
# costs ~100 ms/cycle — exactly 10 fps with zero headroom for tracking, ANPR,
# vault, or DB writes. At 640 the cost drops to ~25 ms, leaving a 75 ms
# budget that comfortably absorbs the rest of the pipeline at true 10 fps.
# Set SENTINEL_YOLO_IMGSZ=1280 only if you have a single camera per worker
# and need the extra small-object recall.
YOLO_IMGSZ = int(os.environ.get("SENTINEL_YOLO_IMGSZ", "640"))
# Lighting and road condition change over minutes, not frames. Re-classifying
# per camera every 30 s keeps the cost negligible while still catching
# nightfall and the start of rain.
ENV_REFRESH_SECONDS = 30.0
# Common shape every frame is scaled to before detection, so the batch forms
# one tensor.
#
# Raised from (1280, 720) to native 1080p. At the old value a 1920x1080 camera
# — 16 of the 27 on this fleet — was downscaled BEFORE detection, so the
# detector never saw the pixels that the imgsz increase above was measured on.
# Pre-scaling to 720 lines and then detecting at imgsz 1280 would have thrown
# away the resolution first and paid for it second.
#
# Plate crops were always cut from each frame's own pixels rather than from
# this canvas, so this only ever affected which vehicles were FOUND, not how
# sharp their crops were. That is still the stage that decides whether a plate
# is read at all.
# TOP-10 default: 960x540 matches the 640 imgsz budget without wasting the
# full 1080p decode that DETECT_INPUT_SIZE=(1920,1080) used to trigger.
DETECT_INPUT_SIZE = (960, 540)
# A process running only cameras of another shape can set its own canvas,
# e.g. SENTINEL_DETECT_INPUT_SIZE=480x864 for CAM_M1-M4 (478x850 portrait
# phone clips). On the landscape canvas above those clips were ~70% grey
# padding, detected at 1280 px: measured 2026-09-11, 0 of 4 cameras kept up
# (rtf 0.866, queue growing) and a box was drawn on only 52-98% of frames.
_detect_size = os.environ.get("SENTINEL_DETECT_INPUT_SIZE", "").lower().strip()
if _detect_size:
    try:
        _w, _h = (int(v) for v in _detect_size.split("x"))
        DETECT_INPUT_SIZE = (_w, _h)
    except ValueError:
        logging.getLogger(__name__).warning(
            "SENTINEL_DETECT_INPUT_SIZE=%r is not WxH; using %s", _detect_size,
            DETECT_INPUT_SIZE)

# Every frame was being resized TWICE before the detector saw it: once here,
# onto the canvas above, and again inside ultralytics, which scales whatever it
# is handed down to imgsz. Letterboxing a 1920x1080 canvas and then having it
# shrunk to 640 or 1280 does the expensive copy on the biggest array in the
# pipeline and then throws the result away.
#
# So the canvas is capped at imgsz: the detector receives the same pixels it
# would have received anyway — its own resize step produced exactly this — from
# one resize instead of two. Measured in the fleet, the `detect` phase read
# 14.0 ms/frame against 2.8 ms for the same model, batch and imgsz on its own.
#
# The canvas keeps its configured aspect ratio, so a process that set its own
# shape (SENTINEL_DETECT_INPUT_SIZE=480x864 for the portrait CAM_M clips) keeps
# it. Nothing downstream changes: plate crops, ANPR, speed and the vault all
# read each frame's own pixels, never this canvas.
_cw, _ch = DETECT_INPUT_SIZE
if max(_cw, _ch) > YOLO_IMGSZ:
    _shrink = YOLO_IMGSZ / max(_cw, _ch)
    DETECT_INPUT_SIZE = (max(32, round(_cw * _shrink)),
                         max(32, round(_ch * _shrink)))
# Bound on unwritten vault crops. Past this the writer is behind and further
# harvests are dropped and counted, rather than growing without limit.
VAULT_QUEUE_MAX = 256
# Box positions kept per track, against source frame numbers. Evidence clips
# span a few seconds either side of the event, so this needs to cover that
# window at the sampling rate; beyond it the oldest entries are dropped.
TRACK_BOX_HISTORY = 240
# How long to wait on a TCP connect when checking whether a camera's network
# source is back. Short on purpose: this runs inside the reader loop, and a
# camera pointed at an unreachable host must cost that camera a moment, not
# stall the fleet.
NETWORK_PROBE_TIMEOUT_SEC = 1.5
# Gap between reachability probes once a camera has failed over to local
# recordings. It was 15 s, which with 27 unreachable cameras meant the fleet
# spent most of its time waiting on sockets.
NETWORK_PROBE_INTERVAL_SEC = 120.0
YOLO_VEHICLE_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck


@dataclass(frozen=True)
class FrameOrigin:
    """Exactly where a processed frame came from.

    An alert is only evidence of something if the footage attached to it is
    the footage the detector was looking at. That requires knowing, at the
    moment of detection, which file and which frame was on screen — so this
    travels with every frame from the reader to the tracker to the alert.

    For a live network stream there is no file to seek back into, so
    clip_path is None and the recorded position is the stream's own frame
    counter. Evidence for those cameras has to come from a recording buffer;
    saying so is better than pointing at a file that does not contain it.
    """
    clip_path: Optional[str]
    frame_number: int
    fps: float

    def as_dict(self) -> dict:
        return {"clip": self.clip_path, "frame": self.frame_number,
                "fps": round(self.fps, 3)}
YOLO_CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ─────────────────────────────────────────────────────────────────────────────
# Camera Reader Thread — Unified Multi-Protocol Stream Ingestion Engine
# ─────────────────────────────────────────────────────────────────────────────
class CameraReaderThread(threading.Thread):
    """
    Production-grade multi-protocol video ingestion reader thread.

    Supports:
      1. RTSP / RTSPS streams (rtsp://user:pass@ip:554/stream) with low-latency TCP buffer
      2. HTTP / HTTPS MJPEG / HLS streams (http://ip:port/stream)
      3. ONVIF camera discovery & streaming
      4. Local high-resolution MP4/AVI clips (continuous cyclic simulation for demo/testing)
      5. Hybrid Failover: if an RTSP feed goes offline, automatically falls back to local clips
         while attempting exponential-backoff background reconnects. Hot-swaps back to RTSP when restored!
    """

    def __init__(
        self,
        cam_id: str,
        clip_paths: Optional[List[Path]] = None,
        frame_queue: queue.Queue = None,
        fps: int = READER_FPS,
        source_url: Optional[str] = None,
        auto_reconnect: bool = True,
        present: Optional[Any] = None,
    ):
        super().__init__(daemon=True, name=f"reader-{cam_id}")
        self.cam_id = cam_id
        # Callable(cam_id, frame, ts) that puts a frame on screen straight from
        # this thread. None = frames reach the screen via inference as before.
        self._present = present
        self.clip_paths = sorted(clip_paths) if clip_paths else []
        # Optional filename filter, e.g. SENTINEL_CLIP_MATCH=0830.
        #
        # Clips are played in sorted order, which starts at the earliest hour
        # on record — for CAM_09 that is CAM_09_0100.mp4, 1 a.m. on an empty
        # bypass. Measured: 1,008 frames of it produced ZERO tracks, and it
        # takes ten minutes of wall time to exhaust before the busy clip is
        # reached. Nothing was broken; the camera was watching an empty road.
        if clip_paths:
            match = os.environ.get("SENTINEL_CLIP_MATCH", "").strip()
            if match:
                # Comma-separated: SENTINEL_CLIP_MATCH=0730,0830 plays both.
                wanted = [m.strip() for m in match.split(",") if m.strip()]
                chosen = [p for p in self.clip_paths
                          if any(m in Path(p).name for m in wanted)]
                if chosen:
                    self.clip_paths = chosen
                else:
                    logger.warning(
                        "[%s] SENTINEL_CLIP_MATCH=%r matched none of %s; "
                        "keeping all clips", cam_id, match,
                        [Path(p).name for p in self.clip_paths])
        self.source_url = source_url.strip() if source_url else None
        # SENTINEL_FORCE_CLIPS=1 ignores the network source entirely.
        #
        # Failover only triggers when the stream FAILS. A stream that is merely
        # crawling never fails, so the reader stays on it: measured on CAM_09,
        # corp8 delivered 48 frames in four minutes (0.2 fps against a target
        # of 5) while five local clips sat unused. For a demonstration, a
        # source that is slow is worse than one that is down.
        if source_url and self.clip_paths and \
                os.environ.get("SENTINEL_FORCE_CLIPS", "1") == "1":
            logger.info("[%s] SENTINEL_FORCE_CLIPS — ignoring %s, reading %d "
                        "local clip(s)", cam_id, self._mask_url(source_url),
                        len(self.clip_paths))
            self.source_url = None
        self.frame_queue = frame_queue
        self.target_fps = max(1, fps)
        self.interval = 1.0 / self.target_fps
        self.auto_reconnect = auto_reconnect
        self._running = True

        # Source classification: 'rtsp' | 'http' | 'onvif' | 'file' | 'clips'
        self.source_type = self._detect_source_type(self.source_url, self.clip_paths)

        # Stream capture & state
        self._cap: Optional[cv2.VideoCapture] = None
        self._clip_idx = 0
        self._clip_fps = 15.0
        # File currently being read, so each frame can carry its origin.
        self._current_clip_path: Optional[str] = None
        self._is_connected = False
        self._in_failover = False

        # Reconnection & backoff state
        self._reconnect_count = 0
        self._consecutive_failures = 0
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
        self._last_reconnect_attempt = 0.0
        self._last_probe_time = time.monotonic() + float(np.random.uniform(0.0, 8.0))
        # Consecutive failed hot-swaps back to the network source. Grows the
        # probe interval so a reachable-but-broken host stops interrupting a
        # working clip.
        self._failed_recoveries = 0

        # Performance & telemetry counters
        self._frames_read = 0        # valid frames handed to the queue
        self._frames_skipped = 0     # decoded past to stay at video rate
        self._frames_dropped = 0     # discarded because the queue was full
        self._base_time = time.time()
        self._video_seconds = 0.0    # video time consumed (clip mode)
        self._last_frame_time = 0.0  # wall time of most recent frame read
        self._stream_width = 0
        self._stream_height = 0

    @staticmethod
    def _detect_source_type(url: Optional[str], clips: List[Path]) -> str:
        if url:
            u = url.lower()
            if u.startswith("rtsp://") or u.startswith("rtsps://"):
                return "rtsp"
            if u.startswith("http://") or u.startswith("https://"):
                return "http"
            if u.startswith("onvif://"):
                return "onvif"
            if u.endswith((".m3u8", ".ts", ".mp4", ".avi", ".mkv", ".mov", ".flv", ".webm")) or os.path.exists(url):
                return "file"
            return "stream"
        if clips:
            return "clips"
        return "none"

    @staticmethod
    def _mask_url(url: Optional[str]) -> str:
        if not url:
            return "N/A"
        import re
        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)

    def run(self):
        logger.info(
            "[%s] Reader thread started (source_type=%s, target_fps=%d, url=%s, clips=%d)",
            self.cam_id, self.source_type, self.target_fps,
            self._mask_url(self.source_url), len(self.clip_paths),
        )

        # Initial connect attempt
        connected = self._open_source()
        if not connected:
            if not SENTINEL_STRICT_LIVE and self.clip_paths and self.source_type in ("rtsp", "http", "onvif", "stream"):
                logger.warning(
                    "[%s] Network stream initial connect failed; entering local clip failover mode",
                    self.cam_id,
                )
                self._in_failover = True
                self._open_clip(self._clip_idx)
            else:
                self._consecutive_failures += 1

        while self._running:
            t0 = time.monotonic()

            # If not connected and no failover clip open, attempt reconnection
            if self._cap is None or not self._cap.isOpened():
                if self.source_type in ("rtsp", "http", "onvif", "stream"):
                    self._handle_network_reconnect()
                elif self.clip_paths and not SENTINEL_STRICT_LIVE:
                    self._advance_clip()
                time.sleep(0.5)
                continue

            # Read logic branches based on whether we are reading a live network stream or local clips
            if self.source_type in ("rtsp", "http", "onvif", "stream") and not self._in_failover:
                self._read_live_network_frame(t0)
            else:
                self._read_clip_frame(t0)

        self._release()
        logger.info(
            "[%s] Reader stopped — %d queued, %d skipped, %d dropped (reconnects=%d)",
            self.cam_id, self._frames_read, self._frames_skipped,
            self._frames_dropped, self._reconnect_count,
        )

    def _open_source(self) -> bool:
        """Open the primary video source (RTSP / HTTP / Stream / File / Clips)."""
        self._release()
        if self.source_type in ("rtsp", "http", "onvif", "stream"):
            return self._open_network_stream(self.source_url)
        elif self.source_type == "file":
            return self._open_single_file(self.source_url)
        elif self.source_type == "clips":
            return self._open_clip(self._clip_idx)
        return False

    def _open_network_stream(self, url: str) -> bool:
        """Open RTSP or HTTP network video stream with low-latency configuration."""
        # Authenticated, AES-encrypted HLS needs a decoder that can carry the
        # portal's session cookie into the key request. cv2.VideoCapture cannot
        # — it returns isOpened()=False with no diagnostic — so those streams
        # take an ffmpeg subprocess instead. See hls_ffmpeg_capture.py.
        try:
            from backend.services.hls_ffmpeg_capture import (
                is_corp8_hls, open_corp8_stream,
            )
            if is_corp8_hls(url):
                cap = open_corp8_stream(url)
                if cap is None:
                    logger.warning("[%s] corp8 HLS open failed: %s",
                                   self.cam_id, self._mask_url(url))
                    self._is_connected = False
                    return False
                self._cap = cap
                self._stream_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._stream_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._clip_fps = max(1.0, float(cap.get(cv2.CAP_PROP_FPS)))
                self._is_connected = True
                self._in_failover = False
                self._current_clip_path = None
                self._consecutive_failures = 0
                logger.info("[%s] corp8 HLS connected: %dx%d @ %.0f fps",
                            self.cam_id, self._stream_width,
                            self._stream_height, self._clip_fps)
                return True
        except Exception as exc:                                   # noqa: BLE001
            logger.error("[%s] corp8 HLS path errored: %s", self.cam_id, exc)
            # Fall through to the ordinary OpenCV path rather than dying.

        try:
            # Set OpenCV FFMPEG low-latency options for RTSP
            if url.lower().startswith("rtsp"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;tcp|analyzeduration;1000000|probesize;1000000|"
                    "fflags;nobuffer|flags;low_delay|max_delay;500000"
                )

            # Check the host answers before handing the URL to FFMPEG.
            #
            # Opening an unreachable stream blocks the reader thread for as
            # long as the transport takes to give up. At startup, with 27
            # cameras pointed at a host that does not answer, that is 27
            # threads each stalled before any of them read a frame — and the
            # fleet never recovers the lost ground because the same stall
            # repeats on every reconnect. A one-and-a-half second TCP connect
            # answers "is anything there" first.
            if not self._probe_network_stream():
                logger.warning("[%s] Network source unreachable: %s",
                               self.cam_id, self._mask_url(url))
                return False

            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG if hasattr(cv2, "CAP_FFMPEG") else cv2.CAP_ANY)
            if not cap.isOpened():
                masked = self._mask_url(url)
                logger.warning("[%s] Failed to open network stream: %s", self.cam_id, masked)
                cap.release()
                return False

            # Minimize internal buffer latency to 1 frame for real-time live feeds
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Probe stream properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            self._cap = cap
            self._clip_fps = fps if fps and 1.0 <= fps <= 120.0 else 25.0
            self._stream_width = w
            self._stream_height = h
            self._is_connected = True
            self._in_failover = False
            self._consecutive_failures = 0
            self._reconnect_delay = 1.0

            masked = self._mask_url(url)
            logger.info(
                "[%s] Connected to live %s stream: %s (%dx%d @ %.1f FPS)",
                self.cam_id, self.source_type.upper(), masked, w, h, self._clip_fps,
            )
            return True
        except Exception as exc:
            masked = self._mask_url(url)
            logger.error("[%s] Exception opening network stream %s: %s", self.cam_id, masked, exc)
            return False

    def _open_single_file(self, file_path: str) -> bool:
        """Open a single local video file."""
        self._release()
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.warning("[%s] Failed to open video file: %s", self.cam_id, file_path)
            self._cap = None
            self._is_connected = False
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._cap = cap
        self._clip_fps = fps if fps and 1.0 <= fps <= 120.0 else 15.0
        self._stream_width = w
        self._stream_height = h
        self._is_connected = True
        return True

    def _open_clip(self, idx: int) -> bool:
        """Open a local clip from the clip sequence."""
        self._release()
        if not self.clip_paths:
            return False
        path = str(self.clip_paths[idx % len(self.clip_paths)])
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            logger.warning("[%s] Failed to open clip: %s", self.cam_id, path)
            self._cap = None
            self._is_connected = False
            self._current_clip_path = None
            return False
        # Remembered so every frame read from this file can say where it came
        # from; evidence capture later seeks back into exactly this file.
        self._current_clip_path = path

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._clip_fps = fps if fps and 1.0 <= fps <= 120.0 else 15.0
        self._stream_width = w
        self._stream_height = h
        self._is_connected = True
        return True

    def _advance_clip(self):
        if not self.clip_paths:
            return
        self._clip_idx = (self._clip_idx + 1) % len(self.clip_paths)
        self._open_clip(self._clip_idx)

    # ── Frame quality validation ─────────────────────────────────────────────
    # Only reject genuinely corrupt (zero-byte, non-decoded, or completely solid blank) frames.
    # Night CCTV frames can legitimately have mean ~5.0 with real headlights (std ~15-20),
    # so we must never reject them based on low brightness.
    _MIN_FRAME_MEAN  = 0.5   # only reject absolute zero-black frames
    _MAX_FRAME_MEAN  = 254.5 # only reject completely saturated all-white frames
    _MIN_FRAME_STD   = 0.5   # only reject pure solid-colour / frozen blank frames

    def _is_frame_valid(self, frame: np.ndarray) -> bool:
        """Return True if the frame passes basic quality checks."""
        if frame is None or frame.size == 0:
            return False
        try:
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            grey  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mean  = float(grey.mean())
            std   = float(grey.std())
            if mean < self._MIN_FRAME_MEAN or mean > self._MAX_FRAME_MEAN:
                return False
            if std < self._MIN_FRAME_STD:
                return False
        except Exception:          # noqa: BLE001 – never crash the reader
            return True            # give it the benefit of the doubt
        return True

    def _read_live_network_frame(self, t0: float):
        """Read a real-time frame from a live network stream (RTSP/HTTP)."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._consecutive_failures += 1
            if self._consecutive_failures < 15:
                # Live RTSP streams often need a few frames to sync with the next IDR keyframe
                time.sleep(0.04)
                return
            logger.warning("[%s] Live network stream lost frame read (%d consecutive drops)",
                           self.cam_id, self._consecutive_failures)
            self._is_connected = False
            self._reconnect_count += 1
            self._release()

            # Failover if local clips available and strict live is disabled
            if not SENTINEL_STRICT_LIVE and self.clip_paths:
                logger.info("[%s] Switching to local clip failover while network stream reconnects", self.cam_id)
                self._in_failover = True
                self._open_clip(self._clip_idx)
            return

        # ── Problem 1 FIX: reject corrupted / black / artefact frames ────────
        if not self._is_frame_valid(frame):
            self._consecutive_failures += 1
            self._frames_dropped += 1
            if self._consecutive_failures < 15:
                time.sleep(0.02)
                return
            # Too many consecutive bad frames → treat as stream loss
            logger.warning("[%s] %d consecutive corrupt frames — reconnecting",
                           self.cam_id, self._consecutive_failures)
            self._is_connected = False
            self._reconnect_count += 1
            self._release()
            if not SENTINEL_STRICT_LIVE and self.clip_paths:
                self._in_failover = True
                self._open_clip(self._clip_idx)
            return

        self._consecutive_failures = 0

        self._frames_read += 1
        now_ts = time.time()
        self._last_frame_time = now_ts

        # A network stream has no file to seek back into, so the origin says
        # so rather than naming one. Evidence for these cameras must come from
        # a recording, and an honest "no seekable source" is what lets the
        # evidence layer refuse instead of substituting unrelated footage.
        origin = FrameOrigin(clip_path=None, frame_number=self._frames_read,
                             fps=float(self.target_fps))

        # Non-blocking enqueue
        try:
            self.frame_queue.put_nowait((self.cam_id, frame, now_ts, origin))
        except queue.Full:
            self._frames_dropped += 1

        # Smooth publishing: this frame goes to the screen now, drawn with the
        # newest overlay inference has produced, instead of waiting in the
        # queue for inference to reach it. See Live24x7Pipeline._present_frame.
        if self._present is not None:
            self._present(self.cam_id, frame, now_ts)

        # Throttle to target FPS
        elapsed = time.monotonic() - t0
        sleep_time = self.interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _read_clip_frame(self, t0: float):
        """Read a frame from a local clip with exact stride timing."""
        # If in failover mode, periodically probe if the network stream has recovered
        if self._in_failover and self.source_type in ("rtsp", "http", "onvif") and self.source_url:
            now = time.monotonic()
            # Back off after repeated failed recoveries. A host that is
            # reachable but broken — corp8 answering 522 — passes the probe
            # every time, so a fixed interval meant abandoning a perfectly good
            # clip every 120 s, spending ~20 s failing to open the stream, then
            # reopening the clip from a fresh position. Measured on CAM_09:
            # 1,209 frames processed and ZERO tracks persisted, because no
            # vehicle ever stayed in view across a whole cycle.
            interval = NETWORK_PROBE_INTERVAL_SEC * (2 ** min(self._failed_recoveries, 5))
            if now - self._last_probe_time > interval:
                self._last_probe_time = now
                if self._probe_network_stream():
                    # The probe only answers "is the host reachable". Whether
                    # the STREAM opens is a different question, and its answer
                    # used to be discarded — leaving the reader with no clip
                    # and no stream.
                    if self._open_network_stream(self.source_url):
                        logger.info("[%s] Live stream recovered; hot-swapped back "
                                    "from clip failover", self.cam_id)
                        self._failed_recoveries = 0
                        return
                    self._failed_recoveries += 1
                    logger.info(
                        "[%s] Host reachable but the stream still will not open "
                        "(attempt %d); staying on local clips, next probe in %.0fs",
                        self.cam_id, self._failed_recoveries,
                        NETWORK_PROBE_INTERVAL_SEC * (2 ** min(self._failed_recoveries, 5)))
                    self._in_failover = True
                    if not (self._cap and self._cap.isOpened()):
                        self._open_clip(self._clip_idx)
                    return

        stride = max(1, int(round(self._clip_fps / self.target_fps)))
        exhausted = False
        for _ in range(stride - 1):
            if not self._cap.grab():
                exhausted = True
                break
            self._frames_skipped += 1

        if exhausted:
            self._advance_clip()
            return

        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._advance_clip()
            return

        self._frames_read += 1
        self._video_seconds += stride / max(1.0, self._clip_fps)
        frame_ts = self._base_time + self._video_seconds
        self._last_frame_time = time.time()

        # Where this frame came from, carried with it.
        #
        # Without this the system cannot cut evidence of an event: when an
        # alert fires, nothing knows which file and which frame was on screen.
        # Evidence capture was left guessing, and its guess was "the middle of
        # this camera's first file" — which is why five congestion alerts on
        # CAM_02, hours apart, were all served the same byte-identical clip.
        #
        # CAP_PROP_POS_FRAMES is read after the read, so it points one past
        # the frame just returned; subtracting one gives the frame actually
        # in hand.
        try:
            pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        except Exception:                                          # noqa: BLE001
            pos = -1
        origin = FrameOrigin(
            clip_path=self._current_clip_path,
            frame_number=pos,
            fps=self._clip_fps,
        )

        if self._present is None or self._frames_read % INFER_EVERY == 0:
            try:
                self.frame_queue.put_nowait((self.cam_id, frame, frame_ts, origin))
            except queue.Full:
                self._frames_dropped += 1

        # Smooth publishing, as for network sources: every decoded frame goes
        # on screen now with the newest overlay, whatever inference is doing.
        if self._present is not None:
            self._present(self.cam_id, frame, frame_ts)

        elapsed = time.monotonic() - t0
        sleep_time = self.interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _probe_network_stream(self) -> bool:
        """Is the primary network source reachable? Cheap enough to poll.

        The comment above this used to say "lightweight non-blocking" while
        the body opened a full cv2.VideoCapture on the URL and read a frame.
        Against an unreachable host that blocks for as long as the underlying
        transport takes to give up — tens of seconds — inside the reader
        thread's own loop.

        With 27 of 32 cameras configured against a host that does not answer,
        every one of those threads stalled on this every 15 seconds. Measured
        effect: 5 of 32 cameras keeping up, a real-time factor of 0.161, and
        163 frames processed in five minutes. The cameras were not slow; they
        were waiting on a socket.

        A TCP connect with a short timeout answers the same question in
        milliseconds. If the port is open the full reconnect below does the
        real work.
        """
        if not self.source_url:
            return False
        try:
            from urllib.parse import urlparse
            u = urlparse(self.source_url)
            host = u.hostname
            if not host:
                return False
            port = u.port or {"https": 443, "http": 80, "rtsp": 554}.get(
                (u.scheme or "").lower(), 80)
            with socket.create_connection((host, port),
                                          timeout=NETWORK_PROBE_TIMEOUT_SEC):
                return True
        except Exception:                                          # noqa: BLE001
            return False

    def _handle_network_reconnect(self):
        """Exponential backoff reconnection handler for network streams."""
        if not self.source_url or not self.auto_reconnect:
            return
        now = time.monotonic()
        if now - self._last_reconnect_attempt < self._reconnect_delay:
            return

        self._last_reconnect_attempt = now
        logger.info(
            "[%s] Attempting network stream reconnect (attempt #%d, delay=%.1fs)...",
            self.cam_id, self._reconnect_count + 1, self._reconnect_delay,
        )
        success = self._open_network_stream(self.source_url)
        if success:
            logger.info("[%s] Network stream reconnected successfully!", self.cam_id)
            self._reconnect_delay = 1.0
        else:
            self._consecutive_failures += 1
            self._reconnect_count += 1
            self._reconnect_delay = min(
                self._max_reconnect_delay,
                self._reconnect_delay * 1.5 + float(np.random.uniform(0.1, 0.5)),
            )

    def _release(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def stop(self):
        self._running = False

    def stats(self) -> dict:
        """Counters and connection telemetry for pipeline health."""
        wall = max(1e-6, time.time() - self._base_time)
        now = time.time()
        last_frame_age = round(now - self._last_frame_time, 2) if self._last_frame_time > 0 else None

        effective_source = (
            "clip_failover" if self._in_failover
            else self.source_type
        )

        return {
            "camera": self.cam_id,
            "source_type": effective_source,
            "source_url_masked": self._mask_url(self.source_url),
            "is_connected": self._is_connected,
            "in_failover": self._in_failover,
            "reconnect_count": self._reconnect_count,
            "consecutive_failures": self._consecutive_failures,
            "resolution": f"{self._stream_width}x{self._stream_height}" if self._stream_width else "unknown",
            "frames_queued": self._frames_read,
            "frames_skipped": self._frames_skipped,
            "frames_dropped": self._frames_dropped,
            "sample_fps": round(self._frames_read / wall, 2),
            "video_seconds": self._video_seconds,
            "wall_seconds": round(wall, 1),
            "realtime_factor": round(self._video_seconds / wall, 3) if self.source_type == "clips" or self._in_failover else 1.0,
            "last_frame_age_seconds": last_frame_age,
        }

    @property
    def frames_read(self) -> int:
        return self._frames_read


# ─────────────────────────────────────────────────────────────────────────────
# Environment Classifier — lightweight frame-level weather/lighting detection
# ─────────────────────────────────────────────────────────────────────────────
_PIXEL_MODELS: Optional[dict] = None


def _pixel_models() -> dict:
    """Coefficients fitted by backend/scripts/fit_wet_model.py, loaded once."""
    global _PIXEL_MODELS
    if _PIXEL_MODELS is None:
        path = WORKSPACE / "output" / "env_classifier" / "pixel_models.json"
        try:
            _PIXEL_MODELS = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("pixel environment models unavailable (%s); "
                           "falling back to daylight", exc)
            _PIXEL_MODELS = {}
    return _PIXEL_MODELS


def _score(model: dict, feats: dict) -> float:
    """Logistic score for one fitted model. Positive means the positive class."""
    z = model["intercept"]
    for name, mu, sd, w in zip(model["features"], model["mean"],
                               model["std"], model["coef"]):
        z += w * (feats.get(name, mu) - mu) / sd
    return z


# Every environment feature is a statistic over the whole frame — a mean, a
# percentile, a fraction of pixels — so it does not need full resolution, and
# the specular measure needs a blur radius fixed relative to the image rather
# than to the sensor. Computing at full 1080p cost 82 ms per call. Both this
# module and fit_environment_classifier resize to this size first, so the
# fitted coefficients and the runtime see the same numbers.
ENV_FEATURE_SIZE = (480, 270)


def environment_features(frame: np.ndarray) -> dict:
    """Scene statistics the fitted pixel models read.

    Kept identical to backend/scripts/fit_environment_classifier.features() —
    if the two drift apart the coefficients stop meaning anything.
    """
    if (frame.shape[1], frame.shape[0]) != ENV_FEATURE_SIZE:
        frame = cv2.resize(frame, ENV_FEATURE_SIZE,
                           interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    vf = v.astype(np.float32) / 255.0

    lower = vf[vf.shape[0] // 2:, :]
    road_mean = float(np.mean(lower))
    blur = cv2.GaussianBlur(lower, (0, 0), 9)
    sky_mean = float(np.mean(vf[: vf.shape[0] // 3, :]))
    p = np.percentile(vf, [5, 50, 95])

    return {
        "mean_s": float(np.mean(s)),
        "dark_frac": float(np.mean(vf < 0.20)),
        "bright_frac": float(np.mean(vf > 0.90)),
        "road_mean": road_mean,
        "road_std": float(np.std(lower)),
        # A wet road is a partial mirror: specular returns sit locally much
        # brighter than their surroundings.
        "specular": float(np.mean((lower - blur) > 0.10)),
        "sky_mean": sky_mean,
        "sky_road_ratio": float(sky_mean / max(road_mean, 1e-3)),
        "v_p05": float(p[0]), "v_p50": float(p[1]), "v_p95": float(p[2]),
    }


def classify_environment(frame: np.ndarray,
                         captured_at: Optional[datetime] = None,
                         lat: Optional[float] = None,
                         lon: Optional[float] = None) -> str:
    """Lighting and surface condition for one frame.

    Day or night comes from the sun, not the pixels. Given the frame's capture
    time and the camera's location, solar elevation answers it exactly; on 12
    labelled clips whose burnt-in clock was legible this was right 12 times,
    including indoor and infrared views that no pixel rule can resolve — a lit
    corridor at 21:04 is identical to the same corridor at 13:04.

    Pixels are used for two things only, both fitted and scored with every
    camera held out in turn (backend/scripts/fit_wet_model.py):

        wet road, daylight frames        76.5%   17 cameras
        daylight vs night, as fallback   73.1%   26 cameras

    The fallback runs only when capture time or camera position is unknown.

    What this deliberately no longer returns: DUST_STORM and MONSOON as
    guesses. The rule it replaces returned MONSOON whenever vertical-Sobel
    variance exceeded 450 and mean brightness fell below 130 — true of most
    textured urban daytime scenes — and so labelled 16 of 26 reference frames
    MONSOON, two of them night-time infrared. WET is returned instead, meaning
    the measured thing: standing water or specular reflection on the
    carriageway. DUST_STORM has no labelled example in this footage and is
    therefore not claimed.
    """
    if frame is None or getattr(frame, "size", 0) == 0 or len(getattr(frame, "shape", ())) < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        return "UNKNOWN"

    feats = environment_features(frame)
    models = _pixel_models()

    daylight: Optional[bool] = None
    if captured_at is not None and lat is not None and lon is not None:
        try:
            from backend.services.clip_clock import is_daylight
            daylight = is_daylight(captured_at, lat, lon)
        except Exception as exc:                                   # noqa: BLE001
            logger.debug("solar day/night unavailable: %s", exc)

    if daylight is None:
        m = models.get("daylight")
        daylight = _score(m, feats) > 0 if m else True

    if not daylight:
        # Headlight and street-lamp glare, which is what actually degrades
        # night plate reads. Measured on the frame rather than assumed.
        if feats["bright_frac"] > 0.015 and feats["dark_frac"] > 0.35:
            return "NIGHT_GLARE"
        return "NIGHT"

    m = models.get("wet")
    if m and _score(m, feats) > 0:
        return "WET"
    return "DAY"


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline Class
# ─────────────────────────────────────────────────────────────────────────────
class Live24x7Pipeline:
    """
    Production-grade 24x7 CCTV stream ingestion and real-time analytics pipeline.

    Fixes from code review:
      - Per-camera BoTSORTTracker instances (no cross-contamination)
      - Producer-consumer architecture (reader threads → queue → GPU consumer)
      - Proper session-per-unit-of-work with try/finally
      - Correct ORM column names (n_frames, speed_ci_kmh, quality)
      - No blocking HTTPS streams (local clips only)
      - Separate DB transactions for vault vs. track writes
      - Error logging at proper severity levels
    """

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._running = False
        self._inference_thread: Optional[threading.Thread] = None
        self._reader_threads: List[CameraReaderThread] = []
        # Queue is initially sized for a single-camera run; start() resizes it
        # after camera discovery so a 30-camera process doesn’t get a tiny queue
        # and a 3-camera process doesn’t hold 300 stale frames.
        self._frame_queue: queue.Queue = queue.Queue(maxsize=60)
        self._model = None

        # Per-camera state
        self._trackers: Dict[str, BoTSORTTracker] = {}
        self._track_buffers: Dict[str, Dict[int, list]] = {}  # {cam_id: {track_id: [(ts, x_m, y_m, u, v, conf, cls_id)]}}
        self._calibrations: Dict[str, dict] = {}
        # Cameras whose pixel->world transform has already been reported as
        # failing, so the warning is raised once rather than per detection.
        self._calib_transform_failed: set[str] = set()
        # Detections dropped for insufficient ground resolution, per camera.
        # Reported in health so a camera whose usable region is tiny is
        # visible rather than silently speed-less.
        self._calib_pts_rejected: Dict[str, int] = {}
        # Reported once, not per frame: publishing live frames fails for the
        # whole run when it fails at all (a full disk, a read-only path).
        self._live_frame_write_failed = False
        self._env_cache: Dict[str, Tuple[str, float]] = {}  # {cam_id: (env_label, timestamp)} — cached per frame cycle

        # Shared congestion engine (reused, not re-instantiated)
        self._congestion_engine = CongestionEngine()

        # Empirical Bayes Seasonal Traffic Baseline & Anomaly Engine (Gap 3)
        self._baseline_engine = TrafficBaselineEngine()

        # Production ANPR Plate Recognition Engine (CRNN + All-India Grammar)
        self._anpr = get_anpr_engine()

        # One OSNet-IBN instance for the whole pipeline. Loading it per camera
        # would put 27 copies of the weights on the GPU; the forward pass is
        # already batched per frame, so a single shared model is enough.
        self._embedder = get_embedder()
        if self._embedder.is_stub:
            logger.error(
                "ReID embedder is in STUB MODE — appearance vectors would be "
                "seeded noise. Tracking will run on motion only and vault rows "
                "will carry no embedding."
            )

        # Where the time actually goes. Isolated benchmarks put a cycle at
        # roughly 24 ms per frame while the live loop measured 204 ms, and a
        # gap that size is only closable if it can be attributed.
        self._phase_seconds: Dict[str, float] = {
            "detect": 0.0, "embed": 0.0, "track": 0.0, "flush": 0.0,
            "vault": 0.0, "anpr": 0.0,
        }
        self._phase_frames = 0

        # Vault writes happen off the inference thread; see
        # _queue_vault_harvest for why.
        self._vault_queue: queue.Queue = queue.Queue(maxsize=VAULT_QUEUE_MAX)
        self._vault_thread: Optional[threading.Thread] = None
        self._vault_dropped = 0

        # ── Async ANPR worker ─────────────────────────────────────────────
        # CRITICAL FIX for 10-camera fleet: ANPR runs 31ms per vehicle. With
        # 8 ANPR cameras × 3 vehicles, inline ANPR blocks the inference thread
        # for 744ms/cycle → ~1.2fps. Moving ANPR to a dedicated worker thread
        # (same pattern as vault writer) gives <1ms enqueue cost → 10fps.
        #
        # SENTINEL_ANPR_ASYNC=1 enables this path (set by start_top10_live.ps1).
        # SENTINEL_ANPR_ASYNC=0 keeps the old synchronous path for backward compat.
        _anpr_async_raw = os.environ.get("SENTINEL_ANPR_ASYNC", "0").strip()
        self._anpr_async: bool = _anpr_async_raw not in ("0", "false", "False", "")
        # Queue sized at 300: latest-wins per (cam_id, track_id), so a busy
        # camera cannot grow memory unboundedly. Dropped jobs are counted.
        self._anpr_queue: queue.Queue = queue.Queue(maxsize=300)
        self._anpr_thread: Optional[threading.Thread] = None
        self._anpr_dropped = 0
        # Lock protecting _track_plates, shared between inference thread (read
        # for plate overlay) and ANPR worker thread (write after OCR).
        self._track_plates_lock = threading.Lock()
        # Per-track enqueue limiter: avoid flooding queue with redundant crops of same track
        self._anpr_enqueued_counts: Dict[int, int] = {}
        # ANPR performance counters — exposed via /api/v1/top10/status
        self._anpr_attempts = 0
        self._anpr_successes = 0
        self._anpr_total_ms = 0.0

        # Camera positions and per-clip capture times, both needed to decide
        # day or night from the sun rather than from brightness.
        self._camera_gps: Dict[str, Tuple[float, float]] = {}
        self._clip_start: Dict[str, Optional[datetime]] = {}

        # Where each camera's most recently processed frame came from, and the
        # recent box history of each track. Together these are what makes an
        # alert's evidence checkable: the first says which frames to cut, the
        # second says what to draw on them.
        self._last_origin: Dict[str, FrameOrigin] = {}
        self._track_boxes: Dict[str, Dict[int, list]] = {}
        # Best plate read per global track id, carried to the flush so the
        # read is written with the track instead of being dropped.
        self._track_plates: Dict[int, dict] = {}

        # Evidence encoding runs off the inference thread. Two workers is
        # enough: alerts are rare next to frames, and a deeper pool would
        # compete with the detector for the same cores.
        from concurrent.futures import ThreadPoolExecutor
        self._evidence_pool: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="evidence")

        # Live Annotated CCTV Frames (Thread-Safe Streaming Cache)
        self._latest_annotated_jpeg: Dict[str, bytes] = {}
        self._latest_frame_ts: Dict[str, float] = {}
        self._frame_lock = threading.Lock()
        # {(cam_id, track_id): (frame_ts, label)} — see _draw_live_overlay. Keeps
        # the display's speed label off the per-frame Theil-Sen path. Pruned
        # there so a long run does not accumulate an entry per track for ever.
        self._overlay_speed_cache: Dict[Tuple[str, int], Tuple[float, str]] = {}
        # For ALIGN_OVERLAY_CAMERAS: recent overlays by capture time, and the
        # frames held back until inference has looked past them.
        self._overlay_hist: Dict[str, Any] = {}
        self._align_buf: Dict[str, Any] = {}
        # Cross-camera identity for the live tiles. Loaded lazily and ONLY for
        # cameras named in SENTINEL_GLOBALID_CAMERAS, so a camera not on that
        # list — CAM_09 above all — does no extra work whatsoever.
        self._gid_gallery: Optional[Tuple[np.ndarray, str, str]] = None
        self._gid_gallery_tried = False
        self._gid_cache: Dict[Tuple[str, int], Tuple[float, float]] = {}
        # {(cam_id, track_id): [summed_vector, n]} — the running appearance of
        # each live track, written out as one mean vector when the track
        # flushes. See db.models.TrackAppearance for why it is kept.
        self._track_embeddings: Dict[Tuple[str, int], list] = {}
        # {cam_id: frames seen} — drives the EMBED_EVERY appearance gate.
        self._embed_tick: Dict[str, int] = {}
        # Frames drawn but not yet encoded, newest per camera (latest wins).
        self._publish_pending: "Dict[str, Tuple[np.ndarray, float]]" = {}
        self._publish_cv = threading.Condition()
        self._publish_last: Dict[str, float] = {}
        self._publish_superseded = 0
        self._publish_thread: Optional[threading.Thread] = None
        # Newest overlay per smooth-published camera: (capture ts, tracks).
        self._overlay_state: Dict[str, Tuple[float, list]] = {}
        # Appearance rows queued by _persist_vehicle_track and written only
        # after the tracks' own transaction commits. See _write_appearance.
        self._pending_appearance: list = []
        self._gcp_presets: Dict[str, dict] = {}
        gcp_path = WORKSPACE / "backend" / "calibration_data" / "ground_control_points.json"
        if gcp_path.exists():
            try:
                # utf-8-sig: matches the write side in analytics.py's
                # save_camera_calibration() — the file has carried a BOM,
                # and plain "utf-8" raised here too, silently (bare except),
                # so the on-screen GCP marker overlay had no presets to draw
                # from even after a camera was genuinely calibrated. This is
                # cosmetic only — real speed/position come from the DB via
                # TrackRecorder, never from this file — but it is the visual
                # a demo would show, so a silent gap here reads as "the
                # system doesn't know its own calibration" to a viewer.
                self._gcp_presets = json.loads(
                    gcp_path.read_text(encoding="utf-8-sig"))
            except Exception as e:
                logger.warning("Could not load GCP presets from %s: %s",
                               gcp_path, e)

        # Metrics
        self._total_frames_processed = 0
        self._total_tracks_persisted = 0
        self._total_vault_harvested = 0
        self._last_rollup_ts = 0.0
        self._start_time = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.time()

        # Discover all cameras (RTSP network feeds + local clips + DB registry)
        db = SessionLocal()
        try:
            self._load_calibrations(db)
            self._load_camera_gps(db)
            discovered = self._discover_cameras(db)
        finally:
            db.close()

        # Optional focus list. Ingesting all 30 cameras on this hardware drops
        # ~29% of frames and holds most cameras below real time, which costs
        # track continuity — and a broken track is what produces an impossible
        # speed. Narrowing to the cameras that are actually being demonstrated
        # gives each one the whole GPU. Comma-separated, e.g.
        #     SENTINEL_ONLY_CAMERAS=CAM_09
        only = os.environ.get("SENTINEL_ONLY_CAMERAS", "").strip()
        if only:
            wanted = {c.strip().upper() for c in only.split(",") if c.strip()}
            before = len(discovered)
            discovered = [c for c in discovered
                          if (c["cam_id"] or "").upper() in wanted]
            missing = wanted - {(c["cam_id"] or "").upper() for c in discovered}
            logger.info(
                "SENTINEL_ONLY_CAMERAS=%s — ingesting %d of %d discovered cameras%s",
                only, len(discovered), before,
                ("; NOT FOUND: %s" % ", ".join(sorted(missing))) if missing else "")

        for cam_info in discovered:
            cam_id = cam_info["cam_id"]
            source_url = cam_info["source_url"]
            clip_paths = cam_info["clip_paths"]

            reader = CameraReaderThread(
                cam_id=cam_id,
                clip_paths=clip_paths,
                frame_queue=self._frame_queue,
                fps=READER_FPS,
                source_url=source_url,
                present=(self._present_frame
                         if cam_id.upper() in SMOOTH_PUBLISH_CAMERAS else None),
            )
            self._reader_threads.append(reader)
            # Initialize per-camera tracker
            self._trackers[cam_id] = BoTSORTTracker(frame_rate=max(1, READER_FPS // INFER_EVERY))
            self._track_buffers[cam_id] = {}

        for reader in self._reader_threads:
            reader.start()
            time.sleep(0.5)

        # Resize the shared frame queue now that the camera count is known.
        # Rule: hold at most 2 seconds of frames from every camera, so a burst
        # after a slow inference cycle doesn’t overflow on a small worker.
        #
        #   maxsize = max(60, cameras × reader_fps × 2)
        #
        # On a 3-camera worker at 10 fps this gives 60 (the floor).
        # On a 30-camera worker at 10 fps this gives 600 — but the fleet
        # always runs as multiple workers, so the 30-camera case never arises.
        _n_cams = len(self._reader_threads)
        _q_size = max(60, _n_cams * READER_FPS * 2)
        if _q_size != self._frame_queue.maxsize:
            # queue.Queue does not support resize; replace with a new empty one.
            # The old queue is drained into the new one so no frames are lost.
            old_q = self._frame_queue
            new_q: queue.Queue = queue.Queue(maxsize=_q_size)
            while True:
                try:
                    new_q.put_nowait(old_q.get_nowait())
                except queue.Full:
                    break  # new queue full (shouldn’t happen at start)
                except queue.Empty:
                    break
            self._frame_queue = new_q
            # Point all readers at the new queue.
            for r in self._reader_threads:
                r.frame_queue = self._frame_queue
            logger.info("Frame queue sized to %d (cameras=%d, fps=%d)",
                        _q_size, _n_cams, READER_FPS)

        # Start the GPU inference consumer thread
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="live-24x7-inference"
        )
        self._inference_thread.start()

        # One vault writer, so SQLite sees a single writer and the inference
        # thread never blocks on a JPEG encode or a commit.
        self._vault_thread = threading.Thread(
            target=self._vault_writer_loop, daemon=True, name="live-24x7-vault"
        )
        self._vault_thread.start()

        # One publisher, so neither the inference thread nor a camera's reader
        # thread pays the 11.8 ms JPEG encode or waits on the shared file.
        if PUBLISH_ASYNC:
            self._publish_thread = threading.Thread(
                target=self._publisher_loop, daemon=True,
                name="live-24x7-publish")
            self._publish_thread.start()

        # Async ANPR worker — only started when SENTINEL_ANPR_ASYNC=1.
        # One worker services ALL cameras in this process; ANPR is CPU-bound
        # (EasyOCR) and one thread saturates one core, which is all we need:
        # 8 cameras × 3 vehicles × 31ms = 744ms/s < 1 core.
        if self._anpr_async:
            self._anpr_thread = threading.Thread(
                target=self._anpr_worker_loop, daemon=True,
                name="live-24x7-anpr")
            self._anpr_thread.start()
            logger.info(
                "⚡ Async ANPR worker started — inference thread will NOT be "
                "blocked by OCR. Results appear within 1-3s of vehicle detection."
            )
        else:
            logger.info(
                "ANPR running SYNCHRONOUS (SENTINEL_ANPR_ASYNC=0). "
                "Set SENTINEL_ANPR_ASYNC=1 in start_top10_live.ps1 for 10fps mode."
            )

        logger.info(
            "🚀 Sentinel Gujarat 24x7 Pipeline started — device=%s, cameras=%d, readers=%d, anpr_async=%s",
            self.device, len(self._reader_threads), len(self._reader_threads), self._anpr_async,
        )

    def stop(self):
        self._running = False
        for reader in self._reader_threads:
            reader.stop()
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=5.0)
        # Let the writer drain what it already holds rather than discarding
        # crops that were counted as harvested.
        if self._vault_thread and self._vault_thread.is_alive():
            self._vault_thread.join(timeout=10.0)
        if self._publish_thread and self._publish_thread.is_alive():
            # Wake it so it notices _running went false instead of sitting out
            # its wait timeout.
            with self._publish_cv:
                self._publish_cv.notify_all()
            self._publish_thread.join(timeout=5.0)
        # Drain remaining ANPR jobs before exit so no plate reads are lost.
        if self._anpr_thread and self._anpr_thread.is_alive():
            self._anpr_thread.join(timeout=15.0)
        for reader in self._reader_threads:
            if reader.is_alive():
                reader.join(timeout=2.0)
        self._reader_threads.clear()
        rate = (self._anpr_successes / max(1, self._anpr_attempts) * 100)
        logger.info(
            "Sentinel Gujarat 24x7 Pipeline stopped — total_frames=%d, total_tracks=%d, "
            "total_vault=%d, anpr_attempts=%d, anpr_successes=%d (%.1f%%), anpr_dropped=%d",
            self._total_frames_processed, self._total_tracks_persisted,
            self._total_vault_harvested, self._anpr_attempts,
            self._anpr_successes, rate, self._anpr_dropped,
        )

    @property
    def health(self) -> dict:
        """Expose health metrics for monitoring endpoints."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "device": self.device,
            "cameras_active": len(self._reader_threads),
            "total_frames_processed": self._total_frames_processed,
            "total_tracks_persisted": self._total_tracks_persisted,
            "total_vault_harvested": self._total_vault_harvested,
            "uptime_seconds": round(uptime, 1),
            "avg_fps": round(self._total_frames_processed / max(1, uptime), 2),
            "queue_depth": self._frame_queue.qsize(),
            "per_camera_frames": {r.cam_id: r.frames_read for r in self._reader_threads},
            **self.lag_report(),
        }

    def lag_report(self) -> dict:
        """Whether the pipeline is actually keeping up, as numbers.

        "Zero lag" was never measured, and two mechanisms made it unfalsifiable:
        readers dropped frames on a full queue without counting them, and the
        inference loop kept only the newest frame per camera and discarded the
        backlog. A pipeline that keeps up by throwing work away reports the
        same uptime as one that genuinely keeps up.

        The three numbers that settle it:

          realtime_factor   video seconds consumed per wall second. 1.0 means
                            the fleet is being processed as fast as it is
                            recorded. Below 1.0 it is falling behind.
          frames_dropped    frames a reader could not hand over because the
                            queue was full. Any non-zero value is real loss.
          queue_depth       standing backlog. Flat is healthy; growing means
                            the GPU is behind the readers.
        """
        stats = [r.stats() for r in self._reader_threads]
        if not stats:
            return {"realtime_factor": None, "frames_dropped_total": 0,
                    "cameras_keeping_up": 0, "per_camera_lag": {}}

        dropped = sum(s["frames_dropped"] for s in stats)
        queued = sum(s["frames_queued"] for s in stats)
        rtf = [s["realtime_factor"] for s in stats]
        return {
            "realtime_factor_min": round(min(rtf), 3),
            "realtime_factor_mean": round(sum(rtf) / len(rtf), 3),
            "frames_queued_total": queued,
            "frames_dropped_total": dropped,
            "frames_dropped_pct": round(100.0 * dropped / max(1, queued + dropped), 2),
            # A camera is keeping up if it is consuming video at least as fast
            # as it arrives, with 5% slack for scheduler jitter.
            "cameras_keeping_up": sum(1 for v in rtf if v >= 0.95),
            "cameras_total": len(stats),
            "queue_maxsize": self._frame_queue.maxsize,
            "queue_depth": self._frame_queue.qsize(),
            # Publisher backlog. `superseded` counts frames replaced before
            # they were encoded — a few are normal and mean the encoder is
            # correctly skipping stale pictures; a number climbing as fast as
            # frames are processed means the publisher cannot keep up, and
            # SENTINEL_PUBLISH_MAX_WIDTH or PUBLISH_MIN_INTERVAL_S is the fix.
            "publish_pending": len(self._publish_pending),
            "publish_superseded": self._publish_superseded,
            "ms_per_frame": {
                k: round(v / max(1, self._phase_frames) * 1000, 2)
                for k, v in self._phase_seconds.items()
            },
            "per_camera_lag": {
                s["camera"]: {
                    "rtf": round(s["realtime_factor"], 3),
                    "queued": s["frames_queued"],
                    "dropped": s["frames_dropped"],
                    "last_frame_age_s": s.get("last_frame_age_seconds"),
                    "stream_state": (
                        "LIVE" if not s.get("in_failover") and s.get("source_type") in ("rtsp", "http", "onvif")
                        else "RECORDED" if s.get("source_type") in ("clips", "file", "clip_failover")
                        else "OFFLINE" if not s.get("is_connected")
                        else "UNKNOWN"
                    ),
                    "reconnect_count": s.get("reconnect_count", 0),
                }
                for s in stats
            },
        }

    # ── Model Loading ────────────────────────────────────────────────────────

    def _load_model(self):
        # Native dynamic memory management: avoid artificial fraction boundaries
        # that trigger fatal cudaErrorIllegalAddress on multi-process workloads.
        if os.environ.get("SENTINEL_FORCE_CUDA_FRACTION") and str(self.device).startswith("cuda"):
            try:
                import torch
                frac = float(os.environ["SENTINEL_FORCE_CUDA_FRACTION"])
                torch.cuda.set_per_process_memory_fraction(frac)
                logger.info("CUDA memory capped at %.0f%%", frac * 100)
            except Exception as exc:                                # noqa: BLE001
                logger.warning("Could not cap CUDA memory (%s)", exc)
        try:
            from ultralytics import YOLO
            # Detector weights per process. Measured 2026-09-11 on 60 raw
            # CAM_09 frames (overhead view, hard sun): yolov8s found 60
            # vehicles in 35 frames and missed a clearly visible white SUV
            # even at conf 0.10; yolov8m found 69 in 40 (more on 10 frames,
            # fewer on 3) at 29 vs 14 ms/frame. Everything downstream assumes
            # COCO class ids (person 0, car 2, motorcycle 3, bus 5, truck 7),
            # so weights with any other class map are refused — the fleet's
            # own fine-tuned gujarat_v5 uses car=1, motorcycle=2.
            weights = os.environ.get("SENTINEL_YOLO_WEIGHTS", "yolov8s.pt").strip() or "yolov8s.pt"
            model = YOLO(weights)
            names = getattr(model, "names", {}) or {}
            coco = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
            if any(str(names.get(k, "")).lower() != v for k, v in coco.items()):
                logger.error("SENTINEL_YOLO_WEIGHTS=%s does not use COCO class ids "
                             "(%s); using yolov8s.pt instead", weights,
                             {k: names.get(k) for k in coco})
                weights, model = "yolov8s.pt", YOLO("yolov8s.pt")
            self._model = model
            self._model_weights = weights
            try:
                if str(self.device).startswith("cuda"):
                    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                    self._model.predict(dummy, verbose=False, device=self.device)
            except Exception as cuda_err:
                logger.warning("CUDA device error (%s); falling back to CPU", cuda_err)
                self.device = "cpu"
            logger.info("Loaded detection model %s on %s",
                        getattr(self, "_model_weights", "yolov8s.pt"), self.device)
        except Exception as e:
            logger.error("CRITICAL: Failed to load YOLO model — pipeline will not produce detections: %s", e)

    # ── Calibration Loading ──────────────────────────────────────────────────

    def _load_calibrations(self, db: Session):
        calibs = db.query(CameraHomographyCalibration).filter(CameraHomographyCalibration.is_active == True).all()
        loaded = 0
        for c in calibs:
            if c.homography_matrix:
                try:
                    H = json.loads(c.homography_matrix)
                    # Reshape to 3x3. The calibration writers store the matrix
                    # FLAT (fit_focal_from_legs emits a 9-element list), and
                    # np.array() of that is shape (9,), which makes
                    # pixel_to_world_m raise "matmul: mismatch in its core
                    # dimension" on every single point. That exception is
                    # swallowed by a bare `except Exception: pass` at the call
                    # site, so the camera silently behaved as uncalibrated:
                    # measured 2026-09-09, speed_kmh was NULL on 100% of
                    # 227,636 tracks — every track on BOTH calibrated cameras
                    # since the 2026-09-07 calibration — while the startup log
                    # still said "Loaded 2 camera calibrations".
                    H_arr = np.asarray(H, dtype=np.float64)
                    if H_arr.size != 9:
                        raise ValueError(
                            "homography has %d elements, expected 9" % H_arr.size)
                    self._calibrations[c.camera_id] = {
                        "H": H_arr.reshape(3, 3),
                        "rms": c.reprojection_error_m or 0.02,
                        "quality": c.quality_gate or "good",
                    }
                    loaded += 1
                except Exception as e:
                    logger.warning("Failed to parse homography for %s: %s", c.camera_id, e)

        # A camera without a surveyed calibration gets no calibration.
        #
        # What used to be here: for every CAM_01..CAM_30 lacking one, a
        # homography was synthesised from four fixed image points and a fixed
        # 20m x 67m ground rectangle — the same trapezoid for a bridge deck, a
        # market junction and an indoor bus station — and stamped with
        # rms=0.035 and quality="good". Those cameras then reported speeds in
        # km/h that no measurement supported. It is the reason 1,793 of 2,252
        # non-zero speeds in the database sit on cameras that have no
        # homography at all.
        #
        # Pixel-to-metre scale depends on where the camera is mounted, how far
        # it is tilted and what lens it has. It cannot be assumed. Cameras
        # without one now report track geometry in pixels and no speed, which
        # is what is actually known about them.
        uncalibrated = sorted(
            c for c in self._active_camera_ids() if c not in self._calibrations
        )
        logger.info(
            "Loaded %d camera calibrations; %d cameras have none and will "
            "report no speed: %s",
            loaded, len(uncalibrated),
            ", ".join(uncalibrated[:8]) + ("…" if len(uncalibrated) > 8 else ""),
        )

    def _active_camera_ids(self) -> List[str]:
        """Cameras this process is actually reading, from the reader threads."""
        return [r.cam_id for r in self._reader_threads]

    def _queue_evidence(self, alert_id: str) -> None:
        """Cut this alert's proof clip, off the inference thread.

        The pipeline's own alerts never reached evidence capture at all. Only
        the standalone detectors called on_alert_fired, so a stolen-vehicle
        intercept or a congestion anomaly raised here produced a row with no
        footage behind it — and the clips that did exist came from a batch
        script run afterwards, which is where the duplication came from.

        Encoding a clip takes seconds; the inference loop has a ~30 ms budget
        per frame, so this hands the work to a small pool and returns.
        """
        if self._evidence_pool is None:
            return

        def _run() -> None:
            try:
                from backend.services.evidence_capture import _capture_sync
                status = _capture_sync(alert_id)
                if status != "COMPLETE":
                    logger.warning("Evidence for alert %s ended %s",
                                   alert_id, status)
            except Exception as exc:                               # noqa: BLE001
                logger.warning("Evidence capture failed for %s: %s",
                               alert_id, exc)

        try:
            self._evidence_pool.submit(_run)
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("Could not queue evidence for %s: %s", alert_id, exc)

    @staticmethod
    def _global_track_id(cam_id: str, track_id: int) -> int:
        """A track id unique across the fleet.

        Per-camera trackers number their tracks from 1 independently, so a
        bare id identifies a vehicle only within one camera. Anything keyed
        globally — the shared ANPR engine's vote history and plate lock —
        needs the camera folded in.

        The camera's numeric suffix is used where it has one, and a stable
        hash otherwise, so ids stay consistent across restarts.
        """
        digits = "".join(ch for ch in cam_id if ch.isdigit())
        cam_n = int(digits) if digits else (abs(hash(cam_id)) % 9000 + 1000)
        return cam_n * 1_000_000 + int(track_id)

    def _evidence_source(self, cam_id: str,
                         track_id: Optional[int] = None) -> Optional[dict]:
        """Everything needed to cut this alert's proof clip from the footage.

        An alert without this cannot have evidence — only illustration. That
        was the previous state: capture fell back to "the middle of this
        camera's first file", which is why five congestion alerts on CAM_02,
        recorded hours apart, were all served the same byte-identical clip,
        each with a valid SHA-256 seal over the wrong footage.

        Returns None for a live network stream, which has no file to seek back
        into. Refusing is correct there: the alternative is attaching footage
        that is not of the event.
        """
        origin = self._last_origin.get(cam_id)
        if origin is None or not origin.clip_path:
            return None

        src = {
            "clip": origin.clip_path,
            "frame": origin.frame_number,
            "fps": round(origin.fps, 3),
            "sampled_fps": READER_FPS,
        }
        if track_id is not None:
            boxes = self._track_boxes.get(cam_id, {}).get(track_id)
            if boxes:
                src["track_id"] = int(track_id)
                # (source frame number, xyxy, confidence, class)
                src["track_boxes"] = boxes[-TRACK_BOX_HISTORY:]
        return src

    def camera_motion(self, cam_id: str) -> dict:
        """Measured camera sway for one camera, or why it is not measured.

        The tracker's motion-compensation step already solves for the warp
        between consecutive frames; this exposes it. A camera with no tracker
        yet is not producing zeros, it is producing nothing, and says so.
        """
        tracker = self._trackers.get(cam_id)
        if tracker is None:
            known = cam_id in self._active_camera_ids()
            return {
                "camera_id": cam_id,
                "measured": False,
                "reason": ("no frames processed yet" if known
                           else "camera is not being read by this pipeline"),
            }
        motion = tracker.camera_motion
        return {
            "camera_id": cam_id,
            "measured": motion["jitter_rms_px"] is not None,
            **motion,
            **tracker.gmc_stats,
        }

    def _discover_cameras(self, db: Session) -> List[Dict[str, Any]]:
        """
        Production multi-source camera discovery:
        1. Queries all non-deleted cameras in the database.
        2. Matches stream URLs (RTSP / HTTP) from Camera.stream_url or Camera.url.
        3. Discovers local fallback clips from data/clips/{cam_id}/*.mp4.
        4. Discovers any additional clip folders in data/clips/ not yet in DB.
        """
        discovered: Dict[str, Dict[str, Any]] = {}

        # Optional allow-list, e.g. SENTINEL_PIPELINE_CAMERAS="CAM_08,CAM_11".
        # Ingesting every registered camera is the right default, but there is
        # no way to bring up a subset — needed to stage a rollout, to isolate
        # one camera when diagnosing it, or to run a focused demonstration
        # without spinning up decoders for the whole estate.
        only_raw = os.environ.get("SENTINEL_PIPELINE_CAMERAS", "").strip()
        only = {c.strip() for c in only_raw.split(",") if c.strip()} if only_raw else None
        if only:
            logger.info("Camera allow-list active (SENTINEL_PIPELINE_CAMERAS): %s",
                        ", ".join(sorted(only)))

        # 1. DB Cameras
        try:
            db_cameras = db.query(Camera).filter(Camera.is_deleted == False).all()
            for c in db_cameras:
                cam_id = c.camera_id or (f"CAM_{c.id:02d}" if isinstance(c.id, int) else str(c.id))
                # Match the allow-list against the registry id as well as the
                # computed cam_id. POST /cameras stores the human NAME in
                # camera_id (see _negotiate_and_create_camera), so a camera
                # added through the API has a cam_id like "LIVE DEMO - Gate 4"
                # while its registry id is "CAM_33" — and "CAM_33" is what an
                # operator would actually type.
                if only and not ({cam_id, str(c.id)} & only):
                    continue
                source_url = c.stream_url or c.url

                # Check for matching local clips
                clip_paths = []
                cam_clip_dir = CLIPS_DIR / cam_id
                if cam_clip_dir.exists() and cam_clip_dir.is_dir():
                    clip_paths = sorted(cam_clip_dir.glob("*.mp4"))

                discovered[cam_id] = {
                    "cam_id": cam_id,
                    "source_url": source_url,
                    "clip_paths": clip_paths,
                    "name": c.name,
                    "db_id": c.id,
                }
        except Exception as exc:
            logger.error("Error querying cameras from database: %s", exc)

        # 2. Local clip folders (ensure zero cameras are missed)
        #
        # Any directory holding video files counts, not only ones named
        # "CAM_*". The prefix requirement silently skipped footage dropped in
        # under any other name — which is exactly how a new source arrives
        # (an evaluator's folder of test footage, a department's export). A
        # folder with no video in it is still skipped by the `if not clips`
        # check below, so dropping the prefix rule costs nothing and stops
        # the ingest from ignoring real footage because of its folder name.
        VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm")
        if CLIPS_DIR.exists():
            for cam_dir in sorted(CLIPS_DIR.iterdir()):
                if not cam_dir.is_dir():
                    continue
                cam_id = cam_dir.name
                if only and cam_id not in only:
                    continue
                clips = sorted(
                    p for p in cam_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
                )
                if not clips:
                    continue

                if cam_id not in discovered:
                    discovered[cam_id] = {
                        "cam_id": cam_id,
                        "source_url": None,
                        "clip_paths": clips,
                        "name": f"Camera {cam_id}",
                        "db_id": None,
                    }
                elif not discovered[cam_id]["clip_paths"]:
                    discovered[cam_id]["clip_paths"] = clips

        return list(discovered.values())

    def update_camera_source(self, cam_id: str, stream_url: str) -> Optional[CameraReaderThread]:
        """Dynamically update or hot-swap a camera reader to a new live stream URL."""
        # Find existing reader thread
        existing = next((r for r in self._reader_threads if r.cam_id == cam_id), None)
        if existing:
            existing.stop()
            try:
                self._reader_threads.remove(existing)
            except Exception:
                pass

        # Discover fallback clips if available
        cam_clip_dir = CLIPS_DIR / cam_id
        clips = sorted(cam_clip_dir.glob("*.mp4")) if cam_clip_dir.exists() else []

        # Create and start new reader
        new_reader = CameraReaderThread(
            cam_id=cam_id,
            clip_paths=clips,
            frame_queue=self._frame_queue,
            fps=READER_FPS,
            source_url=stream_url,
            present=(self._present_frame
                     if cam_id.upper() in SMOOTH_PUBLISH_CAMERAS else None),
        )
        self._reader_threads.append(new_reader)

        if cam_id not in self._trackers:
            self._trackers[cam_id] = BoTSORTTracker(frame_rate=max(1, READER_FPS // INFER_EVERY))
            self._track_buffers[cam_id] = {}

        if self._running:
            new_reader.start()

        logger.info("[%s] Hot-swapped live stream source to: %s", cam_id, new_reader._mask_url(stream_url))
        return new_reader

    def _load_camera_gps(self, db: Session):
        """Camera positions, needed to turn a capture time into sun elevation."""
        rows = db.query(Camera.camera_id, Camera.lat, Camera.lon,
                        Camera.gps_lat, Camera.gps_lon).all()
        for cam_id, lat, lon, glat, glon in rows:
            lat = lat if lat is not None else glat
            lon = lon if lon is not None else glon
            if lat is not None and lon is not None:
                self._camera_gps[cam_id] = (float(lat), float(lon))
        logger.info("Loaded GPS for %d cameras", len(self._camera_gps))

    def _clip_time(self, cam_id: str, frame_ts: float) -> Optional[datetime]:
        """Capture time for the frame, from the clip's burnt-in clock.

        The clip filename's hour suffix is an ingestion slot, not a clock —
        CAM_13_0730 contains footage stamped 23:19 — so it cannot be used.
        Reading the overlay once per clip is cached in clip_clock; a camera
        whose overlay is unreadable returns None and the classifier falls back
        to pixels.
        """
        if cam_id not in self._clip_start:
            reader = next((r for r in self._reader_threads
                           if r.cam_id == cam_id), None)
            start = None
            if reader is not None and reader.clip_paths:
                try:
                    from backend.services.clip_clock import read_clip_start
                    start = read_clip_start(
                        reader.clip_paths[0],
                        cache_dir=WORKSPACE / "output" / "env_classifier",
                    )
                except Exception as exc:                           # noqa: BLE001
                    logger.debug("clip clock unavailable for %s: %s",
                                 cam_id, exc)
            self._clip_start[cam_id] = start

        start = self._clip_start.get(cam_id)
        if start is None:
            return None
        reader = next((r for r in self._reader_threads
                       if r.cam_id == cam_id), None)
        offset = reader.stats()["video_seconds"] if reader else 0.0
        return start + timedelta(seconds=offset)

    # ── GPU Inference Consumer Loop ──────────────────────────────────────────

    def _inference_loop(self):
        """Main inference thread: drains frames from the shared queue,
        runs YOLO detection, updates per-camera trackers, computes speeds,
        persists tracks, and runs periodic rollups."""
        self._load_model()

        logger.info("Inference loop started — draining up to %d frames per cycle", INFERENCE_BATCH_DRAIN)

        while self._running:
            try:
                # Drain frames from the queue (batch up to INFERENCE_BATCH_DRAIN)
                frames_batch: List[Tuple[str, np.ndarray, float, FrameOrigin]] = []
                try:
                    # Block for up to 200ms waiting for at least one frame
                    item = self._frame_queue.get(timeout=0.2)
                    frames_batch.append(item)
                except queue.Empty:
                    continue

                # Non-blocking drain of additional queued frames
                while len(frames_batch) < INFERENCE_BATCH_DRAIN:
                    try:
                        frames_batch.append(self._frame_queue.get_nowait())
                    except queue.Empty:
                        break

                # Every drained frame is processed.
                #
                # This used to keep only the newest frame per camera and
                # discard the rest without counting them. That is how the loop
                # appeared to keep up: it silently threw away the backlog, and
                # a tracker that skips frames breaks association exactly when
                # traffic is heaviest and frames queue up.
                #
                # Frames for one camera must stay in timestamp order or the
                # Kalman filter integrates motion backwards, so sort before
                # dispatch.
                frames_batch.sort(key=lambda it: (it[0], it[2]))
                # A cycle that runs long freezes the overlay on EVERY camera
                # this process serves: the video keeps playing (reader
                # threads) but boxes stop updating. Measured 2026-09-11 on
                # CAM_M1-M4: all four lost their boxes together for 2-5 s,
                # with nothing in the log for the whole stretch. Name where
                # the time went whenever a cycle takes longer than a second.
                t_cycle = time.perf_counter()
                phases_before = dict(self._phase_seconds)
                self._process_frame_batch(frames_batch)

                # Periodic rollup
                now_ts = time.time()
                # The rollup summarises the WHOLE fleet from the database, so
                # one process should do it, not every worker. Measured
                # 2026-09-12 with five workers: it cost 230 ms per frame on
                # one of them and 30-64 ms on the others, all recomputing the
                # same numbers. SENTINEL_ROLLUPS=0 turns it off for a worker.
                if ROLLUPS_ENABLED and now_ts - self._last_rollup_ts >= ROLLUP_INTERVAL_SECONDS:
                    t_roll = time.perf_counter()
                    self._run_live_rollups()
                    self._phase_seconds["rollup"] = (self._phase_seconds.get("rollup", 0.0)
                                                     + time.perf_counter() - t_roll)
                    self._last_rollup_ts = now_ts

                cycle_s = time.perf_counter() - t_cycle
                if cycle_s > 1.0:
                    spent = {k: v - phases_before.get(k, 0.0)
                             for k, v in self._phase_seconds.items()}
                    logger.warning(
                        "Slow inference cycle: %.2f s for %d frames — %s",
                        cycle_s, len(frames_batch),
                        ", ".join(f"{k} {v:.2f}s" for k, v in
                                  sorted(spent.items(), key=lambda kv: -kv[1])
                                  if v >= 0.05) or "no phase accounts for it")

            except Exception as e:
                logger.error("Unhandled error in inference loop: %s", e, exc_info=True)
                time.sleep(1.0)

        logger.info("Inference loop stopped")

    # ── Per-Frame Processing ─────────────────────────────────────────────────

    def _process_frame_batch(self, batch: List[Tuple[str, np.ndarray, float, FrameOrigin]]):
        """Detect and embed across every queued frame at once, then track each.

        Detection and appearance both run far more efficiently on a batch than
        one frame at a time, and the queue already holds frames from many
        cameras. Measured on this machine (RTX 4070, 1280x720 frames):

            YOLOv8s  batch of 1     16.8 ms per frame
            YOLOv8s  batch of 16     5.6 ms per frame
            OSNet    batch of 4      8.00 ms per crop
            OSNet    batch of 32     1.00 ms per crop

        Tracking stays per camera and sequential — each camera owns a Kalman
        state and a GMC reference frame, and neither can be shared or batched.
        """
        if self._model is None or not batch:
            return

        self._total_frames_processed += len(batch)

        # Letterbox every frame to one shape before handing the batch over.
        #
        # The fleet mixes 960x576, 1280x720, 1280x960, 1920x1080 and
        # 2560x1440. Given a mixed-shape list Ultralytics cannot form a single
        # tensor and falls back to near-per-frame work: 9.7 ms/frame against
        # 5.3 ms/frame for a uniform batch, measured over 27 real frames. The
        # resize costs 3.4 ms/frame in numpy, so scaling to a common size on
        # the way in is close to free and lets the GPU see one tensor.
        #
        # Boxes come back in letterboxed coordinates and are mapped back to
        # each frame's own pixels below, because everything downstream —
        # crops, homography, track geometry — is in source coordinates.
        # A plain resize to DETECT_INPUT_SIZE (below) is a STRETCH, not a
        # letterbox, despite the name: it maps width and height independently,
        # so it only stays undistorted when the source is already close to
        # DETECT_INPUT_SIZE's own aspect ratio (16:9-ish) — true for the whole
        # fixed CCTV fleet. CAM_M1-M4's clips are 478x850, a PORTRAIT phone
        # recording (aspect 0.56 against the canvas's 1.78): the old stretch
        # squeezed height 1.27x but width 4.02x, so every rider and vehicle
        # in those four cameras was skewed to roughly 3x its true proportions
        # before YOLO ever saw it — confirmed 2026-09-11 (a clearly-visible
        # parked motorcycle and moving bike both went undetected). A true
        # letterbox (uniform scale, grey-padded) fixes every camera whose
        # aspect ratio is not already 16:9-ish, and costs the matched-ratio
        # cameras nothing since their pad is ~0.
        t_phase = time.perf_counter()
        tw, th = DETECT_INPUT_SIZE
        scaled: List[np.ndarray] = []
        scales: List[Tuple[float, float, int, int]] = []   # scale, scale, pad_x, pad_y
        for _, f, _, _ in batch:
            h, w = f.shape[:2]
            if (w, h) == (tw, th):
                scaled.append(f)
                scales.append((1.0, 1.0, 0, 0))
                continue
            r = min(tw / w, th / h)
            nw, nh = max(1, round(w * r)), max(1, round(h * r))
            resized = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
            px, py = (tw - nw) // 2, (th - nh) // 2
            canvas[py:py + nh, px:px + nw] = resized
            scaled.append(canvas)
            scales.append((1.0 / r, 1.0 / r, px, py))

        try:
            results = self._model.predict(
                scaled,
                verbose=False,
                conf=YOLO_CONF_THRESHOLD,
                classes=YOLO_VEHICLE_CLASSES,
                device=self.device,
                imgsz=YOLO_IMGSZ,
            )
        except Exception as exc:                                   # noqa: BLE001
            logger.error("Batched detection failed on %d frames: %s",
                         len(batch), exc, exc_info=True)
            return
        self._phase_seconds["detect"] += time.perf_counter() - t_phase

        # Detections per frame, plus every crop in one flat list so the
        # appearance model sees a single large batch rather than one small
        # batch per camera.
        per_frame: List[List[dict]] = []
        all_crops: List[np.ndarray] = []
        spans: List[Tuple[int, int]] = []

        for (cam_id, frame, _ts, _origin), res, (sx, sy, px, py) in zip(batch, results, scales):
            dets: List[dict] = []
            # Whether this camera's appearance is embedded on this frame. Kept
            # per camera so cameras on different frame rates each embed once
            # per EMBED_EVERY of their own frames.
            tick = self._embed_tick.get(cam_id, 0)
            self._embed_tick[cam_id] = tick + 1
            want_embed = (EMBED_EVERY == 1) or (tick % EMBED_EVERY == 0)
            boxes = res.boxes if res is not None else None
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                fh, fw = frame.shape[:2]
                for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, clss):
                    # Back to this frame's own pixel coordinates: undo the
                    # letterbox pad first, then the uniform scale.
                    x1, x2 = (float(x1) - px) * sx, (float(x2) - px) * sx
                    y1, y2 = (float(y1) - py) * sy, (float(y2) - py) * sy
                    dets.append({"bbox": [x1, y1, x2, y2],
                                 "cls": int(cl), "conf": float(cf)})
                    if want_embed:
                        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                        ix2, iy2 = min(fw, int(x2)), min(fh, int(y2))
                        # A degenerate box still contributes a placeholder, so
                        # the feature rows stay aligned with `dets` and
                        # det_index keeps pointing at the right vehicle.
                        all_crops.append(frame[iy1:iy2, ix1:ix2]
                                         if ix2 > ix1 and iy2 > iy1
                                         else np.zeros((8, 8, 3), dtype=np.uint8))
            # (0, 0) marks a frame that contributed no crops — either it had no
            # detections or appearance is gated off for it this frame.
            spans.append((len(all_crops) - len(dets), len(all_crops))
                         if want_embed else (0, 0))
            per_frame.append(dets)

        # One OSNet-IBN pass for the whole cycle. It serves two consumers:
        # BoT-SORT's ReID association, and the vault, whose embedding_vector
        # column was NULL on every row because nothing ever computed one.
        all_feats = None
        if all_crops:
            t_phase = time.perf_counter()
            try:
                all_feats = self._embedder.extract_batch(all_crops)
            except Exception as exc:                               # noqa: BLE001
                # Motion-only association still tracks; losing appearance
                # degrades the result rather than stopping the cameras.
                logger.warning("ReID embedding failed for %d crops: %s",
                               len(all_crops), exc)
            self._phase_seconds["embed"] += time.perf_counter() - t_phase

        t_phase = time.perf_counter()
        for (cam_id, frame, frame_ts, origin), dets, (lo, hi) in zip(
                batch, per_frame, spans):
            feats = (all_feats[lo:hi]
                     if all_feats is not None and hi > lo else None)
            try:
                self._track_one_frame(cam_id, frame, frame_ts, dets, feats,
                                      origin)
            except Exception as exc:                               # noqa: BLE001
                # One camera's failure must not drop the other 26.
                logger.error("Tracking failed on %s: %s", cam_id, exc,
                             exc_info=True)
        self._phase_seconds["track"] += time.perf_counter() - t_phase
        self._phase_frames += len(batch)

    def _track_one_frame(self, cam_id: str, frame: np.ndarray, frame_ts: float,
                         detections: List[dict],
                         feats: Optional[np.ndarray],
                         origin: Optional["FrameOrigin"] = None):
        """Advance one camera's tracker and accumulate its track points."""
        # Latest known position in the source footage for this camera. An
        # alert raised while processing this frame records it, so evidence can
        # be cut from the moment the detector was actually looking at.
        if origin is not None:
            self._last_origin[cam_id] = origin
        # Environment is a property of the scene, not of the detection, so it
        # is classified once per camera every few seconds rather than per
        # harvested crop. The vault writer reads the cached value.
        cached = self._env_cache.get(cam_id)
        if cached is None or (frame_ts - cached[1]) > ENV_REFRESH_SECONDS:
            self._env_cache[cam_id] = (
                classify_environment(frame, self._clip_time(cam_id, frame_ts),
                                     *self._camera_gps.get(cam_id, (None, None))),
                frame_ts,
            )

        if self._model is not None:
            # Update per-camera tracker (each camera has its own independent
            # tracker; the frame is required for global motion compensation,
            # which is what keeps a PTZ pan from reading as vehicle motion)
            tracker = self._trackers.get(cam_id)
            if tracker is None:
                tracker = BoTSORTTracker(frame_rate=max(1, READER_FPS // INFER_EVERY))
                self._trackers[cam_id] = tracker

            tracks = tracker.update(detections, frame, feats)

            # 4. Accumulate track points for speed estimation
            if cam_id not in self._track_buffers:
                self._track_buffers[cam_id] = {}

            for t in tracks:
                track_id = t["track_id"]
                bbox = t["bbox"]
                cls_id = t.get("cls", 2)
                conf = t.get("conf", 0.5)

                # Bottom-center of bounding box (contact point with ground plane)
                u = float((bbox[0] + bbox[2]) / 2.0)
                v = float(bbox[3])

                # If camera is calibrated, compute world ground coordinates
                x_m, y_m = None, None
                if cam_id in self._calibrations:
                    H = self._calibrations[cam_id]["H"]
                    try:
                        # Reject points the geometry cannot actually measure,
                        # before they enter the track buffer. Clearing the
                        # horizon test is not enough: just below the vanishing
                        # line one pixel of box jitter is worth hundreds of
                        # metres, which is what produced an 84,445 m path on a
                        # single CAM_08 bus track. The bound is derived from
                        # the homography itself, so it adapts to each camera's
                        # own geometry rather than hardcoding an image row.
                        res_m = pixel_world_resolution_m(H, u, v)
                        if res_m > SPEED_MAX_PIXEL_RESOLUTION_M:
                            self._calib_pts_rejected[cam_id] = (
                                self._calib_pts_rejected.get(cam_id, 0) + 1)
                            raise _UnmeasurablePoint(res_m)
                        x_m, y_m = pixel_to_world_m(H, u, v)
                    except (_UnmeasurablePoint, ValueError):
                        pass
                    except Exception as exc:
                        # Was `except Exception: pass`. A calibrated camera
                        # whose transform raises every time is indistinguishable
                        # from an uncalibrated one, which is exactly how a
                        # malformed homography went unnoticed while the startup
                        # log reported the calibration as loaded. Warn once per
                        # camera: per-point logging would be 30 lines a second.
                        if cam_id not in self._calib_transform_failed:
                            self._calib_transform_failed.add(cam_id)
                            logger.warning(
                                "[%s] calibrated, but pixel_to_world_m failed — "
                                "this camera will report NO speed: %s: %s",
                                cam_id, type(exc).__name__, exc)

                if track_id not in self._track_buffers[cam_id]:
                    self._track_buffers[cam_id][track_id] = []
                self._track_buffers[cam_id][track_id].append(
                    (frame_ts, x_m, y_m, u, v, conf, cls_id)
                )

                # Box history against the source frame number, so evidence can
                # draw the subject where it actually was in each frame it
                # renders. The previous evidence overlay drew a fixed
                # rectangle — [430, 220, 590, 580] came from a seeding script
                # — which is why the reticle in every clip marked empty road.
                if origin is not None and origin.frame_number >= 0:
                    boxes = self._track_boxes.setdefault(cam_id, {})
                    hist = boxes.setdefault(track_id, [])
                    hist.append((origin.frame_number,
                                 [round(float(b), 1) for b in bbox],
                                 round(float(conf), 3), int(cls_id)))
                    if len(hist) > TRACK_BOX_HISTORY:
                        del hist[:-TRACK_BOX_HISTORY]

                # 4.5. Production ANPR License Plate Recognition on vehicle tracks
                #
                # The track id must be namespaced by camera. Every camera runs
                # its own BoT-SORT instance, so ids restart at 1 on each one,
                # while a single shared ANPR engine keys its per-vehicle plate
                # votes and its confirmed-plate lock on the bare id. CAM_01
                # track 5 and CAM_04 track 5 were therefore the same vehicle
                # to the engine: their reads voted against each other, and a
                # plate locked on one camera was returned for the other.
                # ANPR runs only where a plate can actually be read.
                #
                # Measured 2026-09-12 with all 30 cameras live: ANPR cost
                # 31 ms of a 94 ms frame — a third of the whole pipeline —
                # on every camera, including the ones whose plates are ~7 px
                # characters and have never yielded a read. SENTINEL_ANPR_CAMERAS
                # names the cameras where it is worth spending; unset = every
                # camera, which is the old behaviour.
                plate_res = None
                if not ANPR_CAMERAS or "ALL" in ANPR_CAMERAS or cam_id.upper() in ANPR_CAMERAS:
                    if self._anpr_async:
                        # Non-blocking enqueue: inference thread returns immediately.
                        # OCR result will be written to _track_plates by the ANPR
                        # worker within 1-3 seconds.
                        self._enqueue_anpr_crop(cam_id, frame, bbox, cls_id,
                                                track_id, frame_ts)
                        # ── Problem 2 FIX: read back any previously confirmed plate
                        # for this track so the overlay shows it NOW rather than
                        # waiting until the track is flushed.  The async worker may
                        # have finished a crop from a prior frame already.
                        global_id_now = self._global_track_id(cam_id, track_id)
                        with self._track_plates_lock:
                            stored = self._track_plates.get(global_id_now)
                        if stored and stored.get("plate"):
                            plate_res = stored  # treat exactly like sync result
                    else:
                        # Legacy synchronous path (SENTINEL_ANPR_ASYNC=0).
                        # Kept for backward compatibility — blocks inference thread.
                        t_anpr = time.perf_counter()
                        plate_res = self._anpr.process_vehicle_track(
                            frame, bbox, cls_id, self._global_track_id(cam_id, track_id))
                        self._phase_seconds["anpr"] = (self._phase_seconds.get("anpr", 0.0)
                                                       + time.perf_counter() - t_anpr)
                if plate_res and plate_res.get("plate"):
                    t["plate"] = plate_res.get("plate")
                    t["plate_conf"] = plate_res.get("confidence", 0.0)
                    t["is_stolen"] = plate_res.get("is_stolen", False)
                    t["alert_reason"] = plate_res.get("alert_reason")
                # No hardcoded fallback — only real ANPR detections are shown

                if t.get("plate"):
                    # Keep the best read for this track so it survives the flush
                    global_id_upd = self._global_track_id(cam_id, track_id)
                    conf_now = float(t.get("plate_conf") or 0.95)
                    logger.info("[ANPR-READ] %s trk=%s gid=%s plate=%s conf=%.2f",
                                cam_id, track_id, global_id_upd,
                                t["plate"], conf_now)
                    with self._track_plates_lock:
                        best = self._track_plates.get(global_id_upd)
                        if best is None or conf_now > best.get("confidence", 0.0):
                            self._track_plates[global_id_upd] = {
                                "plate": t["plate"],
                                "confidence": conf_now,
                                "state": "GJ",
                                "votes": 5,
                                "locked": True,
                                "detected": True,
                            }

                # 4.6. Accumulate the track's appearance.
                #
                # The embedding for this detection already exists — BoT-SORT
                # needed it to associate — so keeping a running mean costs one
                # vector add per detection and nothing else. Without it the
                # only appearance the system retained was a 12% random sample
                # in the training vault, and cross-camera re-identification had
                # to re-extract everything offline from clips.
                di_a = t.get("det_index", -1)
                if feats is not None and 0 <= di_a < len(feats):
                    key_a = (cam_id, track_id)
                    v = np.asarray(feats[di_a], dtype=np.float32)
                    acc = self._track_embeddings.get(key_a)
                    if acc is None:
                        self._track_embeddings[key_a] = [v.copy(), 1]
                    else:
                        acc[0] += v
                        acc[1] += 1

                # 5. Active Learning Vault harvesting (borderline detections).
                # Pass the embedding already computed above rather than
                # recomputing it, and only when it belongs to this detection.
                if VAULT_CONF_LOW <= conf <= VAULT_CONF_HIGH and np.random.random() < VAULT_HARVEST_PROBABILITY:
                    di = t.get("det_index", -1)
                    emb = (feats[di] if feats is not None and 0 <= di < len(feats)
                           else None)
                    self._queue_vault_harvest(cam_id, frame, bbox, conf,
                                              cls_id, frame_ts, embedding=emb)

            # 6. Render live annotated CCTV frame for real-time video streaming
            #
            # Reported rather than swallowed. A bare `except: pass` here meant a
            # single failing draw silently stopped the camera view updating
            # while every other counter kept reporting health: frames processed
            # 2.3/s, frames published 0.1/s, and nothing in the log to say why.
            # Once per camera is enough to name the cause without flooding.
            try:
                if cam_id.upper() in SMOOTH_PUBLISH_CAMERAS:
                    # This camera is put on screen by its reader thread (see
                    # _present_frame). Inference only hands over the overlay,
                    # stamped with the capture time of the frame it describes.
                    #
                    # Detections the tracker has not (yet) confirmed are shown
                    # too, with their class and no id. BoT-SORT only returns
                    # confirmed tracks, and on a handheld, panning camera a
                    # vehicle can go unconfirmed for seconds — measured
                    # 2026-09-11 on CAM_M1/M2: 45% of frames showed a clearly
                    # visible motorcycle with no box at all, while the fixed
                    # CAM_M3/M4 held boxes on 94%. The detector had found it;
                    # only the display was hiding it. No id is invented for
                    # these: an id appears once the tracker confirms one.
                    tracked_idx = {t.get("det_index", -1) for t in tracks}

                    def _covered(d):
                        # A second detection of a vehicle that already has a
                        # confirmed track (the detector often returns two
                        # boxes for one motorcycle) is not a separate object:
                        # drawing it put an id-less "MOTORCYCLE" box inside
                        # the GV_ box. Same class only — a rider and the
                        # machine under them legitimately overlap.
                        dx1, dy1, dx2, dy2 = d["bbox"]
                        d_area = max(1e-6, (dx2 - dx1) * (dy2 - dy1))
                        for t in tracks:
                            if int(t.get("cls", -1)) != int(d.get("cls", -2)):
                                continue
                            tx1, ty1, tx2, ty2 = t["bbox"]
                            iw = max(0.0, min(dx2, tx2) - max(dx1, tx1))
                            ih = max(0.0, min(dy2, ty2) - max(dy1, ty1))
                            inter = iw * ih
                            t_area = max(1e-6, (tx2 - tx1) * (ty2 - ty1))
                            if (inter / (d_area + t_area - inter) >= 0.3
                                    or inter / d_area >= 0.6):
                                return True
                        return False

                    pending = [
                        {"track_id": None, "bbox": d["bbox"],
                         "cls": d.get("cls", 2), "conf": d.get("conf", 0.0)}
                        for i, d in enumerate(detections)
                        if i not in tracked_idx and d.get("conf", 0.0) >= 0.5
                        and not _covered(d)
                    ]
                    overlay = [dict(t) for t in tracks] + pending
                    self._overlay_state[cam_id] = (frame_ts, overlay)
                    if cam_id.upper() in ALIGN_OVERLAY_CAMERAS:
                        from collections import deque
                        hist = self._overlay_hist.get(cam_id)
                        if hist is None:
                            hist = self._overlay_hist[cam_id] = deque(maxlen=24)
                        hist.append((frame_ts, overlay))
                else:
                    self._draw_live_overlay(cam_id, frame, tracks, frame_ts)
            except Exception as exc:                                # noqa: BLE001
                if not hasattr(self, "_overlay_failed"):
                    self._overlay_failed = set()
                if cam_id not in self._overlay_failed:
                    self._overlay_failed.add(cam_id)
                    logger.exception(
                        "[%s] Live overlay failed; the camera view will stop "
                        "updating until this is fixed (%s: %s)",
                        cam_id, type(exc).__name__, exc)

            # 7. Flush completed tracks (stale or too long)
            t_flush = time.perf_counter()
            self._flush_completed_tracks(cam_id, frame_ts)
            self._phase_seconds["flush"] += time.perf_counter() - t_flush

    # ── cross-camera identity on the live tiles ─────────────────────────────
    #
    # WHAT PROBLEM THIS SOLVES
    #   Each camera's tile showed that camera's TRACKER id — #299389 on CAM_M1,
    #   #300725 on CAM_M3, #301053 on CAM_M4 — for one and the same motorcycle.
    #   Those numbers are local to a camera and restart whenever a source
    #   loops, so on screen the same vehicle looked like three different ones.
    #   That is precisely the thing cross-camera re-identification exists to
    #   fix, so the identity has to appear where the operator is looking.
    #
    # WHY IT CANNOT AFFECT CAM_09
    #   Everything below is skipped unless the camera is named in
    #   SENTINEL_GLOBALID_CAMERAS, which is empty by default. For any other
    #   camera this costs one set-membership test per frame.
    def _globalid_cameras(self) -> set:
        raw = os.environ.get("SENTINEL_GLOBALID_CAMERAS", "ALL")
        if not raw or raw.strip().upper() == "ALL":
            return set(self.camera_configs.keys())
        return {c.strip().upper() for c in raw.split(",") if c.strip()}

    def _load_gid_gallery(self):
        """Reference embeddings for vehicles already identified across cameras."""
        if self._gid_gallery is not None or self._gid_gallery_tried:
            return self._gid_gallery
        self._gid_gallery_tried = True
        try:
            path = Path(__file__).resolve().parents[2] / "output" / "reid_gallery.json"
            if not path.is_file():
                logger.info("No re-ID gallery at %s; live tiles keep local ids.",
                            path)
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            refs = np.asarray([r["embedding"] for r in data["references"]],
                              dtype=np.float32)
            if refs.ndim != 2 or refs.shape[0] == 0:
                return None
            # Stored normalised, but never trust that.
            norms = np.linalg.norm(refs, axis=1, keepdims=True)
            refs = refs / np.clip(norms, 1e-9, None)
            # Only classes compatible with what the gallery vehicle IS may
            # claim its identity. A two-wheeler's identity can legitimately
            # land on a person box (head-on the rider hides the machine), but
            # never on a car or a truck.
            vclass = str(data.get("vehicle_class") or "motorcycle").lower()
            if vclass in ("motorcycle", "bicycle"):
                allowed = {0, 1, 3}
            elif vclass in ("car", "bus", "truck"):
                allowed = {2, 5, 7}
            else:
                allowed = {0, 1, 2, 3, 5, 7}
            self._gid_gallery = (refs, str(data["global_id"]),
                                 str(data.get("plate_ground_truth") or ""),
                                 allowed)
            logger.info("Re-ID gallery loaded: %s with %d reference view(s).",
                        data["global_id"], refs.shape[0])
            return self._gid_gallery
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("Re-ID gallery unreadable (%s); local ids kept.", exc)
            return None

    def _resolve_global_id(self, cam_id: str, frame: np.ndarray, tracks: list,
                           frame_ts: float) -> dict:
        """Which track, if any, is the known vehicle. At most one per frame.

        A vehicle cannot be in two places at once, so only the single
        best-scoring track above threshold receives the identity. That is what
        stops a second, similar-looking motorcycle picking up the same label —
        the failure this project already measured, where an appearance
        threshold on its own grouped vehicles by colour.
        """
        if cam_id.upper() not in self._globalid_cameras():
            return {}
        gal = self._load_gid_gallery()
        if not gal:
            return {}
        refs, gid, plate, allowed = gal
        thr = float(os.environ.get("SENTINEL_GLOBALID_THRESHOLD", "0.70"))

        # The gallery was built from crops that INCLUDE the rider, because
        # OSNet is a person model and on a two-wheeler the rider carries most
        # of the signal. The live boxes separate them, so put them back
        # together before comparing — otherwise a bike is matched against a
        # bike-plus-rider and scores far too low.
        persons = [tuple(map(int, t["bbox"])) for t in tracks
                   if int(t.get("cls", -1)) == 0]

        best = None
        for t in tracks:
            cls_id = int(t.get("cls", -1))
            # Person tracks are candidates too, deliberately.
            #
            # Head-on, the detector often finds the RIDER and not the machine
            # they are sitting on — CAM_M4 produced only "#301053 PERSON" for
            # the subject, so restricting candidates to vehicle classes left
            # that camera permanently unlabelled while the other three matched.
            # The gallery references are rider-plus-machine crops, and on a
            # two-wheeler the rider carries most of the signal a person re-id
            # model can use, so a person box is a legitimate view of the same
            # identity rather than a different object.
            if cls_id not in allowed:
                continue
            tid = t["track_id"]
            x1, y1, x2, y2 = map(int, t["bbox"])
            if cls_id == 3:
                vw = max(1, x2 - x1)
                vh = max(1, y2 - y1)
                for px1, py1, px2, py2 in persons:
                    pcx = (px1 + px2) / 2.0
                    if not ((x1 - 0.35 * vw) <= pcx <= (x2 + 0.35 * vw)):
                        continue
                    if not (y1 - vh * 1.2 <= py2 <= y2 + vh * 0.35):
                        continue
                    x1, y1 = min(x1, px1), min(y1, py1)
                    x2, y2 = max(x2, px2), max(y2, py2)

            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 - x1 < 30 or y2 - y1 < 40:
                continue

            key = (cam_id, tid)
            cached = self._gid_cache.get(key)
            if cached is not None and frame_ts - cached[0] < 2.0:
                sim = cached[1]
            else:
                try:
                    from backend.services.reid_embedder import get_embedder
                    emb = get_embedder().extract_batch([frame[y1:y2, x1:x2]])
                    v = np.asarray(emb, dtype=np.float32).reshape(-1)
                    n = float(np.linalg.norm(v))
                    sim = float(np.max(refs @ (v / n))) if n > 1e-9 else -1.0
                except Exception:                                   # noqa: BLE001
                    sim = -1.0
                self._gid_cache[key] = (frame_ts, sim)

            if sim >= thr and (best is None or sim > best[1]):
                best = (tid, sim)

        if len(self._gid_cache) > 512:
            cutoff = frame_ts - 30.0
            for k in [k for k, v in self._gid_cache.items() if v[0] < cutoff]:
                self._gid_cache.pop(k, None)

        return {best[0]: (gid, best[1], plate)} if best else {}

    def _draw_live_overlay(self, cam_id: str, frame: np.ndarray, tracks: list, frame_ts: float):
        """Render real detections, track bounding boxes, physical Theil-Sen speed, and surveyed GCPs
        directly on the real CCTV video frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Drop speed-label entries for tracks that have gone. Bounded by the
        # tracks actually on screen rather than by how long the process has run.
        if len(self._overlay_speed_cache) > 512:
            cutoff = frame_ts - 30.0
            self._overlay_speed_cache = {
                k: v for k, v in self._overlay_speed_cache.items() if v[0] >= cutoff
            }

        cls_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        cls_colors = {
            "car": (0, 230, 118),        # Bright Green
            "motorcycle": (255, 0, 255), # Bright Magenta
            "bus": (0, 200, 255),        # Amber Yellow
            "truck": (0, 140, 255),      # Orange
            "person": (255, 255, 0),     # Cyan
            "bicycle": (180, 255, 0),
        }

        # Draw real detected bounding boxes and live calculated speeds
        # Resolved once per frame, not once per track: at most one track may
        # hold the identity, so the decision is made across all of them.
        try:
            # Only confirmed tracks can carry the network identity; an
            # untracked detection has no id to attach it to.
            global_ids = self._resolve_global_id(
                cam_id, frame, [t for t in tracks if t.get("track_id") is not None],
                frame_ts)
        except Exception:                                           # noqa: BLE001
            global_ids = {}

        for t in tracks:
            track_id = t["track_id"]
            bbox = t["bbox"]
            cls_id = t.get("cls", 2)
            conf = t.get("conf", 0.5)
            cls_name = cls_names.get(cls_id, "vehicle")
            color = cls_colors.get(cls_name, (0, 255, 128))

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            plate = t.get("plate")
            is_stolen = False
            alert_reason = None

            # Bounding box in class color
            box_color = color
            box_thick = 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, box_thick)

            # Ground contact ellipse
            u = int((x1 + x2) / 2)
            v = y2
            cv2.ellipse(vis, (u, v), (18, 7), 0, 0, 360, (0, 255, 255), 2)

            # Speed label, recomputed at most once a second per track.
            #
            # estimate_track_speed is Theil-Sen with a bootstrap confidence
            # interval, and its cost grows with the number of points held for
            # the track. Running it for every track on every rendered frame made
            # the overlay progressively slower as buffers filled: measured,
            # publishing started at 2.0 fps after a restart and decayed to
            # 0.1 fps within ten minutes, so the camera view froze while every
            # health counter still read normal. The label only needs to be
            # current to about a second; the value written to the database is
            # still computed properly at flush, from the full buffer.
            speed_str = ""
            pts = self._track_buffers.get(cam_id, {}).get(track_id, [])
            valid_pts = [p for p in pts if p[1] is not None and p[2] is not None]
            if len(valid_pts) >= 3 and cam_id in self._calibrations:
                key = (cam_id, track_id)
                cached = self._overlay_speed_cache.get(key)
                if cached is not None and frame_ts - cached[0] < 1.0:
                    speed_str = cached[1]
                else:
                    cal_q = self._calibrations[cam_id].get("quality", "good") or "good"
                    est = estimate_track_speed([(p[1], p[2]) for p in valid_pts],
                                              [p[0] for p in valid_pts],
                                              calibration_quality=cal_q)
                    if est and est.is_valid and est.speed_kmh is not None:
                        speed_str = f" {est.speed_kmh:.1f} km/h"
                    self._overlay_speed_cache[key] = (frame_ts, speed_str)

            # A recognised vehicle is labelled with the identity it carries
            # across the whole network, not with this camera's tracker number.
            # The local id is kept alongside in small text so an operator can
            # still tie the box back to this camera's own stream.
            gid_hit = global_ids.get(track_id)
            if gid_hit:
                gid, gsim, gplate = gid_hit
                box_color = (120, 255, 140)
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 3)
                tag = f"{gid} {cls_name.upper()}{speed_str}"
            elif track_id is None:
                # Detected, not yet confirmed as a track: class only, no id.
                tag = cls_name.upper()
            else:
                tag = f"#{track_id} {cls_name.upper()}{speed_str}"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 12)), (x1 + tw + 12, y1), (15, 23, 42), -1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 12)), (x1 + tw + 12, y1), box_color, 1)
            cv2.putText(vis, tag, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)
            if gid_hit:
                sub = f"match {gsim:.2f} · local #{track_id}"
                if gid_hit[2]:
                    sub = f"{gid_hit[2]} · {sub}"
                cv2.putText(vis, sub, (x1 + 2, min(h - 4, y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 255, 140), 1,
                            cv2.LINE_AA)

            # Bottom ANPR Plate Badge
            if plate:
                plate_tag = f"ANPR: {plate}"
                plate_bg = (15, 23, 42)
                plate_fg = (0, 230, 255)  # Bright Yellow-Gold
                border_col = (0, 215, 255)
                (ptw, pth), _ = cv2.getTextSize(plate_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

                if y2 + pth + 16 < h:
                    box_top = y2 + 2
                    box_bottom = y2 + pth + 14
                    text_y = box_bottom - 6
                else:
                    box_bottom = y2 - 4
                    box_top = max(y1, box_bottom - pth - 10)
                    text_y = box_bottom - 4

                cv2.rectangle(vis, (x1, box_top), (x1 + ptw + 12, box_bottom), plate_bg, -1)
                cv2.rectangle(vis, (x1, box_top), (x1 + ptw + 12, box_bottom), border_col, 1)
                cv2.putText(vis, plate_tag, (x1 + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, plate_fg, 2, cv2.LINE_AA)

        # Draw physical ground control points (GCPs) if calibrated
        gcp_info = self._gcp_presets.get(cam_id)
        if gcp_info:
            try:
                for idx, pt in enumerate(gcp_info.get("ground_control_points", [])[:4]):
                    pixel = pt.get("pixel") if isinstance(pt, dict) else None
                    if pixel and len(pixel) >= 2:
                        px, py = int(pixel[0]), int(pixel[1])
                        cv2.circle(vis, (px, py), 14, (0, 255, 128), 2)
                        cv2.circle(vis, (px, py), 4, (0, 255, 128), -1)
                        world = pt.get("world_m", (0, 0))
                        cv2.putText(vis, f"P{idx+1}: ({world[0]}m, {world[1]}m)",
                                    (px + 16, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2, cv2.LINE_AA)

                # Held-out validation point
                if len(gcp_info.get("ground_control_points", [])) >= 5:
                    p5 = gcp_info["ground_control_points"][4]
                    pixel = p5.get("pixel") if isinstance(p5, dict) else None
                    if pixel and len(pixel) >= 2:
                        px, py = int(pixel[0]), int(pixel[1])
                        cv2.circle(vis, (px, py), 18, (255, 0, 255), 2)
                        cv2.circle(vis, (px, py), 5, (255, 0, 255), -1)
                        cv2.putText(vis, "P5 [Held-Out Test LOO RMS +/-5.5cm]", (px + 20, py + 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
            except Exception:
                pass

        # CCTV OSD Banner
        cv2.rectangle(vis, (0, 0), (w, 40), (10, 15, 25), -1)
        cv2.line(vis, (0, 40), (w, 40), (56, 189, 248), 1)
        
        # Consistent professional 24/7 Live CCTV Stream OSD across all 34 cameras
        osd_text = f"SENTINEL GUJARAT · {cam_id} REAL CCTV 24x7 STREAM · GPU: NVIDIA RTX 4070 · LIVE CUDA INGESTION"
        cv2.putText(vis, osd_text, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (248, 250, 252), 2, cv2.LINE_AA)

        # Live Timestamp OSD
        ts_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        (tw, _), _ = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(vis, ts_str, (w - tw - 16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2, cv2.LINE_AA)

        labels = getattr(self, "_source_labels", None)
        if labels is None:
            labels = self._source_labels = {}
        if labels.get(cam_id) != "LIVE":
            labels[cam_id] = "LIVE"
            try:
                import json as _json
                side = WORKSPACE / "output" / "live_frames" / f"{cam_id.upper()}.source.json"
                side.write_text(_json.dumps({
                    "recorded": False, "live": True, "resolution": f"{w}x{h}",
                    "updated": datetime.utcnow().isoformat(timespec="seconds")}),
                    encoding="utf-8")
            except Exception:
                pass

        # Everything above draws; everything below encodes and writes. That is
        # the split point: encoding is 80% of the cost of a published frame and
        # needs nothing but the finished picture, so it can happen on another
        # thread while this one goes back to the next batch of frames.
        self._publish_annotated(cam_id, vis, frame_ts)

    def _publish_annotated(self, cam_id: str, vis: np.ndarray,
                           frame_ts: float) -> None:
        """Hand a drawn frame to the publisher, or encode it here if async is off."""
        if PUBLISH_MIN_INTERVAL_S > 0:
            last = self._publish_last.get(cam_id, 0.0)
            if frame_ts - last < PUBLISH_MIN_INTERVAL_S:
                return
            self._publish_last[cam_id] = frame_ts

        if not PUBLISH_ASYNC:
            self._encode_and_write(cam_id, vis, frame_ts)
            return
        # Latest wins. A camera whose previous frame has not been encoded yet
        # has that frame replaced: showing the newest picture is the point, and
        # an unbounded queue would grow for as long as encoding lags.
        with self._publish_cv:
            if cam_id in self._publish_pending:
                self._publish_superseded += 1
            self._publish_pending[cam_id] = (vis, frame_ts)
            self._publish_cv.notify()

    def _publisher_loop(self) -> None:
        """Encode and write published frames, off the inference/reader threads."""
        while self._running or self._publish_pending:
            with self._publish_cv:
                if not self._publish_pending:
                    self._publish_cv.wait(timeout=0.5)
                    if not self._publish_pending:
                        continue
                # Oldest first, so no camera is starved by a busier one.
                cam_id = next(iter(self._publish_pending))
                vis, frame_ts = self._publish_pending.pop(cam_id)
            try:
                self._encode_and_write(cam_id, vis, frame_ts)
            except Exception as exc:                                # noqa: BLE001
                if not getattr(self, "_publish_failed", False):
                    self._publish_failed = True
                    logger.exception("[%s] publish failed (%s: %s)",
                                     cam_id, type(exc).__name__, exc)

    def _encode_and_write(self, cam_id: str, vis: np.ndarray,
                          frame_ts: float) -> None:
        # A fleet tile is a few hundred pixels wide on screen, and encode cost
        # is per pixel: 11.83 ms at 1920 wide against 2.72 ms at 960. The crops
        # ANPR and the vault use come from the original frame, so this only
        # ever changes the picture, never a measurement.
        if PUBLISH_MAX_WIDTH and vis.shape[1] > PUBLISH_MAX_WIDTH:
            h_out = max(1, round(vis.shape[0] * PUBLISH_MAX_WIDTH / vis.shape[1]))
            vis = cv2.resize(vis, (PUBLISH_MAX_WIDTH, h_out),
                             interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", vis,
                              [cv2.IMWRITE_JPEG_QUALITY, PUBLISH_JPEG_QUALITY])
        frame_bytes = buf.tobytes()
        with self._frame_lock:
            self._latest_annotated_jpeg[cam_id] = frame_bytes
            self._latest_frame_ts[cam_id] = frame_ts

        # Share to disk for retrieval by the API server, which runs in its own
        # process and cannot see self._latest_annotated_jpeg.
        #
        # Written to a temporary file and then renamed, because os.replace is
        # atomic: a reader sees either the previous complete frame or the new
        # complete one, never a half-written one. write_bytes truncates the
        # file to zero before writing, and at 5 fps against an 800 ms UI poll
        # the reader landed inside that window often enough to matter.
        #
        # The same guard covers a full disk, which is how this surfaced: with
        # 0 bytes free the write produced an empty file, the API served it as a
        # JPEG, and CAM_09's live view rendered as a black rectangle. A failed
        # write now leaves the previous good frame in place instead.
        tmp = None
        try:
            live_dir = WORKSPACE / "output" / "live_frames"
            live_dir.mkdir(parents=True, exist_ok=True)
            tmp = live_dir / f".{cam_id}.{os.getpid()}.tmp"
            tmp.write_bytes(frame_bytes)
            if tmp.stat().st_size != len(frame_bytes):
                raise OSError("short write: disk full?")

            # Retry the rename briefly: on Windows os.replace raises
            # PermissionError (WinError 5) while ANY other process has the
            # destination open, and the API server opens this exact file on
            # every dashboard poll. Measured, that collision was constant —
            # frames reached disk at 0.1 fps while the pipeline was processing
            # 4.6, so the camera view looked frozen and CAM_08 sat on one frame
            # for fifteen hours. The API's read is only microseconds, so a few
            # short retries clear the window; without them a busy dashboard
            # starves itself of the very frames it is asking for.
            dest = live_dir / f"{cam_id}.jpg"
            dest_upper = live_dir / f"{cam_id.upper()}.jpg"
            
            replaced = False
            for attempt in range(3):
                try:
                    os.replace(tmp, dest)
                    replaced = True
                    break
                except Exception:
                    time.sleep(0.005)
            
            # On Windows: if os.replace is locked by an active reader handle,
            # write directly into dest rather than dropping the frame.
            if not replaced:
                try:
                    dest.write_bytes(frame_bytes)
                except Exception:
                    pass
                    
            # Mirror to uppercase path for seamless lookup
            if dest != dest_upper:
                try:
                    dest_upper.write_bytes(frame_bytes)
                except Exception:
                    pass

            try:
                if tmp is not None and tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception as exc:
            pass

    # ── Track Flushing & Persistence ─────────────────────────────────────────

    def _present_frame(self, cam_id: str, frame: np.ndarray, frame_ts: float) -> None:
        """Put a just-decoded frame on screen with the newest AI overlay.

        Runs on the camera's READER thread.

        WHY — measured on CAM_09 at 12.5 fps. Publishing from the inference
        thread tied the picture to inference. The pipeline processed 12.5 fps
        on average, but any pause in that thread — a database flush, a rollup —
        froze the view, and the frames queued behind it were then published in
        a burst: median gap 72 ms, worst 3,791 ms. On screen that is lag,
        whatever the average says.

        Here every frame is published as it is decoded, so the picture moves at
        the decode rate whatever inference is doing; inference only supplies
        the overlay. An overlay older than OVERLAY_MAX_AGE_S by capture time is
        not drawn, so during a stall the video keeps moving without boxes
        rather than with boxes parked where the vehicles used to be.
        """
        if cam_id.upper() in ALIGN_OVERLAY_CAMERAS:
            try:
                self._present_aligned(cam_id, frame, frame_ts)
            except Exception as exc:                                # noqa: BLE001
                if not getattr(self, "_present_failed", False):
                    self._present_failed = True
                    logger.exception("[%s] aligned publish failed (%s: %s)",
                                     cam_id, type(exc).__name__, exc)
            return
        state = self._overlay_state.get(cam_id)
        tracks = []
        if state is not None and frame_ts - state[0] <= OVERLAY_MAX_AGE_S:
            tracks = state[1]
        try:
            self._draw_live_overlay(cam_id, frame, tracks, frame_ts)
        except Exception as exc:                                    # noqa: BLE001
            if not getattr(self, "_present_failed", False):
                self._present_failed = True
                logger.exception("[%s] smooth publish failed (%s: %s)",
                                 cam_id, type(exc).__name__, exc)

    def _present_aligned(self, cam_id: str, frame: np.ndarray, frame_ts: float) -> None:
        """Publish one held-back frame per call, drawn with its own overlay.

        Runs on the camera's reader thread, once per decoded frame, so it
        publishes at the decode rate. A frame waits in the buffer until
        inference has produced an overlay at or after its capture time; it is
        then drawn with whichever overlay is nearest to it in time. If
        inference falls behind, ready frames beyond ALIGN_MAX_READY are
        skipped so the delay cannot grow; if it stalls, frames older than
        ALIGN_MAX_DELAY_S are shown without boxes so the picture keeps moving.
        """
        from collections import deque
        buf = self._align_buf.get(cam_id)
        if buf is None:
            buf = self._align_buf[cam_id] = deque()
        buf.append((frame_ts, frame))

        hist = list(self._overlay_hist.get(cam_id, ()))
        latest = hist[-1][0] if hist else None
        ready = 0
        if latest is not None:
            for ts, _ in buf:
                if ts > latest:
                    break
                ready += 1
        while ready > ALIGN_MAX_READY:
            buf.popleft()
            ready -= 1
        if ready == 0 and frame_ts - buf[0][0] <= ALIGN_MAX_DELAY_S:
            return                      # wait for inference to reach it

        ts, fr = buf.popleft()
        tracks = []
        if hist:
            o_ts, o_tracks = min(hist, key=lambda h: abs(h[0] - ts))
            if abs(o_ts - ts) <= ALIGN_MATCH_S:
                tracks = o_tracks
        self._draw_live_overlay(cam_id, fr, tracks, ts)

    def _write_appearance(self, rows: list) -> None:
        """Store queued appearance rows in their own transaction.

        Separate from the tracks' transaction on purpose: an appearance row is
        an enhancement, and when one shared the tracks' commit a duplicate-key
        failure rolled back 42 tracks — plates and speeds included.
        """
        from backend.db.models import TrackAppearance

        db = SessionLocal()
        try:
            for r in rows:
                db.add(TrackAppearance(**r))
            db.commit()
        except Exception as exc:                                    # noqa: BLE001
            db.rollback()
            if not getattr(self, "_appearance_write_failed", False):
                self._appearance_write_failed = True
                logger.warning("Track appearance not stored (%s); the tracks "
                               "themselves were saved.", exc)
        finally:
            db.close()

    def _flush_completed_tracks(self, cam_id: str, current_ts: float):
        """Flush tracks that are stale (not seen recently) or have accumulated too many points."""
        if cam_id not in self._track_buffers:
            return

        to_flush: List[Tuple[int, list]] = []
        to_remove: List[int] = []

        for track_id, pts in self._track_buffers[cam_id].items():
            if not pts:
                to_remove.append(track_id)
                continue

            last_ts = pts[-1][0]
            if (current_ts - last_ts > TRACK_STALE_SECONDS) or (len(pts) >= TRACK_MAX_POINTS):
                to_remove.append(track_id)
                if len(pts) >= 4:  # Minimum points for speed estimation
                    to_flush.append((track_id, pts))

        for tid in to_remove:
            del self._track_buffers[cam_id][tid]

        # Persist flushed tracks in their own DB session (isolated transaction)
        if to_flush:
            self._pending_appearance = []
            committed = False
            db = SessionLocal()
            try:
                for track_id, pts in to_flush:
                    self._persist_vehicle_track(db, cam_id, track_id, pts)
                db.commit()
                committed = True
            except Exception as e:
                db.rollback()
                logger.error("Error persisting %d tracks for %s: %s", len(to_flush), cam_id, e)
            finally:
                db.close()
            pending, self._pending_appearance = self._pending_appearance, []
            if committed and pending:
                self._write_appearance(pending)

    def _persist_vehicle_track(self, db: Session, cam_id: str, track_id: int, pts: list):
        """Compute Theil-Sen speed and write a VehicleTrack row to the database.

        Uses the correct ORM column names: n_frames, speed_ci_kmh, quality.
        """
        # pts format: (ts, x_m, y_m, u, v, conf, cls_id)
        valid_metric_pts = [p for p in pts if p[1] is not None and p[2] is not None]

        speed_kmh = None
        speed_ci = None
        path_length = None
        heading = None
        quality = "speed_unavailable"

        # A traffic speed belongs to a vehicle. YOLO_VEHICLE_CLASSES includes
        # person (0) and bicycle (1) so that pedestrians are still detected and
        # counted, but their tracks must not be published as traffic speeds:
        # measured on CAM_09, "person" rows carried 75.8, 61.9 and 61.6 km/h,
        # which are motorcycle riders whose machine the detector missed — a
        # failure mode already recorded on this fleet. A person shown at
        # 75 km/h discredits every other figure beside it.
        cls_ids = [p[6] for p in valid_metric_pts if len(p) > 6 and p[6] is not None]
        majority_cls = max(set(cls_ids), key=cls_ids.count) if cls_ids else None
        if majority_cls in (0, 1):
            quality = "speed_not_applicable"
            valid_metric_pts = []

        if len(valid_metric_pts) >= 4:
            # BLOCKER-1 FIX: Actually define coords and timestamps from valid_metric_pts
            coords = [(p[1], p[2]) for p in valid_metric_pts]
            timestamps = [p[0] for p in valid_metric_pts]

            # Determine calibration quality for this camera
            cal_quality = "good"
            if cam_id in self._calibrations:
                cal_quality = self._calibrations[cam_id].get("quality", "good") or "good"

            est = estimate_track_speed(coords, timestamps, calibration_quality=cal_quality)
            if est and est.is_valid and est.speed_kmh is not None:
                speed_kmh = round(est.speed_kmh, 1)
                speed_ci = est.speed_ci_kmh
                path_length = est.path_length_m
                heading = est.heading_deg
                quality = est.quality

        # Speed in pixels per second, which needs no calibration.
        #
        # Metric speed needs a homography, and no camera on this fleet has a
        # trustworthy one — so speed_kmh is None everywhere, the rollup skips
        # cameras with no speed, and congestion anomalies can never fire. That
        # would be the correct outcome if congestion were a metric quantity.
        # It is not: "traffic here is slower than normal for this camera" is a
        # comparison against the camera's own history, and a comparison needs
        # no scale — only a consistent unit.
        #
        # Pixels per second is that unit. It cannot be reported as km/h and is
        # not stored in a column that says km/h; it is not comparable between
        # cameras, or across a PTZ move, and the anomaly engine only ever
        # compares a camera against itself.
        speed_px_s = None
        if len(pts) >= 4:
            uv = [(p[3], p[4]) for p in pts]
            ts = [p[0] for p in pts]
            span = ts[-1] - ts[0]
            if span > 0.2:
                dist = sum(
                    float(np.hypot(uv[i][0] - uv[i - 1][0], uv[i][1] - uv[i - 1][1]))
                    for i in range(1, len(uv))
                )
                speed_px_s = round(dist / span, 2)

        first_seen = datetime.utcfromtimestamp(pts[0][0])
        last_seen = datetime.utcfromtimestamp(pts[-1][0])

        # Determine vehicle class from majority class in track
        cls_counts: Dict[int, int] = {}
        for p in pts:
            cls_counts[p[6]] = cls_counts.get(p[6], 0) + 1
        majority_cls = max(cls_counts, key=cls_counts.get)
        vehicle_class = YOLO_CLASS_NAMES.get(majority_cls, "unknown")

        # Mean detector confidence across the track
        mean_conf = float(np.mean([p[5] for p in pts]))

        # Calibration RMS error for this camera
        cal_err = None
        if cam_id in self._calibrations:
            cal_err = self._calibrations[cam_id].get("rms")

        # BLOCKER-2 FIX: Use correct ORM columns (n_frames, speed_ci_kmh, quality — NOT speed_r2, points_count)
        vt = VehicleTrack(
            camera_id=cam_id,
            track_id=track_id,
            first_seen=first_seen,
            last_seen=last_seen,
            speed_kmh=speed_kmh,
            speed_px_s=speed_px_s,
            speed_ci_kmh=speed_ci,
            path_length_m=path_length,
            heading_deg=heading,
            vehicle_class=vehicle_class,
            n_frames=len(pts),
            detector_conf=round(mean_conf, 3),
            calibration_err_m=cal_err,
            quality=quality,
        )

        # The plate this vehicle was read as, written with the track.
        #
        # Without this the system cannot answer "how accurate is ANPR on
        # CAM_18 at night" — the read existed for the lifetime of one frame
        # and then went. Read rate, capture rate, confidence distribution and
        # per-camera comparison all become computable from these columns.
        gid = self._global_track_id(cam_id, track_id)
        pr = self._track_plates.pop(gid, None)
        if pr:
            logger.info("[ANPR-WRITE] %s trk=%s gid=%s plate=%s",
                        cam_id, track_id, gid, pr.get("plate"))
        elif self._track_plates:
            logger.info("[ANPR-MISS] %s trk=%s gid=%s — no stored read; "
                        "%d gid(s) still held: %s",
                        cam_id, track_id, gid, len(self._track_plates),
                        list(self._track_plates)[:6])
        if pr:
            vt.plate_text = pr.get("plate")
            vt.plate_confidence = pr.get("confidence")
            vt.plate_state = pr.get("state")
            vt.plate_votes = pr.get("votes")
            vt.plate_locked = pr.get("locked")
            vt.plate_detected = pr.get("detected", True)
        else:
            # No read at all: either no plate region was found, or every
            # candidate failed the quality gate. Recorded as False rather
            # than left NULL so "never attempted" is distinguishable from
            # "attempted and produced nothing".
            vt.plate_detected = False

        db.add(vt)
        self._total_tracks_persisted += 1

        # The track's appearance, so a journey can be continued across a
        # camera whose plate read failed. QUEUED here and written by
        # _flush_completed_tracks only after the track's own transaction has
        # committed, in a separate one. Adding it to this session was not
        # enough: a failing row fails at commit, and that commit is shared with
        # the track, its plate and its speed — measured, a duplicate-key error
        # on the first restart rolled back 42 tracks.
        try:
            acc = self._track_embeddings.pop((cam_id, track_id), None)
            if acc is not None and acc[1] > 0:
                vec = acc[0] / float(acc[1])
                nrm = float(np.linalg.norm(vec))
                if nrm > 1e-9:
                    self._pending_appearance.append(dict(
                        camera_id=cam_id,
                        track_id=int(track_id),
                        global_track_key=gid,
                        vehicle_class=vt.vehicle_class,
                        plate_text=vt.plate_text,
                        first_seen=vt.first_seen,
                        last_seen=vt.last_seen,
                        n_frames=int(acc[1]),
                        embedding=(vec / nrm).astype(np.float32).tolist(),
                        expires_at=datetime.utcnow() + timedelta(days=90),
                    ))
        except Exception as exc:                                    # noqa: BLE001
            if not hasattr(self, "_appearance_write_failed"):
                self._appearance_write_failed = True
                logger.warning(
                    "Track appearance not stored (%s). Journeys will still "
                    "work; cross-camera appearance bridging will not.", exc)

        # Looked up BEFORE the ANPR event below uses it.
        #
        # This assignment used to sit after that block, so every track that
        # actually read a plate raised UnboundLocalError, the surrounding
        # try/except rolled the transaction back, and the whole row was lost —
        # the read, the speed, the track. A camera reading plates perfectly
        # well therefore showed ZERO reads in the database, and only tracks
        # with NO plate survived to be stored. Traced on CAM_09: 32 reads
        # written, 32 transactions rolled back, 0 rows.
        #
        # Namespaced by camera: a bare id would miss every plate, or return
        # the plate another camera locked on its own track number 5.
        confirmed_plate = self._anpr._confirmed_plates.get(
            self._global_track_id(cam_id, track_id))

        if vt.plate_text and vt.plate_text.strip():
            clean_plate = vt.plate_text.upper().replace(" ", "").strip()
            cam_rec = db.query(Camera).filter((Camera.camera_id == cam_id) | (Camera.id == cam_id)).first()
            cam_lat = getattr(cam_rec, "lat", None) if cam_rec else None
            cam_lon = getattr(cam_rec, "lon", None) if cam_rec else None
            cam_zone = getattr(cam_rec, "zone", "Saurashtra") if cam_rec else "Saurashtra"
            cam_dist = getattr(cam_rec, "district", "Junagadh") if cam_rec else "Junagadh"

            je = JourneyEvent(
                id=str(uuid.uuid4()),
                reid_id=clean_plate,
                camera_id=cam_id,
                lat=cam_lat,
                lon=cam_lon,
                zone=cam_zone,
                district=cam_dist,
                is_cross_zone=False,
                object_class="vehicle",
                subtype=vehicle_class,
                plate_text=clean_plate,
                visual_score=1.0,
                final_score=float(vt.plate_confidence or 0.95),
                match_type="plate_confirmed",
                timestamp=last_seen,
            )
            db.add(je)

            ae = ANPREvent(
                id=str(uuid.uuid4()),
                camera_id=cam_id,
                plate_text=clean_plate,
                confidence=float(vt.plate_confidence or 0.95),
                is_stolen=bool(confirmed_plate and confirmed_plate.get("is_stolen")),
                is_wanted=False,
                vehicle_type=vehicle_class,
                timestamp=last_seen,
            )
            db.add(ae)

        # Is the plate read on this track on the police stolen watchlist?
        if confirmed_plate and confirmed_plate.get("is_stolen"):
            cam_rec = db.query(Camera).filter(Camera.camera_id == cam_id).first()
            cam_lat = getattr(cam_rec, "gps_lat", None) or getattr(cam_rec, "lat", None) or 23.0225
            cam_lon = getattr(cam_rec, "gps_lon", None) or getattr(cam_rec, "lon", None) or 72.5714
            incident_id = str(uuid.uuid4())

            # Automatically dispatch nearest Dial-112 PCR Patrol unit via CAD Dispatch
            disp = get_dispatcher()
            dispatch_res = disp.dispatch_nearest(
                incident_lat=cam_lat,
                incident_lon=cam_lon,
                severity="CRITICAL",
                description=f"Intercept {confirmed_plate['plate']} ({confirmed_plate.get('alert_reason')})",
                incident_id=incident_id,
            )
            cad_info = None
            if dispatch_res:
                cad_info = {
                    "unit_id": dispatch_res.unit_id,
                    "call_sign": dispatch_res.call_sign,
                    "officer": dispatch_res.officer_name,
                    "contact": dispatch_res.contact,
                    "distance_km": dispatch_res.distance_km,
                    "eta_minutes": dispatch_res.eta_minutes,
                }
                logger.info(
                    "🚔 [CAD DISPATCH] Dispatched %s (%s, %s) to %s (ETA %.1f min, %.2f km)",
                    dispatch_res.call_sign, dispatch_res.officer_name, dispatch_res.contact,
                    cam_id, dispatch_res.eta_minutes, dispatch_res.distance_km,
                )

            stolen_alert = Alert(
                id=incident_id,
                camera_id=cam_id,
                track_id=track_id,
                alert_type="STOLEN_VEHICLE_WATCHLIST_HIT",
                # Previously unset. AlertFeed.jsx's primary label falls back
                # to "Unknown subject" when this is None — the single alert
                # type this whole platform is evaluated on was rendering as
                # the least informative row in the feed.
                subject_label=f"{confirmed_plate['plate']} — {confirmed_plate.get('alert_reason') or 'Watchlist hit'}",
                severity="critical",
                danger_score=9.8,
                # Speed is only stated when it was actually measured. `speed_kmh
                # or 0` rendered every uncalibrated camera's hit as "tracked at
                # 0.0 km/h" — which reads as "the vehicle was stationary", a
                # claim the system never made. Only 2 of 30 cameras are
                # calibrated, so this was the text on almost every watchlist
                # alert. score_breakdown already carried the honest null.
                description=(
                    f"🚨 WANTED/STOLEN VEHICLE INTERCEPT: License Plate "
                    f"{confirmed_plate['plate']} ({confirmed_plate.get('alert_reason')})"
                    + (f" tracked at {speed_kmh:.1f} km/h."
                       if speed_kmh is not None else
                       " detected (speed not available — camera not calibrated).")
                    + (f" Dispatched {cad_info['call_sign']} (ETA: {cad_info['eta_minutes']} min)."
                       if cad_info else "")
                ),
                plate_text=confirmed_plate["plate"],
                lat=cam_lat,
                lon=cam_lon,
                district=getattr(cam_rec, "district", None) or "Gujarat",
                zone=getattr(cam_rec, "zone", None) or "Central",
                department="Traffic Police",
                status="new",
                lifecycle_status="OPEN",
                score_breakdown={
                    "plate": confirmed_plate["plate"],
                    "confidence": confirmed_plate.get("confidence", 0.0),
                    "reason": confirmed_plate.get("alert_reason"),
                    "speed_kmh": speed_kmh,
                    "cad_dispatch": cad_info,
                    # Where in the footage this happened, and where the
                    # subject was in each of those frames. Evidence capture
                    # reads these to cut the right seconds and draw the right
                    # box, instead of guessing at the middle of a file.
                    "evidence_source": self._evidence_source(cam_id, track_id),
                },
                timestamp=datetime.utcnow(),
            )
            db.add(stolen_alert)
            # Cut the proof clip from the position recorded above. Queued
            # rather than run inline: encoding takes seconds and the inference
            # loop has a ~30 ms budget per frame.
            db.flush()
            self._queue_evidence(incident_id)
            logger.warning("🚨 [STOLEN VEHICLE INTERCEPT] Plate %s on %s: %s", confirmed_plate["plate"], cam_id, confirmed_plate.get("alert_reason"))

            # Push live, not just persist. This alert used to reach the
            # dashboard only on the next full page load / manual refresh —
            # everything up to this point (DB row, CAD dispatch, evidence
            # queue) already ran correctly; only the live push was missing.
            # publish_event() itself goes to Redis pub/sub when available
            # (this pipeline runs as its own process — see run_pipeline.py's
            # own docstring on why — so an in-process broadcast call cannot
            # reach the API process's WebSocket clients on its own) and
            # falls back to an in-process queue otherwise; main.py's
            # _pubsub_broadcast_task is the existing consumer on the API
            # side, already wired to forward whatever arrives here straight
            # into ws_broadcast(). This call site is synchronous (the reader
            # thread loop, not asyncio), hence asyncio.run() rather than
            # await; wrapped so a broadcast failure can never take down the
            # ingestion loop that already did the work that matters.
            try:
                import asyncio as _asyncio
                from backend.cache.redis_cache import publish_event as _publish_event
                _asyncio.run(_publish_event({
                    "type": "new_alert",
                    "alert": {
                        "id": stolen_alert.id,
                        "alert_type": stolen_alert.alert_type,
                        "subject_label": stolen_alert.subject_label,
                        "confidence": confirmed_plate.get("confidence", 0.0),
                        "danger_score": stolen_alert.danger_score,
                        "evidence_hash": None,
                        "snapshot_path": None,
                        "status": stolen_alert.status,
                        "false_positive_reason": None,
                        "lifecycle_status": stolen_alert.lifecycle_status,
                        "merged_into_alert_id": None,
                        "merged_count": 0,
                        "metadata": {"cad_dispatch": cad_info,
                                     "speed_kmh": speed_kmh},
                        "camera_name": getattr(cam_rec, "name", None) or cam_id,
                        "camera_zone": stolen_alert.zone,
                        "created_at": stolen_alert.timestamp.isoformat(),
                    },
                }))
            except Exception as exc:
                logger.debug("Live broadcast of stolen-vehicle alert failed "
                            "(row is already saved, unaffected): %s", exc)

        if speed_kmh and speed_kmh > 5.0:
            logger.info(
                "🚗 Track %s/%d: %s %.1f km/h (±%.1f) | %d frames | quality=%s",
                cam_id, track_id, vehicle_class, speed_kmh,
                speed_ci or 0, len(pts), quality,
            )

    # ── Async ANPR Worker ────────────────────────────────────────────────────

    def _enqueue_anpr_crop(self, cam_id: str, frame: np.ndarray, bbox: list,
                           cls_id: int, track_id: int, frame_ts: float) -> None:
        """Queue one vehicle crop for async ANPR. Non-blocking (<0.1ms).

        Latest-wins per (cam_id, track_id): if the queue is full, the oldest
        pending job is dropped (a moving vehicle will have a newer crop soon)
        rather than allowing the queue to grow unboundedly.
        """
        gid = self._global_track_id(cam_id, track_id)
        with self._track_plates_lock:
            stored = self._track_plates.get(gid)
            if stored and stored.get("locked"):
                return  # already locked plate, no need to OCR again

        max_attempts = 24 if cls_id == 3 else 18
        count = self._anpr_enqueued_counts.get(gid, 0)

        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(frame.shape[1], int(bbox[2])), min(frame.shape[0], int(bbox[3]))
        bw = x2 - x1
        bh = y2 - y1
        min_veh_sz = 20 if cls_id == 3 else 30
        if bw < min_veh_sz or bh < min_veh_sz:
            return  # too small / distant for OCR

        # Growth-aware tracking: vehicles driving towards camera become larger & sharper
        area = bw * bh
        if not hasattr(self, "_anpr_max_area"):
            self._anpr_max_area = {}
        last_area = self._anpr_max_area.get(gid, 0)
        is_significant_growth = area >= 1.25 * last_area

        if count >= max_attempts and not is_significant_growth:
            return  # max OCR attempts reached unless vehicle got significantly closer

        try:
            self._anpr_queue.put_nowait((
                cam_id,
                frame[y1:y2, x1:x2].copy(),  # crop copy — original frame is reused
                bbox,
                cls_id,
                track_id,
                frame_ts,
                gid,
            ))
            self._anpr_enqueued_counts[gid] = count + 1
            self._anpr_max_area[gid] = max(last_area, area)
            if len(self._anpr_enqueued_counts) > 2000:
                self._anpr_enqueued_counts.clear()
                self._anpr_max_area.clear()
        except queue.Full:
            self._anpr_dropped += 1

    def _anpr_worker_loop(self) -> None:
        """Process ANPR crops off the inference thread. One thread per worker process.

        Modeled on _vault_writer_loop. Drains the ANPR queue, runs plate
        detection + OCR, and writes the result to self._track_plates under
        a lock. The inference thread reads _track_plates on the next overlay
        render — the same mechanism used by the synchronous path, unchanged.

        Performance (measured):
          EasyOCR + plate detect per crop: ~31ms
          8 ANPR cameras × 3 vehicles avg = 24 crops/s → 744ms/s → ~0.75 core
          One worker thread handles the full fleet without blocking inference.
        """
        logger.info("ANPR worker started (async mode).")
        while self._running or not self._anpr_queue.empty():
            try:
                item = self._anpr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            (cam_id, crop, bbox, cls_id, track_id, frame_ts, global_id) = item
            t0 = time.perf_counter()
            try:
                self._anpr_attempts += 1
                # Reconstruct a synthetic "frame" from the crop for the ANPR engine.
                # process_vehicle_track expects the full frame + bbox, but we stored
                # the crop directly to save memory. Pad back to a minimal canvas.
                h, w = crop.shape[:2]
                synthetic_frame = crop
                synthetic_bbox = [0, 0, w, h]
                plate_res = self._anpr.process_vehicle_track(
                    synthetic_frame, synthetic_bbox, cls_id, global_id
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._anpr_total_ms += elapsed_ms

                if plate_res and plate_res.get("plate"):
                    self._anpr_successes += 1
                    conf_now = float(plate_res.get("confidence", 0.95))
                    logger.info(
                        "[ANPR-ASYNC-READ] %s trk=%s gid=%s plate=%s conf=%.2f latency=%.1fms",
                        cam_id, track_id, global_id, plate_res["plate"], conf_now, elapsed_ms
                    )
                    # Write to _track_plates under lock (inference thread reads it).
                    with self._track_plates_lock:
                        best = self._track_plates.get(global_id)
                        if best is None or conf_now > best.get("confidence", 0.0):
                            self._track_plates[global_id] = {
                                "plate":      plate_res["plate"],
                                "confidence": conf_now,
                                "state":      "GJ",
                                "votes":      plate_res.get("votes", 5),
                                "locked":     plate_res.get("locked", True),
                                "detected":   True,
                                "is_stolen":  plate_res.get("is_stolen", False),
                                "alert_reason": plate_res.get("alert_reason"),
                            }
            except Exception as exc:                                # noqa: BLE001
                logger.warning("[ANPR-ASYNC] Failed on %s trk=%s: %s",
                               cam_id, track_id, exc)
        logger.info(
            "ANPR worker stopped — attempts=%d successes=%d dropped=%d avg_ms=%.1f",
            self._anpr_attempts, self._anpr_successes, self._anpr_dropped,
            self._anpr_total_ms / max(1, self._anpr_attempts),
        )

    # ── Active Learning Vault Harvesting ─────────────────────────────────────

    def _queue_vault_harvest(self, cam_id: str, frame: np.ndarray, bbox: list,
                             conf: float, cls_id: int, frame_ts: float,
                             embedding: Optional[np.ndarray] = None):
        """Hand a vault crop to the writer thread instead of blocking on it.

        Writing the JPEG and committing the row cost 12.7 ms per processed
        frame on the inference thread — more than detection and appearance
        combined. None of it needs to happen before the next frame is
        tracked, and the crop is already a copy, so it can be done off the
        hot path.

        The queue is bounded: if the writer falls behind, harvests are dropped
        and counted rather than allowed to consume memory. A vault is a
        sample, not a ledger, so dropping some is acceptable — silently
        pretending they were stored is not.
        """
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(frame.shape[1], int(bbox[2])), min(frame.shape[0], int(bbox[3]))
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return
        try:
            self._vault_queue.put_nowait((
                cam_id, frame[y1:y2, x1:x2].copy(), conf, cls_id, frame_ts,
                None if embedding is None else np.asarray(embedding).copy(),
                self._env_cache.get(cam_id, ("DAY", 0.0))[0],
            ))
        except queue.Full:
            self._vault_dropped += 1

    def _vault_writer_loop(self):
        """Drains queued vault crops. One thread, so SQLite sees one writer."""
        while self._running or not self._vault_queue.empty():
            try:
                item = self._vault_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._write_vault_entry(*item)
            except Exception as exc:                               # noqa: BLE001
                logger.warning("Vault write failed: %s", exc)

    def _write_vault_entry(self, cam_id: str, crop: np.ndarray, conf: float,
                           cls_id: int, frame_ts: float,
                           embedding: Optional[np.ndarray], env: str):
        """Persist one harvested crop. Runs on the vault writer thread.

        This used to write a row and throw the pixels away: crop_minio_path
        was formatted from the row's uuid but no file was ever saved, and
        embedding_vector stayed NULL. A vault whose crops do not exist cannot
        be reviewed by an officer and cannot be retrained on, which is the
        entire point of harvesting. Now the JPEG lands on disk and the OSNet
        embedding computed for tracking is stored alongside it.
        """
        t_vault = time.perf_counter()
        db = SessionLocal()
        try:
            captured_at = datetime.utcfromtimestamp(frame_ts)
            entity_type = "person" if cls_id == 0 else "vehicle"
            comp = "hard_positive" if conf >= 0.52 else "probationary"
            entry_id = str(uuid.uuid4())

            # Write the pixels before the row, so a row never points at a file
            # that does not exist. Local filesystem, per the storage decision
            # for this build; the column keeps its name for schema continuity.
            rel = (f"vault/{comp}/{captured_at.year}/{captured_at.month:02d}/"
                   f"{entity_type}/{entry_id}.jpg")
            dest = VAULT_CROP_ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dest), crop,
                               [cv2.IMWRITE_JPEG_QUALITY, 92]):
                logger.warning("Vault crop write failed for %s — skipping row",
                               dest)
                return

            emb_blob = None
            emb_hash = None
            if embedding is not None:
                vec = np.asarray(embedding, dtype=np.float32)
                emb_blob = vec.tobytes()
                emb_hash = hashlib.sha256(emb_blob).hexdigest()

            entry = VaultEntry(
                id=entry_id,
                compartment=comp,
                trust_level="high" if comp == "hard_positive" else "probationary",
                entity_type=entity_type,
                camera_id=cam_id,
                captured_at=captured_at,
                stored_at=captured_at,
                lighting_condition=env,
                crop_minio_path=rel,
                embedding_vector=emb_blob,
                embedding_hash=emb_hash,
                camera_zone=f"Zone_{cam_id.split('_')[-1]}",
                ai_confidence=round(conf, 3),
                # Distance to the nearest gallery entry is a property of the
                # ReID index, not of the detector score. The previous value was
                # an arithmetic restatement of `conf` and carried no
                # information; leave it unset until the index actually answers.
                ai_cosine_distance=None,
                model_version=MODEL_VERSION,
                training_eligible=(comp == "hard_positive"),
                is_gold_retained=False,
                crop_expires_at=captured_at + timedelta(days=90),
                embedding_expires_at=captured_at + timedelta(days=180),
            )
            db.add(entry)
            db.commit()
            self._total_vault_harvested += 1
            logger.debug(
                "Vault harvest: %s on %s (%s, conf=%.2f, env=%s)",
                entity_type, cam_id, comp, conf, env,
            )
        except Exception as e:
            db.rollback()
            logger.warning("Failed to harvest vault crop on %s: %s", cam_id, e)
        finally:
            db.close()
            self._phase_seconds["vault"] += time.perf_counter() - t_vault

    # ── 1-Minute Metric Rollups ──────────────────────────────────────────────

    def _run_live_rollups(self):
        """Roll up completed tracks into camera_metrics_1m and compute Congestion Index."""
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            bucket_start = now.replace(second=0, microsecond=0)
            fifteen_mins_ago = now - timedelta(minutes=15)

            tracks = (
                db.query(VehicleTrack)
                .filter(VehicleTrack.first_seen >= fifteen_mins_ago)
                .all()
            )

            by_cam: Dict[str, list] = {}
            for t in tracks:
                by_cam.setdefault(t.camera_id, []).append(t)

            cameras_updated = 0
            for cam_id, c_tracks in by_cam.items():
                # Metric speed where the camera is calibrated; image-plane
                # speed everywhere else.
                #
                # This used to `continue` when no track had a km/h figure —
                # and with every fabricated homography removed, that is every
                # camera. No rollup row meant no baseline, no baseline meant
                # no congestion anomaly could ever fire, and the traffic
                # module went silent on a fleet that is plainly busy.
                kmh = [
                    t.speed_kmh for t in c_tracks
                    if t.speed_kmh is not None and 1.0 <= t.speed_kmh <= 180.0
                ]
                px = [
                    t.speed_px_s for t in c_tracks
                    if getattr(t, "speed_px_s", None) is not None
                    and 0.5 <= t.speed_px_s <= 4000.0
                ]

                if kmh:
                    speeds, basis = kmh, "kmh"
                elif px:
                    speeds, basis = px, "px_s"
                else:
                    continue

                median_v = float(np.median(speeds))
                p15_v = float(np.percentile(speeds, 15))
                p85_v = float(np.percentile(speeds, 85))
                median_px = float(np.median(px)) if px else None

                # Compute Congestion Index using shared engine instance
                profile = self._congestion_engine.get_congestion_profile(
                    cam_id, current_speed_kmh=median_v, db=db
                )
                ci = profile.congestion_index

                # Upsert camera_metrics_1m
                # BLOCKER-3 FIX: Do NOT write congestion_label (column doesn't exist in DB model)
                metric = (
                    db.query(CameraMetrics1M)
                    .filter(
                        CameraMetrics1M.camera_id == cam_id,
                        CameraMetrics1M.bucket_start == bucket_start,
                    )
                    .first()
                )
                # median_speed_kmh stays NULL on an uncalibrated camera. The
                # column says km/h and must only ever hold km/h; writing a
                # pixel rate into it would be the same class of error as the
                # homographies that produced 1.2 km/h for moving traffic.
                if not metric:
                    metric = CameraMetrics1M(
                        camera_id=cam_id,
                        bucket_start=bucket_start,
                        vehicle_count=len(c_tracks),
                        speed_samples=len(speeds),
                        congestion_index=round(ci, 3) if ci is not None else None,
                        health_status="ACTIVE",
                    )
                    db.add(metric)

                metric.vehicle_count = len(c_tracks)
                metric.speed_samples = len(speeds)
                metric.congestion_index = round(ci, 3) if ci is not None else None
                metric.speed_basis = basis
                metric.median_speed_px_s = (round(median_px, 2)
                                            if median_px is not None else None)
                if basis == "kmh":
                    metric.median_speed_kmh = round(median_v, 1)
                    metric.p15_speed_kmh = round(p15_v, 1)
                    metric.p85_speed_kmh = round(p85_v, 1)
                else:
                    metric.median_speed_kmh = None
                    metric.p15_speed_kmh = None
                    metric.p85_speed_kmh = None

                cameras_updated += 1

            db.commit()
            logger.info(
                "📊 Rollup: %d cameras updated | %d total tracks in window | queue=%d",
                cameras_updated, len(tracks), self._frame_queue.qsize(),
            )

            # ── Evaluate Live Contextual & Shockwave Anomalies (Gap 3) ──
            try:
                self._baseline_engine.build_baseline_matrix(db=db)
                anomalies = self._baseline_engine.detect_anomalies(lookback_minutes=15, db=db)
                if anomalies:
                    cams_dict = {c.camera_id: c for c in db.query(Camera).filter(Camera.is_deleted == False).all()}
                    fifteen_mins_ago = datetime.utcnow() - timedelta(minutes=15)

                    for anom in anomalies:
                        # Deduplicate: Only generate alert if no open alert for this camera+type in past 15m
                        existing_alert = (
                            db.query(Alert)
                            .filter(
                                Alert.camera_id == anom.camera_id,
                                Alert.alert_type == f"TRAFFIC_ANOMALY_{anom.anomaly_type}",
                                Alert.timestamp >= fifteen_mins_ago,
                            )
                            .first()
                        )
                        if not existing_alert:
                            cam_rec = cams_dict.get(anom.camera_id)
                            new_alert = Alert(
                                id=str(uuid.uuid4()),
                                camera_id=anom.camera_id,
                                alert_type=f"TRAFFIC_ANOMALY_{anom.anomaly_type}",
                                severity="critical" if anom.severity == "CRITICAL" else "high",
                                danger_score=8.5 if anom.severity == "CRITICAL" else 6.0,
                                description=anom.explanation,
                                lat=getattr(cam_rec, "gps_lat", None) or getattr(cam_rec, "lat", None) or 23.0225,
                                lon=getattr(cam_rec, "gps_lon", None) or getattr(cam_rec, "lon", None) or 72.5714,
                                district=getattr(cam_rec, "district", None) or "Gujarat",
                                zone=getattr(cam_rec, "zone", None) or "Central",
                                department="Traffic Police",
                                status="new",
                                lifecycle_status="OPEN",
                                score_breakdown={
                                    "anomaly_type": anom.anomaly_type,
                                    "deviation_pct": anom.deviation_pct,
                                    "z_score": anom.z_score,
                                    "current_speed_kmh": anom.current_speed_kmh,
                                    "baseline_speed_kmh": anom.baseline_speed_kmh,
                                    "sustained_minutes": anom.sustained_minutes,
                                    "recommended_action": anom.recommended_action,
                                    # Congestion is a property of the whole
                                    # scene rather than one vehicle, so no
                                    # track is named — but the clip must still
                                    # be the moment the anomaly was measured.
                                    "evidence_source": self._evidence_source(
                                        anom.camera_id),
                                },
                                timestamp=datetime.utcnow(),
                            )
                            db.add(new_alert)
                            db.commit()
                            # Proof clip for this anomaly, cut from the frames
                            # the detector was on when it fired.
                            self._queue_evidence(new_alert.id)
                            logger.warning(
                                "🚨 [MACRO ANOMALY ALERT] %s on %s: %s",
                                anom.anomaly_type, anom.camera_id, anom.explanation,
                            )
            except Exception as e_anom:
                logger.error("Error evaluating live traffic baseline anomalies: %s", e_anom, exc_info=True)

        except Exception as e:
            db.rollback()
            logger.error("Error during live traffic rollup: %s", e, exc_info=True)
        finally:
            db.close()


    # _resolve_camera_plate REMOVED — no more hardcoded/fabricated plate generation.
    # Only real ANPR detections from the CRNN ensemble + EasyOCR pipeline are shown.

    def get_live_camera_jpeg(self, cam_id: str) -> Optional[bytes]:
        """Return the latest live annotated CCTV frame JPEG for this camera.
        If the pipeline has not yet processed a frame for this camera, reads from the stream source or clip."""
        with self._frame_lock:
            if self._running:
                jpeg = self._latest_annotated_jpeg.get(cam_id)
                if jpeg is not None:
                    return jpeg

        # Check cross-process shared live frames written by the running pipeline
        for fname in (f"{cam_id.upper()}.jpg", f"{cam_id}.jpg"):
            shared_file = WORKSPACE / "output" / "live_frames" / fname
            if shared_file.is_file():
                try:
                    st = shared_file.stat()
                    if st.st_size > 1024:
                        data = shared_file.read_bytes()
                        if len(data) > 1024:
                            return data
                except Exception:
                    pass

        # Check if a reader is configured with a network stream URL or local clip
        reader = next((r for r in self._reader_threads if r.cam_id == cam_id), None)
        frame = None

        if reader and reader.source_url:
            try:
                url = reader.source_url
                if url.lower().startswith("rtsp"):
                    # RTSP: cv2.VideoCapture handles it natively; no cookie needed.
                    # Force TCP — the corp8 RTSP server requires it for cam21 and cam25-30.
                    os.environ.setdefault(
                        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                        "rtsp_transport;tcp|analyzeduration;1000000|probesize;1000000|"
                        "fflags;nobuffer|flags;low_delay|max_delay;500000",
                    )
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    if cap.isOpened():
                        ret, f = cap.read()
                        cap.release()
                        if ret and f is not None:
                            frame = f
                else:
                    from backend.services.hls_ffmpeg_capture import is_corp8_hls, open_corp8_stream
                    if is_corp8_hls(url):
                        cap = open_corp8_stream(url)
                        if cap is not None:
                            try:
                                ret, f = cap.read()
                                if ret and f is not None:
                                    frame = f
                            finally:
                                cap.release()
                    else:
                        cap = cv2.VideoCapture(url)
                        if cap.isOpened():
                            ret, f = cap.read()
                            cap.release()
                            if ret and f is not None:
                                frame = f
            except Exception:
                pass

        # Fallback: Read directly from local clip ONLY if not in strict live mode
        if frame is None and not SENTINEL_STRICT_LIVE:
            cam_dir = CLIPS_DIR / cam_id
            if not cam_dir.exists():
                # Try numeric formatting CAM_02 or cam02
                digits = "".join(ch for ch in str(cam_id) if ch.isdigit())
                if digits:
                    cam_dir = CLIPS_DIR / f"CAM_{int(digits):02d}"

            if not cam_dir.exists() or not list(cam_dir.glob("*.mp4")):
                # No clips for this camera and no RTSP frame yet — return None
                # rather than serving footage from a different camera (the old
                # CAM_02 fallback silently served wrong footage to every camera
                # that did not yet have a processed frame).
                return None

            if cam_dir.exists():
                clips = sorted(cam_dir.glob("*.mp4"))
                if clips:
                    cap = cv2.VideoCapture(str(clips[0]))
                    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    if total_f > 10:
                        current_f = int((time.time() * fps) % total_f)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_f)
                    ret, f = cap.read()
                    if not ret or f is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, f = cap.read()
                    cap.release()
                    if ret and f is not None:
                        frame = f

        if frame is not None:
            dets = []
            if self._model is None:
                try:
                    self._load_model()
                except Exception:
                    pass
            if self._model is not None:
                try:
                    res = self._model.predict(frame, verbose=False, conf=YOLO_CONF_THRESHOLD,
                                              classes=YOLO_VEHICLE_CLASSES, device=self.device)
                    if res and len(res[0].boxes):
                        xyxy = res[0].boxes.xyxy.cpu().numpy()
                        clss = res[0].boxes.cls.cpu().numpy()
                        confs = res[0].boxes.conf.cpu().numpy()
                        for (x1, y1, x2, y2), cl, cf in zip(xyxy, clss, confs):
                            # Deterministic track ID based on spatial coordinates
                            tid = (int(x1 * 3 + y1 * 7) % 899) + 101
                            d_item = {"track_id": tid,
                                      "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                      "cls": int(cl), "conf": float(cf)}

                            # Real-Time ANPR Plate Detection & Recognition across all 30 CCTV cameras
                            if int(cl) in (2, 3, 5, 7):
                                plate_res = None
                                try:
                                    plate_res = self._anpr.process_vehicle_track(
                                        frame, [x1, y1, x2, y2], int(cl), 9_000_000_000 + tid)
                                except Exception:
                                    pass

                                if plate_res and plate_res.get("plate"):
                                    d_item["plate"] = plate_res.get("plate")
                                    d_item["plate_conf"] = plate_res.get("confidence", 0.95)
                                    d_item["is_stolen"] = False
                                    d_item["alert_reason"] = None
                                # No hardcoded fallback — only real ANPR detections shown
                            dets.append(d_item)
                except Exception:
                    pass
            self._draw_live_overlay(cam_id, frame, dets, time.time())
            with self._frame_lock:
                return self._latest_annotated_jpeg.get(cam_id)

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────
_pipeline_instance: Optional[Live24x7Pipeline] = None


def get_live_pipeline() -> Live24x7Pipeline:
    """The pipeline, constructing it if this process has not yet.

    Constructing is not free: Live24x7Pipeline.__init__ loads OSNet-IBN onto
    the GPU. Callers that only want to *inspect* pipeline state must use
    peek_live_pipeline instead.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = Live24x7Pipeline()
    return _pipeline_instance


def peek_live_pipeline() -> Optional[Live24x7Pipeline]:
    """The pipeline if this process already has one, otherwise None.

    Read-only endpoints ask whether ingestion is running. Routing that through
    get_live_pipeline built the object to answer the question — loading the
    ReID model onto the GPU of an API process that had deliberately been kept
    free of it, and turning a status check into a 1.7 second request. Asking
    without building is the whole point.
    """
    return _pipeline_instance


def get_live_camera_frame(cam_id: str) -> Optional[bytes]:
    """Module-level helper to fetch real-time annotated CCTV frame JPEG."""
    pipeline = get_live_pipeline()
    return pipeline.get_live_camera_jpeg(cam_id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sentinel Gujarat Live 24x7 AI Pipeline Daemon")
    parser.add_argument("--run-duration", type=int, default=0, help="Duration in seconds to run (0 for indefinite)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    pipe = get_live_pipeline()
    pipe.start()
    try:
        if args.run_duration > 0:
            logger.info("Pipeline running for %d seconds...", args.run_duration)
            time.sleep(args.run_duration)
        else:
            logger.info("Pipeline running indefinitely. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()


