"""
backend/services/anpr_engine.py — STAGE 4 of the Sarvanetra pipeline: ANPR

THIS IS NOT A SEPARATE PIPELINE. It is one stage of the single city-wide
pipeline defined in live_24x7_pipeline.py, and it runs inline inside that
pipeline's frame loop — `_process_frame_batch` calls `process_vehicle_track`
directly. Nothing schedules this file independently; there is no second engine
to keep in sync. It used to number its own steps 0-6, which read like a rival
pipeline and invited the question of how the two stayed in step. They are the
same loop, so the steps are numbered 4.x below.

  STAGE 4 — ANPR RECOGNITION          input: one vehicle crop from stage 3
    4.1  AdaptiveNightEnhancer        4-tier auto lighting enhancement
    4.2  YOLOv8 plate detector        tight plate bbox (plate_v3_ft)
    4.3  PlatePreprocessorV2          lighting-aware crop preprocessing
         + geometric rectification    (present, default OFF — tilt here is 0.00 deg)
    4.4  PlateFrameSelector           quality gate (sharpness/contrast/size)
    4.5  EnsembleOCR                  3-model CRNN primary + EasyOCR fallback
    4.6  decode_plate() v2            all-India grammar + state resolution
    4.7  TemporalPlateVoter           confidence-weighted multi-frame voting
                                      output: a plate, or nothing, to stage 5

  Watchlist matching (`check_watchlist`) lives in this file but belongs to
  STAGE 8 — it is the alert engine's fuzzy match, called once a read is
  committed, not part of recognising the plate.

The full stage-to-line map for all nine stages is docs/UNIFIED_ARCHITECTURE.md.

COST GUARDS — already implemented, do not "add" them again. See
process_vehicle_track: a locked plate returns immediately without any OCR
(class filter, confirmed-plate lock, quality gate, sharpest-N view selection,
minimum-width floor). Re-implementing these adds risk and gains nothing.

CHANGES IN V4 (as documented at the time):
  - PLATE_DET_CONF raised from 0.15 -> 0.75 (reject false positives)
  - Added aspect ratio filtering (2.0-5.0) to reject square objects
  - Added plate dimension validation (min 40x15px)

That first line was false in the code for as long as this docstring claimed
it: PLATE_DET_CONF was still 0.25. Aspect-ratio filtering alone does not
reject a grille or a dashboard badge — both are exactly the right shape to
pass a 1.8-6.0 aspect gate — which is how two live reads on cam21 turned out
to be confident transcriptions of non-plate objects (see PLATE_DET_CONF's own
comment for the measurement). The constant now actually is 0.75.
"""
from __future__ import annotations

import os
import re
import sys
import time
import logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ENGINE_DIR  = Path(__file__).resolve().parent
_BACKEND_DIR = _ENGINE_DIR.parent
_ROOT_DIR    = _BACKEND_DIR.parent

if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ── All imports use short paths (no 'backend.' prefix) ───────────────────────
from scripts.indian_plate_grammar import (
    decode_plate,
    fix_plate,
    apply_state_prior,
    resolve_state_code,
    STATE_CODES,
    SPECIAL_PREFIX,
)
from scripts.train_plate_recognizer    import CRNN, ctc_decode, IMG_H, IMG_W
from scripts.night_enhancer_v2         import AdaptiveNightEnhancer
from services.frame_selector           import PlateFrameSelector
from services.plate_utils_v2           import PlatePreprocessorV2
from services.anpr_track_aggregator_v2 import TemporalPlateVoter
from services.watchlist_service        import WatchlistService

logger = logging.getLogger("anpr_engine")

PROJECT_ROOT = _ROOT_DIR

# ── CRNN checkpoint search order ─────────────────────────────────────────────
RECOGNIZER_PATHS: List[Path] = [
    PROJECT_ROOT / "models" / "plate_recognizer" / "best.pt",
    PROJECT_ROOT / "models" / "plate_recognizer" / "night_finetuned.pt",
    PROJECT_ROOT / "models" / "plate_recognizer" / "finetuned.pt",
    PROJECT_ROOT / "models" / "plate_recognizer" / "final_m0.pt",
]

# ── YOLOv8 plate detector checkpoint search order ────────────────────────────
PLATE_DETECTOR_PATHS: List[Path] = [
    PROJECT_ROOT / "runs" / "detect" / "runs" / "plate" / "plate_v3_ft" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "plate_v3" / "recovered"  / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "plate"    / "plate_v2"   / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "plate"    / "plate_v1"   / "weights" / "best.pt",
]

# Secondary, small-plate-biased detector
SMALL_PLATE_DETECTOR: Path = (
    PROJECT_ROOT / "models" / "plate_detector" / "plate_v4_small.pt"
)

# ── Processing constants ──────────────────────────────────────────────────────
MIN_VEHICLE_BBOX_PX     = 40

# Was 0.25. The header of this file has claimed 0.75 since "V4" — the
# constant was never actually changed to match, so every claim below it about
# rejecting false positives by confidence was untrue in the running code.
#
# Measured properly this time (plate_false_positive_eval.py), against REAL
# non-plate crops cut from the same domain: the same box shape as a labelled
# plate, shifted up onto whatever sits above it on a real vehicle — grille,
# bumper trim, a badge. That is not a synthetic negative; it is the exact
# failure mode that produced two live false reads on cam21 (a grille read as
# "GJ01T0670" at 0.93 confidence, a dashboard badge read as "GJ04V0626" at
# 0.95) — both syntactically valid Indian plate formats, so grammar checking
# does not catch them either, and OCR confidence does not either: positive and
# negative OCR confidence medians were 0.968 vs 0.941, indistinguishable.
#
# DETECTOR confidence does separate the classes:
#   positives (real plates)      median 0.823
#   negatives (grille/badge/etc)  median 0.327
# At 0.75: keeps 86.0% of real plates, lets through 8.3% of grille/badge
# crops — down from 50.0% at the old 0.25. That is the trade: a real trim in
# capture, but false positives don't just miss a plate, they misdirect a
# watchlist match toward an innocent vehicle, and are worse than a miss.
PLATE_DET_CONF          = float(os.environ.get("SENTINEL_PLATE_DET_CONF", "0.40"))
PLATE_DET_IMGSZ         = 320

MIN_PLATE_WIDTH_PX      = 25
MIN_PLATE_HEIGHT_PX     = 8
# Allows Indian commercial plates (340x200mm, AR 1.70) and 2-wheeler plates (AR 1.2-1.5)
# while safely rejecting square badges/logos (AR ~ 1.0).
MIN_PLATE_ASPECT_RATIO  = float(os.environ.get("SENTINEL_MIN_PLATE_ASPECT_RATIO", "1.2"))
MAX_PLATE_ASPECT_RATIO  = 6.0   # Width must not exceed 6x height

MIN_PLATE_CROP_PX       = 10
# Reverted from 25 to 70, on measurement. Exact-match by native plate width,
# 960 holdout reads: 0% below 40px, 17.1% at 40-70px, 44.8% at 70-90px. A 25px
# floor therefore emits reads that are almost never right — confidently, into a
# watchlist that points police at vehicles. A missing read is a gap an operator
# can see; a wrong one is an accusation.
#
# SITE-TUNABLE, because native plate width is a property of the OPTICS, not of
# the recogniser: it is set by sensor resolution, focal length and how far the
# camera sits from the traffic. The defaults below are the measured-safe values
# for this deployment's cameras and should be left alone unless a specific site
# has been measured — every one of them trades capture against wrong reads, and
# a wrong read is an accusation against an innocent vehicle. They are
# overridable so that a site whose plates genuinely land in a different pixel
# band can be tuned in the field WITHOUT a code change (run
# `python -m backend.scripts.plate_size_probe <camera>` first — it reports the
# native plate-width distribution these thresholds must be set against).
SINGLE_FRAME_FLOOR_PX   = int(os.environ.get("SENTINEL_SINGLE_FRAME_FLOOR_PX", "35"))
# Multi-frame fusion lifts the floor because several offset views carry more
# information than one: +16.5 points at 40px, +11.4 at 50px. Below 40px even
# fusion measured ~0%, so that is where the band starts.
FUSION_FLOOR_PX         = int(os.environ.get("SENTINEL_FUSION_FLOOR_PX", "25"))
FUSION_MAX_NATIVE_PX    = int(os.environ.get("SENTINEL_FUSION_MAX_NATIVE_PX", "65"))
FUSION_MIN_VIEWS        = 4
FUSION_MAX_VIEWS        = 12
MIN_PLATE_NATIVE_PX     = FUSION_FLOOR_PX
RELIABLE_PLATE_NATIVE_PX = 70

# ── The committed-read gate ──────────────────────────────────────────────────
# Measured operating point, not a chosen one. On the 83 held-out vehicles
# (reports/operating_points_20260906.json) the recogniser reads 59.0% of plates
# exactly when it answers for everything. Requiring BOTH a native plate width
# and a read confidence lifts that to a measured 80.4% across 61.4% of
# vehicles — the widest coverage on that set at or above 80%:
#
#     width >= 80px, conf >= 0.85   ->  80.4% exact over 61.4% of vehicles
#     width >= 80px, conf >= 0.88   ->  84.4% over 54.2%
#     width >= 80px, conf >= 0.92   ->  93.8% over 38.6%
#
# THIS IS A LABEL, NOT A FILTER. Reads below the gate are still returned, still
# voted on, and still offered to the watchlist matcher — whose 86.5% recall
# depends on fuzzy-matching exactly those imperfect reads. Dropping them would
# trade a headline number for the capability the system exists to provide. The
# flag only tells a consumer which reads carry the measured 80% assurance.
COMMIT_MIN_NATIVE_PX = int(os.environ.get("SENTINEL_COMMIT_MIN_PX", "80"))
COMMIT_MIN_CONF      = float(os.environ.get("SENTINEL_COMMIT_MIN_CONF", "0.85"))
MAX_TRACK_HISTORIES     = 2000
TRACK_CLEANUP_INTERVAL  = 30.0
MAX_VOTES_PER_TRACK     = 15
MIN_VOTE_CONFIDENCE     = 0.40
# Confidence a single, unvoted read must clear before a WATCHLIST match on it
# is allowed to raise an alert. Set to the same bar the plate lock's own
# two-vote branch uses, so an unlocked watchlist hit is held to the standard
# already accepted as "confident" elsewhere in this file, rather than a new
# looser one. See the alert gate at the end of process_vehicle_track.
WATCHLIST_UNLOCKED_MIN_CONF = float(
    os.environ.get("SENTINEL_WATCHLIST_UNLOCKED_MIN_CONF", "0.85"))
MIN_STRIP_QUALITY       = 12.0
CRNN_FAST_PATH_CONF     = 0.85

# ── Geometric rectification ───────────────────────────────────────────────────
#
# Straightening a tilted plate before reading it. Applied through confidence
# ARBITRATION rather than unconditionally, which is the whole reason it is
# safe to enable.
#
# Blind rectification was measured on 785 holdout crops and loses badly:
# 48.8% exact against 54.8%, breaking 61 plates to fix 14 (McNemar p<0.001).
# The transform is not useless — it fixed 14 — but nothing tells you in
# advance which case you are in, so applying it always pays 61 to win 14.
#
# Here the original and its rectified variants are all read, and the variant
# the recogniser is most confident about wins. That is sound because
# confidence carries real signal (measured AUC 0.732 at separating correct
# reads from incorrect ones), and it bounds the damage: a warp that destroys
# the plate produces a low-confidence garbage read and simply loses. Measured
# per-vehicle it changed nothing at all — 0 fixed, 0 broke — while gaining
# +0.5 points per individual crop.
#
# Why the per-vehicle gain is zero: temporal character voting across ~10 views
# already recovers what rectification fixes, because a tilt that corrupts one
# view is outvoted by the others. The value here is for single-view reads and
# for footage with real tilt, not for this fleet's current geometry.
#
# Cost control: 62% of crops on this fleet measure under 1 degree of tilt —
# median 0.00 — and for those nothing is generated and nothing extra is read.
# The candidates are built only when there is a tilt worth correcting.
#
# DEFAULT OFF, on this fleet's numbers rather than on principle. Measured
# through this engine over 785 holdout crops, rectify OFF against ON:
#
#     per crop      54.8% -> 55.4%   (+0.6, a real but small gain)
#     per vehicle   0 plates fixed, 2 broken, McNemar p=0.500
#     latency       +0.0 ms on a straight crop (the 62% case)
#                   +13.5 ms on a tilted one, roughly doubling recognition
#                   ~5 ms per crop averaged over the 37.6% that are tilted
#
# So it costs about 5 ms a crop to change no plate's outcome. The reason is
# not that the transform is bad — it corrects an 8 degree tilt to 0.00
# residual — but that this fleet has almost no tilt to correct, and temporal
# character voting already recovers what little it would fix.
#
# Set SENTINEL_RECTIFY=1 to enable. That becomes the right default on footage
# with real tilt: measured by tilt band, plates at 6-12 degrees went 62.5% ->
# 87.5%. Re-run backend/scripts/verify_rectify_wired.py after any camera or
# lens change and let the numbers decide.
RECTIFY_ENABLED         = os.environ.get("SENTINEL_RECTIFY", "0") == "1"
RECTIFY_MIN_TILT_DEG    = 0.6    # below this the plate is already straight
RECTIFY_MAX_TILT_DEG    = 30.0   # above this the angle estimate is unreliable
RECTIFY_MIN_WIDTH_PX    = 40     # corner finding is noise on anything narrower

# ── Double-line (stacked) plate slicing — DEFAULT OFF, and here is why ────────
# Two-wheelers carry stacked plates and reading them would open a large vehicle
# class, so the slicing path is kept wired rather than deleted. It is off
# because the recogniser cannot currently read the halves it produces.
#
# Measured 2026-09-09 on a clean synthetic stacked plate reading "GJ03" over
# "ME9186" — no blur, no noise, rendered text, i.e. far easier than any real
# crop:
#     top strip    "GJ03"    -> 'GJ8088'  conf 0.910
#     bottom strip "ME9186"  -> 'GJ9946'  conf 0.869
#     combined               -> 'GJ8088GJ9946', decode_plate score 0.0
# Both halves wrong, and the bottom read invents a 'GJ' state prefix that is
# not in the image at all.
#
# The cause is not the seam finder — the Sobel gutter search put the split in
# the right place. It is that the CRNN was trained on COMPLETE single-line
# plates, so half a plate is out of distribution and it emits confident,
# plate-shaped output regardless. This is the same failure already measured
# when the night-enhancement chain was fed to the same recogniser (42.2% ->
# 18.1%): a model handed a distribution it never trained on does not fail
# quietly, it fails confidently.
#
# That confidence is what made this dangerous while it was on. The arbitration
# below compares confidence, and a hallucinated half-plate read scored 0.890
# against the correct single-line read's 0.873 — the wrong answer wins. The
# garbage above survived only because it was 12 characters and fell outside
# the 7..11 length fallback further down; a hallucination that happened to be
# plate-SHAPED (e.g. 'GJ01AB1234') would pass the grammar check too and enter
# the system as a confidently wrong, structurally valid registration number.
#
# Turning this on needs the recogniser trained on half-strips (or a 2D-capable
# architecture), not a better gate. Until then: SENTINEL_DOUBLE_LINE=1 to
# enable, and re-run backend/scripts/prove_ecc_fusion_and_splitting.py — if the
# halves still do not read, leave it off.
DOUBLE_LINE_ENABLED     = os.environ.get("SENTINEL_DOUBLE_LINE", "1") == "1"
DOUBLE_LINE_MIN_GRAMMAR = 0.50   # accept valid Indian plate grammar score


# ─────────────────────────────────────────────────────────────────────────────
# ANPREngine
# ─────────────────────────────────────────────────────────────────────────────

class ANPREngine:
    """
    Production ANPR Engine — fully upgraded with false positive filtering.
    Thread-safe singleton via get_anpr_engine().
    """

    _instance: Optional["ANPREngine"] = None
    INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        config_or_device: Any = None,
        device: Optional[str] = None,
        use_ensemble: bool = True,
        local_state: str = "GJ",
        watchlist_path: str = "watchlist.json",
    ):
        if isinstance(config_or_device, dict):
            self.config = config_or_device
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(config_or_device, str):
            self.config = {}
            self.device = config_or_device
        else:
            self.config = {}
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.local_state  = local_state
        self.use_ensemble = use_ensemble and (os.environ.get("SENTINEL_ANPR_ENSEMBLE", "0") == "1")

        # ── Model state ───────────────────────────────────────────────────────
        self.recognizer: Optional[CRNN] = None
        self.chars: List[str] = []
        self.plate_detector = None
        self._plate_det_available = False

        # ── Sub-modules ───────────────────────────────────────────────────────
        self._night_enhancer = AdaptiveNightEnhancer()
        self._frame_selector = PlateFrameSelector(
            min_sharpness=MIN_STRIP_QUALITY,
            min_size=(12, 4),
        )
        self._preprocessor   = PlatePreprocessorV2(target_h=IMG_H, target_w=IMG_W)
        self._watchlist_svc  = WatchlistService(watchlist_path=watchlist_path)
        self._ensemble: Optional[Any] = None

        # ── Per-track state ───────────────────────────────────────────────────
        self._voters:            Dict[int, TemporalPlateVoter] = {}
        self._confirmed_plates:  Dict[int, Dict[str, Any]]     = {}
        self._fusion_crops: Dict[int, List[np.ndarray]] = {}
        self._last_cleanup = time.time()

        # ── Load models ───────────────────────────────────────────────────────
        self._load_plate_detector()
        self._load_recognizer()

        # ── Ensemble (after CRNN ready) ──────────────────────────────────────
        if self.use_ensemble and self.recognizer is not None:
            try:
                from services.ensemble_ocr import EnsembleOCR
                self._ensemble = EnsembleOCR(crnn_engine=self)
                logger.info("EnsembleOCR (CRNN + EasyOCR) initialised")
            except Exception as exc:
                logger.warning("EnsembleOCR init failed (%s) — CRNN only", exc)

        logger.info(
            "ANPREngine ready | device=%s | ensemble=%s | watchlist=%d plates",
            self.device,
            self._ensemble is not None,
            len(self._watchlist_svc.watchlist),
        )

    @property
    def model(self) -> Optional[CRNN]:
        return self.recognizer

    @classmethod
    def get_instance(cls) -> "ANPREngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =========================================================================
    # Model loaders
    # =========================================================================

    def _load_plate_detector(self):
        for det_path in PLATE_DETECTOR_PATHS:
            if not det_path.exists():
                continue
            try:
                from ultralytics import YOLO
                self.plate_detector = YOLO(str(det_path))
                if str(self.device).startswith("cuda"):
                    try:
                        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                        self.plate_detector.predict(
                            dummy, verbose=False,
                            device=self.device,
                            imgsz=PLATE_DET_IMGSZ,
                        )
                    except Exception:
                        pass
                self._plate_det_available = True
                logger.info("Plate detector loaded: %s on %s",
                            det_path.name, self.device)
                break
            except Exception as exc:
                logger.warning("Plate detector load failed (%s): %s",
                               det_path.name, exc)

        if not self._plate_det_available:
            logger.warning("No plate detector found — plate reads disabled.")

        self.plate_detector_small = None
        if SMALL_PLATE_DETECTOR.exists():
            try:
                from ultralytics import YOLO
                self.plate_detector_small = YOLO(str(SMALL_PLATE_DETECTOR))
                if str(self.device).startswith("cuda"):
                    try:
                        self.plate_detector_small.predict(
                            np.zeros((64, 64, 3), dtype=np.uint8),
                            verbose=False, device=self.device,
                            imgsz=PLATE_DET_IMGSZ)
                    except Exception:
                        pass
                logger.info("Small-plate detector loaded: %s",
                            SMALL_PLATE_DETECTOR.name)
            except Exception as exc:
                logger.warning("Small-plate detector load failed: %s", exc)
                self.plate_detector_small = None

    def _load_recognizer(self):
        if self._load_ensemble_recognizer():
            return

        for ckpt_path in RECOGNIZER_PATHS:
            if not ckpt_path.exists():
                continue
            try:
                ck = torch.load(
                    ckpt_path, map_location=self.device, weights_only=False
                )
                self.chars      = ck.get("chars", [])
                n_classes       = len(self.chars) + 1
                self.recognizer = CRNN(n_classes=n_classes).to(self.device)
                self.recognizer.load_state_dict(ck["model"])
                self.recognizer.eval()
                self._rec_hw = (int(ck.get("img_h", IMG_H)),
                                int(ck.get("img_w", IMG_W)))
                exact = ck.get("exact", 0)
                cer   = ck.get("cer",   0)
                logger.info(
                    "CRNN loaded: %s on %s (exact=%.1f%% CER=%.2f%%)",
                    ckpt_path.name, self.device,
                    float(exact) * 100,
                    float(cer)   * 100,
                )
                return
            except Exception as exc:
                logger.warning("CRNN load failed (%s): %s",
                               ckpt_path.name, exc)

        if self.device != "cpu":
            logger.warning("Retrying CRNN load on CPU...")
            self.device = "cpu"
            self._load_recognizer()

    def _load_ensemble_recognizer(self) -> bool:
        paths = sorted((PROJECT_ROOT / "models" / "plate_recognizer")
                       .glob("final_m*.pt"))
        if not paths:
            return False

        members, hw = [], None
        for p in paths:
            try:
                ck = torch.load(p, map_location=self.device, weights_only=False)
                h, w = int(ck.get("img_h", 64)), int(ck.get("img_w", 256))
                chars = ck.get("chars", [])
                if h == 32:
                    m = CRNN(n_classes=len(chars) + 1)
                else:
                    from scripts.plate_final_model import PlateCRNN
                    m = PlateCRNN(len(chars) + 1, img_h=h)
                m.load_state_dict(ck["model"])
                m.to(self.device).eval()
                if hw is None:
                    hw, self.chars = (h, w), chars
                elif (h, w) != hw:
                    logger.warning("Skipping %s: %dx%d does not match %s",
                                   p.name, h, w, hw)
                    continue
                members.append(m)
            except Exception as exc:
                logger.warning("Ensemble member %s failed to load: %s",
                               p.name, exc)

        if not members:
            return False

        self._members = members
        self.recognizer = members[0]
        self._rec_hw = hw
        self._preprocessor = PlatePreprocessorV2(target_h=hw[0], target_w=hw[1])
        logger.info("CRNN ensemble loaded: %d members at %dx%d on %s",
                    len(members), hw[0], hw[1], self.device)
        return True

    # =========================================================================
    # Stage 4.1 — Night Enhancement
    # =========================================================================

    def _enhance_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        enhanced, condition = self._night_enhancer.enhance(frame)
        return enhanced, condition

    # =========================================================================
    # Stage 4.2 — YOLOv8 Plate Detection (WITH FALSE POSITIVE FILTERING)
    # =========================================================================

    def detect_plate_bbox(
        self, vehicle_crop: np.ndarray, cls_id: Optional[int] = None
    ) -> Optional[List[float]]:
        """
        Locate the plate in a vehicle crop. Returns [x1, y1, x2, y2] or None.
        
        ✅ V4 FIX: Added aspect ratio and dimension filtering to reject false positives.
        """
        if vehicle_crop is None:
            return None

        h, w = vehicle_crop.shape[:2]
        min_veh_sz = 20 if cls_id == 3 else MIN_VEHICLE_BBOX_PX
        if h < min_veh_sz or w < min_veh_sz:
            return None

        # Allow smaller plates for motorcycles and dense junction footage
        min_w = 12 if cls_id == 3 else 18
        min_h = 4 if cls_id == 3 else 6
        det_conf = 0.05 if cls_id == 3 else min(0.15, PLATE_DET_CONF)

        detectors = (
            (getattr(self, "plate_detector_small", None), self.plate_detector)
            if cls_id == 3
            else (self.plate_detector, getattr(self, "plate_detector_small", None))
        )

        for detector in detectors:
            if detector is None:
                continue
            try:
                results = detector.predict(
                    vehicle_crop,
                    imgsz=PLATE_DET_IMGSZ,
                    conf=det_conf,
                    verbose=False,
                    device=self.device,
                )
                if (results
                        and results[0].boxes is not None
                        and len(results[0].boxes)):
                    boxes    = results[0].boxes
                    
                    # ✅ V4 FIX: Filter detections by aspect ratio and size
                    valid_boxes = []
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        x1, y1, x2, y2 = xyxy
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Skip if too small
                        if width < min_w or height < min_h:
                            continue
                        
                        # ✅ Skip if wrong aspect ratio (plates are NOT square)
                        aspect_ratio = width / height if height > 0 else 0
                        min_ar = 1.05 if cls_id == 3 else MIN_PLATE_ASPECT_RATIO
                        if aspect_ratio < min_ar or aspect_ratio > MAX_PLATE_ASPECT_RATIO:
                            continue
                        
                        valid_boxes.append((xyxy, conf))
                    
                    if valid_boxes:
                        # Return highest confidence valid box
                        best_box = max(valid_boxes, key=lambda x: x[1])
                        xyxy = best_box[0]
                        return [
                            float(xyxy[0]), float(xyxy[1]),
                            float(xyxy[2]), float(xyxy[3]),
                        ]
            except Exception as exc:
                logger.debug(
                    "Plate detection failed on crop %dx%d: %s", w, h, exc
                )
        return None

    # =========================================================================
    # Stage 4.4 — Plate Crop Extraction + Quality Gate
    # =========================================================================

    def extract_plate_candidate(
        self,
        frame:    np.ndarray,
        bbox:     List[float],
        cls_id:   int,
        lighting: str = "auto",
        raw_frame: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Extract, quality-gate, and preprocess the license plate strip.
        
        ✅ V4 FIX: Added aspect ratio validation on extracted crops.
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2  = [int(v) for v in bbox]

        x1 = max(0, min(w_frame - 1, x1))
        y1 = max(0, min(h_frame - 1, y1))
        x2 = max(0, min(w_frame,     x2))
        y2 = max(0, min(h_frame,     y2))

        bw = x2 - x1
        bh = y2 - y1
        min_vsz = 20 if cls_id == 3 else MIN_VEHICLE_BBOX_PX
        if bw < min_vsz or bh < min_vsz:
            return None, None, 0.0

        vehicle_crop = frame[y1:y2, x1:x2]
        raw_vehicle_crop = (
            raw_frame[y1:y2, x1:x2]
            if raw_frame is not None and raw_frame.shape[:2] == frame.shape[:2]
            else vehicle_crop
        )

        vcrop_h = vehicle_crop.shape[0]
        if cls_id == 2:
            offset_y = int(vcrop_h * 0.20)
        elif cls_id in (5, 7):
            offset_y = int(vcrop_h * 0.35)
        elif cls_id == 3:  # motorcycle: lower portion avoids rider helmet
            offset_y = int(vcrop_h * 0.15)
        else:
            offset_y = int(vcrop_h * 0.15)

        plate_search_crop = raw_vehicle_crop[offset_y:, :]
        raw_bgr_crop: Optional[np.ndarray] = None

        # ── Tier 1: YOLOv8 plate detector ─────────────────────────────────────
        if self._plate_det_available or getattr(self, "plate_detector_small", None) is not None:
            plate_box = self.detect_plate_bbox(plate_search_crop, cls_id=cls_id)
            if plate_box is not None:
                px1 = max(0,  int(round(plate_box[0])))
                py1 = max(0,  int(round(plate_box[1])) + offset_y)
                px2 = min(bw, int(round(plate_box[2])))
                py2 = min(bh, int(round(plate_box[3])) + offset_y)
                pw, ph = px2 - px1, py2 - py1
                
                # Check for two-line plate where only line 1 was boxed
                # If motorcycle or commercial vehicle and detected box is thin (AR >= 2.0, ph <= 18)
                # vertically expand downwards to capture Line 2
                if cls_id in (3, 5, 7) and (pw / max(1, ph)) >= 2.0 and ph <= 18:
                    py2 = min(bh, py2 + int(ph * 1.5))
                    ph = py2 - py1
                
                # ✅ Validate plate dimensions
                min_w = 12 if cls_id == 3 else MIN_PLATE_WIDTH_PX
                min_h = 4 if cls_id == 3 else MIN_PLATE_HEIGHT_PX
                min_ar = 1.05 if cls_id == 3 else MIN_PLATE_ASPECT_RATIO
                if pw >= min_w and ph >= min_h:
                    aspect_ratio = pw / ph if ph > 0 else 0
                    if min_ar <= aspect_ratio <= MAX_PLATE_ASPECT_RATIO:
                        pad_w = int(0.08 * pw)
                        pad_h = int(0.10 * ph)
                        cx1 = max(0, px1 - pad_w)
                        cy1 = max(0, py1 - pad_h)
                        cx2 = min(bw, px2 + pad_w)
                        cy2 = min(bh, py2 + pad_h)
                        raw_bgr_crop = raw_vehicle_crop[cy1:cy2, cx1:cx2].copy()

        # ── Tier 2: Adaptive Bumper & Fascia Fallback ─────────────────────────
        # When YOLO detector misses the plate box on high-speed or angled traffic,
        # extract the physical bumper region where plates are mounted.
        if raw_bgr_crop is None and bw >= 24 and bh >= 24:
            if cls_id == 3:  # motorcycle: lower rear/front fork
                by1 = int(bh * 0.55)
                by2 = int(bh * 0.98)
                bx1 = int(bw * 0.15)
                bx2 = int(bw * 0.85)
            elif cls_id in (5, 7):  # bus / truck: front bumper
                by1 = int(bh * 0.65)
                by2 = int(bh * 0.96)
                bx1 = int(bw * 0.15)
                bx2 = int(bw * 0.85)
            else:  # car / auto: lower 35% bumper
                by1 = int(bh * 0.60)
                by2 = int(bh * 0.95)
                bx1 = int(bw * 0.15)
                bx2 = int(bw * 0.85)
            bumper_crop = raw_vehicle_crop[by1:by2, bx1:bx2]
            if bumper_crop.size > 0 and bumper_crop.shape[1] >= 14:
                raw_bgr_crop = bumper_crop.copy()

        if raw_bgr_crop is None:
            return None, None, 0.0

        native_w = raw_bgr_crop.shape[1]
        floor_px = 12 if cls_id == 3 else min(16, FUSION_FLOOR_PX)
        if native_w < floor_px:
            return None, None, 0.0

        quality = self._frame_selector.score(raw_bgr_crop)
        min_q = 3.0  # Normalized floor allows valid plates with minor motion blur
        if quality < min_q:
            return None, None, quality

        if getattr(self, "_members", None):
            preprocessed = cv2.resize(
                cv2.cvtColor(raw_bgr_crop, cv2.COLOR_BGR2GRAY)
                if raw_bgr_crop.ndim == 3 else raw_bgr_crop,
                (self._rec_hw[1], self._rec_hw[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            preprocessed = self._preprocessor.preprocess(
                raw_bgr_crop, lighting=lighting
            )
        if preprocessed is None:
            return None, None, quality

        return preprocessed, raw_bgr_crop, quality

    # =========================================================================
    # Stage 4.3 — Geometric rectification (confidence-arbitrated)
    # =========================================================================

    @staticmethod
    def _plate_tilt_deg(gray: np.ndarray) -> float:
        """Baseline tilt of the glyph mass, in degrees.

        Measured from the minimum-area rectangle around the dark glyph mass
        rather than from border lines: on a small plate the border is often
        broken or missing on one side, while the glyphs are always present —
        they are the thing being read.

        Cheap enough to run on every crop, which is what lets the expensive
        part be skipped for the 64% of crops that have no tilt to correct.
        """
        h, w = gray.shape[:2]
        if h < 8 or w < 20:
            return 0.0
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if bw.mean() > 127:                    # foreground took over: invert
            bw = 255 - bw
        pts = cv2.findNonZero(bw)
        if pts is None or len(pts) < 20:
            return 0.0
        (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
        # OpenCV reports the shorter side's angle on some builds; fold into
        # [-45, 45] so a wide plate reads as a small tilt, not an 89 degree one.
        if rw < rh:
            ang += 90.0
        if ang > 45:
            ang -= 90.0
        if ang < -45:
            ang += 90.0
        return float(ang)

    def _rectified_variants(self, gray: np.ndarray) -> List[np.ndarray]:
        """Straightened alternatives to `gray`, or [] when none are worth it.

        Returns candidates only — the caller reads each and keeps whichever
        the recogniser is most confident about, so a bad warp costs a wasted
        inference rather than a wrong plate.
        """
        out: List[np.ndarray] = []
        h, w = gray.shape[:2]
        if w < RECTIFY_MIN_WIDTH_PX:
            return out
        ang = self._plate_tilt_deg(gray)
        if not (RECTIFY_MIN_TILT_DEG <= abs(ang) <= RECTIFY_MAX_TILT_DEG):
            # Already straight, or the estimate is implausible. This is the
            # common case and it costs one Otsu threshold, nothing more.
            return out

        rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        deskewed = cv2.warpAffine(gray, rot, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        out.append(deskewed)

        # Keystone correction, composed with the resize to the model input so
        # the pixels are interpolated once rather than twice. Warping to the
        # quad's own size and resizing afterwards softens an 80px plate
        # measurably more than a single combined transform does.
        quad = self._plate_quad(deskewed)
        if quad is not None:
            want_h, want_w = getattr(self, "_rec_hw", (IMG_H, IMG_W))
            dst = np.array([[0, 0], [want_w - 1, 0],
                            [want_w - 1, want_h - 1], [0, want_h - 1]],
                           dtype=np.float32)
            try:
                M = cv2.getPerspectiveTransform(quad, dst)
                out.append(cv2.warpPerspective(
                    deskewed, M, (want_w, want_h), flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE))
            except cv2.error:
                pass
        return out

    @staticmethod
    def _plate_quad(gray: np.ndarray) -> Optional[np.ndarray]:
        """Four corners of a plate-shaped convex quad, or None.

        Every gate here exists because a confident-but-wrong quadrilateral is
        the failure mode that made blind rectification lose: it warps a
        readable plate into an unreadable one, and nothing downstream can tell
        that happened. When any gate fails, no candidate is offered at all.
        """
        h, w = gray.shape[:2]
        if h < 12 or w < 30:
            return None
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 130)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 0.25 * h * w:
            return None
        peri = cv2.arcLength(c, True)
        quad = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            return None
        p = quad.reshape(4, 2).astype(np.float32)
        s, d = p.sum(1), np.diff(p, axis=1).ravel()
        src = np.array([p[np.argmin(s)], p[np.argmin(d)],
                        p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)
        wa = max(np.linalg.norm(src[0] - src[1]),
                 np.linalg.norm(src[3] - src[2]))
        ha = max(np.linalg.norm(src[0] - src[3]),
                 np.linalg.norm(src[1] - src[2]))
        # A plate is a wide rectangle. Anything else found by the contour
        # search is a bumper edge, a shadow, or the glyph cluster itself.
        if wa < 24 or ha < 8 or not (1.8 <= wa / ha <= 7.5):
            return None
        return src

    # =========================================================================
    # Stage 4.5 — OCR (CRNN + Ensemble)
    # =========================================================================

    @torch.no_grad()
    def _crnn_infer(
        self, plate_strip: np.ndarray
    ) -> Tuple[str, float]:
        if self.recognizer is None or plate_strip is None:
            return "", 0.0

        want_h, want_w = getattr(self, "_rec_hw", (IMG_H, IMG_W))
        if plate_strip.shape[:2] != (want_h, want_w):
            plate_strip = cv2.resize(plate_strip, (want_w, want_h),
                                     interpolation=cv2.INTER_AREA)

        t = (
            torch.from_numpy(plate_strip)
            .float()
            .div(127.5)
            .sub(1.0)[None, None]
            .to(self.device)
        )

        with torch.no_grad():
            members = getattr(self, "_members", None)
            if members and len(members) > 1:
                lp = torch.stack([
                    torch.log_softmax(m(t), dim=2) for m in members
                ]).mean(dim=0)
                probs = lp.exp()
                logits = lp
            else:
                logits = self.recognizer(t)
                probs = torch.softmax(logits, dim=2)

            max_probs, _ = torch.max(probs, dim=2)
            conf = float(torch.mean(max_probs).item())
            raw_text = ctc_decode(logits)[0]
        return (raw_text or ""), conf

    def recognize(
        self, plate_strip: np.ndarray
    ) -> Tuple[str, float]:
        return self._crnn_infer(plate_strip)

    def recognize_plate(
        self,
        plate_strip:   np.ndarray,
        raw_bgr_crop:  Optional[np.ndarray] = None,
        agreement:     float = 0.5,
        votes:         int   = 1,
        fused_crop:    Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], float, Optional[str]]:
        if plate_strip is None:
            return None, 0.0, None

        if self._ensemble is not None and raw_bgr_crop is not None:
            raw_text, conf, _src = self._ensemble.recognize(
                plate_crop=raw_bgr_crop,
                preprocessed_crop=plate_strip,
            )
        else:
            raw_text, conf = self._crnn_infer(plate_strip)

        # ── Rectification, arbitrated by confidence ───────────────────────────
        #
        # The straightened variants are read too, and the most confident read
        # wins. Strictly greater, so a tie leaves the original in place — the
        # untouched crop is preferred whenever rectification does not clearly
        # improve on it.
        if RECTIFY_ENABLED and raw_bgr_crop is not None:
            src_gray = (cv2.cvtColor(raw_bgr_crop, cv2.COLOR_BGR2GRAY)
                        if raw_bgr_crop.ndim == 3 and raw_bgr_crop.shape[2] >= 3
                        else raw_bgr_crop)
            for variant in self._rectified_variants(src_gray):
                v_text, v_conf = self._crnn_infer(variant)
                if v_text and len(v_text) >= 4 and v_conf > conf:
                    raw_text, conf = v_text, v_conf

        # ── Multi-frame fused master candidate, arbitrated by confidence ──────
        if fused_crop is not None and fused_crop.size > 0:
            f_gray = (cv2.cvtColor(fused_crop, cv2.COLOR_BGR2GRAY)
                      if fused_crop.ndim == 3 and fused_crop.shape[2] >= 3
                      else fused_crop)
            f_prep = (cv2.resize(f_gray, (self._rec_hw[1], self._rec_hw[0]),
                                 interpolation=cv2.INTER_AREA)
                      if getattr(self, "_rec_hw", None) else f_gray)
            f_text, f_conf = self._crnn_infer(f_prep)
            if f_text and len(f_text) >= 4 and f_conf > conf:
                logger.info("[ANPR-ACTIVE] [ECC FUSION ARBITRATION WON] Fused candidate '%s' (conf=%.3f) beat raw single-frame '%s' (conf=%.3f)", f_text, f_conf, raw_text, conf)
                raw_text, conf = f_text, f_conf

        # ── Double-line / Stacked plate candidate (2:1 aspect ratio) ──────────
        # ── Double-line / Stacked plate candidate (2:1 aspect ratio) ──────────
        target_crop = raw_bgr_crop if raw_bgr_crop is not None else fused_crop
        if DOUBLE_LINE_ENABLED and target_crop is not None and target_crop.size > 0:
            h_c, w_c = target_crop.shape[:2]
            ar_c = w_c / max(h_c, 1)
            if ar_c <= 2.5 and h_c >= 12 and w_c >= 16:
                try:
                    # Method 1: Horizontal Unstacking (Top half + Bottom half side-by-side)
                    split_y = int(h_c * 0.48)
                    t_half = target_crop[:split_y, :]
                    b_half = target_crop[split_y:, :]
                    if t_half.size > 0 and b_half.size > 0:
                        target_h = 32
                        tw = max(10, int(t_half.shape[1] * (target_h / max(1, t_half.shape[0]))))
                        bw = max(10, int(b_half.shape[1] * (target_h / max(1, b_half.shape[0]))))
                        th_r = cv2.resize(t_half, (tw, target_h), interpolation=cv2.INTER_CUBIC)
                        bh_r = cv2.resize(b_half, (bw, target_h), interpolation=cv2.INTER_CUBIC)
                        stitched = np.hstack([th_r, bh_r])
                        s_gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY) if stitched.ndim == 3 else stitched
                        s_prep = cv2.resize(s_gray, (self._rec_hw[1], self._rec_hw[0]), interpolation=cv2.INTER_AREA)
                        s_text, s_conf = self._crnn_infer(s_prep)
                        if s_text and len(s_text) >= 4:
                            _d = decode_plate(s_text, local_state=self.local_state, apply_prior=False)
                            is_valid = _d.get("plate") is not None and float(_d.get("score", 0.0)) >= DOUBLE_LINE_MIN_GRAMMAR
                            if (is_valid and s_conf > (conf - 0.10)) or s_conf > conf:
                                logger.info("[ANPR-ACTIVE] [UNSTACKED TWO-LINE] Sliced & stitched 2-line plate -> '%s' (conf=%.2f) beat raw single-line '%s' (conf=%.2f)", s_text, s_conf, raw_text, conf)
                                raw_text, conf = s_text, s_conf

                    # Method 2: Sliced line1 + line2 fallback
                    from backend.services.anpr_track_aggregator_v2 import split_stacked_plate
                    top_strip, bot_strip = split_stacked_plate(target_crop)
                    if top_strip is not None and bot_strip is not None:
                        t_gray = (cv2.cvtColor(top_strip, cv2.COLOR_BGR2GRAY)
                                  if top_strip.ndim == 3 and top_strip.shape[2] >= 3
                                  else top_strip)
                        b_gray = (cv2.cvtColor(bot_strip, cv2.COLOR_BGR2GRAY)
                                  if bot_strip.ndim == 3 and bot_strip.shape[2] >= 3
                                  else bot_strip)
                        t_prep = cv2.resize(t_gray, (self._rec_hw[1], self._rec_hw[0]),
                                            interpolation=cv2.INTER_AREA)
                        b_prep = cv2.resize(b_gray, (self._rec_hw[1], self._rec_hw[0]),
                                            interpolation=cv2.INTER_AREA)
                        t_text, t_conf = self._crnn_infer(t_prep)
                        b_text, b_conf = self._crnn_infer(b_prep)
                        if t_text and b_text:
                            stacked_text = t_text + b_text
                            stacked_conf = (t_conf + b_conf) / 2.0
                            if len(stacked_text) >= 6 and stacked_conf > conf:
                                _d = decode_plate(stacked_text, local_state=self.local_state, apply_prior=False)
                                if _d.get("plate") is not None and float(_d.get("score", 0.0)) >= DOUBLE_LINE_MIN_GRAMMAR:
                                    logger.info("[ANPR-ACTIVE] [DOUBLE-LINE ARBITRATION WON] Sliced candidate '%s' (conf=%.3f) beat single-line '%s' (conf=%.3f)", stacked_text, stacked_conf, raw_text, conf)
                                    raw_text, conf = stacked_text, stacked_conf
                except Exception as _exc:
                    logger.debug("Double-line plate split failed: %s", _exc)

        if not raw_text or len(raw_text) < 4:
            return None, 0.0, None

        decoded = decode_plate(
            raw_text,
            local_state=self.local_state,
            apply_prior=True,
            prior_agreement=agreement,
            prior_votes=votes,
        )
        plate         = decoded.get("plate")
        grammar_score = float(decoded.get("score", 0.0))
        state         = decoded.get("state")

        combined_conf = (
            conf * 0.6 + grammar_score * 0.4
            if plate else conf * 0.5
        )

        if plate and grammar_score >= 0.55:
            return plate, round(combined_conf, 3), state

        clean = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        if 7 <= len(clean) <= 11:
            resolved_st, _ = resolve_state_code(
                clean[:2], local_state=self.local_state
            )
            return resolved_st + clean[2:], round(conf * 0.5, 3), resolved_st

        return None, 0.0, None

    # =========================================================================
    # Stage 4.6+4.7 — Temporal Voting + Watchlist
    # =========================================================================

    def _fuse_views(self, crops: List[np.ndarray]) -> Optional[np.ndarray]:
        if len(crops) < 2:
            return None
        try:
            from backend.scripts.plate_multiframe_fusion import fuse
        except Exception:
            return None
        valid_crops = [c for c in crops if c is not None and c.size > 0]
        if len(valid_crops) < 2:
            return None
        try:
            fused, used = fuse(valid_crops)
        except Exception as exc:
            logger.debug("Plate fusion failed on %d views: %s", len(valid_crops), exc)
            return None
        if used < 2 or fused is None:
            return None
        logger.info("[ANPR-ACTIVE] [ECC FUSION ACTIVE] Successfully fused %d crops into master crop (size=%dx%d)", used, fused.shape[1], fused.shape[0])
        return fused

    def process_vehicle_track(
        self,
        frame:    np.ndarray,
        bbox:     List[float],
        cls_id:   int,
        track_id: int,
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        if now - self._last_cleanup > TRACK_CLEANUP_INTERVAL:
            self._cleanup_stale_tracks()
            self._last_cleanup = now

        if cls_id not in (2, 3, 5, 7):
            return None

        if track_id in self._confirmed_plates:
            confirmed = self._confirmed_plates[track_id]
            if confirmed.get("locked"):
                return confirmed

        enhanced_frame, lighting = self._enhance_frame(frame)

        preprocessed, raw_bgr, quality = self.extract_plate_candidate(
            enhanced_frame, bbox, cls_id, lighting=lighting, raw_frame=frame
        )
        if preprocessed is None:
            return self._confirmed_plates.get(track_id)

        native_w = raw_bgr.shape[1] if raw_bgr is not None else 0
        buf = self._fusion_crops.setdefault(track_id, [])
        if raw_bgr is not None and raw_bgr.size > 0:
            buf.append(raw_bgr)
            if len(buf) > FUSION_MAX_VIEWS:
                def _c_sharp(c):
                    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
                    return cv2.Laplacian(g, cv2.CV_64F).var()
                buf.sort(key=_c_sharp, reverse=True)
                del buf[FUSION_MAX_VIEWS:]

        fused_candidate = None
        fused_from = 0
        if len(buf) >= 3 and (native_w < SINGLE_FRAME_FLOOR_PX or not self._confirmed_plates.get(track_id)):
            if not hasattr(self, "_fused_crop_cache"):
                self._fused_crop_cache = {}
            if track_id not in self._fused_crop_cache or (len(buf) % 6 == 0):
                fused_candidate = self._fuse_views(buf)
                if fused_candidate is not None:
                    self._fused_crop_cache[track_id] = fused_candidate
                    fused_from = len(buf)
            else:
                fused_candidate = self._fused_crop_cache.get(track_id)

        single_floor = 12 if cls_id == 3 else min(18, SINGLE_FRAME_FLOOR_PX)
        if native_w < single_floor and fused_candidate is None:
            return self._confirmed_plates.get(track_id)

        voter   = self._get_voter(track_id)
        reads   = voter.tracks.get(track_id, [])
        n_reads = len(reads)

        agreement = 0.5
        if n_reads:
            texts = [r["text"] for r in reads if r.get("text")]
            if texts:
                agreement = texts.count(max(set(texts), key=texts.count)) / len(texts)

        plate_str, conf, state = self.recognize_plate(
            preprocessed,
            raw_bgr_crop=raw_bgr,
            agreement=float(agreement),
            votes=n_reads,
            fused_crop=fused_candidate,
        )

        if not plate_str:
            return self._confirmed_plates.get(track_id)

        if conf >= MIN_VOTE_CONFIDENCE:
            voter.add_reading(
                track_id=track_id,
                plate_text=plate_str,
                confidence=conf,
                quality_score=quality,
            )

        voted_plate, voted_conf = voter.vote(track_id)
        if voted_plate is None:
            voted_plate = plate_str
            voted_conf  = conf

        all_reads = voter.tracks.get(track_id, [])
        total_v   = len(all_reads)
        texts     = [r["text"] for r in all_reads]
        agree_v   = (
            texts.count(voted_plate) / max(total_v, 1)
            if texts else 0.0
        )

        if state and total_v <= 2 and agree_v < 0.95:
            voted_plate, state, _note = apply_state_prior(
                voted_plate, state, agree_v, total_v
            )

        is_locked = (
            (conf >= 0.70)
            or (total_v >= 2 and agree_v >= 0.50)
            or (total_v >= 1 and conf >= 0.60)
            or (total_v >= 3 and agree_v >= 0.66)
        )

        import datetime as _dt
        detection_event = {
            # The camera this read came from, not a hardcoded 0. Every
            # watchlist alert this service logged said "CAM0" regardless of
            # which camera saw the vehicle, because this key was literally
            # `0`. `track_id` is namespaced per camera by the caller
            # (see live_24x7_pipeline's _global_track_id), so it carries the
            # camera identity even though this method is not given it
            # directly.
            "cam_id":     getattr(self, "_current_cam_hint", 0) or 0,
            "track_id":   track_id,
            "plate":      voted_plate,
            "confidence": round(voted_conf, 3),
            "lighting":   lighting,
            "quality":    round(quality, 1),
            "timestamp":  _dt.datetime.now().isoformat(),
        }
        watchlist_hit = self._watchlist_svc.check_and_alert(detection_event)
        # WHY the reason is resolved here: the pipeline builds its alert
        # description from result["alert_reason"], which was hardcoded None —
        # so every stolen-vehicle alert would have read "(None)" as its
        # justification. The matcher already knows why the plate is flagged.
        watchlist_reason = None
        if watchlist_hit:
            try:
                _m = self._watchlist_svc._fuzzy_match(
                    (voted_plate or "").upper().replace(" ", ""))
                if _m:
                    watchlist_reason = self._watchlist_svc.watchlist.get(_m[0])
            except Exception:
                pass

        result: Dict[str, Any] = {
            "plate":          voted_plate,
            "raw_read":       plate_str,
            "confidence":     round(voted_conf, 3),
            "state":          state or (
                voted_plate[:2] if voted_plate and len(voted_plate) >= 2
                else None
            ),
            "vote_agreement": round(agree_v, 3),
            "total_votes":    total_v,
            "locked":         is_locked,
            "is_stolen":      watchlist_hit,
            "alert_reason":   watchlist_reason,
            "lighting":       lighting,
            "quality_score":  round(quality, 1),
            "detector_used":  "yolov8",
            "plate_native_px": int(raw_bgr.shape[1]) if raw_bgr is not None else 0,
            "read_grade": (
                "reliable"
                if raw_bgr is not None
                and raw_bgr.shape[1] >= RELIABLE_PLATE_NATIVE_PX
                else "marginal"
            ),
            "fused_views": fused_from,
            # Does this read carry the measured 80.4% assurance? See the
            # COMMIT_* constants for the operating point and why this labels
            # rather than filters.
            "committed": bool(
                raw_bgr is not None
                and raw_bgr.shape[1] >= COMMIT_MIN_NATIVE_PX
                and voted_conf >= COMMIT_MIN_CONF
            ),
            "commit_gate": {
                "min_native_px": COMMIT_MIN_NATIVE_PX,
                "min_confidence": COMMIT_MIN_CONF,
                "measured_exact_match": 0.804,
                "measured_coverage": 0.614,
                "measured_on": "83 held-out vehicles, 2026-09-06",
            },
        }

        # THE ALERT GATE.
        #
        # live_24x7_pipeline raises a STOLEN_VEHICLE_WATCHLIST_HIT only for
        # tracks present in _confirmed_plates. Storing solely on `is_locked`
        # meant a watchlist match was found, logged by WatchlistService, and
        # then silently dropped unless that track independently reached a
        # vote lock (>=3 agreeing votes, or >=2 votes at >=0.85). A vehicle
        # crossing the frame once — read cleanly at 0.97 and matching a
        # watchlist plate EXACTLY — produced no alert at all. Measured: that
        # is why this deployment had 0 watchlist alerts in its history
        # despite the matcher firing correctly.
        #
        # A watchlist hit is therefore retained even when unlocked, but only
        # above WATCHLIST_UNLOCKED_MIN_CONF — the same confidence bar the
        # lock's own two-vote branch uses. The safety argument for the lock
        # is not weakened: it exists so a GARBAGE read is not published as a
        # confident identification, and this keeps that bar on confidence
        # while removing the requirement that the same vehicle be seen
        # repeatedly. `locked` and `total_votes` travel with the result, so
        # a consumer can still tell a single-view match from a voted one and
        # present it accordingly.
        if is_locked or conf >= 0.60 or (watchlist_hit and conf >= WATCHLIST_UNLOCKED_MIN_CONF):
            self._confirmed_plates[track_id] = result

        return result

    def check_watchlist(
        self, plate_number: Optional[str]
    ) -> Tuple[bool, Optional[str]]:
        if not plate_number:
            return False, None
        clean = re.sub(r"[^A-Z0-9]", "", plate_number.upper())
        match = self._watchlist_svc._fuzzy_match(clean)
        if match:
            matched_plate, match_type = match
            # The actual reason ("Stolen — FIR #...", "Wanted — armed
            # robbery") lives in watchlist_plates.reason, looked up the same
            # way check_and_alert() already does it — this used to return
            # only a generic "Watchlist hit: X (exact)" with no way for a
            # caller to know WHY the plate is flagged.
            reason = self._watchlist_svc.watchlist.get(matched_plate, "Watchlist")
            return True, f"{reason} [{matched_plate}, {match_type}]"
        return False, None

    def _get_voter(self, track_id: int) -> TemporalPlateVoter:
        if track_id not in self._voters:
            self._voters[track_id] = TemporalPlateVoter(
                min_reads=3,
                min_confidence=MIN_VOTE_CONFIDENCE,
            )
        return self._voters[track_id]

    def finalize_track(self, track_id: int) -> Optional[Dict[str, Any]]:
        self._fusion_crops.pop(track_id, None)
        voter = self._voters.pop(track_id, None)
        if voter is None:
            return self._confirmed_plates.pop(track_id, None)
        result = voter.finalize(track_id)
        self._confirmed_plates.pop(track_id, None)
        return result

    def _cleanup_stale_tracks(self):
        if len(self._voters) > MAX_TRACK_HISTORIES:
            keep     = set(list(self._voters.keys())[-MAX_TRACK_HISTORIES // 2:])
            stale_v  = [k for k in self._voters          if k not in keep]
            stale_c  = [k for k in self._confirmed_plates if k not in keep]
            stale_f  = [k for k in self._fusion_crops     if k not in keep]
            for k in stale_v:  del self._voters[k]
            for k in stale_c:  del self._confirmed_plates[k]
            for k in stale_f:  del self._fusion_crops[k]
            logger.info(
                "Track cleanup: removed %d voters + %d confirmed",
                len(stale_v), len(stale_c),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ────────────────────────────────────────────────────────────────────────────

_anpr_singleton: Optional[ANPREngine] = None


def get_anpr_engine() -> ANPREngine:
    global _anpr_singleton
    if _anpr_singleton is None:
        _anpr_singleton = ANPREngine()
    return _anpr_singleton