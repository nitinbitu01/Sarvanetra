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
import threading
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

# Sentinel for "run the plate detector here" (a precomputed None means "the
# batched detector found no plate", which must not trigger a second run).
_DETECT = object()

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
    PROJECT_ROOT / "models" / "plate_detector" / "plate_v4_small.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "plate" / "plate_v3_ft" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "plate_v3" / "recovered"  / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "plate"    / "plate_v2"   / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "plate"    / "plate_v1"   / "weights" / "best.pt",
]

# Secondary, small-plate-biased detector
SMALL_PLATE_DETECTOR: Path = (
    PROJECT_ROOT / "models" / "plate_detector" / "plate_v4_small.pt"
)
PLATE_DET_ENGINE: Path = PROJECT_ROOT / "models" / "plate_detector" / "plate_v3_ft_320_fp16.engine"
# Plate / not-plate classifier (backend/scripts/train_plate_verifier.py).
# SENTINEL_PLATE_VERIFIER=1 turns it on; crops scoring below the threshold
# are treated as "no plate here" instead of being read.
PLATE_VERIFIER_PATH: Path = Path(os.environ.get("SENTINEL_PLATE_VERIFIER_PATH", "").strip()
                                 or PROJECT_ROOT / "models" / "plate_verifier" / "verifier.pt")
PLATE_VERIFIER_THR = float(os.environ.get("SENTINEL_PLATE_VERIFIER_THR", "0.9"))
# Report gate (see process_vehicle_track): minimum CRNN confidence and minimum
# read stability under perturbation for a read to be voted on and reported.
REPORT_GATE = os.environ.get("SENTINEL_REPORT_GATE", "0") == "1"
GATE_MIN_CONF = float(os.environ.get("SENTINEL_GATE_MIN_CONF", "0.90"))
GATE_MIN_STAB = float(os.environ.get("SENTINEL_GATE_MIN_STAB", "0.5"))
GATE_MIN_MARGIN = float(os.environ.get("SENTINEL_GATE_MIN_MARGIN", "0.5"))
# Plate-style routing to the expert recogniser (see _plate_style).
# real green views 0.49-0.59 in the 82-95 hue band; held-out non-green max 0.27
STYLE_GREEN_FRAC = float(os.environ.get("SENTINEL_STYLE_GREEN_FRAC", "0.4"))
# Green EV plates read inverted by the shipped recogniser (see
# _process_vehicle_track_impl). On by default.
GREEN_INVERT = os.environ.get("SENTINEL_GREEN_INVERT", "1") == "1"
# Verifier threshold for colourless (infrared night) crops; unset = same as
# PLATE_VERIFIER_THR. GRAY_SAT_MAX: mean HSV saturation below which a crop
# counts as colourless.
_vtg = os.environ.get("SENTINEL_VERIFIER_THR_GRAY", "").strip()
VERIFIER_THR_GRAY = float(_vtg) if _vtg else None
GRAY_SAT_MAX = float(os.environ.get("SENTINEL_GRAY_SAT_MAX", "12"))
# Report gate's second route: the style expert agrees on the voted plate
# (needs SENTINEL_STYLE_EXPERT=1 so the expert is loaded).
CROSS_MODEL_GATE = os.environ.get("SENTINEL_GATE_CROSS_MODEL", "0") == "1"
# Report only what a second, differently trained recogniser also reads.
REQUIRE_AGREE = os.environ.get("SENTINEL_GATE_REQUIRE_AGREE", "0") == "1"
# Lower margin for stable, confident reads (unset = off).
_gms = os.environ.get("SENTINEL_GATE_MARGIN_STABLE", "").strip()
GATE_MARGIN_STABLE = float(_gms) if _gms else None
GATE_STAB_HIGH = float(os.environ.get("SENTINEL_GATE_STAB_HIGH", "0.8"))
GATE_CONF_HIGH = float(os.environ.get("SENTINEL_GATE_CONF_HIGH", "0.95"))
# Read the unpadded plate box as well and keep the more confident read.
TIGHT_CROP_ARB = os.environ.get("SENTINEL_TIGHT_CROP_ARB", "0") == "1"
# Confidence the tight read must beat the padded one by. Swapping on any gain
# cost 3 points of held-out precision (93.1 -> 90.0); the framed-plate case it
# is meant for had 0.48 -> 0.97.
TIGHT_CROP_GAP = float(os.environ.get("SENTINEL_TIGHT_CROP_GAP", "0.15"))
# Clear a track's reads when its vehicle crop changes appearance (tracker
# identity switch). Distance is Bhattacharyya on an HS histogram.
APPEARANCE_RESET = os.environ.get("SENTINEL_APPEARANCE_RESET", "0") == "1"
APPEARANCE_DIST = float(os.environ.get("SENTINEL_APPEARANCE_DIST", "0.5"))
# Off (0): crop aspect does not separate two-row plates - single-row eye crops
# span 1.58-2.63+ (p1-p50) and two-row views 1.46-2.68; at 2.3 it routed 43
# held-out single-row crops and cut legible recall 77.8% -> 66.7%.
STYLE_TWOROW_AR = float(os.environ.get("SENTINEL_STYLE_TWOROW_AR", "0"))
# Views taking part in a track's vote: plate width >= this fraction of the
# widest plate the track has shown (0 = every view, the old behaviour).
# Measured 2026-09-18 on the answer key + held-out set: 0.75 gained nothing
# (key 3/7 -> 2/7 at margin 0.5), so it is off by default.
GATE_NEAR_FRAC = float(os.environ.get("SENTINEL_GATE_NEAR_FRAC", "0"))
# Window in which two tracks on one camera with plates one character apart
# are treated as one vehicle split by the tracker (_consistent_with_recent).
CONSIST_WINDOW_S = float(os.environ.get("SENTINEL_CONSIST_WINDOW_S", "20"))
SMALL_PLATE_DET_ENGINE: Path = PROJECT_ROOT / "models" / "plate_detector" / "plate_v4_small_320_fp16.engine"

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
PLATE_DET_CONF          = float(os.environ.get("SENTINEL_PLATE_DET_CONF", "0.15"))
PLATE_DET_IMGSZ         = int(os.environ.get("SENTINEL_PLATE_DET_IMGSZ", "640"))

MIN_PLATE_WIDTH_PX      = 25
MIN_PLATE_HEIGHT_PX     = 8
# Allows Indian commercial plates (340x200mm, AR 1.70), 2-row plates (AR ~0.85-1.2)
# and 2-wheeler plates (AR 1.2-1.5) while rejecting truly square badges (AR < 0.75).
# Default lowered from 1.2 → 0.85: Indian taxi, auto, commercial, motorcycle 2-row
# plates all have AR 0.85-1.18 and were silently rejected at 1.2 (fix: 2026-09-22).
MIN_PLATE_ASPECT_RATIO  = float(os.environ.get("SENTINEL_MIN_PLATE_ASPECT_RATIO", "0.85"))
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
# SINGLE_FRAME_FLOOR_PX: minimum native plate width to attempt single-frame OCR.
# Lowered from 35 → 22 px: overhead cameras (CAM_07) and busy intersections
# (CAM_04) have many mid-distance vehicles whose plates land at 22-34px.
# Below 22px the CRNN is still noise and must wait for multi-frame fusion.
# Overridable so per-site calibration can tighten it back if needed.
SINGLE_FRAME_FLOOR_PX   = int(os.environ.get("SENTINEL_SINGLE_FRAME_FLOOR_PX", "22"))
# Multi-frame fusion is OFF by default. SENTINEL_PLATE_FUSION=1 turns it back on.
#
# The old note here credited fusion with "+16.5 points at 40px, +11.4 at 50px".
# Those figures came from plate_multiframe_fusion --evaluate, which used the old
# finetuned.pt 32x128 recogniser on vehicles drawn from ALL labelled data,
# including the ones the shipped ensemble was trained on. Measured 2026-09-16
# with the shipped 64x256 ensemble on vehicles it never saw, simulating this
# engine's own arbitration (eval_fusion_arbitration.py):
#
#                          with fusion   without fusion
#     validation, 83         55.4%          65.1%    fixed 8, broke 0
#     test, 83               55.4%          56.6%    fixed 2, broke 1
#     committed wrong        22 / 12        16 / 11
#
# The fused image reads with systematically HIGHER confidence, so it won ~390
# frame arbitrations per split while being read less accurately — it replaced
# correct single-frame reads. It also cost 54% of the ANPR stage on CAM_09
# (~114 ms of CPU ECC per fusion, 44 ms per frame).
PLATE_FUSION_ENABLED    = os.environ.get("SENTINEL_PLATE_FUSION", "0") == "1"
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
MIN_STRIP_QUALITY       = float(os.environ.get("SENTINEL_MIN_STRIP_QUALITY", "12.0"))
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
# Re-measured 2026-09-17 on the 83 held-out vehicles through recognize_plate
# (backend/scripts/eval_engine_recognize_ab.py): ON 56.6% exact / 83.1%
# within-1, OFF 56.6% / 83.1% — identical — while ON cost 2.12 recogniser calls
# per crop against 1.00. The default had been flipped to "1" with no
# measurement behind it; back to "0".
DOUBLE_LINE_ENABLED     = os.environ.get("SENTINEL_DOUBLE_LINE", "0") == "1"
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

        # ── Concurrency guards ────────────────────────────────────────────────
        # The pipeline runs two ANPR worker threads per process against this one
        # engine (live_24x7_pipeline: n_threads = 2 when there is no shared ANPR
        # service), and its comment claimed the engine "is thread-safe: its
        # internal locks are per-track_id". There were no locks in this module
        # at all, so two crops of the same track — normal, since cars queue 3
        # and two-wheelers 6 crops per track — could interleave their votes and
        # their lock decisions. One lock per track keeps the parallelism the
        # second thread exists for (different tracks still read concurrently)
        # while serialising same-track work.
        self._track_locks: Dict[int, threading.Lock] = {}
        self._track_lock_guard = threading.Lock()
        # The plate-style expert swaps self.recognizer/_members for one plate
        # and swaps them back in a finally — process-global state, so while an
        # expert is loaded every call takes one lock. SENTINEL_ANPR_SERIALIZE=1
        # forces that single-lock behaviour everywhere, which is the safe choice
        # on a TensorRT/CUDA worker where concurrent forward passes are not
        # proven safe.
        self._serialize_lock = threading.Lock()
        self._serialize_all = os.environ.get("SENTINEL_ANPR_SERIALIZE", "0") == "1"

        # ── Load models ───────────────────────────────────────────────────────
        self._load_plate_detector()
        # Deterministic kernels: a read that sits on a gate threshold flipped
        # between identical runs (green GJ32AG2883, live_2239 track 77).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self._load_recognizer()
        self._load_verifier()
        self.style_routed: Dict[str, int] = {}
        self._load_style_expert()

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

        # SENTINEL_PLATE_DET_TRT=1: the same two detectors as FP16 TensorRT
        # engines (backend/scripts/build_trt_engine.py, 320x320, batch 1-32),
        # for the shared ANPR service on a GPU also running seven detection
        # workers. Falls back to the .pt models if an engine will not load.
        if os.environ.get("SENTINEL_PLATE_DET_TRT", "0") == "1" and str(self.device).startswith("cuda"):
            from ultralytics import YOLO
            for attr, eng in (("plate_detector", PLATE_DET_ENGINE),
                              ("plate_detector_small", SMALL_PLATE_DET_ENGINE)):
                if getattr(self, attr) is None or not eng.exists():
                    continue
                try:
                    m = YOLO(str(eng), task="detect")
                    m.predict([np.zeros((64, 64, 3), dtype=np.uint8)] * 2, verbose=False,
                              device=self.device, imgsz=PLATE_DET_IMGSZ)
                    m._sentinel_trt = True
                    setattr(self, attr, m)
                    logger.info("Plate detector %s: TensorRT engine %s", attr, eng.name)
                except Exception as exc:                            # noqa: BLE001
                    logger.warning("TensorRT plate engine %s not used (%s); keeping .pt", eng.name, exc)

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
        # SENTINEL_RECOGNIZER_DIR: a candidate ensemble scored in the live path
        # without replacing the shipped one.
        rec_dir = Path(os.environ.get("SENTINEL_RECOGNIZER_DIR", "").strip()
                       or PROJECT_ROOT / "models" / "plate_recognizer")
        paths = sorted(rec_dir.glob("final_m*.pt"))
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

        # ── Night-finetuned CRNN ensemble ─────────────────────────────────────
        night_dir = PROJECT_ROOT / "models" / "plate_recognizer_night"
        night_paths = sorted(night_dir.glob("final_m*.pt"))
        members_night = []
        for np_path in night_paths:
            try:
                ck_n = torch.load(np_path, map_location=self.device, weights_only=False)
                hn, wn = int(ck_n.get("img_h", 64)), int(ck_n.get("img_w", 256))
                chars_n = ck_n.get("chars", [])
                from scripts.plate_final_model import PlateCRNN
                mn = PlateCRNN(len(chars_n) + 1, img_h=hn)
                mn.load_state_dict(ck_n["model"])
                mn.to(self.device).eval()
                members_night.append(mn)
            except Exception as n_exc:
                logger.debug("Night ensemble member %s not loaded: %s", np_path.name, n_exc)
        self._members_night = members_night
        if members_night:
            logger.info("Night CRNN ensemble loaded: %d members on %s", len(members_night), self.device)
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
        min_veh_sz = 16 if cls_id in (1, 3) else MIN_VEHICLE_BBOX_PX
        if h < min_veh_sz or w < min_veh_sz:
            return None

        det_conf = 0.05 if cls_id in (1, 3) else min(0.15, PLATE_DET_CONF)

        # Dynamic low-light enhancement for dark vehicle crops (zero hardcoding)
        crop_mean = float(np.mean(vehicle_crop))
        crops_to_try = [vehicle_crop]
        if crop_mean < 75.0:
            try:
                from backend.scripts.night_enhancer import enhance_crop_adaptive
                crops_to_try.append(enhance_crop_adaptive(vehicle_crop, crop_mean))
                det_conf = max(0.04, det_conf * 0.7)
            except Exception:
                pass

        detectors = (
            (getattr(self, "plate_detector_small", None), self.plate_detector)
            if cls_id in (1, 3)
            else (self.plate_detector, getattr(self, "plate_detector_small", None))
        )

        for cur_crop in crops_to_try:
            for detector in detectors:
                if detector is None:
                    continue
                try:
                    results = detector.predict(
                        cur_crop,
                        imgsz=PLATE_DET_IMGSZ,
                        conf=det_conf,
                        verbose=False,
                        device=self.device,
                    )
                    box = self._best_plate_box(results[0] if results else None, cls_id)
                    if box is not None:
                        return box
                except Exception as exc:
                    logger.debug(
                        "Plate detection failed on crop %dx%d: %s", w, h, exc
                    )

        # Multi-scale pass: if crop is small, upscale to resolve tiny plates
        if w < 320 and h >= 18:
            scale = min(3.0, 320.0 / max(1, w))
            up_w = int(w * scale)
            up_h = int(h * scale)
            try:
                up_crop = cv2.resize(vehicle_crop, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
                for detector in detectors:
                    if detector is None:
                        continue
                    results = detector.predict(
                        up_crop,
                        imgsz=PLATE_DET_IMGSZ,
                        conf=max(0.04, det_conf * 0.7),
                        verbose=False,
                        device=self.device,
                    )
                    up_box = self._best_plate_box(results[0] if results else None, cls_id)
                    if up_box is not None:
                        return [
                            float(up_box[0] / scale),
                            float(up_box[1] / scale),
                            float(up_box[2] / scale),
                            float(up_box[3] / scale),
                        ]
            except Exception:
                pass

        return None


    @staticmethod
    def _best_plate_box(result, cls_id: Optional[int]) -> Optional[List[float]]:
        """The highest-confidence plate-shaped box in one detector result.

        Accepts 2-row / square Indian plates (AR >= 0.75) as well as standard
        rectangular plates (AR 1.0-6.0). Previously the 1.2 floor silently
        dropped taxi, auto, motorcycle 2-row and high-mount commercial plates.
        """
        if result is None or result.boxes is None or not len(result.boxes):
            return None
        min_w = 8 if cls_id in (1, 3) else 16
        min_h = 3 if cls_id in (1, 3) else 5
        # Two-wheelers allow slightly squarer 2-row plates; for all
        # others use MIN_PLATE_ASPECT_RATIO (default 0.85 from env).
        min_ar = 0.70 if cls_id in (1, 3) else MIN_PLATE_ASPECT_RATIO
        xyxys = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        best = None
        for (x1, y1, x2, y2), conf in zip(xyxys, confs):
            width, height = x2 - x1, y2 - y1
            if width < min_w or height < min_h:
                continue
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < min_ar or aspect_ratio > MAX_PLATE_ASPECT_RATIO:
                continue
            if best is None or conf > best[1]:
                best = ([float(x1), float(y1), float(x2), float(y2)], float(conf))
        return best[0] if best else None

    @staticmethod
    def _plate_search_offset(cls_id: int, crop_h: int) -> int:
        """Rows skipped from the top of a vehicle crop before looking for a plate.

        cls_id=1 (bicycle) is used for scooties / scooters by YOLOv8 on
        Indian traffic data. Treat it identically to motorcycle (cls_id=3):
        skip top 30% to avoid rider helmet, shoulders, and chest pattern noise,
        focusing the plate detector directly on the vehicle body and mudguard.
        """
        if cls_id in (1, 3):              # motorcycle, scootie (YOLO may label as bicycle)
            return int(crop_h * 0.30)     # avoid rider helmet / chest / headlamp noise
        if cls_id == 2:                   # car / auto
            return int(crop_h * 0.20)
        if cls_id in (5, 7):              # bus / truck: tall cab
            return int(crop_h * 0.35)
        return int(crop_h * 0.20)         # default

    def detect_plate_bboxes_batch(
        self, vehicle_crops: List[np.ndarray], cls_ids: List[int]
    ) -> List[Optional[List[float]]]:
        """detect_plate_bbox for many vehicle crops, with one detector call per pass.

        Same result per crop as detect_plate_bbox: the class's preferred
        detector first, the other only for crops the first found nothing on.
        Each crop is the plate SEARCH crop (offset already applied). Calling the
        detector once per crop was the GPU cost that made per-worker ANPR on 30
        cameras stall detection (2026-09-17: 280-630 ms per read, 8 copies).
        """
        n = len(vehicle_crops)
        out: List[Optional[List[float]]] = [None] * n
        eligible = []
        for i, (crop, cls_id) in enumerate(zip(vehicle_crops, cls_ids)):
            if crop is None:
                continue
            h, w = crop.shape[:2]
            min_veh_sz = 16 if cls_id in (1, 3) else MIN_VEHICLE_BBOX_PX
            if h >= min_veh_sz and w >= min_veh_sz:
                eligible.append(i)

        big = self.plate_detector
        small = getattr(self, "plate_detector_small", None)
        # pass 1: preferred detector; pass 2: the other one, for misses only
        for pass_no in (0, 1):
            # (detector, is_motorcycle) -> crop indices. The detection
            # confidence differs by class, so the two classes run separately.
            groups: Dict[Tuple[int, bool], Tuple[Any, List[int]]] = {}
            for i in eligible:
                if out[i] is not None:
                    continue
                is_moto = cls_ids[i] in (1, 3)
                det = ((small, big) if is_moto else (big, small))[pass_no]
                if det is None:
                    continue
                groups.setdefault((id(det), is_moto), (det, []))[1].append(i)
            for (_, is_moto), (det, idxs) in groups.items():
                det_conf = 0.05 if is_moto else min(0.15, PLATE_DET_CONF)
                imgs = [vehicle_crops[i] for i in idxs]
                try:
                    results = det.predict(
                        imgs, imgsz=PLATE_DET_IMGSZ,
                        conf=det_conf, verbose=False, device=self.device)
                except Exception as exc:                            # noqa: BLE001
                    logger.debug("Batched plate detection failed (%d crops): %s", len(idxs), exc)
                    continue
                for i, res in zip(idxs, results):
                    out[i] = self._best_plate_box(res, cls_ids[i])
        return out

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
        plate_box: Any = _DETECT,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Extract, quality-gate, and preprocess the license plate strip.

        `plate_box`: the detector's answer for this crop's search region when a
        batch already ran it (detect_plate_bboxes_batch) — a box, or None for
        "no plate found". Left out, the detector runs here as before.

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
        min_vsz = 16 if cls_id in (1, 3) else MIN_VEHICLE_BBOX_PX
        if bw < min_vsz or bh < min_vsz:
            return None, None, 0.0

        vehicle_crop = frame[y1:y2, x1:x2]
        raw_vehicle_crop = (
            raw_frame[y1:y2, x1:x2]
            if raw_frame is not None and raw_frame.shape[:2] == frame.shape[:2]
            else vehicle_crop
        )

        offset_y = self._plate_search_offset(cls_id, vehicle_crop.shape[0])

        plate_search_crop = raw_vehicle_crop[offset_y:, :]
        raw_bgr_crop: Optional[np.ndarray] = None

        # ── Tier 1: YOLOv8 plate detector ─────────────────────────────────────
        if self._plate_det_available or getattr(self, "plate_detector_small", None) is not None:
            if plate_box is _DETECT:
                plate_box = self.detect_plate_bbox(plate_search_crop, cls_id=cls_id)
                # Full-crop retry: if the offset search found nothing, retry on the
                # entire raw vehicle crop. High-mounted front plates on cars and
                # commercial vehicles sit in the top 20-35% and are clipped by the
                # offset. The retry uses a lower confidence floor (0.04) to catch
                # partial detections on small crops.
                if plate_box is None and offset_y > 0:
                    plate_box = self.detect_plate_bbox(raw_vehicle_crop, cls_id=cls_id)
                    if plate_box is not None:
                        # Coordinates are already relative to raw_vehicle_crop (no offset),
                        # so we must NOT add offset_y later. Set offset_y=0 for this path.
                        offset_y = 0
            if plate_box is not None:
                px1 = max(0,  int(round(plate_box[0])))
                py1 = max(0,  int(round(plate_box[1])) + offset_y)
                px2 = min(bw, int(round(plate_box[2])))
                py2 = min(bh, int(round(plate_box[3])) + offset_y)
                pw, ph = px2 - px1, py2 - py1
                
                # Check for two-line plate where only line 1 was boxed
                # If motorcycle or commercial vehicle and detected box is thin
                # vertically expand downwards to capture Line 2
                if cls_id in (1, 3) and (pw / max(1, ph)) >= 1.6 and ph <= 26:
                    py2 = min(bh, py2 + int(ph * 1.8))
                    ph = py2 - py1
                elif cls_id in (5, 7) and (pw / max(1, ph)) >= 2.0 and ph <= 20:
                    py2 = min(bh, py2 + int(ph * 1.2))
                    ph = py2 - py1
                
                # ✅ Validate plate dimensions
                min_w = 8 if cls_id in (1, 3) else MIN_PLATE_WIDTH_PX
                min_h = 3 if cls_id in (1, 3) else MIN_PLATE_HEIGHT_PX
                # Accept 2-row / square plates: AR floor lowered to 0.70 for two-wheelers,
                # MIN_PLATE_ASPECT_RATIO (0.85 default) for all other classes.
                min_ar = 0.70 if cls_id in (1, 3) else MIN_PLATE_ASPECT_RATIO
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
            if cls_id in (1, 3):  # motorcycle / scootie (cls 1=bicycle also used for scooties)
                # Two-wheeler plates are at: rear fork (60-95%) or front fork (75-100%)
                # Try rear-mount zone first (most common for Indian two-wheelers)
                by1 = int(bh * 0.55)
                by2 = int(bh * 0.97)
                bx1 = int(bw * 0.15)
                bx2 = int(bw * 0.85)
            elif cls_id in (5, 7):  # bus / truck: front bumper
                by1 = int(bh * 0.70)
                by2 = int(bh * 0.95)
                bx1 = int(bw * 0.25)
                bx2 = int(bw * 0.75)
            else:  # car / auto: lower center bumper
                by1 = int(bh * 0.68)
                by2 = int(bh * 0.92)
                bx1 = int(bw * 0.25)
                bx2 = int(bw * 0.75)
            bumper_crop = raw_vehicle_crop[by1:by2, bx1:bx2]
            if bumper_crop.size > 0 and bumper_crop.shape[1] >= 14:
                raw_bgr_crop = bumper_crop.copy()

        if raw_bgr_crop is None:
            return None, None, 0.0

        native_w = raw_bgr_crop.shape[1]
        floor_px = 10 if cls_id in (1, 3) else min(16, FUSION_FLOOR_PX)
        if native_w < floor_px:
            return None, None, 0.0

        quality = self._frame_selector.score(raw_bgr_crop)
        # Two-wheelers at speed have more motion blur; lower the quality floor
        # so small-but-legible plates aren't rejected by the Laplacian score.
        min_q = 2.0 if cls_id in (1, 3) else 3.0
        if quality < min_q:
            return None, None, quality

        if getattr(self, "_members", None):
            # ── Optional super-resolution on small plate crops ─────────────────
            # When SENTINEL_PLATE_SR=1, upscale crops narrower than the target
            # width (default 128px) before preprocessing. Dramatically improves
            # exact-match rate at sub-50px native plate widths.
            _sr_target = int(os.environ.get("SENTINEL_PLATE_SR_TARGET_WIDTH", "128"))
            if os.environ.get("SENTINEL_PLATE_SR", "0") == "1" and raw_bgr_crop.shape[1] < _sr_target:
                try:
                    from backend.scripts.plate_super_res import upscale_plate
                    raw_bgr_crop = upscale_plate(raw_bgr_crop, target_width=_sr_target)
                except Exception:
                    pass  # fail-safe: use original crop
            crop_to_use = raw_bgr_crop
            if crop_to_use is not None and crop_to_use.size > 0:
                plate_b = float(np.mean(crop_to_use))
                if plate_b < 75.0:
                    try:
                        from backend.scripts.night_enhancer import enhance_crop_adaptive
                        crop_to_use = enhance_crop_adaptive(crop_to_use, plate_b)
                    except Exception:
                        pass
            preprocessed = cv2.resize(
                cv2.cvtColor(crop_to_use, cv2.COLOR_BGR2GRAY)
                if crop_to_use.ndim == 3 else crop_to_use,
                (self._rec_hw[1], self._rec_hw[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            # ── Optional super-resolution on small plate crops ─────────────────
            _sr_target = int(os.environ.get("SENTINEL_PLATE_SR_TARGET_WIDTH", "128"))
            if os.environ.get("SENTINEL_PLATE_SR", "0") == "1" and raw_bgr_crop.shape[1] < _sr_target:
                try:
                    from backend.scripts.plate_super_res import upscale_plate
                    raw_bgr_crop = upscale_plate(raw_bgr_crop, target_width=_sr_target)
                except Exception:
                    pass  # fail-safe: use original crop
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
        cache = getattr(self, "_crnn_batch_cache", None)
        if cache and not getattr(self, "_expert_active", False):   # cache holds default-model reads
            hit = cache.get(self._strip_key(plate_strip))
            if hit is not None:
                return hit

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

            # Dynamic low-light arbitration with night-finetuned CRNN ensemble
            members_night = getattr(self, "_members_night", None)
            strip_mean = float(plate_strip.mean()) if hasattr(plate_strip, "mean") else 100.0
            if members_night and strip_mean < 80.0:
                try:
                    lp_n = torch.stack([
                        torch.log_softmax(m(t), dim=2) for m in members_night
                    ]).mean(dim=0)
                    probs_n = lp_n.exp()
                    max_probs_n, _ = torch.max(probs_n, dim=2)
                    conf_n = float(torch.mean(max_probs_n).item())
                    text_n = ctc_decode(lp_n)[0]
                    if (conf_n > conf and len(text_n or "") >= 6) or (conf < 0.85 and conf_n >= 0.85):
                        raw_text, conf = text_n, conf_n
                except Exception:
                    pass
        return (raw_text or ""), conf

    @staticmethod
    def _strip_key(strip: np.ndarray):
        return (strip.shape, hash(strip.tobytes()))

    @torch.no_grad()
    def _crnn_infer_batch(self, strips: List[np.ndarray]) -> List[Tuple[str, float]]:
        """_crnn_infer for many strips in one forward pass per ensemble member."""
        if self.recognizer is None or not strips:
            return [("", 0.0)] * len(strips)
        want_h, want_w = getattr(self, "_rec_hw", (IMG_H, IMG_W))
        arr = np.stack([
            s if s.shape[:2] == (want_h, want_w)
            else cv2.resize(s, (want_w, want_h), interpolation=cv2.INTER_AREA)
            for s in strips])
        t = torch.from_numpy(arr).float().div(127.5).sub(1.0)[:, None].to(self.device)
        members = getattr(self, "_members", None)
        if members and len(members) > 1:
            lp = torch.stack([torch.log_softmax(m(t), dim=2) for m in members]).mean(dim=0)
            probs, logits = lp.exp(), lp
        else:
            logits = self.recognizer(t)
            probs = torch.softmax(logits, dim=2)
        confs = torch.max(probs, dim=2)[0].mean(dim=1).tolist()
        texts = ctc_decode(logits)

        # Dynamic low-light arbitration with night ensemble across batch
        members_night = getattr(self, "_members_night", None)
        if members_night:
            try:
                low_light_indices = [idx for idx, s in enumerate(strips) if float(s.mean()) < 80.0]
                if low_light_indices:
                    t_sub = t[low_light_indices]
                    lp_n = torch.stack([torch.log_softmax(m(t_sub), dim=2) for m in members_night]).mean(dim=0)
                    confs_n = torch.max(lp_n.exp(), dim=2)[0].mean(dim=1).tolist()
                    texts_n = ctc_decode(lp_n)
                    for k, idx in enumerate(low_light_indices):
                        if (confs_n[k] > confs[idx] and len(texts_n[k] or "") >= 6) or (confs[idx] < 0.85 and confs_n[k] >= 0.85):
                            texts[idx] = texts_n[k]
                            confs[idx] = confs_n[k]
            except Exception:
                pass

        return [((tx or ""), float(c)) for tx, c in zip(texts, confs)]

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

        # ── Adaptive Night & Low-Contrast Plate Arbitration ───────────────────
        # In night traffic or low-contrast situations (headlight glare, shadow),
        # license plate characters lose local contrast against the plate background.
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) restores stroke
        # sharpness without blurring edges.
        # FAST-PATH: If raw read already has high confidence (conf >= 0.85),
        # skip CLAHE entirely (zero overhead on clean daytime plates).
        if raw_bgr_crop is not None and raw_bgr_crop.size > 0 and conf < 0.85:
            c_gray = (cv2.cvtColor(raw_bgr_crop, cv2.COLOR_BGR2GRAY)
                      if raw_bgr_crop.ndim == 3 and raw_bgr_crop.shape[2] >= 3
                      else raw_bgr_crop)
            if c_gray.size > 0:
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(2, 8))
                c_enh = clahe.apply(c_gray)
                want_h, want_w = getattr(self, "_rec_hw", (IMG_H, IMG_W))
                c_prep = cv2.resize(c_enh, (want_w, want_h), interpolation=cv2.INTER_AREA)
                c_text, c_conf = self._crnn_infer(c_prep)
                if c_text and len(c_text) >= 4 and c_conf > conf:
                    c_clean = re.sub(r"[^A-Z0-9]", "", c_text.upper())
                    c_dec = decode_plate(c_text, local_state=self.local_state, apply_prior=False)
                    r_dec = decode_plate(raw_text or "", local_state=self.local_state, apply_prior=False)
                    c_score = float(c_dec.get("score", 0.0))
                    r_score = float(r_dec.get("score", 0.0))
                    # Accept CLAHE read if its grammar score is >= raw score
                    if c_score >= r_score or (c_score >= 0.5 and len(c_clean) >= 7):
                        logger.info("[ANPR-NIGHT-ARB] CLAHE night arbitration won: '%s' (conf=%.3f, score=%.2f) beat raw '%s' (conf=%.3f, score=%.2f)",
                                    c_text, c_conf, c_score, raw_text, conf, r_score)
                        raw_text, conf = c_text, c_conf

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
            if ar_c <= 2.5 and h_c >= 8 and w_c >= 12:
                try:
                    # Method 1: Horizontal Unstacking (Top half + Bottom half side-by-side)
                    from backend.services.anpr_track_aggregator_v2 import split_stacked_plate
                    t_half, b_half = split_stacked_plate(target_crop)
                    if t_half is None or b_half is None:
                        split_y = int(h_c * 0.48)
                        t_half = target_crop[:split_y, :]
                        b_half = target_crop[split_y:, :]
                    if t_half.size > 0 and b_half.size > 0:
                        target_h = 32
                        tw = max(10, int(t_half.shape[1] * (target_h / max(1, t_half.shape[0]))))
                        bw = max(10, int(b_half.shape[1] * (target_h / max(1, b_half.shape[0]))))
                        th_r = cv2.resize(t_half, (tw, target_h), interpolation=cv2.INTER_CUBIC)
                        bh_r = cv2.resize(b_half, (bw, target_h), interpolation=cv2.INTER_CUBIC)
                        
                        # Add a neutral white spacer (12px) between Line 1 and Line 2
                        # to prevent character strokes fusing together across lines
                        spacer = np.full((target_h, 12, 3), 255, dtype=np.uint8) if th_r.ndim == 3 else np.full((target_h, 12), 255, dtype=np.uint8)
                        stitched = np.hstack([th_r, spacer, bh_r])
                        
                        s_gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY) if stitched.ndim == 3 else stitched
                        s_prep = cv2.resize(s_gray, (self._rec_hw[1], self._rec_hw[0]), interpolation=cv2.INTER_AREA)
                        s_text, s_conf = self._crnn_infer(s_prep)
                        if s_text and len(s_text) >= 4:
                            s_clean = re.sub(r"[^A-Z0-9]", "", s_text.upper())
                            _d = decode_plate(s_text, local_state=self.local_state, apply_prior=False)
                            is_valid = _d.get("plate") is not None and float(_d.get("score", 0.0)) >= DOUBLE_LINE_MIN_GRAMMAR
                            # Strictly arbitrate: only accept if valid grammar OR long clean format (>=7 chars), never truncated fragments (<7 chars)
                            if is_valid and (s_conf > (conf - 0.10) or len(s_clean) > len(re.sub(r"[^A-Z0-9]", "", raw_text.upper()))):
                                logger.info("[ANPR-ACTIVE] [UNSTACKED TWO-LINE] Sliced & stitched 2-line plate -> '%s' (conf=%.2f) beat raw single-line '%s' (conf=%.2f)", s_text, s_conf, raw_text, conf)
                                raw_text, conf = s_text, s_conf
                            elif len(s_clean) >= 7 and s_conf > conf:
                                raw_text, conf = s_text, s_conf

                    # Method 2: Sliced line1 + line2 fallback
                    top_strip, bot_strip = t_half, b_half
                    if top_strip is not None and bot_strip is not None:
                        def pad_to_crnn(strip):
                            sh, sw = strip.shape[:2]
                            target_h = self._rec_hw[0]
                            scaled_w = max(10, min(self._rec_hw[1], int(sw * (target_h / max(1, sh)))))
                            resized = cv2.resize(strip, (scaled_w, target_h), interpolation=cv2.INTER_CUBIC)
                            if resized.ndim == 3:
                                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                            padded = np.full((self._rec_hw[0], self._rec_hw[1]), 255, dtype=np.uint8)
                            padded[:, :scaled_w] = resized
                            return padded

                        t_prep = pad_to_crnn(top_strip)
                        b_prep = pad_to_crnn(bot_strip)
                        t_text, t_conf = self._crnn_infer(t_prep)
                        b_text, b_conf = self._crnn_infer(b_prep)
                        if t_text and b_text:
                            stacked_text = t_text + b_text
                            stacked_conf = (t_conf + b_conf) / 2.0
                            stacked_clean = re.sub(r"[^A-Z0-9]", "", stacked_text.upper())
                            if len(stacked_clean) >= 7:
                                _d = decode_plate(stacked_text, local_state=self.local_state, apply_prior=False)
                                is_valid = _d.get("plate") is not None and float(_d.get("score", 0.0)) >= DOUBLE_LINE_MIN_GRAMMAR
                                if is_valid and (stacked_conf > (conf - 0.10) or len(stacked_clean) > len(re.sub(r"[^A-Z0-9]", "", raw_text.upper()))):
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

    def _lock_for(self, track_id: int) -> threading.Lock:
        """The lock that serialises engine work for one track.

        Two ANPR worker threads share this engine. Different tracks are
        independent (their votes, confirmations and view buffers are all keyed
        by track_id), so they take different locks and still run in parallel;
        two crops of the *same* track cannot interleave, which is what the vote
        counter and the plate lock assume.

        One lock for everything when a style expert is loaded, because swapping
        the recogniser is process-global state, or when SENTINEL_ANPR_SERIALIZE=1
        pins the worker to a single inference at a time.
        """
        if self._serialize_all or getattr(self, "_expert_members", None):
            return self._serialize_lock
        with self._track_lock_guard:
            lock = self._track_locks.get(track_id)
            if lock is None:
                lock = threading.Lock()
                self._track_locks[track_id] = lock
            return lock

    def process_vehicle_track(
        self,
        frame:    np.ndarray,
        bbox:     List[float],
        cls_id:   int,
        track_id: int,
        plate_box: Any = _DETECT,
    ) -> Optional[Dict[str, Any]]:
        # Serialised per track: see _lock_for. The style expert may be swapped
        # in for one plate, so it is always swapped back before the lock is
        # released.
        with self._lock_for(track_id):
            default = (getattr(self, "_members", None), self.recognizer)
            try:
                return self._process_vehicle_track_impl(frame, bbox, cls_id, track_id, plate_box)
            finally:
                if getattr(self, "_expert_active", False):
                    self._members, self.recognizer = default
                    self._expert_active = False

    # ── Plate-style expert ────────────────────────────────────────────────────
    def _load_style_expert(self) -> None:
        """Recogniser for the plate styles the shipped one cannot read.

        finetune_relay_styles.py: green EV 0% -> 68%, two-row 1% -> 89%, but
        normal plates 59.0 -> 55.4% (test A). So it is used only for plates
        whose crop IS green or two-row; everything else keeps the shipped model.
        SENTINEL_STYLE_EXPERT=1 turns it on.
        """
        self._expert_members = None
        self._expert_active = False
        if os.environ.get("SENTINEL_STYLE_EXPERT", "0") != "1":
            return
        d = Path(os.environ.get("SENTINEL_STYLE_EXPERT_DIR", "").strip()
                 or PROJECT_ROOT / "models" / "plate_recognizer_relay")
        try:
            from scripts.plate_final_model import PlateCRNN
            ms = []
            for p in sorted(d.glob("final_m*.pt")):
                ck = torch.load(p, map_location=self.device, weights_only=False)
                m = PlateCRNN(len(ck.get("chars", self.chars)) + 1, img_h=int(ck.get("img_h", 64)))
                m.load_state_dict(ck["model"])
                ms.append(m.to(self.device).eval())
            if ms:
                self._expert_members = ms
                logger.info("Plate-style expert loaded: %d members from %s", len(ms), d)
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("Plate-style expert not loaded: %s", exc)

    @torch.no_grad()
    def _expert_read(self, strips: List[np.ndarray]) -> List[str]:
        """Grammar-decoded reads of recogniser-sized strips by the expert."""
        hh, ww = getattr(self, "_rec_hw", (IMG_H, IMG_W))
        arr = np.stack([s if s.shape[:2] == (hh, ww) else cv2.resize(s, (ww, hh), interpolation=cv2.INTER_AREA)
                        for s in strips])
        t = torch.from_numpy(arr).float().div(127.5).sub(1.0)[:, None].to(self.device)
        lp = torch.stack([torch.log_softmax(m(t), dim=2) for m in self._expert_members]).mean(0)
        out = []
        for raw in ctc_decode(lp):
            d = decode_plate(raw or "", local_state=self.local_state, apply_prior=True) if raw and len(raw) >= 4 else {}
            out.append(d.get("plate") or raw or "")
        return out

    @staticmethod
    def _plate_style(bgr: np.ndarray) -> Optional[str]:
        """'green' (EV plate), 'tworow' (stacked plate) or None."""
        if bgr is None or bgr.size == 0 or bgr.ndim != 3:
            return None
        h, w = bgr.shape[:2]
        core = bgr[int(h * 0.15):int(h * 0.85), int(w * 0.08):int(w * 0.92)]
        if core.size:
            hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
            # Live green EV plates: green share 0.58-0.69, median hue 86-87
            # (OpenCV 0-180), median saturation 97-124. The first band (hue
            # 60-110, share 0.35) also caught 61 held-out crops - blue-cyan
            # tints at hue 100-108 and low-saturation greys, nearly all not
            # plates - and routed crops skip the verifier.
            mask = (hsv[..., 0] >= 82) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 40)
            if mask.mean() >= STYLE_GREEN_FRAC and float(np.median(hsv[..., 1])) >= 90:
                return "green"
        if w / max(h, 1) <= STYLE_TWOROW_AR:
            return "tworow"
        return None

    @staticmethod
    def _unstack(bgr: np.ndarray) -> np.ndarray:
        """Two-row plate -> one row: top 48% and bottom beside it, height 32
        (identical to finetune_relay_styles.unstack, which the expert learned)."""
        h = bgr.shape[0]
        sy = int(h * 0.48)
        t, b = bgr[:sy], bgr[sy:]
        tw = max(10, int(t.shape[1] * 32 / max(1, t.shape[0])))
        bw = max(10, int(b.shape[1] * 32 / max(1, b.shape[0])))
        return np.hstack([cv2.resize(t, (tw, 32), interpolation=cv2.INTER_CUBIC),
                          cv2.resize(b, (bw, 32), interpolation=cv2.INTER_CUBIC)])

    def _process_vehicle_track_impl(
        self,
        frame:    np.ndarray,
        bbox:     List[float],
        cls_id:   int,
        track_id: int,
        plate_box: Any = _DETECT,
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        if now - self._last_cleanup > TRACK_CLEANUP_INTERVAL:
            self._cleanup_stale_tracks()
            self._last_cleanup = now

        if cls_id not in (1, 2, 3, 5, 7):
            return None

        if track_id in self._confirmed_plates:
            confirmed = self._confirmed_plates[track_id]
            if confirmed.get("locked"):
                return confirmed

        # Only the lighting LABEL is needed here, not the enhanced image.
        #
        # This used to run the full enhancer — CLAHE over the whole 1920x1080
        # frame, once per VEHICLE — and pass the result as `frame`. But
        # extract_plate_candidate is also given raw_frame, and with it every
        # pixel it uses comes from the raw frame: the plate detector, the crop,
        # the quality score and the recogniser input. The enhanced image was
        # only ever read for its shape. Measured 2026-09-16 on CAM_09 daytime
        # footage: 22.7 ms per call, 12% of the ANPR stage, for pixels that
        # were discarded. The label comes from the enhancer's own brightness
        # classification, so it is identical to what enhance() returned, and
        # scripts that want the enhanced image still call _enhance_frame.
        lighting = self._night_enhancer._classify(self._night_enhancer._brightness(frame))

        preprocessed, raw_bgr, quality = self.extract_plate_candidate(
            frame, bbox, cls_id, lighting=lighting, raw_frame=frame,
            plate_box=plate_box,
        )
        if preprocessed is None:
            return self._confirmed_plates.get(track_id)

        # Plate verifier: is this crop a number plate at all? Signboards and
        # grilles that the detector boxes read as confident, grammar-valid
        # plates ("Bhavani" -> GJ18KA8558 at 0.93); see train_plate_verifier.
        # Green EV plates and square 2-row motorcycle/scooty plates are checked
        # by their own recognizer and grammar gates instead: the verifier learned
        # rectangular white/yellow car plates only and frequently rejects square 2-row
        # motorcycle plates (AR 1.0-1.8) as non-plates (fix: 2026-09-22).
        is_green = GREEN_INVERT and raw_bgr is not None and self._plate_style(raw_bgr) == "green"
        is_two_wheeler = cls_id in (1, 3)
        if self._verifier is not None and raw_bgr is not None and not is_green and not is_two_wheeler:
            vs = self._verifier_score(raw_bgr)
            # Infrared night crops (no colour) get their own threshold: the
            # verifier learned colour daytime plates and scored CAM_07's IR
            # GJ32K9870 at 163 px 0.06 and GJ32AG0416 at 148 px 0.01.
            thr = PLATE_VERIFIER_THR
            if VERIFIER_THR_GRAY is not None and raw_bgr.ndim == 3:
                sat = float(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)[..., 1].mean())
                if sat < GRAY_SAT_MAX:
                    thr = VERIFIER_THR_GRAY
            if vs < thr:
                self.verifier_rejected += 1
                self._audit_reject(track_id, "verifier", raw_bgr, verifier=vs)
                return self._confirmed_plates.get(track_id)

        # One track, two vehicles: the tracker swaps identity and the second
        # vehicle's views join the first one's vote (live CAM_06 track 5 holds
        # a Maruti van and a Hyundai; GJ37AB9445 was reported for GJ11BH2735).
        # A sudden change in the vehicle crop's colour profile starts the
        # track's reads afresh, so each vehicle is voted on its own views.
        if APPEARANCE_RESET and frame is not None and frame.size:
            sig = self._appearance_sig(frame)
            prev_sig = self._track_sig.get(track_id)
            self._track_sig[track_id] = sig
            if prev_sig is not None and float(cv2.compareHist(prev_sig, sig, cv2.HISTCMP_BHATTACHARYYA)) > APPEARANCE_DIST:
                if self._gate_reads.pop(track_id, None):
                    self.appearance_resets += 1
                self._confirmed_plates.pop(track_id, None)
                self._audit_best.pop(track_id, None)
            if len(self._track_sig) > 5000:
                for k in list(self._track_sig)[:1000]:
                    self._track_sig.pop(k, None)

        # The plate box is padded by 8%/10% before recognition, which on a
        # framed plate pulls the frame in with it: live CAM_06 GJ11VV7338 read
        # "GJ11VV338" at 0.48 padded and "GJ11VV7338" at 0.97 tight. Read the
        # unpadded box too and keep the more confident of the two.
        if TIGHT_CROP_ARB and raw_bgr is not None and raw_bgr.shape[1] > 40:
            ph, pw = raw_bgr.shape[:2]
            ax, ay = round(pw * 0.08 / 1.16), round(ph * 0.10 / 1.20)
            tight = raw_bgr[ay:ph - ay, ax:pw - ax] if ay > 0 and ax > 0 else None
            if tight is not None and tight.size and tight.shape[0] >= 8 and tight.shape[1] >= 24:
                hh, ww = getattr(self, "_rec_hw", (IMG_H, IMG_W))
                t_prep = cv2.resize(cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY), (ww, hh),
                                    interpolation=cv2.INTER_AREA)
                p_text, c_pad = self._crnn_infer(preprocessed)
                t_text, c_tight = self._crnn_infer(t_prep)
                # Confidence cannot separate these: the framed plate read
                # "GJ11VV338" (a character lost) at 0.955 padded against the
                # correct "GJ11VV7338" at 0.965 tight. The grammar can - the
                # first is not a valid plate (0.0) and the second is (0.98) -
                # so a read that parses beats one that does not, and
                # confidence only decides when both or neither parse.
                # Letting a grammar-valid tight read override a padded one that
                # does not parse was measured too: live-key precision 80.6 ->
                # 76.3% for the same 29 recognised, so confidence alone decides.
                del p_text
                if t_text and c_tight > c_pad + TIGHT_CROP_GAP:
                    preprocessed, raw_bgr = t_prep, tight
                    self.tight_crop_wins += 1

        # Green EV plate: white text on green is low-contrast light-on-dark in
        # grayscale. Inverted it is dark text on a light plate - what the
        # shipped recogniser reads best. Live GJ32AG2883 views: not read at all
        # before; inverted, exact at 0.97 on both larger views. (Inverting every
        # "mostly dark" crop was rejected - it broke 29 normal plates - so this
        # is applied only to crops that are green by COLOUR.)
        if is_green:
            raw_bgr = 255 - raw_bgr
            hh, ww = getattr(self, "_rec_hw", (IMG_H, IMG_W))
            preprocessed = cv2.resize(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY), (ww, hh),
                                      interpolation=cv2.INTER_AREA)
            self.style_routed["green"] = self.style_routed.get("green", 0) + 1

        # Two-row plates go to the style expert (read, stability and margin all
        # use it), unstacked into one row. Off while no detector separates them.
        if getattr(self, "_expert_members", None) and raw_bgr is not None and not is_green:
            style = self._plate_style(raw_bgr)
            if style:
                self._members, self.recognizer = self._expert_members, self._expert_members[0]
                self._expert_active = True
                if style == "tworow":
                    raw_bgr = self._unstack(raw_bgr)
                g = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
                hh, ww = getattr(self, "_rec_hw", (IMG_H, IMG_W))
                preprocessed = cv2.resize(g, (ww, hh), interpolation=cv2.INTER_AREA)
                self.style_routed[style] = self.style_routed.get(style, 0) + 1

        native_w = raw_bgr.shape[1] if raw_bgr is not None else 0
        # With fusion off nothing is buffered, so no crop copies are held per
        # track and the sharpness sort below never runs.
        buf = self._fusion_crops.setdefault(track_id, []) if PLATE_FUSION_ENABLED else []
        if PLATE_FUSION_ENABLED and raw_bgr is not None and raw_bgr.size > 0:
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

        single_floor = 10 if cls_id in (1, 3) else min(18, SINGLE_FRAME_FLOOR_PX)
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

        if REPORT_GATE and raw_bgr is not None:
            # Report gate, decided per VEHICLE over all of its verified reads:
            # the majority read is reported only if its best confidence is
            # >= GATE_MIN_CONF and its reads survive small perturbations of the
            # crop on average (stability >= GATE_MIN_STAB). Held-out June-13
            # plates (eval_read_stability.py): 90.0% precision at 84.4%
            # legible-plate recall. Gating each crop separately and locking on
            # the first pass measured 80.6% — a one-off misread got locked.
            reads = self._gate_reads.setdefault(track_id, [])
            reads.append((plate_str, conf, self._read_stability(raw_bgr, plate_str),
                          self._min_char_margin(preprocessed), int(raw_bgr.shape[1]),
                          preprocessed.copy() if (CROSS_MODEL_GATE or REQUIRE_AGREE) else None))
            if len(reads) > 16:
                del reads[0]
            if len(self._gate_reads) > 5000:
                for k in list(self._gate_reads)[:1000]:
                    self._gate_reads.pop(k, None)
            # Decide from the views nearest the camera. A track is read at every
            # size on its way through the frame; on the relay's night footage the
            # far views (60-90 px) read garbage (GJ14MB4141, GJ15AC4352 for a
            # GJ32AG4959 read correctly at 96-164 px) and outvoted the near ones.
            # Only views whose plate is at least GATE_NEAR_FRAC of the widest
            # plate this track has shown take part in the vote and the gate.
            widest = max(r[4] for r in reads)
            near = [r for r in reads if r[4] >= GATE_NEAR_FRAC * widest]
            texts = [r[0] for r in near]
            voted_plate = max(set(texts), key=texts.count)
            sel = [r for r in near if r[0] == voted_plate]
            voted_conf = max(r[1] for r in sel)
            stab = sum(r[2] for r in sel) / len(sel)
            # weakest character's top-vs-second margin: a one-character
            # misread (GJ08 -> GJ09) is confident overall but doubtful there
            margin = sum(r[3] for r in sel) / len(sel)
            total_v = len(near)
            agree_v = len(sel) / total_v
            # A read that survives the perturbations (stab >= GATE_STAB_HIGH) at
            # high confidence has proved itself twice; it needs a smaller
            # weakest-character margin. Live CAM_06 GJ18BL4660 (conf 0.965,
            # stab 0.83, margin 0.27) and GJ18ZT1782 (0.973, 0.83, 0.45) were
            # correct and refused on margin alone.
            min_margin = GATE_MIN_MARGIN
            if (GATE_MARGIN_STABLE is not None and stab >= GATE_STAB_HIGH
                    and voted_conf >= GATE_CONF_HIGH):
                min_margin = GATE_MARGIN_STABLE
            passed = not (voted_conf < GATE_MIN_CONF or stab < GATE_MIN_STAB or margin < min_margin)
            # Second route: a differently trained recogniser (the relay-style
            # fine-tune) reads the same views and returns the identical plate.
            # Correct live reads were refused on margin alone (GJ32AG5088 0.16,
            # GJ03EK8807 0.35); two models' errors are far less correlated than
            # one model's per-character confidence.
            # Only a margin-only failure can be rescued: rescuing any failure
            # measured recall 53.3 -> 60.0% but precision 76.2 -> 62.1% (keys).
            if (not passed and CROSS_MODEL_GATE and voted_conf >= GATE_MIN_CONF
                    and stab >= GATE_MIN_STAB and getattr(self, "_expert_members", None)):
                strips = [r[5] for r in sel if r[5] is not None]
                if strips:
                    agree = sum(p == voted_plate for p in self._expert_read(strips))
                    if agree >= max(1, (len(strips) + 1) // 2):
                        passed = True
                        self.cross_model_passed += 1
            # Second opinion as a FILTER (not a rescue): a differently trained
            # recogniser must read the same plate on one of the views, or the
            # read is not reported.
            if passed and REQUIRE_AGREE and getattr(self, "_expert_members", None):
                strips = [r[5] for r in sel if r[5] is not None]
                if strips and not any(p == voted_plate for p in self._expert_read(strips)):
                    passed = False
                    self.agree_rejected += 1
            if not passed:
                self.gate_rejected += 1
                self._audit_reject(track_id, "gate", raw_bgr, plate=voted_plate, read=plate_str,
                                   conf=voted_conf, stab=stab, margin=margin)
                # a read reported earlier for this track no longer holds
                self._confirmed_plates.pop(track_id, None)
                return None
            # Never frozen: every further view can still change the verdict.
            is_locked = False
            voted_plate = self._consistent_with_recent(track_id, voted_plate, len(sel), voted_conf)
            # the audit crop is the most confident view that gave the reported text
            best = self._audit_best.get(track_id)
            if plate_str == voted_plate and (best is None or best[0] != voted_plate or conf >= best[1]):
                self._audit_best[track_id] = (voted_plate, conf)
                if len(self._audit_best) > 5000:
                    for k in list(self._audit_best)[:1000]:
                        self._audit_best.pop(k, None)
                self._audit_report(track_id, voted_plate, voted_conf, stab, margin, raw_bgr)
        else:
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

    def process_vehicle_tracks_batch(
        self, items: List[Tuple[np.ndarray, int, int]]
    ) -> List[Optional[Dict[str, Any]]]:
        """process_vehicle_track for many (vehicle_crop, cls_id, track_id) at once.

        The plate detector runs once per batch instead of once per crop; the
        rest — crop, quality gate, recogniser, voting, watchlist — is the same
        per-vehicle code, so results match calling process_vehicle_track on
        each crop in order.
        """
        need = []
        for i, (crop, cls_id, track_id) in enumerate(items):
            # Include cls_id=1 (bicycle): YOLOv8 on Indian traffic often assigns
            # scooties and light scooters to cls 1. Excluding it silently drops
            # all such two-wheelers from batch ANPR (fix: 2026-09-22).
            if crop is None or cls_id not in (1, 2, 3, 5, 7):
                continue
            confirmed = self._confirmed_plates.get(track_id)
            if confirmed and confirmed.get("locked"):
                continue
            need.append(i)
        boxes: Dict[int, Any] = {}
        if need and (self._plate_det_available
                     or getattr(self, "plate_detector_small", None) is not None):
            search = []
            for i in need:
                crop, cls_id, _ = items[i]
                search.append(crop[self._plate_search_offset(cls_id, crop.shape[0]):, :])
            found = self.detect_plate_bboxes_batch(search, [items[i][1] for i in need])
            boxes = dict(zip(need, found))
        # Detector outcome per item, for diagnostics (None = not attempted).
        self.last_batch_plate_boxes = [boxes.get(i) if i in boxes else None for i in range(len(items))]
        self.last_batch_attempted = [i in boxes for i in range(len(items))]
        # Recognise every plate strip of the batch in one CRNN pass. The strips
        # are re-derived exactly as process_vehicle_track derives them, and the
        # results are looked up by content when it asks for them.
        strips = []
        plate_crops = []
        for i in need:
            crop, cls_id, _ = items[i]
            h, w = crop.shape[:2]
            lighting = self._night_enhancer._classify(self._night_enhancer._brightness(crop))
            prep, raw_bgr, _q = self.extract_plate_candidate(
                crop, [0, 0, w, h], cls_id, lighting=lighting, raw_frame=crop,
                plate_box=boxes.get(i, _DETECT) if i in boxes else _DETECT)
            if prep is not None:
                strips.append(prep)
                if raw_bgr is not None:
                    plate_crops.append(raw_bgr)
        # Verifier scores for the whole batch in one pass, looked up by content.
        self._verifier_cache = {}
        if self._verifier is not None and plate_crops:
            for c, s in zip(plate_crops, self._verifier_scores(plate_crops)):
                self._verifier_cache[self._strip_key(c)] = s
        self._crnn_batch_cache = {}
        if strips:
            want_h, want_w = getattr(self, "_rec_hw", (IMG_H, IMG_W))
            sized = [s if s.shape[:2] == (want_h, want_w)
                     else cv2.resize(s, (want_w, want_h), interpolation=cv2.INTER_AREA)
                     for s in strips]
            for s, r in zip(sized, self._crnn_infer_batch(sized)):
                self._crnn_batch_cache[self._strip_key(s)] = r

        out: List[Optional[Dict[str, Any]]] = []
        try:
            for i, (crop, cls_id, track_id) in enumerate(items):
                if crop is None:
                    out.append(None)
                    continue
                h, w = crop.shape[:2]
                out.append(self.process_vehicle_track(
                    crop, [0, 0, w, h], cls_id, track_id,
                    plate_box=boxes.get(i, _DETECT) if i in boxes else _DETECT))
        finally:
            self._crnn_batch_cache = None
            self._verifier_cache = None
        return out

    def _consistent_with_recent(self, track_id, plate: str, votes: int, conf: float) -> str:
        """One vehicle, two track ids, two readings one character apart.

        The tracker splits a vehicle into several tracks (see
        tracker-id-switch notes); each track votes on its own views, so one of
        them can settle on a one-character misread of the other's plate. Live
        on 2026-09-18: GJ18ZT2598 and GJ18ZT2508 reported for one car, seconds
        apart, on one camera. Two different vehicles with plates one character
        apart on the same camera within CONSIST_WINDOW_S are far rarer than
        that, so the better-supported reading (more agreeing views, then
        higher confidence) is reported for both tracks.
        """
        now = time.time()
        cam = str((getattr(self, "_track_cam", None) or {}).get(track_id)
                  or getattr(self, "_current_cam_hint", "") or "")
        recent = self._recent_reports.setdefault(cam, [])
        recent[:] = [r for r in recent if now - r[0] <= CONSIST_WINDOW_S and r[1] != track_id]
        best = (votes, conf, plate)
        partner = None
        for ts, tid, p, v, c in recent:
            if len(p) == len(plate) and p != plate and sum(a != b for a, b in zip(p, plate)) == 1:
                if (v, c) > best[:2]:
                    best, partner = (v, c, p), tid
                elif partner is None:
                    partner = tid
        recent.append((now, track_id, plate, votes, conf))
        if partner is not None:
            self.consistency_merged += 1
            if best[2] != plate:
                logger.info("consistency: track %s %s -> %s (track %s)", track_id, plate, best[2], partner)
            elif partner in self._confirmed_plates:
                self._confirmed_plates[partner]["plate"] = plate
        return best[2]

    @staticmethod
    def _appearance_sig(bgr: np.ndarray) -> np.ndarray:
        """Normalised hue-saturation histogram of a vehicle crop."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()

    def _grammar_score(self, text: str) -> float:
        """How well a raw read parses as an Indian plate (0 = not a plate)."""
        if not text or len(text) < 4:
            return 0.0
        try:
            d = decode_plate(text, local_state=self.local_state, apply_prior=False)
            return float(d.get("score", 0.0)) if d.get("plate") else 0.0
        except Exception:                                           # noqa: BLE001
            return 0.0

    def _audit_report(self, track_id, plate: str, conf: float, stab: float,
                      margin: float, raw_bgr: np.ndarray) -> None:
        """Keep the crop behind every reported read, so live precision can be
        counted by eye (backend/scripts/live_read_audit.py). One file per track,
        overwritten as the track's reported read is refreshed.
        SENTINEL_ANPR_AUDIT_DIR turns it on."""
        root = os.environ.get("SENTINEL_ANPR_AUDIT_DIR", "").strip()
        if not root:
            return
        try:
            import json as _json
            d = Path(root)
            d.mkdir(parents=True, exist_ok=True)
            cam = str((getattr(self, "_track_cam", None) or {}).get(track_id)
                      or getattr(self, "_current_cam_hint", "") or "cam")
            stem = f"{cam}_{track_id}"
            cv2.imwrite(str(d / f"{stem}.jpg"), raw_bgr)
            with open(d / "reports.jsonl", "a", encoding="utf-8") as f:
                f.write(_json.dumps({"ts": time.time(), "cam": cam, "track": str(track_id), "plate": plate,
                                     "conf": round(float(conf), 3), "stab": round(float(stab), 3),
                                     "margin": round(float(margin), 3), "crop": f"{stem}.jpg",
                                     "crop_w": int(raw_bgr.shape[1])}) + "\n")
        except Exception as exc:                                    # noqa: BLE001
            logger.debug("audit write failed: %s", exc)

    def _audit_reject(self, track_id, reason: str, raw_bgr: np.ndarray, **info) -> None:
        """With SENTINEL_ANPR_AUDIT_REJECTS=1, keep refused crops too (one per
        track and reason), so lost recall can be inspected by eye."""
        root = os.environ.get("SENTINEL_ANPR_AUDIT_DIR", "").strip()
        if not root or os.environ.get("SENTINEL_ANPR_AUDIT_REJECTS", "0") != "1" or raw_bgr is None:
            return
        try:
            import json as _json
            d = Path(root) / "rejected"
            d.mkdir(parents=True, exist_ok=True)
            cam = str((getattr(self, "_track_cam", None) or {}).get(track_id) or "cam")
            stem = f"{cam}_{track_id}_{reason}"
            cv2.imwrite(str(d / f"{stem}.jpg"), raw_bgr)
            row = {"ts": time.time(), "cam": cam, "track": str(track_id), "reason": reason,
                   "crop": f"rejected/{stem}.jpg", "crop_w": int(raw_bgr.shape[1])}
            row.update({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
                        for k, v in info.items()})
            with open(Path(root) / "rejected.jsonl", "a", encoding="utf-8") as f:
                f.write(_json.dumps(row) + "\n")
        except Exception as exc:                                    # noqa: BLE001
            logger.debug("reject audit failed: %s", exc)

    @torch.no_grad()
    def _min_char_margin(self, prep: np.ndarray) -> float:
        """Smallest top-1 minus top-2 probability over the read's characters.

        Each emitted character is scored at its most confident timestep; the
        plate is as trustworthy as its weakest character (eval_char_margin.py).
        """
        from scripts.train_plate_recognizer import BLANK
        want_h, want_w = getattr(self, "_rec_hw", (IMG_H, IMG_W))
        if prep.shape[:2] != (want_h, want_w):
            prep = cv2.resize(prep, (want_w, want_h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(prep).float().div(127.5).sub(1.0)[None, None].to(self.device)
        members = getattr(self, "_members", None) or [self.recognizer]
        p = torch.stack([torch.softmax(m(t), dim=2) for m in members]).mean(0)[0].cpu().numpy()
        ids = p.argmax(1)
        margins, prev, best = [], -1, None
        for tstep, k in enumerate(ids):
            if k != prev and best is not None:
                margins.append(best)
                best = None
            if k != BLANK:
                top2 = np.sort(p[tstep])[-2:]
                m = float(top2[1] - top2[0])
                best = m if best is None else max(best, m)
            prev = k
        if best is not None:
            margins.append(best)
        return min(margins) if margins else 0.0

    # ── Read stability ──────────────────────────────────────────────────────
    @staticmethod
    def _perturbations(img: np.ndarray) -> List[np.ndarray]:
        """Six small changes a correct read should survive (see eval_read_stability)."""
        h, w = img.shape[:2]
        dx, dy = max(1, w // 40), max(1, h // 12)
        return [img[:, dx:], img[:, :w - dx], img[dy:, :], img[:h - dy, :],
                cv2.copyMakeBorder(img, dy, dy, dx, dx, cv2.BORDER_REPLICATE),
                cv2.GaussianBlur(img, (3, 3), 0)]

    def _read_stability(self, raw_bgr: np.ndarray, plate: str) -> float:
        """Share of perturbed crops that read as the same plate, in one CRNN batch."""
        want_h, want_w = getattr(self, "_rec_hw", (IMG_H, IMG_W))
        variants = self._perturbations(raw_bgr)
        preps = [cv2.resize(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) if v.ndim == 3 else v,
                            (want_w, want_h), interpolation=cv2.INTER_AREA) for v in variants]
        cache = getattr(self, "_crnn_batch_cache", None)
        own = cache is None
        if own:
            self._crnn_batch_cache = cache = {}
        for p, r in zip(preps, self._crnn_infer_batch(preps)):
            cache[self._strip_key(p)] = r
        try:
            same = sum(1 for v, p in zip(variants, preps)
                       if (self.recognize_plate(p, raw_bgr_crop=v)[0] or "") == plate)
        finally:
            if own:
                self._crnn_batch_cache = None
        return same / len(variants)

    # ── Plate verifier ──────────────────────────────────────────────────────
    def _load_verifier(self) -> None:
        self._verifier = None
        self._verifier_cache = None
        self.verifier_rejected = 0
        self.gate_rejected = 0
        self._gate_reads: Dict[int, list] = {}
        self._audit_best: Dict[int, tuple] = {}
        self._recent_reports: Dict[str, list] = {}
        self.consistency_merged = 0
        self.cross_model_passed = 0
        self.agree_rejected = 0
        self.tight_crop_wins = 0
        self.appearance_resets = 0
        self._track_sig: Dict[int, Any] = {}
        if os.environ.get("SENTINEL_PLATE_VERIFIER", "0") != "1" or not PLATE_VERIFIER_PATH.exists():
            return
        try:
            from backend.scripts.train_plate_verifier import build_model
            ck = torch.load(PLATE_VERIFIER_PATH, map_location=self.device, weights_only=False)
            m = build_model()
            m.load_state_dict(ck["state_dict"])
            self._verifier = m.eval().to(self.device)
            logger.info("Plate verifier loaded (%s, threshold %.2f)", PLATE_VERIFIER_PATH.name, PLATE_VERIFIER_THR)
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("Plate verifier not loaded: %s", exc)
            self._verifier = None

    @torch.no_grad()
    def _verifier_scores(self, crops: List[np.ndarray]) -> List[float]:
        from backend.scripts.train_plate_verifier import letterbox, to_tensor
        x = torch.stack([to_tensor(letterbox(c)) for c in crops]).to(self.device)
        return torch.sigmoid(self._verifier(x).squeeze(1)).tolist()

    def _verifier_score(self, crop: np.ndarray) -> float:
        cache = getattr(self, "_verifier_cache", None)
        if cache:
            hit = cache.get(self._strip_key(crop))
            if hit is not None:
                return hit
        return self._verifier_scores([crop])[0]

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
        # Per-track locks follow the same lifecycle as the state they protect:
        # a lock whose track the engine has already forgotten above is dropped
        # too, so a long run does not keep a lock per track it will never read
        # again. Locks for tracks still live are never touched — deleting one
        # that a worker holds would let two threads lock "the same" track with
        # two different objects.
        if len(self._track_locks) > MAX_TRACK_HISTORIES:
            with self._track_lock_guard:
                gone = [k for k in self._track_locks if k not in self._voters]
                for k in gone:
                    del self._track_locks[k]


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ────────────────────────────────────────────────────────────────────────────

_anpr_singleton: Optional[ANPREngine] = None
_anpr_singleton_lock = threading.Lock()


def get_anpr_engine() -> ANPREngine:
    """The process-wide engine, built at most once.

    Guarded because the pipeline reaches the engine through its `_anpr`
    property, which builds it lazily and can be called from the ANPR worker
    threads: two threads arriving together used to construct two engines (two
    full model loads, two sets of caches) and keep whichever assignment landed
    last, so half the reads went to an engine the other half never saw.
    """
    global _anpr_singleton
    if _anpr_singleton is None:
        with _anpr_singleton_lock:
            if _anpr_singleton is None:
                _anpr_singleton = ANPREngine()
    return _anpr_singleton