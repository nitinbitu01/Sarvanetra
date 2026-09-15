"""
sentinel gujarat/scripts/sanity_check.py

Usage:
    python "sentinel gujarat/scripts/sanity_check.py"
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# ── Path bootstrap — ORDER MATTERS ───────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]   # sentinel gujarat/
BACKEND = ROOT / "backend"

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ROOT appended (lower priority)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# BACKEND inserted at front (higher priority) — finds backend/scripts/ first
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# ─────────────────────────────────────────────────────────────────────────────

def hdr(title: str):
    print(f"\n{'─'*58}")
    print(f"  {title}")
    print(f"{'─'*58}")

def ok(label, detail=""):
    print(f"  ✅  {label:<42} {detail}")

def fail(label, detail=""):
    short = str(detail).split("\n")[0][:55]
    print(f"  ❌  {label:<42} {short}")

def warn(label, detail=""):
    print(f"  ⚠️   {label:<42} {detail}")

# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 58)
print("  SENTINEL GUJARAT — Sanity Check (Fixed)")
print(f"  ROOT:    {ROOT}")
print(f"  BACKEND: {BACKEND}")
print(f"  sys.path[0]: {sys.path[0]}")
print(f"  sys.path[1]: {sys.path[1] if len(sys.path) > 1 else 'N/A'}")
print("=" * 58)

failed_imports = []
missing_files  = []

# ─────────────────────────────────────────────────────────────────────────────
# 1. FILE EXISTENCE
# ─────────────────────────────────────────────────────────────────────────────
hdr("1/8  File Existence Check")

EXPECTED_FILES = {
    BACKEND / "scripts" / "indian_plate_grammar.py":
        "backend/scripts/indian_plate_grammar.py",
    BACKEND / "scripts" / "night_enhancer_v2.py":
        "backend/scripts/night_enhancer_v2.py",
    BACKEND / "scripts" / "train_plate_recognizer.py":
        "backend/scripts/train_plate_recognizer.py",
    BACKEND / "services" / "anpr_engine.py":
        "backend/services/anpr_engine.py",
    BACKEND / "services" / "anpr_track_aggregator_v2.py":
        "backend/services/anpr_track_aggregator_v2.py",
    BACKEND / "services" / "ensemble_ocr.py":
        "backend/services/ensemble_ocr.py",
    BACKEND / "services" / "frame_selector.py":
        "backend/services/frame_selector.py",
    BACKEND / "services" / "plate_utils_v2.py":
        "backend/services/plate_utils_v2.py",
    BACKEND / "services" / "watchlist_service.py":
        "backend/services/watchlist_service.py",
    BACKEND / "__init__.py":
        "backend/__init__.py",
    BACKEND / "scripts" / "__init__.py":
        "backend/scripts/__init__.py",
    BACKEND / "services" / "__init__.py":
        "backend/services/__init__.py",
}

for full_path, label in EXPECTED_FILES.items():
    if full_path.exists():
        size = full_path.stat().st_size
        if size == 0 and "__init__" not in label:
            warn(label, f"EXISTS but EMPTY ({size} bytes)")
        else:
            ok(label, f"{size:,} bytes")
    else:
        fail(label, f"MISSING")
        if "__init__" not in label:
            missing_files.append(str(full_path))

if missing_files:
    print(f"\n  ⛔  {len(missing_files)} file(s) MISSING — copy them first")

# ─────────────────────────────────────────────────────────────────────────────
# 2. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
hdr("2/8  Module Imports")

def try_import(label: str, file_path: Path, fn):
    if not file_path.exists():
        warn(label, "skipped — file missing")
        failed_imports.append(label)
        return None
    try:
        result = fn()
        ok(label)
        return result
    except Exception as e:
        fail(label, str(e))
        failed_imports.append(label)
        traceback.print_exc()
        return None

def _imp_grammar():
    from scripts.indian_plate_grammar import (
        fix_plate, decode_plate, is_plausible,
        resolve_state_code, apply_state_prior,
    )
    return fix_plate

fix_plate_fn = try_import(
    "indian_plate_grammar",
    BACKEND / "scripts" / "indian_plate_grammar.py",
    _imp_grammar,
)

def _imp_enhancer():
    from scripts.night_enhancer_v2 import AdaptiveNightEnhancer
    return AdaptiveNightEnhancer

NightEnhancer = try_import(
    "night_enhancer_v2",
    BACKEND / "scripts" / "night_enhancer_v2.py",
    _imp_enhancer,
)

def _imp_crnn():
    from scripts.train_plate_recognizer import CRNN, ctc_decode, IMG_H, IMG_W
    return CRNN

CRNN_cls = try_import(
    "train_plate_recognizer",
    BACKEND / "scripts" / "train_plate_recognizer.py",
    _imp_crnn,
)

def _imp_selector():
    from services.frame_selector import PlateFrameSelector
    return PlateFrameSelector

FrameSelector = try_import(
    "frame_selector",
    BACKEND / "services" / "frame_selector.py",
    _imp_selector,
)

def _imp_preproc():
    from services.plate_utils_v2 import PlatePreprocessorV2
    return PlatePreprocessorV2

Preprocessor = try_import(
    "plate_utils_v2",
    BACKEND / "services" / "plate_utils_v2.py",
    _imp_preproc,
)

def _imp_voter():
    from services.anpr_track_aggregator_v2 import TemporalPlateVoter
    return TemporalPlateVoter

Voter = try_import(
    "anpr_track_aggregator_v2",
    BACKEND / "services" / "anpr_track_aggregator_v2.py",
    _imp_voter,
)

def _imp_ensemble():
    from services.ensemble_ocr import EnsembleOCR
    return EnsembleOCR

EnsembleOCR = try_import(
    "ensemble_ocr",
    BACKEND / "services" / "ensemble_ocr.py",
    _imp_ensemble,
)

def _imp_watchlist():
    from services.watchlist_service import WatchlistService
    return WatchlistService

WatchlistSvc = try_import(
    "watchlist_service",
    BACKEND / "services" / "watchlist_service.py",
    _imp_watchlist,
)

def _imp_engine():
    from services.anpr_engine import ANPREngine, get_anpr_engine
    return get_anpr_engine

engine_fn = try_import(
    "anpr_engine",
    BACKEND / "services" / "anpr_engine.py",
    _imp_engine,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL FILES
# ─────────────────────────────────────────────────────────────────────────────
hdr("3/8  Model Files")

model_files = [
    (ROOT / "models" / "plate_recognizer" / "best.pt",
     "CRNN recognizer (best.pt)"),
    (ROOT / "runs" / "detect" / "runs" / "plate" / "plate_v3_ft" / "weights" / "best.pt",
     "Plate detector (plate_v3_ft)"),
    (ROOT / "yolov8s.pt",
     "Vehicle detector (yolov8s)"),
]
for path, label in model_files:
    if path.exists():
        ok(label, f"{path.stat().st_size/1e6:.1f} MB")
    else:
        fail(label, f"MISSING: {path.name}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET
# ─────────────────────────────────────────────────────────────────────────────
hdr("4/8  Dataset")

gt = ROOT / "data" / "plate_real" / "verified_all.jsonl"
if gt.exists():
    n = sum(1 for _ in open(gt, encoding="utf-8"))
    ok("verified_all.jsonl", f"{n} samples")
else:
    fail("verified_all.jsonl", f"MISSING")

imgs_dir = ROOT / "data" / "plate_real" / "images"
if imgs_dir.exists():
    imgs = list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.png"))
    ok("plate_real/images/", f"{len(imgs)} images")
else:
    warn("plate_real/images/", "not found")

# ─────────────────────────────────────────────────────────────────────────────
# 5. GPU
# ─────────────────────────────────────────────────────────────────────────────
hdr("5/8  GPU / PyTorch")

try:
    import torch
    ok("PyTorch", torch.__version__)
    if torch.cuda.is_available():
        ok("CUDA", f"{torch.cuda.get_device_name(0)}  "
                   f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    else:
        warn("CUDA", "not available — CPU only")
except Exception as e:
    fail("PyTorch", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 6. GRAMMAR TESTS
# ─────────────────────────────────────────────────────────────────────────────
hdr("6/8  Grammar Tests")

if fix_plate_fn is None:
    warn("Grammar tests", "skipped — import failed")
else:
    from scripts.indian_plate_grammar import fix_plate

    tests = [
        ("6J01K0297",  "GJ01K0297",  "6→G"),
        ("CJ01K0297",  "GJ01K0297",  "C→G"),
        ("LJ01BV9921", "GJ01BV9921", "L→G"),
        ("GI01BV9921", "GJ01BV9921", "I→J"),
        ("MH12AB1234", "MH12AB1234", "MH pass-through"),
        ("RJ14CV0002", "RJ14CV0002", "RJ pass-through"),
        ("KA05MK4321", "KA05MK4321", "KA pass-through"),
        ("22BH1234AB", "22BH1234AB", "BH series"),
        ("GARBAGE",    None,          "noise → None"),
    ]
    passed = 0
    for raw, expected, note in tests:
        got = fix_plate(raw)
        if got == expected:
            ok(f"{raw} → {str(expected)}", note)
            passed += 1
        else:
            fail(f"{raw} → {str(got)}", f"expected {expected} | {note}")
    print(f"\n  Grammar: {passed}/{len(tests)} passed")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SUB-MODULE TESTS
# ─────────────────────────────────────────────────────────────────────────────
hdr("7/8  Sub-Module Tests")

import numpy as np

if NightEnhancer:
    try:
        enh  = NightEnhancer()
        dark = np.zeros((480, 640, 3), dtype=np.uint8)
        brt  = np.full((480, 640, 3), 150, dtype=np.uint8)
        _, c1 = enh.enhance(dark)
        _, c2 = enh.enhance(brt)
        assert c1 == "night_dark", f"Expected night_dark got {c1}"
        assert c2 == "day",        f"Expected day got {c2}"
        ok("AdaptiveNightEnhancer", f"dark→{c1}  bright→{c2}")
    except Exception as e:
        fail("AdaptiveNightEnhancer", str(e))
else:
    warn("AdaptiveNightEnhancer", "skipped")

if FrameSelector:
    try:
        sel   = FrameSelector()
        small = np.zeros((5, 5, 3), dtype=np.uint8)
        large = np.full((40, 130, 3), 128, dtype=np.uint8)
        s1 = sel.score(small)
        s2 = sel.score(large)
        assert s1 == 0.0
        assert s2 > 0.0
        ok("PlateFrameSelector", f"small={s1:.1f}  valid={s2:.1f}")
    except Exception as e:
        fail("PlateFrameSelector", str(e))
else:
    warn("PlateFrameSelector", "skipped")

if Preprocessor:
    try:
        prep = Preprocessor()
        crop = np.full((30, 100, 3), 100, dtype=np.uint8)
        out  = prep.preprocess(crop, lighting="day")
        assert out is not None and out.shape == (32, 128), f"shape={out.shape if out is not None else None}"
        ok("PlatePreprocessorV2", f"output={out.shape}")
    except Exception as e:
        fail("PlatePreprocessorV2", str(e))
else:
    warn("PlatePreprocessorV2", "skipped")

if Voter:
    try:
        v = Voter(min_reads=3)
        for _ in range(4):
            v.add_reading(1, "GJ01AB1234", 0.9, 80.0)
        v.add_reading(1, "GJ01AB1235", 0.5, 40.0)
        plate, conf = v.vote(1)
        assert plate == "GJ01AB1234", f"Got '{plate}'"
        ok("TemporalPlateVoter", f"voted='{plate}'  conf={conf:.2f}")
    except Exception as e:
        fail("TemporalPlateVoter", str(e))
else:
    warn("TemporalPlateVoter", "skipped")

if WatchlistSvc:
    try:
        ws = WatchlistSvc()
        h1 = ws._fuzzy_match("GJ05AB1234")
        h2 = ws._fuzzy_match("GJ05AB1235")
        h3 = ws._fuzzy_match("MH99ZZ9999")
        assert h1 and h1[1] == "exact"
        assert h2 and h2[1] == "fuzzy_1char"
        assert h3 is None
        ok("WatchlistService",
           f"exact={h1[1]}  fuzzy={h2[1]}  non-match=None")
    except Exception as e:
        fail("WatchlistService", str(e))
else:
    warn("WatchlistService", "skipped")

# ─────────────────────────────────────────────────────────────────────────────
# 8. FULL ENGINE LOAD
# ─────────────────────────────────────────────────────────────────────────────
hdr("8/8  Full Engine Load")

if failed_imports:
    warn("ANPREngine",
         f"Skipped — fix {len(failed_imports)} import(s): "
         f"{', '.join(failed_imports)}")
elif engine_fn is None:
    fail("ANPREngine", "engine module failed to load")
else:
    try:
        from services.anpr_engine import get_anpr_engine
        t0     = time.time()
        engine = get_anpr_engine()
        dt     = time.time() - t0

        ok("ANPREngine loaded", f"in {dt:.1f}s on {engine.device}")
        ok("  CRNN recognizer",
           "loaded ✅" if engine.recognizer else "NOT LOADED ❌")
        ok("  Plate detector",
           "YOLOv8 ✅" if engine._plate_det_available
           else "heuristic fallback ⚠️")
        ok("  EnsembleOCR",
           "CRNN+EasyOCR ✅" if engine._ensemble
           else "CRNN only ⚠️")
        ok("  Watchlist",
           f"{len(engine._watchlist_svc.watchlist)} plates ✅")

        # Smoke test — blank frame
        dummy  = np.zeros((480, 640, 3), dtype=np.uint8)
        result = engine.process_vehicle_track(
            frame=dummy,
            bbox=[50.0, 50.0, 300.0, 400.0],
            cls_id=2,
            track_id=9999,
        )
        ok("  Smoke test (blank frame)",
           "None ✅ (expected — no plate in black frame)"
           if result is None else f"dict ✅ {result.get('plate')}")

    except Exception as e:
        fail("ANPREngine", str(e))
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 58)
all_ok = not failed_imports and not missing_files

if all_ok:
    print("  ✅  ALL CHECKS PASSED")
    print()
    print("  NEXT STEP:")
    print('  python "sentinel gujarat/scripts/eval_live_v4.py"')
else:
    print(f"  ❌  ISSUES FOUND")
    if missing_files:
        print(f"\n  MISSING FILES ({len(missing_files)}):")
        for mf in missing_files:
            print(f"    ❌  {mf}")
    if failed_imports:
        print(f"\n  FAILED IMPORTS: {', '.join(failed_imports)}")

print("=" * 58)
print()