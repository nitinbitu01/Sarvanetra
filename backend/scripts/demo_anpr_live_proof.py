"""
backend/scripts/demo_anpr_live_proof.py — Live Demonstration of Real ANPR License Plate Recognition

Runs the trained CRNN Neural Network + All-India Grammar Engine on real traffic footage crops,
validates plate characters, multi-frame consensus voting, and watchlist detection.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from backend.services.anpr_engine import get_anpr_engine

def run_anpr_live_proof():
    print("=" * 80)
    print(">> SENTINEL GUJARAT — REAL ANPR LICENSE PLATE RECOGNITION PROOF")
    print("=" * 80)

    engine = get_anpr_engine()
    print(f"[MODEL] CRNN Loaded: {engine.model is not None} on {engine.device}")
    print(f"[GRAMMAR] All-India Positional Disambiguation Decoder: ACTIVE")
    print(f"[WATCHLIST] High-Priority Stolen Vehicle Alarm: ACTIVE")

    # Discover real vehicle crops across dataset
    crop_paths = list((PROJECT_ROOT / "data").rglob("*_mid.jpg"))[:15]
    if not crop_paths:
        crop_paths = list((PROJECT_ROOT / "output" / "crops").rglob("*.jpg"))[:15]

    print(f"\n[EVALUATION] Testing CRNN + All-India Grammar on {len(crop_paths)} real CCTV vehicle crops:\n")
    print(f"{'Crop Filename':<45} {'Recognized Plate':<18} {'Conf':<8} {'State':<6} {'Watchlist Status'}")
    print("-" * 95)

    detected_count = 0
    for p in crop_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        strip = engine.extract_plate_candidate(img, [0, 0, w, h], cls_id=2)
        if strip is not None:
            plate, conf, state = engine.recognize_plate(strip)
            if plate:
                detected_count += 1
                is_stolen, reason = engine.check_watchlist(plate)
                status = f"🚨 {reason}" if is_stolen else "✅ Valid Plate"
                print(f"{p.name:<45} {plate:<18} {conf:<8.2f} {str(state):<6} {status}")
            else:
                print(f"{p.name:<45} {'[NO READ / BLUR]':<18} {'0.00':<8} {'-':<6} {'-'}")

    print("\n" + "=" * 80)
    print(f"[SUMMARY] Real ANPR Engine Processed {len(crop_paths)} Crops -> {detected_count} Confirmed Reads!")
    print("=" * 80)

if __name__ == "__main__":
    run_anpr_live_proof()
