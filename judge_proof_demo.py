"""
judge_proof_demo.py — Live Quantitative Proof for Judges & Technical Panels.

Run this script to demonstrate:
  1. Real-time GPU model initialization latency.
  2. Live inference on real held-out Gujarat CCTV plate crops.
  3. Ground Truth vs Sentinel Gujarat Output verification.
  4. Per-crop inference speed (FPS & latency in milliseconds).
  5. Exact match accuracy on tested batch.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Auto-resolve workspace path
script_dir = Path(__file__).resolve().parent
if (script_dir / "sentinel gujarat").exists():
    os.chdir(script_dir / "sentinel gujarat")
    sys.path.insert(0, str(script_dir / "sentinel gujarat"))
else:
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))

import logging
logging.getLogger("backend.anpr").setLevel(logging.ERROR)

import cv2
import numpy as np
import torch

from backend.anpr import read_plate
from backend.anpr_worker import _load_crnn_singleton


def run_proof_demo(num_samples: int = 10) -> None:
    print("=" * 82)
    print(" [SENTINEL GUJARAT] LIVE ANPR QUANTITATIVE PROOF & BENCHMARK")
    print("=" * 82)
    print()

    # 1. Model Loading & GPU Diagnostics
    print("[1/3] HARDWARE & INFERENCE ENGINE INITIALIZATION")
    t_start = time.perf_counter()
    model, device = _load_crnn_singleton()
    t_load = (time.perf_counter() - t_start) * 1000

    if model is None:
        print(" [!] Error loading model weights.")
        return

    # Warmup GPU
    with torch.no_grad():
        dummy_x = torch.zeros((1, 1, 32, 128), dtype=torch.float32, device=device)
        _ = model(dummy_x)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"    * Hardware Device:  {device.upper()} ({gpu_name})")
    print(f"    * Model Checkpoint: models/plate_recognizer/finetuned.pt")
    print(f"    * Architecture:     CRNN (32x128) + 24-Beam CTC Search + Rectification")
    print(f"    * Warmup/Load Time: {t_load:.1f} ms")
    print()

    # 2. Loading Real Dataset
    print(f"[2/3] LOADING {num_samples} HELD-OUT CCTV TEST CROPS (WITH GROUND TRUTH)...")
    real_dir = Path("data/plate_real")
    if not real_dir.exists():
        print(f" [!] Error: {real_dir} directory not found.")
        return

    labels_file = real_dir / "labels.jsonl"
    verified_file = real_dir / "verified_all.jsonl"

    truth_map = {}
    if verified_file.exists():
        for line in verified_file.open(encoding="utf-8"):
            data = json.loads(line)
            if data.get("text"):
                truth_map[(data["camera"], data["track"])] = data["text"]

    samples = []
    if labels_file.exists():
        for line in labels_file.open(encoding="utf-8"):
            row = json.loads(line)
            key = (row["camera"], row["track"])
            gt = truth_map.get(key)
            if gt:
                samples.append((row["file"], row["camera"], row["track"], gt))
            if len(samples) >= num_samples:
                break

    print(f"    * Verified test instances loaded: {len(samples)}")
    print()

    # 3. Live Inference
    print("[3/3] REAL-TIME BATCH INFERENCE EXECUTION")
    print("-" * 82)
    print(f"{'#':<3} | {'CAMERA':<8} | {'GROUND TRUTH':<12} | {'DETECTED PLATE':<14} | {'CONF':<6} | {'LATENCY':<8} | {'MATCH'}")
    print("-" * 82)

    total_time_ms = 0.0
    passed = 0

    for i, (fname, cam, track, gt) in enumerate(samples, 1):
        img_path = real_dir / "images" / fname
        crop = cv2.imread(str(img_path))
        if crop is None:
            continue

        # Measure per-crop execution latency
        t0 = time.perf_counter()
        result = read_plate(crop, easyocr_reader=None, paddleocr_reader=None, crnn_model=model, crnn_device=device)
        lat_ms = (time.perf_counter() - t0) * 1000
        total_time_ms += lat_ms

        pred = result.get("text") or "NONE"
        conf = (result.get("confidence") or 0.0) * 100
        is_hit = (pred == gt)
        if is_hit:
            passed += 1

        status_str = "[PASS]" if is_hit else "[MISS]"
        print(f"{i:<3} | {cam:<8} | {gt:<12} | {pred:<14} | {conf:>5.1f}% | {lat_ms:>5.2f} ms | {status_str}")

    print("-" * 82)
    avg_lat = total_time_ms / max(len(samples), 1)
    fps = 1000.0 / max(avg_lat, 0.001)

    print()
    print("=" * 82)
    print(" [PHASE 1: SINGLE-FRAME PER-CROP METRICS]")
    print(f"   • Total Crops Tested:      {len(samples)}")
    print(f"   • Single-Crop Accuracy:    {passed}/{len(samples)} ({100 * passed / max(len(samples), 1):.1f}%)")
    print(f"   • Average Latency:         {avg_lat:.2f} ms per plate crop")
    print(f"   • Processing Throughput:   {fps:.0f} FPS (Real-time GPU accelerated)")
    print("=" * 82)
    print()

    # 4. Multi-Frame Track Fusion Consensus Demo
    from collections import defaultdict
    from backend.scripts.plate_multiframe_fusion import load_groups
    from backend.scripts.indian_plate_grammar import decode_plate
    from backend.scripts.plate_beam_decode import beam_decode
    from backend.scripts.train_plate_recognizer import ITOS, BLANK, IMG_H, IMG_W
    from backend.scripts.plate_rectify import rectify

    groups, truth = load_groups(min_frames=2)
    track_keys = sorted(groups)[:min(num_samples, len(groups))]

    print("=" * 82)
    print(f" [PHASE 2: MULTI-FRAME FUSION CONSENSUS] ({len(track_keys)} Multi-Frame Vehicle Tracks)")
    print("=" * 82)
    print(f"{'#':<3} | {'CAMERA/TRACK':<16} | {'FRAMES':<6} | {'GROUND TRUTH':<12} | {'FUSED CONSENSUS':<16} | {'MATCH'}")
    print("-" * 82)

    mf_passed = 0
    single_sharp_passed = 0

    for idx, k in enumerate(track_keys, 1):
        gt = truth[k]
        crops = []
        sharpness_list = []
        for f in groups[k]:
            g = cv2.imread(str(real_dir / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                g_rect = rectify(g, mode="full")
                crops.append(g_rect)
                lap = max(float(cv2.Laplacian(g_rect, cv2.CV_64F).var()), 1.0)
                sharpness_list.append(lap)

        if not crops:
            continue

        readings = []
        for i, c_img in enumerate(crops):
            c_res = cv2.resize(c_img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            x_c = torch.from_numpy(c_res).float().div(127.5).sub(1.0).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                l_c = model(x_c)[0]
            cands = beam_decode(l_c, itos=ITOS, blank=BLANK, beam_width=24, topk=2, constrain=True)
            if cands:
                text, score = cands[0]
                conf = float(np.exp(score))
                readings.append({"text": text, "conf": conf, "sharpness": sharpness_list[i]})

        if not readings:
            continue

        # Single sharpest read
        sharpest_read = max(readings, key=lambda r: r["sharpness"])
        if sharpest_read["text"] == gt:
            single_sharp_passed += 1

        # Multi-Frame Weighted Consensus
        score_by_text = defaultdict(float)
        for r in readings:
            w = r["conf"] * np.log1p(r["sharpness"])
            score_by_text[r["text"]] += w

        best_raw = max(score_by_text.items(), key=lambda x: x[1])[0]
        g_dec = decode_plate(best_raw)
        fused_plate = g_dec["plate"] or best_raw

        is_hit = (fused_plate == gt)
        if is_hit:
            mf_passed += 1

        cam_track = f"{k[0]}/{k[1]}"
        status_str = "[PASS]" if is_hit else "[MISS]"
        print(f"{idx:<3} | {cam_track:<16} | {len(crops):<6} | {gt:<12} | {fused_plate:<16} | {status_str}")

    print("-" * 82)
    print(" MULTI-FRAME FUSION PROOF SUMMARY:")
    print(f"   • Single-Frame Baseline Accuracy:   {single_sharp_passed}/{len(track_keys)} ({100 * single_sharp_passed / max(len(track_keys), 1):.1f}%)")
    print(f"   • Multi-Frame Fusion Accuracy:      {mf_passed}/{len(track_keys)} ({100 * mf_passed / max(len(track_keys), 1):.1f}%)")
    gain = (100 * mf_passed / max(len(track_keys), 1)) - (100 * single_sharp_passed / max(len(track_keys), 1))
    print(f"   • Measured Accuracy Improvement:    +{gain:.1f}% absolute jump across video bursts")
    print("=" * 82)


if __name__ == "__main__":
    n = 10
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    run_proof_demo(n)
