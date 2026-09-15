"""backend/scripts/classify_plate_capable.py — which cameras can see plates AT ALL?

THE FINDING THIS IS BUILT ON
  Fifteen random vehicle crops were sampled from the cameras where plate
  reading had been failing. NONE of the fifteen contained a visible number
  plate - they were buses in profile, cars from directly overhead, trucks
  side-on. Those cameras are mounted to watch traffic FLOW, across the road.
  Plates are on the front and rear of a vehicle. From the side there is
  nothing to read, at any resolution, at any hour.

  Everything that looked like model failure was this:
    - the "camera imbalance" in harvested labels was CORRECT, not a bug
    - detector v2's collapse came from compositing synthetic plates onto
      side-view buses, teaching it to expect plates where none can exist
    - "balance the training set across all cameras" was itself the error

  Viewing geometry is not recoverable by any model, enhancement or fusion.
  A plate the camera is not pointed at is not a hard case; it is absent.

WHAT THIS MEASURES
  For each camera, sample vehicles and ask the working pipeline (plate band ->
  OCR -> all-India grammar) how many yield a structurally valid plate. This is
  an empirical capability test, not a guess from mounting angle: a camera that
  produces reads can see plates, whatever its geometry looks like.

WHY IT MATTERS BEYOND TUNING
  Compute spent on a side-view camera can never produce a plate. Knowing which
  cameras are capable turns a diluted fleet-wide average into an honest
  per-camera capability map - and tells a deployment exactly where an ANPR
  camera needs to be added.

USAGE
  python -m backend.scripts.classify_plate_capable
  python -m backend.scripts.classify_plate_capable --per-cam 60 --time 0830
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "timeout;30000000|stimeout;30000000|rw_timeout;30000000")

import cv2
import numpy as np
import yaml

from backend.scripts.indian_plate_grammar import decode_plate
from backend.scripts.plate_mine_corpus import MotionGate

OUT_JSON = Path("output/plate_capable_cameras.json")
EXAMPLES = Path("output/plate_capable_examples")
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OSD = {"BRIDGE", "BHAI", "CHIMAN", "CSITMS", "CSI", "BRIDC", "GSRTC",
       "CARRIAGE", "EXPERIENCE", "SOMNATH", "CHITRA", "JANPATH", "PTZ"}
FOURWHEEL = {1, 3, 4}


def band(img):
    h, w = img.shape[:2]
    b = img[int(h * 0.40):int(h * 0.98), int(w * 0.06):int(w * 0.94)]
    if b.size == 0 or b.shape[1] < 20:
        return None
    if b.shape[1] < 380:
        f = 380 / b.shape[1]
        b = cv2.resize(b, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-cam", type=int, default=50,
                    help="Vehicles to test per camera (largest first).")
    ap.add_argument("--time", default="0830")
    ap.add_argument("--min-veh-w", type=int, default=170)
    ap.add_argument("--frames", type=int, default=2500)
    ap.add_argument("--conf", type=float, default=0.12)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    from ultralytics import YOLO
    veh_model = YOLO(cfg["model"]["path"])

    import easyocr
    print("loading EasyOCR...", flush=True)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    root = Path("data/clips")
    cams = sorted(d.name for d in root.iterdir() if d.is_dir())
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    print(f"\nclassifying {len(cams)} cameras "
          f"({args.per_cam} vehicles each)\n", flush=True)

    results: dict[str, dict] = {}

    for cam in cams:
        clip = root / cam / f"{cam}_{args.time}.mp4"
        if not clip.is_file():
            found = sorted((root / cam).glob("*.mp4"))
            if not found:
                continue
            clip = found[0]

        # Collect the largest moving vehicles - the best chance this camera
        # has of showing a plate. If none of these work, none will.
        gate = MotionGate(80.0)
        crops: list[tuple[float, np.ndarray]] = []
        try:
            stream = veh_model.track(source=str(clip), stream=True, persist=True,
                                     tracker="botsort.yaml", conf=0.35,
                                     imgsz=640, verbose=False, quantize=16)
            for i, r in enumerate(stream):
                if i > args.frames:
                    break
                if r.boxes is None or r.boxes.id is None:
                    continue
                frame = r.orig_img
                H, W = frame.shape[:2]
                for b in r.boxes:
                    if int(b.cls[0]) not in FOURWHEEL:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                    if x2 - x1 < args.min_veh_w:
                        continue
                    if not gate.passes(int(b.id[0]), (x1 + x2) / 2, (y1 + y2) / 2):
                        continue
                    # .copy() is REQUIRED: a numpy slice is a VIEW into the
                    # frame buffer, and ultralytics reuses that buffer while
                    # streaming. Without the copy every stored crop points at
                    # overwritten memory by the time OCR runs, which reported
                    # 0/50 on every camera in the fleet - including two that
                    # had already produced verified plate reads.
                    c = frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)].copy()
                    if c.size:
                        crops.append((x2 - x1, c))
        except Exception as e:                          # noqa: BLE001
            print(f"  {cam:<8} ERROR {e}", flush=True)
            continue

        crops.sort(key=lambda t: -t[0])
        crops = crops[:args.per_cam]
        reads = 0
        texts = []
        saved = False
        for wpx, c in crops:
            b = band(c)
            if b is None:
                continue
            for box, txt, cf in reader.readtext(b, allowlist=ALLOW,
                                                batch_size=8):
                if cf < args.conf:
                    continue
                cleaned = "".join(ch for ch in txt.upper() if ch.isalnum())
                if len(cleaned) < 8 or any(t in cleaned for t in OSD):
                    continue
                d = decode_plate(cleaned)
                if d["plate"] and d["score"] >= 0.85:
                    reads += 1
                    texts.append(d["plate"])
                    if not saved:
                        cv2.imwrite(str(EXAMPLES /
                                    f"{cam}_{d['plate']}.jpg"), b)
                        saved = True
                    break

        n = len(crops)
        rate = reads / max(n, 1) * 100
        verdict = ("CAPABLE" if rate >= 8 else
                   "MARGINAL" if rate >= 2 else "NOT CAPABLE")
        results[cam] = {"tested": n, "reads": reads, "rate": round(rate, 1),
                        "verdict": verdict, "samples": texts[:5]}
        print(f"  {cam:<8} {reads:>3}/{n:<4} ({rate:>5.1f}%)  {verdict:<12}"
              f"{('  e.g. ' + texts[0]) if texts else ''}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))

    cap = [c for c, v in results.items() if v["verdict"] == "CAPABLE"]
    mar = [c for c, v in results.items() if v["verdict"] == "MARGINAL"]
    no = [c for c, v in results.items() if v["verdict"] == "NOT CAPABLE"]
    print("\n" + "=" * 62)
    print(f"CAPABLE     ({len(cap):>2}): {', '.join(cap) or '-'}")
    print(f"MARGINAL    ({len(mar):>2}): {', '.join(mar) or '-'}")
    print(f"NOT CAPABLE ({len(no):>2}): {', '.join(no) or '-'}")
    print("=" * 62)
    print("\nNOT CAPABLE means the camera never shows a plate - side or")
    print("overhead views. No model, enhancement or fusion changes that;")
    print("it needs a differently-aimed camera. Direct all ANPR compute at")
    print("the capable set and report coverage against IT, not the fleet.")
    print(f"\nsaved -> {OUT_JSON}\nexamples -> {EXAMPLES}")


if __name__ == "__main__":
    main()
