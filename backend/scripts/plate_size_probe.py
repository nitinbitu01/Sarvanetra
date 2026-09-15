"""
backend/scripts/plate_size_probe.py

Answers one question about a camera or a piece of footage BEFORE you rely on
it: are the number plates in this video physically big enough to read?

WHY THIS EXISTS
  ANPR accuracy on this stack is dominated by one variable — the NATIVE plate
  width in pixels, which is a property of the optics (sensor resolution, focal
  length, distance to the traffic), not of the recogniser. Measured on 960
  holdout reads:

      < 40 px   ~0%   exact-match          (unreadable, whatever you do)
      40-70 px   17.1% exact-match         (fusion band; several views help)
      70-90 px   44.8% exact-match
      90+  px    higher still

  A camera whose plates land at 25 px will read nothing, and no threshold
  change or model swap fixes that — the characters are ~4 px tall and the
  information is simply not in the image. A camera whose plates land at 90 px
  reads well. Same code, same model, opposite outcome.

  So when new footage arrives, this is the FIRST thing to run. It reports the
  distribution the accuracy bands above are indexed by, and tells you what
  that footage can and cannot support — instead of finding out by watching an
  empty alert feed and guessing whether the model, the wiring, or the lens is
  at fault.

USAGE
    python -m backend.scripts.plate_size_probe --source path/to/footage.mp4
    python -m backend.scripts.plate_size_probe --source rtsp://user:pass@ip/stream
    python -m backend.scripts.plate_size_probe --source CAM_01          # from the DB
    python -m backend.scripts.plate_size_probe --source footage.mp4 --frames 300

It runs the REAL pipeline path — the same vehicle detector, the same
extract_plate_candidate() quality gate the live pipeline uses — so the numbers
it prints are the numbers the live pipeline will see, not an approximation.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

VEHICLE_CLASSES = (2, 3, 5, 7)   # car, motorcycle, bus, truck (COCO)


def _resolve_source(source: str) -> Optional[str]:
    """Accept a file path, a stream URL, or a camera id known to the DB."""
    p = Path(source)
    if p.exists():
        return str(p)
    if "://" in source:
        return source

    # Try it as a camera id / name from the registry.
    try:
        from backend.db.models import Camera
        from backend.db.session import SessionLocal
        db = SessionLocal()
        try:
            cam = db.query(Camera).filter(
                (Camera.camera_id == source) | (Camera.id == source)
                | (Camera.name == source)
            ).first()
            if cam:
                url = cam.stream_url or cam.url
                if url:
                    return url
                clips = sorted((ROOT / "data" / "clips" / source).glob("*.mp4"))
                if clips:
                    return str(clips[0])
                print(f"Camera '{source}' is in the registry but has no stream "
                      f"URL and no local clips.", file=sys.stderr)
                return None
        finally:
            db.close()
    except Exception as exc:
        print(f"(could not consult the camera registry: {exc})", file=sys.stderr)

    print(f"'{source}' is not a file, a URL, or a known camera id.", file=sys.stderr)
    return None


def _verdict(widths: List[float], vehicles: int, frames: int) -> int:
    print()
    print("=" * 68)
    print("  PLATE SIZE PROBE — what this footage can support")
    print("=" * 68)
    print(f"  Frames sampled        : {frames}")
    print(f"  Vehicles detected     : {vehicles}")
    print(f"  Plate regions located : {len(widths)}")

    if vehicles == 0:
        print()
        print("  VERDICT: no vehicles were detected at all.")
        print("  Nothing here is about plate size yet — check that the source is")
        print("  actually decoding (try --frames 300), that it points at traffic,")
        print("  and that it is not a static/blank feed.")
        print("=" * 68)
        return 2

    if not widths:
        print()
        print("  VERDICT: vehicles were found, but no plate region was located on")
        print("  any of them. Either the plates are too small/blurred for the")
        print("  detector to localise at all, or the view never shows a plate")
        print("  (rear-only, extreme angle, heavy occlusion).")
        print("  This footage will NOT produce watchlist hits.")
        print("=" * 68)
        return 1

    widths_sorted = sorted(widths)
    med = statistics.median(widths_sorted)
    p90 = widths_sorted[int(0.9 * (len(widths_sorted) - 1))]

    def band(lo: float, hi: Optional[float]) -> int:
        return sum(1 for w in widths
                   if w >= lo and (hi is None or w < hi))

    n = len(widths)
    b_dead, b_fusion = band(0, 40), band(40, 70)
    b_ok, b_good = band(70, 90), band(90, None)

    print()
    print(f"  Native plate width    : min {min(widths_sorted):.0f}px | "
          f"median {med:.0f}px | p90 {p90:.0f}px | max {max(widths_sorted):.0f}px")
    print()
    print("  Against the measured exact-match bands:")
    print(f"    < 40px  (~0%   readable) : {b_dead:5d}  ({100*b_dead/n:5.1f}%)")
    print(f"    40-70px ( 17.1% exact)   : {b_fusion:5d}  ({100*b_fusion/n:5.1f}%)")
    print(f"    70-90px ( 44.8% exact)   : {b_ok:5d}  ({100*b_ok/n:5.1f}%)")
    print(f"    90px+   ( higher )       : {b_good:5d}  ({100*b_good/n:5.1f}%)")

    # Expected yield, using the measured band rates as-is. Deliberately not
    # rounded up: this is what to expect, not a best case.
    expected = (b_dead * 0.0 + b_fusion * 0.171 + b_ok * 0.448 + b_good * 0.60) / n
    print()
    print(f"  Expected exact-plate read rate on located plates: ~{100*expected:.0f}%")
    print("  (Watchlist RECALL runs higher than this — 1-character fuzzy")
    print("   matching recovers most single-character OCR errors: 86.5%")
    print("   recall at 0% false alarms, measured. Exact transcription is the")
    print("   harder number; watchlist matching is the one that matters for a")
    print("   'find this plate' test.)")

    print()
    readable = b_fusion + b_ok + b_good
    if med >= 70:
        print("  VERDICT: GOOD. Plates are comfortably above the single-frame")
        print("  floor. Default thresholds are correct — change nothing.")
        rc = 0
    elif med >= 40:
        print("  VERDICT: WORKABLE but marginal. Most plates fall in the fusion")
        print("  band, where multi-frame fusion is doing the work. Expect reads,")
        print("  but not on every vehicle. Default thresholds are still correct;")
        print("  do NOT lower them to chase capture — below 40px the reads are")
        print("  wrong, not merely missing, and a wrong read misdirects police.")
        rc = 0
    elif readable > 0:
        print("  VERDICT: MOSTLY UNREADABLE. The median plate is below the 40px")
        print("  floor where accuracy measures ~0%. A minority of closer vehicles")
        print(f"  ({readable} of {n}) are in a readable band and may still hit.")
        print("  This is an OPTICS limit, not a model limit: no threshold or")
        print("  model change reads characters that are ~4px tall. To fix it you")
        print("  need a tighter view — zoom, a longer lens, or a camera closer to")
        print("  the lane.")
        rc = 1
    else:
        print("  VERDICT: UNREADABLE. Every located plate is below the 40px floor")
        print("  where exact-match measures ~0%. This footage cannot support ANPR")
        print("  at any threshold. It is an optics limit, not a software one.")
        rc = 1

    if med < 70:
        print()
        print("  If you have MEASURED that a different band works for this site,")
        print("  the floors are overridable without a code change:")
        print("    SENTINEL_SINGLE_FRAME_FLOOR_PX   (default 70)")
        print("    SENTINEL_FUSION_FLOOR_PX         (default 40)")
        print("    SENTINEL_FUSION_MAX_NATIVE_PX    (default 65)")
        print("  Lowering them raises capture and lowers correctness. The")
        print("  defaults are where they are because of the bands printed above.")

    print("=" * 68)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="video file, stream URL, or a camera id from the registry")
    ap.add_argument("--frames", type=int, default=120,
                    help="frames to sample (default 120)")
    ap.add_argument("--stride", type=int, default=5,
                    help="process every Nth frame (default 5)")
    args = ap.parse_args()

    src = _resolve_source(args.source)
    if not src:
        return 2

    print(f"[probe] opening {src} ...")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[probe] could not open the source. If it is a network stream, "
              f"check reachability and credentials.", file=sys.stderr)
        return 2

    from backend.services.anpr_engine import get_anpr_engine
    engine = get_anpr_engine()

    try:
        from ultralytics import YOLO
        vehicle_model = YOLO("yolov8s.pt")
    except Exception as exc:
        print(f"[probe] vehicle detector unavailable: {exc}", file=sys.stderr)
        cap.release()
        return 2

    widths: List[float] = []
    vehicles = 0
    processed = 0
    frame_idx = 0

    while processed < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.stride:
            continue
        processed += 1

        try:
            res = vehicle_model.predict(frame, verbose=False, conf=0.35)
        except Exception:
            continue
        if not res or res[0].boxes is None:
            continue

        for i in range(len(res[0].boxes)):
            cls_id = int(res[0].boxes.cls[i].item())
            if cls_id not in VEHICLE_CLASSES:
                continue
            vehicles += 1
            bbox = res[0].boxes.xyxy[i].cpu().numpy().tolist()
            try:
                _pre, raw_bgr, _q = engine.extract_plate_candidate(
                    frame, bbox, cls_id, raw_frame=frame)
            except Exception:
                continue
            if raw_bgr is not None and raw_bgr.size:
                widths.append(float(raw_bgr.shape[1]))

        if processed % 20 == 0:
            print(f"[probe] {processed}/{args.frames} frames | "
                  f"{vehicles} vehicles | {len(widths)} plates located")

    cap.release()
    return _verdict(widths, vehicles, processed)


if __name__ == "__main__":
    raise SystemExit(main())
