"""ANPR measured on the corp8 LIVE feeds, not on the local clips.

WHY THIS IS A DIFFERENT MEASUREMENT
  Every plate number reported so far — 6% capture, 64.9% exact, 86.5%
  watchlist — was measured on data/clips/*.mp4. Those are recordings from
  these same thirty cameras, so the geometry and optics are right, but they
  are not the live HLS stream the pipeline now ingests. Two things could
  differ and both push the same way:

    resolution   the HLS decodes at 1280x720. Sixteen of these cameras are
                 1920x1080 locally, so a plate can arrive smaller here.
    compression  HLS segments are re-encoded, and softer pixels cost the
                 recogniser more than they cost the detector.

  So live numbers are more likely slightly WORSE than the clip numbers. This
  measures rather than assumes.

WHAT CAN AND CANNOT BE MEASURED WITHOUT LABELS
  Capture rate needs no ground truth: either a vehicle yielded a plate crop
  above the readable floor or it did not. That is measured here directly and
  is comparable to the clip figure.

  Exact-match accuracy DOES need ground truth, and live footage has none. So
  every plate crop that produced a read is written to disk alongside what the
  model said, for a human to check. Reporting a confidence average instead
  and calling it accuracy would be exactly the kind of number this project has
  spent days removing.

SESSION BUDGET
  A corp8 session serves roughly eleven stream opens before HTTP 429, and
  logins are themselves limited. Cameras are therefore processed in small
  groups with a pause and one fresh login between them — see
  verify_corp8_live for how that limit was characterised.

Run:  python -m backend.scripts.anpr_live_feed_eval
      python -m backend.scripts.anpr_live_feed_eval --cameras 12 --frames 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

OUT_DIR = ROOT / "output" / "live_anpr_eval"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=10,
                    help="How many cameras to sample, from the start of the "
                         "portal's list. Ignored if --camera-id is given.")
    ap.add_argument("--camera-id", type=str, default=None,
                    help="Test exactly one portal camera id, e.g. cam21.")
    ap.add_argument("--frames", type=int, default=30,
                    help="Frames to pull per camera.")
    ap.add_argument("--per-session", type=int, default=5)
    ap.add_argument("--session-pause", type=float, default=45.0)
    ap.add_argument("--seek-sec", type=float, default=None,
                    help="Start at this offset into the recording instead of "
                         "the wall-clock 'now' position. The wall-clock seek "
                         "exists to feel live, but it can land in a dead "
                         "traffic window (e.g. 3am) purely by chance — use "
                         "this to anchor on a window known to have traffic.")
    args = ap.parse_args()

    from ultralytics import YOLO
    from services.anpr_engine import get_anpr_engine, SINGLE_FRAME_FLOOR_PX
    from services.tracker import BoTSORTTracker
    from backend.services.corp8_session import get_corp8_session
    from backend.services.hls_ffmpeg_capture import FFmpegHLSCapture

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.jpg"):
        old.unlink()

    engine = get_anpr_engine()
    vdet = YOLO(str(ROOT / "yolov8s.pt"))
    sess = get_corp8_session()
    all_cams = sess.cameras()
    if args.camera_id:
        cams = [c for c in all_cams if c["id"] == args.camera_id]
        if not cams:
            print(f"camera id {args.camera_id!r} not found on the portal. "
                  f"Available: {[c['id'] for c in all_cams]}")
            return 1
    else:
        cams = all_cams[:args.cameras]
    print(f"sampling {len(cams)} live cameras, {args.frames} frames each\n",
          flush=True)

    tracks_seen = 0
    tracks_captured = 0
    observations = 0
    plate_boxes = 0
    widths: list[int] = []
    reads: list[dict] = []
    per_cam: list[tuple] = []

    for idx, cam in enumerate(cams):
        if idx and args.per_session and idx % args.per_session == 0:
            print(f"  -- pausing {args.session_pause:.0f}s for a fresh "
                  f"session --", flush=True)
            time.sleep(args.session_pause)
            sess.session(force=True)

        cid = cam["id"]
        info = sess.probe(cid)
        if not info.get("ok"):
            print(f"  {cid}: playlist refused ({info.get('status')})",
                  flush=True)
            continue
        cap = FFmpegHLSCapture(sess.stream_url(cid), sess.cookie_header(),
                               seek_sec=args.seek_sec,
                               duration_sec=info.get("duration_sec"))
        if not cap.isOpened():
            print(f"  {cid}: would not open", flush=True)
            continue

        tracker = BoTSORTTracker(frame_rate=1)
        cam_tracks: dict[int, int] = {}   # track id -> best plate width
        for _ in range(args.frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            res = vdet.predict(frame, conf=0.35, classes=[2, 3, 5, 7],
                               verbose=False, device="cuda:0", imgsz=1280)
            boxes = res[0].boxes
            dets = []
            if boxes is not None and len(boxes):
                for xy, cf, cl in zip(boxes.xyxy.cpu().numpy(),
                                      boxes.conf.cpu().numpy(),
                                      boxes.cls.cpu().numpy()):
                    dets.append({"bbox": [float(v) for v in xy],
                                 "conf": float(cf), "cls": int(cl)})
            enhanced, lighting = engine._enhance_frame(frame)
            for t in tracker.update(dets, frame):
                tid = t.get("track_id")
                if tid is None:
                    continue
                x1, y1, x2, y2 = (int(v) for v in t["bbox"])
                if x2 - x1 < 40 or y2 - y1 < 40:
                    continue
                observations += 1
                cam_tracks.setdefault(tid, 0)
                _, raw, _ = engine.extract_plate_candidate(
                    enhanced, [x1, y1, x2, y2], int(t.get("cls", 2)),
                    lighting=lighting, raw_frame=frame)
                if raw is None:
                    continue
                plate_boxes += 1
                w = raw.shape[1]
                widths.append(w)
                cam_tracks[tid] = max(cam_tracks[tid], w)
                if w >= SINGLE_FRAME_FLOOR_PX:
                    result = engine.process_vehicle_track(
                        frame, [x1, y1, x2, y2], int(t.get("cls", 2)),
                        track_id=int(tid) + idx * 100000)
                    if result and result.get("plate"):
                        name = f"{cid}_t{tid}_{result['plate']}_{w}px.jpg"
                        cv2.imwrite(str(OUT_DIR / name), raw)
                        reads.append({
                            "camera": cid, "track": int(tid),
                            "plate": result["plate"],
                            "confidence": result.get("confidence"),
                            "native_px": w,
                            "grade": result.get("read_grade"),
                            "crop": name,
                        })
        cap.release()

        n_tracks = len(cam_tracks)
        n_cap = sum(1 for w in cam_tracks.values() if w >= SINGLE_FRAME_FLOOR_PX)
        tracks_seen += n_tracks
        tracks_captured += n_cap
        per_cam.append((cid, n_tracks, n_cap))
        print(f"  {cid}: {n_tracks:>3} vehicles, {n_cap:>2} with a readable "
              f"plate", flush=True)

    print("\n" + "=" * 62)
    print("LIVE FEED — DETECTION")
    print("=" * 62)
    print(f"  vehicle observations        {observations}")
    print(f"  unique vehicles tracked     {tracks_seen}")
    print(f"  plate boxes found           {plate_boxes} "
          f"({100*plate_boxes/max(observations,1):.1f}% per observation)")
    print(f"  VEHICLES WITH A READABLE PLATE  {tracks_captured}/{tracks_seen} "
          f"= {100*tracks_captured/max(tracks_seen,1):.1f}%")
    print(f"\n  (local clips measured ~6% on the same cameras)")

    if widths:
        w = np.array(widths)
        print(f"\n  plate width: median {np.median(w):.0f}px, "
              f"90th pct {np.percentile(w, 90):.0f}px")
        for lo, hi in ((0, 40), (40, 70), (70, 90), (90, 130), (130, 10_000)):
            n = int(((w >= lo) & (w < hi)).sum())
            if n:
                lab = f"{lo}-{hi}px" if hi < 10_000 else f"{lo}px+"
                print(f"    {lab:<12}{n:>5}  {100*n/len(w):>5.1f}%")

    print("\n" + "=" * 62)
    print("LIVE FEED — RECOGNITION")
    print("=" * 62)
    print(f"  plates read      {len(reads)}")
    if reads:
        grades = Counter(r["grade"] for r in reads)
        print(f"  read grade       {dict(grades)}")
        print(f"\n  {'camera':<8}{'plate':<14}{'px':>5}{'conf':>7}  crop")
        print("  " + "-" * 58)
        for r in sorted(reads, key=lambda x: -x["native_px"])[:25]:
            print(f"  {r['camera']:<8}{r['plate']:<14}{r['native_px']:>5}"
                  f"{r['confidence']:>7.2f}  {r['crop']}")
        (OUT_DIR / "reads.json").write_text(json.dumps(reads, indent=1),
                                            encoding="utf-8")
        print(f"\n  {len(reads)} crops written to {OUT_DIR}")
        print("""
  These carry NO ground truth. Exact-match accuracy on live footage can only
  be had by reading the saved crops by eye and comparing. A confidence
  average is not accuracy and is not reported as one.""")
    else:
        print("  no plates read in this sample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
