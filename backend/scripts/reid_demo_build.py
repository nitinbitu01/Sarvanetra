"""Build the cross-camera Re-ID demonstration: four clips, one global identity.

WHAT IT PRODUCES
  For each clip, an annotated MP4 in which the subject carries a global vehicle
  id — the same id in all four — plus a JSON artifact holding the similarity
  that earned each link. Open the four videos one after another and the id does
  not change: that is the whole claim of cross-camera Re-ID, shown rather than
  asserted.

HOW THE ID IS EARNED
  The first clip's subject defines the identity. Each later clip is matched
  against it by appearance similarity, and a link is only drawn when the
  similarity clears the highest similarity that ANY of 500 fleet vehicles
  achieves against the same query. So the threshold is not chosen — it is
  whatever the hardest wrong answer scored, which is the honest bar.

  If a clip fails that bar it is labelled NO MATCH and keeps its local id. A
  demonstration that cannot fail proves nothing, so this one can.

  python -m backend.scripts.reid_demo_build --subjects CAM_M1=2,CAM_M2=2,CAM_M3=4,CAM_M4=2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2                                                         # noqa: E402
import numpy as np                                                 # noqa: E402

from backend.scripts.reid_crosscamera_eval import (                # noqa: E402
    CAMS, REID_DIR, OUT_DIR, PERSON_CLASS, VEHICLE_CLASSES,
    best_tracks_from_clip, sample_distractors, log,
)

DEMO_DIR = ROOT / "output" / "reid_demo"


def annotate_clip(model, src: Path, dst: Path, subject_tid: int,
                  label: str, sim_text: str, imgsz: int, matched: bool) -> bool:
    """Re-run the tracker over the clip and draw the identity on the subject."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = dst.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    model.predictor = None
    classes = list(VEHICLE_CLASSES) + [PERSON_CLASS]
    col = (100, 220, 90) if matched else (90, 90, 235)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        n += 1
        r = model.track(frame, persist=True, classes=classes, conf=0.35,
                        tracker="botsort.yaml", imgsz=imgsz, verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            for box, tid in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.id.cpu().numpy().astype(int)):
                if int(tid) != subject_tid:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
                tag = label
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
                ty = max(th + 10, y1 - 8)
                cv2.rectangle(frame, (x1, ty - th - 10), (x1 + tw + 14, ty + 6),
                              (18, 24, 38), -1)
                cv2.rectangle(frame, (x1, ty - th - 10), (x1 + tw + 14, ty + 6),
                              col, 2)
                cv2.putText(frame, tag, (x1 + 7, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            0.72, col, 2, cv2.LINE_AA)
                if sim_text:
                    cv2.putText(frame, sim_text, (x1 + 7, min(h - 8, y2 + 22)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        # camera banner, so a viewer always knows which clip they are watching
        cv2.rectangle(frame, (0, 0), (w, 34), (18, 24, 38), -1)
        cv2.putText(frame, f"{dst.stem}   cross-camera Re-ID demonstration",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 240, 250), 1,
                    cv2.LINE_AA)
        vw.write(frame)
    cap.release()
    vw.release()

    # mp4v does not play in browsers; H.264 does.
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(dst)], check=True)
        tmp.unlink(missing_ok=True)
    except Exception as exc:                                       # noqa: BLE001
        log(f"    (ffmpeg re-encode failed: {exc}; keeping {tmp.name})")
        tmp.replace(dst)
    return n > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--distractors", type=int, default=500)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    from backend.services.reid_embedder import get_embedder

    model = YOLO(str(ROOT / "yolov8s.pt"))
    embedder = get_embedder()
    stub = getattr(embedder, "is_stub", None)
    if (stub() if callable(stub) else stub):
        log("FATAL: Re-ID model is a stub. Refusing to build a demo on noise.")
        return 2

    want = {}
    for part in args.subjects.split(","):
        cam, _, num = part.strip().partition("=")
        want[cam.strip().upper()] = int(num)

    log("extracting subjects ...")
    crops, tids = [], []
    for cam in CAMS:
        vids = sorted((REID_DIR / cam).glob("*.mp4"))
        tr = best_tracks_from_clip(model, vids[0], args.imgsz, True)
        tid = want[cam]
        if tid not in tr:
            log(f"FATAL: {cam} has no track #{tid}")
            return 2
        crops.append(tr[tid]["crop"])
        tids.append(tid)
        log(f"  {cam}: track #{tid}")

    embs = np.asarray(embedder.extract_batch(crops), dtype=np.float32)

    log(f"sampling {args.distractors} fleet distractors for the bar ...")
    d_emb, _ = sample_distractors(model, embedder, args.distractors,
                                  args.imgsz, args.seed)
    d_emb = np.asarray(d_emb, dtype=np.float32)

    # The bar each later clip must clear: the best score any wrong vehicle got
    # against clip 1. Nothing here is tuned — it is the hardest false answer.
    bar = float((d_emb @ embs[0]).max())
    gid = "GV_%06d" % (abs(hash(tuple(np.round(embs[0][:8], 3)))) % 1000000)
    log(f"\n  global id      : {gid}")
    log(f"  bar to clear   : {bar:.3f}  (best of {len(d_emb)} wrong vehicles)")

    manifest = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "global_id": gid, "bar": round(bar, 4),
                "distractors": int(len(d_emb)), "clips": []}

    for i, cam in enumerate(CAMS):
        sim = float(embs[i] @ embs[0])
        matched = (i == 0) or (sim > bar)
        label = gid if matched else f"local #{tids[i]} (NO MATCH)"
        sim_text = "" if i == 0 else f"similarity {sim:.2f} vs bar {bar:.2f}"
        src = sorted((REID_DIR / cam).glob("*.mp4"))[0]
        dst = DEMO_DIR / f"{cam}.mp4"
        log(f"  {cam}: sim {sim:.3f} -> {'MATCH' if matched else 'no match'}; annotating ...")
        annotate_clip(model, src, dst, tids[i], label, sim_text,
                      args.imgsz, matched)
        manifest["clips"].append({
            "camera": cam, "video": f"reid_demo/{cam}.mp4",
            "track": tids[i], "similarity": round(sim, 4),
            "matched": bool(matched),
            "label": label,
        })

    (DEMO_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"\nwrote {DEMO_DIR / 'manifest.json'}")
    matched_n = sum(c["matched"] for c in manifest["clips"])
    log(f"  {matched_n}/4 clips carry {gid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
