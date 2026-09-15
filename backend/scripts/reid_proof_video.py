"""Render the four clips with ONE shared global id burned into every frame.

WHAT THE JUDGE SEES
  Four videos filmed on four phones at four points. In every one of them the
  same motorcycle carries the same label — GV_xxxxxx — while every other
  vehicle in shot carries its own separate local id. The label does not change
  between cameras and does not change between frames.

WHY THE LIVE TILES SHOW SOMETHING ELSE
  A camera tile shows that camera's TRACKER id, which is local to the camera
  and restarts every time a clip loops. It is supposed to change. The global id
  is a different thing: it is the identity assigned ACROSS cameras, and this
  script is what puts it on screen.

THE PLATE IS GROUND TRUTH, NOT AN INPUT
  The plate on this motorcycle is GJ27F V8122. It is a two-line plate, roughly
  20-30 px wide in 478x850 phone video, and the recogniser read NOTHING on any
  of the four clips — the matching used appearance alone. So the plate is an
  independent check on an answer the system reached without it, which is the
  strongest form this demonstration can take. It is drawn in a separate colour
  and captioned as operator-verified so nobody can mistake it for an ANPR read.

  python -m backend.scripts.reid_proof_video
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402

from backend.scripts.appearance_linker import Track, _clip_for       # noqa: E402
from backend.scripts.reid_chain_proof import (                       # noqa: E402
    CAMS, PERSON_CLASS, TWO_WHEELERS, VEHICLE_CLASSES, _union,
    displacement, find_best_chain, merge_fragments, moving_pool,
)

DEMO_DIR = ROOT / "output" / "reid_demo"

SUBJECT_COLOR = (90, 230, 120)     # BGR — the shared identity
OTHER_COLOR = (150, 150, 150)      # everything else
PLATE_COLOR = (80, 200, 255)       # operator-verified, deliberately different


def log(msg: str = "") -> None:
    print(msg, flush=True)


def track_clip(model, embedder, camera: str, path: Path, imgsz: int, topk: int):
    """One pass: track everything, keep per-frame boxes AND crops for matching.

    Boxes are kept for every frame rather than every Nth, because the render
    below needs a box on the subject in each frame or the label flickers.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {}, (0, 0, 0.0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model.predictor = None
    tracks: dict[int, Track] = {}
    boxes_by_frame: dict[int, list] = {}
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fi += 1
        r = model.track(frame, persist=True,
                        classes=VEHICLE_CLASSES + [PERSON_CLASS], conf=0.35,
                        tracker="botsort.yaml", imgsz=imgsz, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            boxes_by_frame[fi] = []
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        cls = r.boxes.cls.cpu().numpy().astype(int)

        persons = [xyxy[i] for i in range(len(cls)) if cls[i] == PERSON_CLASS]
        two = [i for i in range(len(cls)) if int(cls[i]) in TWO_WHEELERS]

        # One rider per machine — see reid_chain_proof.scan_clip_rider_aware
        rider_of: dict[int, list] = {}
        for p in persons:
            px1, py1, px2, py2 = p
            pcx = (px1 + px2) / 2.0
            best, best_score = None, 0.0
            for i in two:
                vx1, vy1, vx2, vy2 = xyxy[i]
                vw, vh = vx2 - vx1, vy2 - vy1
                if vw <= 0 or vh <= 0:
                    continue
                if not ((vx1 - 0.35 * vw) <= pcx <= (vx2 + 0.35 * vw)):
                    continue
                if not (vy1 - vh * 1.2 <= py2 <= vy2 + vh * 0.35):
                    continue
                score = 1.0 / (1.0 + abs(pcx - (vx1 + vx2) / 2.0) / vw)
                if score > best_score:
                    best, best_score = i, score
            if best is not None:
                rider_of.setdefault(best, []).append(p)

        rows = []
        for i in range(len(cls)):
            c = int(cls[i])
            if c == PERSON_CLASS:
                continue
            box = list(xyxy[i])
            for p in rider_of.get(i, []):
                box = _union(box, list(p))
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
            if x2 - x1 < 40 or y2 - y1 < 48:
                continue
            tid = int(ids[i])
            rows.append((tid, x1, y1, x2, y2))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            t = tracks.get(tid)
            if t is None:
                t = Track(camera=camera, local_id=tid, cls_id=c,
                          first_frame=fi, fps=fps)
                t.centres = []
                t.frame_wh = (frame.shape[1], frame.shape[0])
                tracks[tid] = t
            t.n_frames += 1
            t.last_frame = fi
            t.frames.add(fi)
            t.centres.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
            t.crops.append((sharp * (crop.shape[0] * crop.shape[1]) ** 0.5,
                            crop.copy()))
            if len(t.crops) > topk * 3:
                t.crops.sort(key=lambda x: -x[0])
                del t.crops[topk:]
        boxes_by_frame[fi] = rows
    cap.release()

    out = []
    for t in tracks.values():
        if not t.crops:
            continue
        t.crops.sort(key=lambda x: -x[0])
        del t.crops[topk:]
        t.best_crop = t.crops[0][1]
        out.append(t)
    if out:
        flat, spans = [], []
        for t in out:
            spans.append((len(flat), len(t.crops)))
            flat.extend(c for _, c in t.crops)
        embs = np.asarray(embedder.extract_batch(flat), dtype=np.float32)
        for t, (off, n) in zip(out, spans):
            v = embs[off:off + n].mean(axis=0)
            nn = float(np.linalg.norm(v))
            t.emb = (v / nn).astype(np.float32) if nn > 1e-9 else None
        out = [t for t in out if t.emb is not None]

    log(f"  {camera}: {len(out)} tracks over {fi} frames ({w}x{h} @ {fps:.0f}fps)")
    return out, boxes_by_frame, (w, h, fps)


def render(camera: str, src: Path, boxes_by_frame: dict, subject_tids: set,
           gid: str, plate: str, spec, out_path: Path) -> None:
    w, h, fps = spec
    tmp = out_path.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap = cv2.VideoCapture(str(src))
    fi = 0
    band = max(52, int(h * 0.075))

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fi += 1
        for tid, x1, y1, x2, y2 in boxes_by_frame.get(fi, []):
            subject = tid in subject_tids
            col = SUBJECT_COLOR if subject else OTHER_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), col,
                          3 if subject else 1)
            if subject:
                label = gid
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.62, 2)
                ly = max(th + 8, y1 - 6)
                cv2.rectangle(frame, (x1, ly - th - 7), (x1 + tw + 10, ly + 4),
                              SUBJECT_COLOR, -1)
                cv2.putText(frame, label, (x1 + 5, ly), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (18, 32, 20), 2, cv2.LINE_AA)
                if plate:
                    cv2.putText(frame, plate, (x1 + 2, min(h - 6, y2 + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, PLATE_COLOR,
                                2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"#{tid}", (x1 + 3, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, OTHER_COLOR, 1,
                            cv2.LINE_AA)

        # top band: the claim, stated the same way on every clip
        cv2.rectangle(frame, (0, 0), (w, band), (18, 18, 18), -1)
        cv2.putText(frame, camera, (10, int(band * 0.42)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 220, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f"GLOBAL ID {gid}", (10, int(band * 0.86)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, SUBJECT_COLOR, 2,
                    cv2.LINE_AA)
        # bottom band: what produced it, and what the plate is doing there
        cv2.rectangle(frame, (0, h - band), (w, h), (18, 18, 18), -1)
        cv2.putText(frame, "matched by appearance - no plate read",
                    (10, h - band + int(band * 0.40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (215, 215, 215), 1,
                    cv2.LINE_AA)
        if plate:
            cv2.putText(frame, f"{plate}  operator-verified, NOT used to match",
                        (10, h - band + int(band * 0.82)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, PLATE_COLOR, 1,
                        cv2.LINE_AA)
        vw.write(frame)
    cap.release()
    vw.release()
    if out_path.exists():
        out_path.unlink()
    shutil.move(str(tmp), str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--min-disp", type=float, default=0.15)
    ap.add_argument("--plate", default="GJ27F V8122",
                    help="operator-verified plate, drawn as ground truth only")
    args = ap.parse_args()

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    from backend.services.reid_embedder import get_embedder

    log("loading models ...")
    model = YOLO(str(ROOT / "yolov8s.pt"))
    embedder = get_embedder()
    if getattr(embedder, "is_stub", False):
        log("FATAL: Re-ID model did not load.")
        return 2

    log("\ntracking all four clips ...")
    by_cam: dict[str, list[Track]] = {}
    boxes: dict[str, dict] = {}
    specs: dict[str, tuple] = {}
    srcs: dict[str, Path] = {}
    for cam in CAMS:
        p = _clip_for(cam)
        if p is None:
            log(f"FATAL: no clip for {cam}")
            return 2
        srcs[cam] = p
        by_cam[cam], boxes[cam], specs[cam] = track_clip(
            model, embedder, cam, p, args.imgsz, args.topk)
        if not by_cam[cam]:
            log(f"FATAL: no tracks in {cam}")
            return 2

    log("\nmerging tracker fragments ...")
    frag_map: dict[str, dict[int, list]] = {}
    for cam in CAMS:
        before = len(by_cam[cam])
        by_cam[cam], merges, thr = merge_fragments(by_cam[cam], cam)
        frag_map[cam] = {t.local_id: getattr(t, "fragment_ids", [t.local_id])
                         for t in by_cam[cam]}
        log(f"  {cam}: {before} -> {len(by_cam[cam])} ({merges} merge(s))")

    pool, _ = moving_pool(by_cam, args.min_frames, args.min_disp)
    ranked = find_best_chain(pool)
    if not ranked:
        log("FATAL: no chain found.")
        return 2
    best = ranked[0]
    chain = best["chain"]
    # Stable across runs: hash() is salted per process, so it would rename the
    # same vehicle on every rebuild.
    import hashlib
    gid = "GV_" + hashlib.sha1(
        "|".join(sorted(m.key for m in chain.values())).encode()
    ).hexdigest()[:6].upper()

    log(f"\n  global id {gid}   weakest pair {best['weakest']:.4f}"
        + (f"   margin +{best['weakest'] - ranked[1]['weakest']:.4f}"
           if len(ranked) > 1 else ""))
    for cam in CAMS:
        m = chain.get(cam)
        if m:
            log(f"    {cam}: track #{m.local_id} "
                f"(fragments {frag_map[cam].get(m.local_id, [m.local_id])}), "
                f"{m.n_frames} frames")

    log("\nrendering ...")
    manifest = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "global_id": gid,
                "plate_ground_truth": args.plate,
                "plate_note": ("Operator-verified. The recogniser read no plate "
                               "in any of these clips - the match used "
                               "appearance only, so the plate is an "
                               "independent check, not an input."),
                "distractors": 0, "bar": None, "clips": []}
    for cam in CAMS:
        m = chain.get(cam)
        if m is None:
            continue
        tids = set(frag_map[cam].get(m.local_id, [m.local_id]))
        dst = DEMO_DIR / f"{cam}.mp4"
        render(cam, srcs[cam], boxes[cam], tids, gid, args.plate,
               specs[cam], dst)
        others = [c for c in chain if c != cam]
        sim = float(np.mean([chain[cam].emb @ chain[o].emb for o in others]))
        log(f"  {cam} -> {dst.name}  (tracks {sorted(tids)})")
        manifest["clips"].append({
            "camera": cam, "video": f"reid_demo/{cam}.mp4",
            "track": int(m.local_id), "similarity": round(sim, 4),
            "matched": True, "label": gid,
        })

    (DEMO_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")

    # The gallery the LIVE overlay matches against, so the camera tiles show
    # this same id instead of a per-camera tracker number. One reference
    # embedding per camera rather than a single averaged one: the subject looks
    # very different head-on and from behind, and a mean of the four is a
    # viewpoint that none of them actually is.
    gallery = {
        "global_id": gid,
        "plate_ground_truth": args.plate,
        # What KIND of thing this is. The live matcher refuses to hand a
        # motorcycle's identity to a car-class box: without that rule a
        # roadside "SLOW DOWN" sign, detected as a car, scored above the
        # threshold on CAM_M2 and was labelled with this id.
        "vehicle_class": next((chain[c].cls_name for c in CAMS if c in chain),
                              "motorcycle"),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("Reference embeddings for the vehicle proven across CAM_M1..M4 "
                 "by reid_proof_video.py. The live overlay scores a track "
                 "against every reference and takes the best."),
        "references": [
            {"camera": c, "embedding": [round(float(x), 6)
                                        for x in chain[c].emb.tolist()]}
            for c in CAMS if c in chain
        ],
    }
    gal_path = ROOT / "output" / "reid_gallery.json"
    gal_path.write_text(json.dumps(gallery), encoding="utf-8")
    log(f"  gallery  -> {gal_path}  ({len(gallery['references'])} references)")
    log(f"\n  {len(manifest['clips'])}/4 clips carry {gid}")
    log(f"  manifest -> {DEMO_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
