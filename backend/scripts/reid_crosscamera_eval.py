"""Cross-camera Re-ID, scored against distractors from the real fleet.

WHAT THIS MEASURES
  Four phone clips follow one motorcycle past four different points. That gives
  something the live fleet cannot: a cross-camera truth somebody actually
  verified. But matching one vehicle against one vehicle proves nothing — with a
  single candidate the matcher is right by having no alternative.

  So the three genuine matches are hidden among several hundred vehicles cut
  from this project's own CCTV clips, and the question becomes the one Re-ID is
  actually asked in the field: out of everything the network has seen, does the
  right vehicle come first? That is standard Re-ID evaluation — Rank-1, Rank-5,
  mAP — over a gallery the system did not choose.

WHAT IT DOES NOT MEASURE
  Phone footage is cleaner than a roadside pole: closer, steadier, higher
  resolution. This says whether the matching is sound, not what it scores on
  CCTV. And OSNet is a person re-identification model, so on a motorcycle much
  of its signal is the rider, not the machine.

  python -m backend.scripts.reid_crosscamera_eval
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2                                                         # noqa: E402
import numpy as np                                                 # noqa: E402

REID_DIR = ROOT / "data" / "reid_test"
CLIPS_DIR = ROOT / "data" / "clips"
OUT_DIR = REID_DIR / "_output"
CAMS = ["CAM_M1", "CAM_M2", "CAM_M3", "CAM_M4"]
VEHICLE_CLASSES = [2, 3, 5, 7]          # car, motorcycle, bus, truck
PERSON_CLASS = 0                        # a rider is often detected as a person


def log(msg: str) -> None:
    print(msg, flush=True)


def best_tracks_from_clip(model, path: Path, imgsz: int, want_person: bool):
    """Every track in one clip, each reduced to its sharpest large crop.

    A track is kept as ONE vector, not many: the question is whether this
    vehicle matches that vehicle, so a track that lingers must not outvote one
    that passes quickly.
    """
    classes = list(VEHICLE_CLASSES) + ([PERSON_CLASS] if want_person else [])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    model.predictor = None                       # fresh tracker per clip
    tracks: dict[int, dict] = {}
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_i += 1
        if frame_i % 3:                          # 60 fps phone video — every 3rd
            continue
        r = model.track(frame, persist=True, classes=classes, conf=0.35,
                        tracker="botsort.yaml", imgsz=imgsz, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            continue
        for box, tid, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                 r.boxes.id.cpu().numpy().astype(int),
                                 r.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
            if x2 - x1 < 40 or y2 - y1 < 60:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # sharpness x area: prefer a big, in-focus view of the subject
            score = float(cv2.Laplacian(g, cv2.CV_64F).var()) * (crop.shape[0] * crop.shape[1]) ** 0.5
            t = tracks.setdefault(int(tid), {"frames": 0, "score": -1.0,
                                             "crop": None, "cls": int(cls)})
            t["frames"] += 1
            if score > t["score"]:
                t["score"] = score
                t["crop"] = crop.copy()
    cap.release()
    return {k: v for k, v in tracks.items() if v["crop"] is not None}


def sample_distractors(model, embedder, n_target: int, imgsz: int, seed: int):
    """Vehicle crops from the project's own CCTV clips, as the gallery's noise."""
    rng = random.Random(seed)
    clips = sorted(CLIPS_DIR.glob("CAM_*/*.mp4"))
    rng.shuffle(clips)
    crops: list[np.ndarray] = []
    used_clips = 0
    for clip in clips:
        if len(crops) >= n_target:
            break
        cap = cv2.VideoCapture(str(clip))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total < 30:
            cap.release()
            continue
        used_clips += 1
        for idx in np.linspace(total * 0.15, total * 0.85, 14).astype(int):
            if len(crops) >= n_target:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if not ok or f is None:
                continue
            r = model.predict(f, classes=VEHICLE_CLASSES, conf=0.4,
                              imgsz=imgsz, verbose=False)[0]
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                if len(crops) >= n_target:
                    break
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(f.shape[1] - 1, x2), min(f.shape[0] - 1, y2)
                if x2 - x1 < 45 or y2 - y1 < 45:
                    continue
                c = f[y1:y2, x1:x2]
                if c.size:
                    crops.append(c.copy())
        cap.release()
    log(f"  distractors: {len(crops)} crops from {used_clips} fleet clips")
    if not crops:
        return np.zeros((0, 512), np.float32), []
    return embedder.extract_batch(crops), crops


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distractors", type=int, default=500)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--include-person", action="store_true", default=True,
                    help="a motorcycle rider is frequently detected as a person")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="dump every track found, numbered, and stop")
    ap.add_argument("--subjects", default=None,
                    help="which track is the subject in each clip, e.g. "
                         "CAM_M1=4,CAM_M2=11,CAM_M3=2,CAM_M4=9")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    from backend.services.reid_embedder import get_embedder

    log("loading models ...")
    model = YOLO(str(ROOT / "yolov8s.pt"))
    embedder = get_embedder()

    # A stub embedder returns noise. Scoring that would produce a number with
    # no meaning at all, so stop rather than report one. (is_stub and
    # checkpoint_applied are properties on this class, not methods.)
    def _val(obj, name):
        v = getattr(obj, name, None)
        return v() if callable(v) else v

    if _val(embedder, "is_stub"):
        log("FATAL: the Re-ID model did not load — embeddings would be random. "
            "Refusing to produce a score.")
        return 2
    log(f"  Re-ID checkpoint: {_val(embedder, 'checkpoint_applied')}")

    # ── the four clips ──────────────────────────────────────────────────────
    #
    # WHICH TRACK IS THE SUBJECT MUST BE CHOSEN BY A PERSON.
    #
    # The first version of this script guessed: it took the longest track in
    # each clip, on the assumption that the camera follows its subject longest.
    # It was wrong in three clips out of four — it picked a red motorcycle
    # parked in the background of two of them and a row of parked bikes in a
    # third, then scored 0% Rank-1 for comparing three different vehicles. A
    # measurement of the wrong thing is worse than no measurement, because it
    # looks like a result. So the subject is now named on the command line, and
    # --contact-sheet exists to see the candidates first.
    all_tracks: dict[str, dict] = {}
    for cam in CAMS:
        vids = sorted((REID_DIR / cam).glob("*.mp4"))
        if not vids:
            log(f"FATAL: no video in {cam}")
            return 2
        tr = best_tracks_from_clip(model, vids[0], args.imgsz, args.include_person)
        if not tr:
            log(f"FATAL: no track found in {cam} — cannot evaluate")
            return 2
        all_tracks[cam] = tr
        log(f"  {cam}: {len(tr)} track(s) -> "
            + ", ".join(f"#{k}({v['frames']}f)" for k, v in
                        sorted(tr.items(), key=lambda kv: -kv[1]['frames'])[:8]))

    if args.contact_sheet or not args.subjects:
        for cam, tr in all_tracks.items():
            items = sorted(tr.items(), key=lambda kv: -kv[1]["frames"])
            tiles = []
            for tid, t in items[:12]:
                x = cv2.resize(t["crop"], (150, 260))
                cv2.rectangle(x, (0, 0), (149, 259), (60, 60, 60), 2)
                cv2.putText(x, f"#{tid}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 230, 255), 2, cv2.LINE_AA)
                cv2.putText(x, f"{t['frames']}f", (8, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
                            cv2.LINE_AA)
                tiles.append(x)
            sheet = np.hstack(tiles) if tiles else np.zeros((260, 150, 3), np.uint8)
            p = OUT_DIR / f"tracks_{cam}.jpg"
            cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
            log(f"  candidates -> {p}")
        log("\nOpen those four images, find the vehicle that appears in all of "
            "them, and re-run naming its track number in each clip:\n"
            "  python -m backend.scripts.reid_crosscamera_eval "
            "--subjects CAM_M1=<n>,CAM_M2=<n>,CAM_M3=<n>,CAM_M4=<n>")
        return 0

    subjects: dict[str, dict] = {}
    for part in args.subjects.split(","):
        cam, _, num = part.strip().partition("=")
        cam = cam.strip().upper()
        if cam not in all_tracks:
            log(f"FATAL: unknown clip {cam}")
            return 2
        try:
            tid = int(num)
        except ValueError:
            log(f"FATAL: track number for {cam} is not a number: {num!r}")
            return 2
        if tid not in all_tracks[cam]:
            log(f"FATAL: {cam} has no track #{tid}. Available: "
                + ", ".join(str(k) for k in sorted(all_tracks[cam])))
            return 2
        subjects[cam] = {"tid": tid, **all_tracks[cam][tid]}
    missing = [c for c in CAMS if c not in subjects]
    if missing:
        log(f"FATAL: no subject given for {', '.join(missing)}")
        return 2
    for cam in CAMS:
        s = subjects[cam]
        log(f"  {cam}: subject #{s['tid']} ({s['frames']} frames, "
            f"{s['crop'].shape[1]}x{s['crop'].shape[0]})")

    sheet = np.hstack([cv2.resize(subjects[c]["crop"], (170, 300)) for c in CAMS])
    for i, c in enumerate(CAMS):
        cv2.putText(sheet, c, (i * 170 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT_DIR / "subjects.jpg"), sheet)
    log(f"  subject crops -> {OUT_DIR / 'subjects.jpg'}  (check they are the same vehicle)")

    embs = embedder.extract_batch([subjects[c]["crop"] for c in CAMS])
    embs = np.asarray(embs, dtype=np.float32)

    # ── gallery: 3 true matches + fleet distractors ─────────────────────────
    log(f"sampling {args.distractors} distractors from the fleet ...")
    d_emb, d_crops = sample_distractors(model, embedder, args.distractors,
                                        args.imgsz, args.seed)
    if len(d_emb) < 50:
        log("FATAL: too few distractors to make the gallery meaningful.")
        return 2

    results = []
    for qi, qcam in enumerate(CAMS):
        gal_cams = [c for c in CAMS if c != qcam]
        gal = np.vstack([embs[[CAMS.index(c) for c in gal_cams]], d_emb])
        is_true = np.array([True] * len(gal_cams) + [False] * len(d_emb))
        sims = gal @ embs[qi]                       # embeddings are L2-normalised
        order = np.argsort(-sims)
        ranked = is_true[order]
        rank1 = bool(ranked[0])
        rank5 = bool(ranked[:5].any())
        hits = np.where(ranked)[0]
        ap = float(np.mean([(i + 1) / (r + 1) for i, r in enumerate(hits)])) if len(hits) else 0.0
        first = int(hits[0]) + 1 if len(hits) else -1
        results.append({"query": qcam, "rank1": rank1, "rank5": rank5, "ap": ap,
                        "first_true_at": first, "gallery": len(gal),
                        "top_sim": float(sims[order[0]]),
                        "order": order, "is_true": is_true, "sims": sims,
                        "gal_cams": gal_cams})
        log(f"  query {qcam}: first true match at rank {first} of {len(gal)}  "
            f"| Rank-1 {'HIT' if rank1 else 'miss'} | AP {ap:.3f}")

    r1 = sum(r["rank1"] for r in results) / len(results)
    r5 = sum(r["rank5"] for r in results) / len(results)
    mAP = sum(r["ap"] for r in results) / len(results)

    log("\n" + "=" * 62)
    log("CROSS-CAMERA Re-ID — 4 phone clips vs a fleet gallery")
    log("=" * 62)
    log(f"  gallery size per query : {results[0]['gallery']}  "
        f"(3 true + {len(d_emb)} distractors)")
    log(f"  Rank-1                 : {r1*100:.1f}%   ({sum(r['rank1'] for r in results)}/4 queries)")
    log(f"  Rank-5                 : {r5*100:.1f}%")
    log(f"  mAP                    : {mAP*100:.1f}%")
    log("=" * 62)

    # ── visual: query and its top matches ───────────────────────────────────
    rows = []
    for r in results:
        q = cv2.resize(subjects[r["query"]]["crop"], (150, 260))
        cv2.rectangle(q, (0, 0), (149, 259), (0, 200, 255), 3)
        cv2.putText(q, "QUERY", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(q, r["query"], (8, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles = [q, np.full((260, 14, 3), 30, np.uint8)]
        for k in range(5):
            gi = r["order"][k]
            if gi < len(r["gal_cams"]):
                crop = subjects[r["gal_cams"][gi]]["crop"]
                good = True
            else:
                crop = d_crops[gi - len(r["gal_cams"])]
                good = False
            t = cv2.resize(crop, (150, 260))
            col = (80, 220, 100) if good else (80, 80, 220)
            cv2.rectangle(t, (0, 0), (149, 259), col, 3)
            cv2.putText(t, f"#{k+1} {'OK' if good else 'x'} {r['sims'][gi]:.2f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
            tiles.append(t)
        rows.append(np.hstack(tiles))
    w = max(x.shape[1] for x in rows)
    rows = [np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0)), constant_values=30) for x in rows]
    grid = np.vstack([np.pad(x, ((0, 10), (0, 0), (0, 0)), constant_values=30) for x in rows])
    out = OUT_DIR / "reid_ranking.jpg"
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    log(f"\nvisual -> {out}")

    with open(OUT_DIR / "reid_result.txt", "w", encoding="utf-8") as f:
        f.write("Cross-camera Re-ID, 4 phone clips against a fleet gallery\n")
        f.write(f"gallery per query : {results[0]['gallery']} "
                f"(3 true + {len(d_emb)} distractors)\n")
        f.write(f"Rank-1            : {r1*100:.1f}%\n")
        f.write(f"Rank-5            : {r5*100:.1f}%\n")
        f.write(f"mAP               : {mAP*100:.1f}%\n\n")
        for r in results:
            f.write(f"  {r['query']}: first true match at rank {r['first_true_at']}, "
                    f"AP {r['ap']:.3f}\n")
        f.write("\nPhone footage is cleaner than this fleet's CCTV, and OSNet is a\n"
                "person re-id model, so on a motorcycle much of the signal is the\n"
                "rider. This measures whether the matching is sound, not what it\n"
                "would score on a roadside pole.\n")
    log(f"report -> {OUT_DIR / 'reid_result.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
