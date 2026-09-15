"""backend/scripts/journey_build_index.py — build the searchable index every
plate query runs against.

THE IDEA THAT MAKES THIS WORK AT 59% RECOGNITION
  Reading a plate and finding a plate are different problems, and the second is
  far easier. Recognition has to produce the right ten characters from nothing.
  Retrieval is handed the right ten characters by the operator and only has to
  decide which stored track they belong to.

  Almost every system stores the DECODED string and compares text. That throws
  away everything the recogniser was unsure about: if the model was torn 55/45
  between B and 8 at position six, the argmax keeps 8 and the information that
  B was nearly as likely is gone. A query for the plate ending in B then misses
  a track that was, in the model's own view, half a vote away from it.

  So this stores the raw CTC logits instead. At query time the forward
  algorithm gives P(query | this track's pixels) directly - which is exactly
  what CTC was built to compute - and a track whose argmax was wrong still
  scores highly for the correct string. The reading being wrong stops mattering
  as long as the right answer was in contention.

WHAT ELSE IS STORED, AND WHY EACH EARNS ITS SPACE
  timestamp   clip names carry the capture hour and frames carry the offset,
              so every detection gets a real wall-clock time. Without it there
              is no journey, only a set of sightings.
  gps         already populated for all 32 cameras. Turns a list of sightings
              into a route, and lets an implausible speed reject a bad match.
  crops       the evidence an officer will be shown. A link no one can check
              is not usable in an investigation.

SIZE
  64 timesteps x 37 classes x float16 is 4.7 KB per frame. Six frames per
  track across 5,594 tracks is roughly 160 MB - small enough to hold in memory
  at query time, which is what keeps search interactive.

USAGE
  python -m backend.scripts.journey_build_index --frames 6
  python -m backend.scripts.journey_build_index --limit 300   # smoke test
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.scripts.indian_plate_grammar import decode_plate
from backend.scripts.plate_eval_clean import _vote
from backend.scripts.plate_final_model import PlateCRNN
from backend.scripts.train_plate_recognizer import BLANK, CHARS, ITOS

CORPUS = Path("output/plate_corpus")
OUT = Path("output/journey_index")
# The production detector, not the v3 checkpoint this script was written
# against. v4 was retrained 2026-09-02 and finds 2.6x more plates on live
# footage. The index is the sole source of cross-camera legs, and legs are what
# calibration and its validation both run on, so a detector that misses plates
# here starves everything downstream.
DET_W = Path("models/plate_detector/plate_v4_small.pt")
DB = Path("output/sentinel.db")

# CAM_09_0830 -> hour 08, minute 30
CLIP_TIME = re.compile(r"_(\d{2})(\d{2})$")


def camera_meta():
    """camera_id -> (lat, lon, name, department), from the live database."""
    meta = {}
    if not DB.is_file():
        return meta
    c = sqlite3.connect(str(DB))
    for cid, lat, lon, name, dep in c.execute(
            "select camera_id, gps_lat, gps_lon, name, department "
            "from cameras"):
        # Ids appear as CAM-01 in the database and CAM_01 in the corpus.
        for key in {cid, cid.replace("-", "_"), cid.replace("_", "-")}:
            meta[key] = {"lat": lat, "lon": lon, "name": name,
                         "department": dep}
    return meta


def clip_start(clip: str) -> datetime | None:
    m = CLIP_TIME.search(clip)
    if not m:
        return None
    # The date is the same across the fleet for this footage; only the
    # time-of-day distinguishes clips, and only relative time matters for a
    # journey. A fixed date keeps timestamps comparable and honest about that.
    return datetime(2026, 6, 14, int(m.group(1)), int(m.group(2)))


@torch.no_grad()
def read_frames(members, crops, dev, h, w):
    """Return (per-frame text, per-frame mean-max confidence, mean logits)."""
    texts, confs, logits = [], [], []
    for g in crops:
        gg = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(gg).float().div(127.5).sub(1.0)[None, None].to(dev)
        probs = None
        for m in members:
            p = m(x).log_softmax(2)
            probs = p if probs is None else torch.logaddexp(probs, p)
        # Mean of member distributions in log space, so the stored tensor is a
        # proper log-probability the CTC scorer can consume unchanged.
        lp = probs - float(np.log(len(members)))
        logits.append(lp[0].to(torch.float16).cpu().numpy())
        ids = lp.argmax(2)[0].tolist()
        out, prev = [], -1
        for k in ids:
            if k != prev and k != BLANK:
                out.append(ITOS.get(int(k), ""))
            prev = k
        texts.append("".join(out))
        pr = lp.exp()[0]
        top, idx = pr.max(1)
        keep = idx != BLANK
        confs.append(float(top[keep].mean()) if bool(keep.any()) else 0.0)
    return texts, confs, logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=6,
                    help="Frames indexed per vehicle. More improves the vote "
                         "and the CTC evidence; storage grows linearly.")
    ap.add_argument("--scan", type=int, default=10,
                    help="Sharpest frames examined before keeping --frames.")
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test cap.")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    meta = camera_meta()
    print(f"cameras with metadata : {len(meta)//2}")

    tracks = defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        tracks[(r["camera"], r["clip"], r["track"])].append(r)
    keys = sorted(tracks)
    if args.limit:
        keys = keys[:args.limit]
    print(f"vehicles to index     : {len(keys)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO
    det = YOLO(str(DET_W))
    members = []
    for j in range(3):
        p = Path(f"models/plate_recognizer/final_m{j}.pt")
        if p.is_file():
            ck = torch.load(p, map_location=dev, weights_only=False)
            m = PlateCRNN(len(CHARS) + 1, img_h=args.height).to(dev)
            m.load_state_dict(ck["model"])
            m.eval()
            members.append(m)
    print(f"ensemble members      : {len(members)}\n", flush=True)

    records, all_logits = [], []
    no_plate = 0
    for i, key in enumerate(keys, 1):
        cam, clip, track = key
        rows = tracks[key]
        rows.sort(key=lambda r: -(r.get("sharpness", 0) ** 0.5
                                  * r.get("plate_px_est", 0)))
        crops, frames_used, widths = [], [], []
        for fr in rows[:args.scan]:
            if len(crops) >= args.frames:
                break
            img = cv2.imread(fr["path"])
            if img is None:
                continue
            res = det.predict(img, conf=args.conf, verbose=False)[0]
            if not len(res.boxes):
                continue
            H, W = img.shape[:2]
            best = None
            for b in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = b
                bw, bh = x2 - x1, y2 - y1
                if bw < 34 or bh < 9 or not (1.8 <= bw / max(bh, 1) <= 7.5):
                    continue
                if best is None or bw > best[0]:
                    best = (bw, (x1, y1, x2, y2))
            if best is None:
                continue
            bw, (x1, y1, x2, y2) = best
            mx, my = bw * .08, (y2 - y1) * .20
            c = img[int(max(0, y1 - my)):int(min(H, y2 + my)),
                    int(max(0, x1 - mx)):int(min(W, x2 + mx))]
            if c.size == 0 or c.shape[1] < 28:
                continue
            crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))
            frames_used.append(fr)
            widths.append(float(bw))

        if not crops:
            no_plate += 1
            continue

        texts, confs, logits = read_frames(members, crops, dev,
                                           args.height, args.width)
        voted = _vote(texts)
        d = decode_plate(voted)
        plate = d["plate"] or voted

        base = clip_start(clip)
        fno = frames_used[0].get("frame", 0)
        fps = 25.0
        ts = (base + timedelta(seconds=fno / fps)).isoformat() if base else None

        cm = meta.get(cam, {})
        records.append({
            "idx": len(all_logits),
            "camera": cam, "clip": clip, "track": track,
            "timestamp": ts, "frame": fno,
            "plate": plate, "reads": texts, "conf": confs,
            "plate_w": round(max(widths), 1),
            "lat": cm.get("lat"), "lon": cm.get("lon"),
            "camera_name": cm.get("name"),
            "department": cm.get("department"),
            "crops": [f["path"] for f in frames_used],
            "n_frames": len(crops),
        })
        all_logits.append(np.stack(logits))

        if i % 200 == 0:
            print(f"  {i}/{len(keys)}  indexed {len(records)}  "
                  f"no-plate {no_plate}", flush=True)

    with (OUT / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    # Ragged frame counts, so an object array of per-track stacks rather than
    # one padded tensor - padding would triple the file for no benefit.
    np.save(OUT / "logits.npy",
            np.array(all_logits, dtype=object), allow_pickle=True)

    mb = sum(l.nbytes for l in all_logits) / 1e6
    print("\n" + "=" * 58)
    print(f"vehicles indexed : {len(records)} of {len(keys)}")
    print(f"no plate found   : {no_plate}")
    print(f"logits stored    : {mb:.0f} MB")
    print(f"with timestamp   : {sum(1 for r in records if r['timestamp'])}")
    print(f"with GPS         : {sum(1 for r in records if r['lat'])}")
    print(f"cameras covered  : {len({r['camera'] for r in records})}")
    print(f"saved -> {OUT}")
    print("=" * 58)


if __name__ == "__main__":
    main()
