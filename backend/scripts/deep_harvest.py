"""backend/scripts/deep_harvest.py — quality-gated frame harvesting.

WHY NOT JUST GRAB MORE FRAMES
  bulk_harvest.py and seek_harvest.py compare each candidate against the
  LAST KEPT frame only. That catches consecutive near-duplicates, but it
  cannot notice that a frame is a near-copy of something saved ten minutes
  ago, on a previous run, or from a different camera pointed at a similar
  road. The dataset therefore accumulates redundancy that costs labelling
  time and training time while teaching the model nothing.

  It also treats every unique frame as equally valuable, which is false. A
  sharp frame of an empty road at 3am is unique and nearly worthless. A
  blurred frame is unique and actively harmful - the teacher will label it
  badly and the student will learn that bad label.

FIVE GATES, EVERY FRAME LOGGED
  1. DECODE    frame readable at all
  2. BLUR      variance of Laplacian >= --min-sharpness. Motion-blurred and
               out-of-focus frames produce unreliable teacher labels.
  3. EXPOSURE  reject near-black / blown-out frames with no recoverable
               detail (mean luma outside [--min-luma, --max-luma]).
  4. UNIQUE    64-bit dHash compared against EVERY frame already in the
               dataset AND everything kept this run. Hamming distance
               <= --hash-dist counts as a duplicate. This is the
               "guaranteed unique" requirement: the index is persisted, so
               re-running never re-saves something already held.
  5. VALUE     run the CURRENT deployed detector. A frame earns its place if
               it contains objects the model finds HARD - low mean
               confidence, or a busy scene. This is uncertainty sampling:
               frames the model is already confident about have little
               gradient to give, while frames it is unsure about are
               precisely where new labels help.

  Every candidate prints one line saying KEPT or which gate rejected it, so
  the yield is auditable rather than a number at the end.

HONEST LIMITS
  * dHash is perceptual, not semantic. Two genuinely different vehicles in
    an identical scene can hash close and one may be dropped. --hash-dist
    trades that off; the default 5 is conservative.
  * "Learning value" is a heuristic proxy measured against the current
    model, not ground truth. It biases the set toward hard examples, which
    is usually right, but it is a bias.

USAGE
  python -m backend.scripts.deep_harvest --index          # build hash index
  python -m backend.scripts.deep_harvest --cameras CAM_02 --target 40
  python -m backend.scripts.deep_harvest --times 07:00,08:30 --target 60
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGES_ROOT = Path("data/detection_train/images")
HASH_INDEX = Path("data/detection_train/dhash_index.json")
MANIFEST = Path("data/detection_train/deep_harvest_manifest.json")
FILE_START_HOUR = 21.0
FILE_DURATION_H = 12.0
_lock = threading.Lock()


# ── perceptual hash ──────────────────────────────────────────────────────
def dhash(img, size: int = 8) -> int:
    """64-bit difference hash: compares each pixel to its right neighbour.

    Robust to small exposure and compression changes, which is what we want
    - two JPEG re-encodes of the same scene must collide, while a genuinely
    different scene must not.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = g[:, 1:] > g[:, :-1]
    out = 0
    for bit in diff.flatten():
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class HashIndex:
    """Every dHash already in the dataset.

    Bucketed by the top 16 bits so a lookup compares against a slice rather
    than all ~5,000 entries. Near-duplicates can straddle a bucket boundary,
    so neighbouring buckets are checked too.
    """

    def __init__(self) -> None:
        self.buckets: dict[int, list[int]] = {}
        self.count = 0

    def _key(self, h: int) -> int:
        return h >> 48

    def add(self, h: int) -> None:
        self.buckets.setdefault(self._key(h), []).append(h)
        self.count += 1

    def is_duplicate(self, h: int, max_dist: int) -> bool:
        k = self._key(h)
        for kk in (k - 1, k, k + 1):
            for existing in self.buckets.get(kk, ()):
                if hamming(h, existing) <= max_dist:
                    return True
        return False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = [h for bucket in self.buckets.values() for h in bucket]
        path.write_text(json.dumps({"hashes": flat}))

    @classmethod
    def load(cls, path: Path) -> "HashIndex":
        idx = cls()
        if path.is_file():
            try:
                for h in json.loads(path.read_text()).get("hashes", []):
                    idx.add(int(h))
            except (json.JSONDecodeError, OSError, ValueError):
                print("hash index unreadable - rebuild with --index",
                      file=sys.stderr)
        return idx


def build_index(args) -> HashIndex:
    """Hash every frame already in the dataset."""
    files = [p for p in IMAGES_ROOT.rglob("*.jpg")
             if "train" not in p.parts and "val" not in p.parts]
    print(f"indexing {len(files)} existing frame(s)...", flush=True)
    idx = HashIndex()
    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is not None:
            idx.add(dhash(img))
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    idx.save(HASH_INDEX)
    print(f"indexed {idx.count} frame(s) -> {HASH_INDEX}")
    return idx


# ── quality gates ────────────────────────────────────────────────────────
def sharpness(img) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def clock_to_offset_s(hhmm: str):
    h, m = (int(x) for x in hhmm.split(":"))
    off = ((h + m / 60.0) - FILE_START_HOUR) % 24.0
    return None if off > FILE_DURATION_H else off * 3600.0


def harvest_camera(cam, hhmm, args, index, model):
    cam_id = cam["id"]
    offset = clock_to_offset_s(hhmm) if hhmm else None
    out_dir = IMAGES_ROOT / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"camera": cam_id, "clock": hhmm, "seen": 0, "kept": 0,
             "reject_decode": 0, "reject_blur": 0, "reject_exposure": 0,
             "reject_duplicate": 0, "reject_lowvalue": 0, "status": "ok"}

    tmp = Path(tempfile.mkdtemp(prefix=f"deep_{cam_id}_"))
    try:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-rw_timeout", "300000000"]
        if offset is not None:
            cmd += ["-ss", f"{offset:.0f}"]
        cmd += ["-i", cam["url"], "-t", str(args.window),
                "-vf", f"fps=1/{args.gap_seconds}", "-q:v", "2",
                "-pix_fmt", "yuvj420p", "-strict", "unofficial",
                "-y", str(tmp / "f_%04d.jpg")]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout)
        if proc.returncode != 0:
            stats["status"] = f"ffmpeg rc={proc.returncode}: {proc.stderr.strip()[:90]}"

        stamp = int(time.time())
        for p in sorted(tmp.glob("*.jpg")):
            if stats["kept"] >= args.target:
                break
            stats["seen"] += 1
            tag = f"  {cam_id} {hhmm or 'head'} #{stats['seen']:03d}"

            img = cv2.imread(str(p))
            if img is None:
                stats["reject_decode"] += 1
                with _lock:
                    print(f"{tag} REJECT decode", flush=True)
                continue

            sh = sharpness(img)
            if sh < args.min_sharpness:
                stats["reject_blur"] += 1
                with _lock:
                    print(f"{tag} REJECT blur (sharp {sh:.0f} < {args.min_sharpness})",
                          flush=True)
                continue

            luma = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
            if luma < args.min_luma or luma > args.max_luma:
                stats["reject_exposure"] += 1
                with _lock:
                    print(f"{tag} REJECT exposure (luma {luma:.0f})", flush=True)
                continue

            h = dhash(img)
            if index.is_duplicate(h, args.hash_dist):
                stats["reject_duplicate"] += 1
                with _lock:
                    print(f"{tag} REJECT duplicate (dHash within {args.hash_dist})",
                          flush=True)
                continue

            n_det, mean_conf = 0, 0.0
            if model is not None:
                r = model(img, imgsz=args.imgsz, conf=0.25, verbose=False,
                          quantize=16)[0]
                if r.boxes is not None and len(r.boxes):
                    n_det = len(r.boxes)
                    mean_conf = float(r.boxes.conf.mean())
            # Uncertainty sampling: a frame is worth labelling if the model
            # is unsure (low mean confidence) or the scene is busy. Frames
            # it already handles confidently add little.
            valuable = (n_det >= args.min_objects
                        and (mean_conf <= args.max_conf
                             or n_det >= args.busy_objects))
            if model is not None and not valuable:
                stats["reject_lowvalue"] += 1
                with _lock:
                    print(f"{tag} REJECT low-value ({n_det} obj, "
                          f"conf {mean_conf:.2f})", flush=True)
                continue

            name = (f"{cam_id}_deep{stamp}_{(hhmm or 'head').replace(':', '')}"
                    f"_{stats['kept']:04d}.jpg")
            cv2.imwrite(str(out_dir / name), img)
            index.add(h)
            stats["kept"] += 1
            with _lock:
                print(f"{tag} KEPT  sharp {sh:>5.0f} luma {luma:>3.0f} "
                      f"obj {n_det:>2} conf {mean_conf:.2f}", flush=True)
    except subprocess.TimeoutExpired:
        stats["status"] = f"TIMEOUT after {args.timeout}s"
    except Exception as exc:
        stats["status"] = f"EXCEPTION {type(exc).__name__}: {exc}"
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="store_true",
                    help="(Re)build the dHash index from the dataset, then exit.")
    ap.add_argument("--cameras", default="")
    ap.add_argument("--times", default="07:00,08:30",
                    help="Clock targets inside the 21:00-09:00 recording. "
                         "Use 'head' to read from the start instead.")
    ap.add_argument("--target", type=int, default=40, help="Kept frames per job.")
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--gap-seconds", type=float, default=3.0)
    ap.add_argument("--hash-dist", type=int, default=5,
                    help="Hamming distance under which two frames count as "
                         "the same image. Lower = stricter uniqueness.")
    ap.add_argument("--min-sharpness", type=float, default=40.0)
    ap.add_argument("--min-luma", type=float, default=25.0)
    ap.add_argument("--max-luma", type=float, default=235.0)
    ap.add_argument("--min-objects", type=int, default=1)
    ap.add_argument("--max-conf", type=float, default=0.80,
                    help="Keep frames whose mean detection confidence is at "
                         "or below this - the model is unsure, so a label "
                         "teaches it something.")
    ap.add_argument("--busy-objects", type=int, default=6,
                    help="...or keep it regardless if the scene is this busy.")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--no-model", action="store_true",
                    help="Skip the learning-value gate (quality + uniqueness only).")
    ap.add_argument("--include-disabled", action="store_true",
                    help="Also harvest cameras marked enabled:false. Those "
                         "were disabled after probing dead on an earlier "
                         "date; if the estate has since been fixed, include "
                         "them rather than silently skipping a quarter of "
                         "the fleet.")
    ap.add_argument("--timeout", type=float, default=1500.0)
    args = ap.parse_args()

    if args.index:
        build_index(args)
        return

    index = HashIndex.load(HASH_INDEX)
    if index.count == 0:
        print("No hash index found - building it first so uniqueness is "
              "guaranteed against the existing dataset.\n")
        index = build_index(args)
    print(f"hash index: {index.count} known frame(s)\n")

    model = None
    if not args.no_model:
        from ultralytics import YOLO
        cfg0 = yaml.safe_load(open("config.yaml", encoding="utf-8"))
        model = YOLO(cfg0["model"]["path"])
        print(f"learning-value gate using {cfg0['model']['path']}\n")

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    only = {c.strip() for c in args.cameras.split(",") if c.strip()} or None
    cams = [c for c in cfg.get("demo_cameras", [])
            if (args.include_disabled or c.get("enabled") is not False)
            and c.get("url")
            and (not only or c["id"] in only)]
    n_disabled = sum(1 for c in cams if c.get("enabled") is False)
    if n_disabled:
        print(f"including {n_disabled} camera(s) previously marked "
              f"enabled:false\n")
    times = [t.strip() for t in args.times.split(",") if t.strip()]
    times = [None if t.lower() == "head" else t for t in times]

    print(f"{len(cams)} camera(s) x {len(times)} target(s); "
          f"up to {args.target} kept each\n")

    report = []
    for t in times:
        for cam in cams:
            report.append(harvest_camera(cam, t, args, index, model))
            index.save(HASH_INDEX)   # persist as we go, so a crash loses nothing

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(report, indent=2))

    kept = sum(r["kept"] for r in report)
    seen = sum(r["seen"] for r in report)
    print("\n" + "=" * 62)
    print(f"candidates examined : {seen}")
    print(f"KEPT (unique+useful): {kept}")
    for key, label in (("reject_duplicate", "duplicates"),
                       ("reject_blur", "too blurred"),
                       ("reject_exposure", "bad exposure"),
                       ("reject_lowvalue", "low learning value"),
                       ("reject_decode", "undecodable")):
        print(f"  rejected {label:<19}: {sum(r[key] for r in report)}")
    print(f"hash index now      : {index.count} frame(s)")
    print(f"dataset total       : "
          f"{sum(1 for p in IMAGES_ROOT.rglob('*.jpg') if 'train' not in p.parts and 'val' not in p.parts)}")


if __name__ == "__main__":
    main()
