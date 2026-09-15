"""backend/scripts/extract_id_crops.py — turn raw camera video into browsable
candidate-identity crops, ready for a human to pick cross-camera pairs from.

WHY THIS EXISTS
  reid_validate.py and its face-mode counterpart need same/different pairs
  laid out in an exact folder contract (see their own docstrings). Building
  that by hand from raw video — scrubbing footage, freezing frames, cropping
  — is the actual bottleneck once real government-provided footage lands.
  This script does the mechanical part (detect, track, pick the best crop
  per identity) so a human only has to do the part a human actually has to
  do: look at two crops and decide if they're the same person.

  Person mode reuses backend.core.gpu_inference_engine.SentinelGPUEngine
  (the same detector already verified elsewhere in this codebase) rather
  than re-implementing YOLO inference — its built-in per-camera track_id
  continuity is exactly "give me one best crop per identity seen in this
  video," which is the actual thing this script needs.

  Face mode reuses backend.services.face_embedder's InsightFace app —
  same reasoning, don't re-initialize a second face detector.

USAGE
  python -m backend.scripts.extract_id_crops --video cam1.mp4 --camera-id CAM-1
  python -m backend.scripts.extract_id_crops --video cam1.mp4 --camera-id CAM-1 --mode face

OUTPUT
  data/labelling_crops/<mode>/<camera-id>/track_XXXX.jpg  — one crop per
    identity, the highest-confidence frame seen for that track.
  data/labelling_crops/<mode>/<camera-id>/manifest.json   — track_id ->
    {crop, confidence, first_seen_sec, last_seen_sec, frame_count}, so a
    human reviewer (or a gallery viewer) has timing context, not just a
    bare thumbnail.

NEXT STEP
  Open the crops from two or more cameras side by side, pick pairs you
  believe are the same person (or are confidently different), and turn each
  pair into a labelled example with:
    python -m backend.scripts.build_pairs_from_crops --dataset-dir ... \
        --same <crop_a> <crop_b>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def extract_person_crops(video_path: Path, camera_id: str, out_dir: Path,
                          every_nth: int, max_frames: int | None) -> dict:
    from backend.core.gpu_inference_engine import SentinelGPUEngine

    engine = SentinelGPUEngine("yolov8n.onnx")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # track_id -> best-so-far record. Kept in memory; crops are written once
    # at the end so a later, higher-confidence sighting of the same track
    # can replace an earlier, worse one without leaving stale files behind.
    best: dict[int, dict] = {}
    frame_index = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % every_nth == 0:
            result = engine.process_frame(camera_id, frame, frame_index, apply_dedup=False)
            for det in result.detections:
                if det.class_name != "person":
                    continue
                ts_sec = frame_index / fps
                rec = best.get(det.track_id)
                if rec is None:
                    best[det.track_id] = {
                        "crop": det.crop, "confidence": det.confidence,
                        "first_seen_sec": ts_sec, "last_seen_sec": ts_sec,
                        "frame_count": 1,
                    }
                else:
                    rec["last_seen_sec"] = ts_sec
                    rec["frame_count"] += 1
                    if det.confidence > rec["confidence"]:
                        rec["crop"] = det.crop
                        rec["confidence"] = det.confidence
            processed += 1
            if max_frames and processed >= max_frames:
                break
        frame_index += 1
    cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for track_id, rec in best.items():
        name = f"track_{track_id:04d}.jpg"
        cv2.imwrite(str(out_dir / name), rec["crop"])
        manifest[str(track_id)] = {
            "crop": name,
            "confidence": round(float(rec["confidence"]), 3),
            "first_seen_sec": round(rec["first_seen_sec"], 1),
            "last_seen_sec": round(rec["last_seen_sec"], 1),
            "frame_count": rec["frame_count"],
        }
    return manifest


def extract_face_crops(video_path: Path, camera_id: str, out_dir: Path,
                        every_nth: int, max_frames: int | None) -> dict:
    from backend.services.face_embedder import FaceEmbedder

    embedder = FaceEmbedder()
    if embedder.is_stub:
        raise RuntimeError(
            "FaceEmbedder is running in stub mode — InsightFace did not load. "
            "Fix that before extracting face crops; a stub detector finds no "
            "real faces."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # No track continuity here — InsightFace's detector doesn't carry a
    # track_id the way the YOLO engine does, and re-identifying "the same
    # face across frames" is exactly the problem this whole pipeline exists
    # to solve, so it can't be assumed for free. Instead: keep the single
    # highest-confidence face crop per SAMPLED FRAME, numbered sequentially.
    # A human reviewer looking at the gallery decides which frames — even
    # from the same camera — show the same person.
    faces_found = []
    frame_index = 0
    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % every_nth == 0:
            detected = embedder._app.get(frame)
            if detected:
                top = max(detected, key=lambda f: f.det_score)
                x1, y1, x2, y2 = [max(0, int(v)) for v in top.bbox]
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    faces_found.append({
                        "crop": crop, "confidence": float(top.det_score),
                        "ts_sec": frame_index / fps,
                    })
            processed += 1
            if max_frames and processed >= max_frames:
                break
        frame_index += 1
    cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for i, rec in enumerate(faces_found):
        name = f"frame_{i:04d}.jpg"
        cv2.imwrite(str(out_dir / name), rec["crop"])
        manifest[str(i)] = {
            "crop": name,
            "confidence": round(rec["confidence"], 3),
            "ts_sec": round(rec["ts_sec"], 1),
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--camera-id", required=True,
                        help="Label for this camera, e.g. CAM-07 or the "
                             "government's own camera identifier.")
    parser.add_argument("--mode", choices=["person", "face"], default="person")
    parser.add_argument("--output-dir", type=Path, default=Path("data/labelling_crops"))
    parser.add_argument("--every-nth", type=int, default=5,
                        help="Process every Nth frame. Lower = more thorough "
                             "and slower; a person/face rarely changes enough "
                             "frame-to-frame to need every single one.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap on SAMPLED frames processed (after --every-nth "
                             "skipping), for a quick pass over a long video.")
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir / args.mode / args.camera_id
    print(f"Extracting {args.mode} crops from {args.video} -> {out_dir}")

    if args.mode == "person":
        manifest = extract_person_crops(args.video, args.camera_id, out_dir,
                                        args.every_nth, args.max_frames)
    else:
        manifest = extract_face_crops(args.video, args.camera_id, out_dir,
                                      args.every_nth, args.max_frames)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done: {len(manifest)} identity crop(s) written to {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
