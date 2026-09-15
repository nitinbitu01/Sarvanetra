"""
backend/scripts/run_cctv_macro_staging.py — Live CCTV Staging Verification Pipeline.

Processes real harvested Gujarat CCTV video clips across calibrated cameras, running:
  1. YOLOv8 + BoT-SORT vehicle tracking
  2. Homography projection (pixel -> world meters -> WGS-84)
  3. Theil-Sen trajectory speed estimation with 95% confidence intervals
  4. Track persistence into SQLite sentinel.db
  5. 1-Minute aggregated metric rollups (P15, Median, P85, vehicle class breakdown)
  6. Free-flow speed bootstrapping (v_free) & real-time Congestion Index (CI)
  7. Cross-camera corridor transit velocity evaluation (CAM_11 -> CAM_08)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

from backend.db.models import Camera, CameraHomographyCalibration, VehicleTrack, CameraMetrics1M
from backend.db.session import SessionLocal, init_db
from backend.services.calibration import (
    compute_homography,
    evaluate_calibration_quality,
    pixel_to_world_m,
    world_to_wgs84,
)
from backend.services.speed_estimator import estimate_track_speed, TrackSpeedResult
from backend.services.track_recorder import TrackRecorder, get_track_recorder
from backend.services.traffic_rollup import compute_camera_rollup_1m, run_rollups_for_window
from backend.services.congestion_engine import CongestionEngine
from backend.services.traffic_baseline import TrafficBaselineEngine, get_macro_network_summary


TARGET_PILOT_CAMERAS = ["CAM_11", "CAM_08", "CAM_01", "CAM_04", "CAM_05", "CAM_07", "CAM_09", "CAM_14"]
CLASS_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}


def find_clips_for_camera(camera_id: str) -> List[Path]:
    """Finds harvested MP4 clips for a given camera."""
    cam_dir = WORKSPACE / "data" / "clips" / camera_id
    if not cam_dir.exists():
        return []
    return sorted(list(cam_dir.glob("*.mp4")))


def parse_clip_clock_to_datetime(clip_name: str, base_date_str: str = "2026-06-14") -> datetime:
    """Parses CAM_11_0830.mp4 into datetime(2026, 6, 14, 8, 30)."""
    parts = clip_name.replace(".mp4", "").split("_")
    clock_part = parts[-1] if len(parts) >= 2 else "1200"
    try:
        hour = int(clock_part[:2])
        minute = int(clock_part[2:4]) if len(clock_part) >= 4 else 0
    except ValueError:
        hour, minute = 12, 0
    base_date = datetime.strptime(base_date_str, "%Y-%m-%d")
    return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)


def run_cctv_macro_staging():
    print("=" * 80)
    print("  SENTINEL GUJARAT — LIVE CCTV MACRO TRAFFIC STAGING PIPELINE")
    print("=" * 80)

    init_db()
    db = SessionLocal()
    recorder = get_track_recorder()
    recorder.refresh_calibrations(db=db)

    # 1. Check calibration availability
    calibrated_cams = list(recorder._calib_cache.keys())
    print(f"\n[Phase 1] Calibrated Cameras Loaded in Memory: {len(calibrated_cams)}", flush=True)
    for cam_id in calibrated_cams:
        c = recorder._calib_cache[cam_id]
        gate = c.get("quality_gate", "good").upper()
        err = c.get("reprojection_error_m", 0.0) or 0.0
        lat = c.get("gps_anchor_lat", 0.0) or 0.0
        lon = c.get("gps_anchor_lon", 0.0) or 0.0
        print(f"  * {cam_id}: Gate={gate} | RMS=±{err:.3f}m | GPS=({lat:.4f}, {lon:.4f})", flush=True)

    # 2. Select YOLO model
    model_path = WORKSPACE / "models_gujarat_yolov8s.pt"
    if not model_path.exists():
        model_path = WORKSPACE / "yolov8s.pt"
    if not model_path.exists():
        model_path = WORKSPACE / "yolov8n.pt"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Phase 2] Initializing Neural Detector: {model_path.name} on {device.upper()}")
    model = YOLO(str(model_path))

    # 3. Process video clips for pilot cameras
    total_tracks_persisted = 0
    camera_track_counts = defaultdict(int)
    all_extracted_tracks: List[VehicleTrack] = []
    min_timestamp: Optional[datetime] = None
    max_timestamp: Optional[datetime] = None

    print(f"\n[Phase 3] Ingesting Live CCTV Video Feeds...")
    for cam_id in TARGET_PILOT_CAMERAS:
        if cam_id not in recorder._calib_cache:
            continue
        calib = recorder._calib_cache[cam_id]
        clips = find_clips_for_camera(cam_id)
        if not clips:
            print(f"  [!] No clips found for {cam_id} in data/clips/{cam_id}")
            continue

        print(f"\n--- Processing {cam_id} ({len(clips)} clips available) ---")
        for clip_path in clips[:3]:  # Process up to 3 diverse time slots per camera (e.g. 01:00 off-peak, 07:30 peak, 08:30 rush)
            clip_start_dt = parse_clip_clock_to_datetime(clip_path.name)
            if min_timestamp is None or clip_start_dt < min_timestamp:
                min_timestamp = clip_start_dt

            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            sample_stride = max(1, int(round(fps / 5.0)))  # Sample at ~5 FPS for track smoothness

            # Tracklet state: track_id -> [(px, py), ...], [ts, ...], class_id, confs
            active_tracks: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
                "positions": [],
                "timestamps": [],
                "classes": [],
                "confs": [],
                "last_frame": 0,
            })
            completed_tracks: List[Dict[str, Any]] = []

            frame_idx = 0
            max_process_frames = min(total_frames, 300) # Process ~60 seconds of video per clip

            while frame_idx < max_process_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_stride == 0:
                    current_rel_sec = frame_idx / fps
                    current_abs_dt = clip_start_dt + timedelta(seconds=current_rel_sec)
                    if max_timestamp is None or current_abs_dt > max_timestamp:
                        max_timestamp = current_abs_dt

                    # Run YOLO tracking with BoT-SORT
                    results = model.track(
                        frame,
                        persist=True,
                        verbose=False,
                        conf=0.25,
                        classes=[2, 3, 5, 7], # Car, motorcycle, bus, truck
                        device=device,
                    )[0]

                    current_tracked_ids = set()
                    if results.boxes is not None and results.boxes.id is not None:
                        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                        track_ids = results.boxes.id.int().cpu().numpy()
                        cls_ids = results.boxes.cls.int().cpu().numpy()
                        confs = results.boxes.conf.cpu().numpy()

                        for box, tid, cid, conf in zip(boxes_xyxy, track_ids, cls_ids, confs):
                            current_tracked_ids.add(tid)
                            # Base centroid (middle bottom of bounding box)
                            bx = float((box[0] + box[2]) / 2.0)
                            by = float(box[3])
                            active_tracks[tid]["positions"].append((bx, by))
                            active_tracks[tid]["timestamps"].append(current_abs_dt.timestamp())
                            active_tracks[tid]["classes"].append(CLASS_MAP.get(cid, "car"))
                            active_tracks[tid]["confs"].append(float(conf))
                            active_tracks[tid]["last_frame"] = frame_idx

                    # Check for tracks that disappeared > 15 frames ago
                    for tid, data in list(active_tracks.items()):
                        if tid not in current_tracked_ids and (frame_idx - data["last_frame"]) > 15:
                            if len(data["positions"]) >= 5:
                                completed_tracks.append(data)
                            del active_tracks[tid]

                frame_idx += 1

            cap.release()

            # Flush remaining active tracks
            for tid, data in active_tracks.items():
                if len(data["positions"]) >= 5:
                    completed_tracks.append(data)

            # Process completed tracks through Theil-Sen speed estimator and recorder
            clip_persisted = 0
            for tid_idx, tr in enumerate(completed_tracks):
                most_common_class = max(set(tr["classes"]), key=tr["classes"].count)
                avg_conf = float(np.mean(tr["confs"]))

                vt = recorder.process_completed_track(
                    camera_id=cam_id,
                    track_id=tid_idx + 1,
                    vehicle_class=most_common_class,
                    pixel_positions=tr["positions"],
                    timestamps=tr["timestamps"],
                    detector_conf=avg_conf,
                    db=db,
                )
                recorder.save_track(vt, db=db)
                all_extracted_tracks.append(vt)
                clip_persisted += 1
                total_tracks_persisted += 1
                camera_track_counts[cam_id] += 1

            print(f"  ✓ {clip_path.name:<20} | Time: {clip_start_dt.strftime('%H:%M')} | Extracted Tracks: {clip_persisted}")

    print(f"\n[Phase 3 Complete] Total Vehicle Tracks Persisted: {total_tracks_persisted}")
    for cam_id, cnt in camera_track_counts.items():
        print(f"  * {cam_id}: {cnt} vehicle tracks")

    # 4. Run 1-Minute Aggregated Rollups
    print(f"\n[Phase 4] Computing 1-Minute Traffic Rollups across observed time window...")
    if min_timestamp and max_timestamp:
        # Buffer window to minute boundaries
        w_start = min_timestamp.replace(second=0, microsecond=0)
        w_end = (max_timestamp + timedelta(minutes=2)).replace(second=0, microsecond=0)
        rollup_count = run_rollups_for_window(start_time=w_start, end_time=w_end, db=db)
        print(f"  ✓ Generated and merged {rollup_count} 1-minute camera rollups into camera_metrics_1m")
    else:
        print("  [!] No valid time window recorded.")

    # 5. Bootstrap Free-Flow Speeds (v_free) & Congestion Index (CI)
    print(f"\n[Phase 5] Bootstrapping Free-Flow Speeds (v_free) & Congestion Index...")
    cong_engine = CongestionEngine(min_samples_threshold=10) # Staging threshold
    for cam_id in calibrated_cams:
        v_free = cong_engine.compute_free_flow_speed(camera_id=cam_id, db=db)
        status = f"{v_free:.1f} km/h (85th percentile off-peak)" if v_free is not None else "Awaiting more off-peak samples"
        print(f"  * {cam_id}: v_free = {status}")

    updated_ci = cong_engine.update_rollups_congestion_index(db=db)
    print(f"  ✓ Updated Congestion Index on {updated_ci} 1-minute metric records")

    # 6. Seasonal Baseline Matrix & Sustained Slowdown Detection
    print(f"\n[Phase 6] Building Time-of-Day x Day-of-Week Baseline Matrix...")
    base_engine = TrafficBaselineEngine()
    matrix_cells = base_engine.build_baseline_matrix(db=db, force_refresh=True)
    print(f"  ✓ Baseline matrix built with {matrix_cells} historical seasonal cells")

    anomalies = base_engine.detect_anomalies(lookback_minutes=1440, db=db)
    print(f"  ✓ Anomaly Scan Completed: {len(anomalies)} sustained baseline slowdowns detected")
    for a in anomalies:
        print(f"    - [{a.severity}] {a.camera_name} ({a.camera_id}): Speed {a.current_speed_kmh} vs Base {a.baseline_speed_kmh} km/h ({a.deviation_pct:+.1f}%)")

    # 7. Junagadh Corridor Transit Evaluation (CAM_11 -> CAM_08)
    print(f"\n[Phase 7] Evaluating Junagadh Urban Corridor (CAM_11 -> CAM_08, 2.36 km)...")
    c11_tracks = db.query(VehicleTrack).filter(VehicleTrack.camera_id == "CAM_11", VehicleTrack.speed_kmh.isnot(None)).all()
    c08_tracks = db.query(VehicleTrack).filter(VehicleTrack.camera_id == "CAM_08", VehicleTrack.speed_kmh.isnot(None)).all()

    c11_speeds = [t.speed_kmh for t in c11_tracks if t.speed_kmh is not None]
    c08_speeds = [t.speed_kmh for t in c08_tracks if t.speed_kmh is not None]

    med_11 = float(np.median(c11_speeds)) if c11_speeds else 0.0
    med_08 = float(np.median(c08_speeds)) if c08_speeds else 0.0

    print(f"  * CAM_11 Entry Speed (Median): {med_11:.1f} km/h (Samples: {len(c11_speeds)})")
    print(f"  * CAM_08 Exit Speed  (Median): {med_08:.1f} km/h (Samples: {len(c08_speeds)})")
    if med_11 > 0 and med_08 > 0:
        avg_corridor_speed = (med_11 + med_08) / 2.0
        expected_transit_mins = (2.36 / avg_corridor_speed) * 60.0
        print(f"  * Expected 2.36 km Corridor Transit Time: {expected_transit_mins:.2f} minutes (Lower Bound: >= {avg_corridor_speed:.1f} km/h)")

    # 8. Fetch Final Network Summary (API Contract Verification)
    print(f"\n[Phase 8] Verifying Real-Time Macro API Summary...")
    summary = get_macro_network_summary(db=db)
    print(f"  * Total Configured Cameras: {summary['summary']['total_cameras']}")
    print(f"  * Calibrated Cameras:       {summary['summary']['calibrated_cameras']}")
    print(f"  * Reporting Cameras:        {summary['summary']['reporting_cameras']}")
    print(f"  * Network Average Speed:    {summary['summary']['network_avg_speed_kmh']} km/h")
    print(f"  * Network Average CI:       {summary['summary']['network_avg_ci']}")
    print(f"  * Active Baseline Anomalies: {summary['summary']['active_anomalies_count']}")
    print(f"  * Active Corridors:         {summary['summary']['active_corridors_count']}")

    # Save artifact report
    out_dir = WORKSPACE / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "macro_traffic_staging_report.json"
    with open(report_file, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_tracks": total_tracks_persisted,
            "camera_tracks": dict(camera_track_counts),
            "rollups_generated": rollup_count if 'rollup_count' in locals() else 0,
            "baseline_matrix_cells": matrix_cells,
            "anomalies_detected": len(anomalies),
            "api_summary": summary["summary"],
        }, f, indent=2)

    db.close()
    print("\n" + "=" * 80)
    print(f"  LIVE STAGING PIPELINE COMPLETED SUCCESSFULLY! Report saved to {report_file.name}")
    print("=" * 80)


if __name__ == "__main__":
    run_cctv_macro_staging()
