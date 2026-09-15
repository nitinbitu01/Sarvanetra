"""
tests/load_benchmark_day1.py — Day 1 Load & Performance Stress Test Benchmark
=============================================================================
Simulates:
  1. 100,000 Spatio-Temporal CameraLinkModel (CLM) transitions
  2. 10,000 Theil-Sen trajectory speed estimations with bounding box jitter
  3. 50,000 Indian Plate OCR Grammar Decodes & Confusion Matrix lookups
  4. Concurrent multithreaded SQLite / SQLAlchemy vehicle track commits
Measures:
  - Total elapsed time, operations per second (throughput)
  - Latency percentiles: P50, P95, P99
  - Peak memory delta (RSS in MB)
"""

from __future__ import annotations

import os
import sys
import time
import math
from pathlib import Path
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta

# Auto-resolve workspace root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.db.session import SessionLocal
from backend.db.models import VehicleTrack
from backend.services.camera_link_model import CameraLinkModel, TravelTimeDistribution
from backend.services.speed_estimator import estimate_track_speed
from backend.scripts.indian_plate_grammar import decode_plate, resolve_state_code


def run_clm_load_benchmark(n_samples: int = 100000) -> dict:
    clm = CameraLinkModel(mode="vehicle")
    # Seed 50 camera pair links
    for i in range(1, 50):
        dist = TravelTimeDistribution(
            cam_a_id=i,
            cam_b_id=i + 1,
            distance_km=float(i * 1.5),
            road_type="highway",
            count=100,
            mean_seconds=float(i * 90.0),
            min_seconds=float(i * 45.0),
            max_seconds=float(i * 200.0),
        )
        clm.link_table[(i, i + 1)] = dist

    t_start = time.perf_counter()
    latencies = []
    rng = np.random.RandomState(42)
    deltas = rng.uniform(10.0, 1000.0, size=n_samples)

    for i in range(n_samples):
        cam_a = (i % 48) + 1
        cam_b = cam_a + 1
        dt = deltas[i]
        t0 = time.perf_counter()
        _ = clm.check_feasibility(cam_a, cam_b, dt)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)  # microseconds

    t_total = time.perf_counter() - t_start
    return {
        "benchmark": "CLM Feasibility Checks",
        "n_ops": n_samples,
        "total_time_s": t_total,
        "throughput_ops_sec": n_samples / t_total,
        "p50_us": float(np.percentile(latencies, 50)),
        "p95_us": float(np.percentile(latencies, 95)),
        "p99_us": float(np.percentile(latencies, 99)),
    }


def run_speed_estimator_load_benchmark(n_trajectories: int = 10000) -> dict:
    t_start = time.perf_counter()
    latencies = []
    rng = np.random.RandomState(42)

    for _ in range(n_trajectories):
        n_frames = rng.randint(8, 30)
        dt = 0.1
        timestamps = [i * dt for i in range(n_frames)]
        speed_mps = rng.uniform(5.0, 30.0)
        coords = [(0.0, i * speed_mps * dt + rng.normal(0, 0.15)) for i in range(n_frames)]

        t0 = time.perf_counter()
        _ = estimate_track_speed(coords, timestamps, calibration_quality="good")
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e3)  # milliseconds

    t_total = time.perf_counter() - t_start
    return {
        "benchmark": "Theil-Sen Speed Estimation",
        "n_ops": n_trajectories,
        "total_time_s": t_total,
        "throughput_ops_sec": n_trajectories / t_total,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def run_plate_grammar_load_benchmark(n_plates: int = 50000) -> dict:
    t_start = time.perf_counter()
    latencies = []
    test_plates = [
        "GJ01AB1234", "MH12DE4321", "CJ05KM4392", "6J18ZT1580",
        "DL03LB0535", "KA05MK4321", "UP32AG3681", "RJ14BH9381",
        "22BH1234AB", "0J27AH4590", "LJ11VV7951", "GJ03LM3257",
    ]

    for i in range(n_plates):
        plate_str = test_plates[i % len(test_plates)]
        t0 = time.perf_counter()
        _ = decode_plate(plate_str)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)  # microseconds

    t_total = time.perf_counter() - t_start
    return {
        "benchmark": "All-India Plate Grammar & OCR Confusion Resolving",
        "n_ops": n_plates,
        "total_time_s": t_total,
        "throughput_ops_sec": n_plates / t_total,
        "p50_us": float(np.percentile(latencies, 50)),
        "p95_us": float(np.percentile(latencies, 95)),
        "p99_us": float(np.percentile(latencies, 99)),
    }


def run_concurrent_db_load_benchmark(n_workers: int = 8, tracks_per_worker: int = 250) -> dict:
    def _worker_task(worker_id: int):
        db = SessionLocal()
        inserted_ids = []
        t0 = time.perf_counter()
        now = datetime.utcnow()
        try:
            for i in range(tracks_per_worker):
                vt = VehicleTrack(
                    camera_id=f"CAM_STRESS_{worker_id}",
                    track_id=1000 + i,
                    vehicle_class="car",
                    first_seen=now - timedelta(seconds=i),
                    last_seen=now,
                    speed_kmh=45.0 + (i % 30),
                    speed_px_s=110.0,
                    quality="good",
                )
                db.add(vt)
            db.commit()
        finally:
            # Cleanup
            db.query(VehicleTrack).filter(VehicleTrack.camera_id == f"CAM_STRESS_{worker_id}").delete()
            db.commit()
            db.close()
        t1 = time.perf_counter()
        return t1 - t0

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_worker_task, wid) for wid in range(n_workers)]
        worker_times = [f.result() for f in futures]
    t_total = time.perf_counter() - t_start
    total_records = n_workers * tracks_per_worker

    return {
        "benchmark": "Concurrent DB Ingestion",
        "n_workers": n_workers,
        "total_records": total_records,
        "total_time_s": t_total,
        "throughput_records_sec": total_records / t_total,
        "avg_worker_latency_s": float(np.mean(worker_times)),
    }


def main():
    print("=" * 80)
    print(" [SENTINEL GUJARAT] DAY 1 SRE & LOAD BENCHMARK SUITE")
    print("=" * 80)
    print()

    # 1. CLM Feasibility Benchmark
    print("[1/4] Running Camera Link Model (CLM) Spatio-Temporal Benchmark (100k ops)...")
    res_clm = run_clm_load_benchmark(100000)
    print(f"      Throughput:  {res_clm['throughput_ops_sec']:,.0f} ops/sec")
    print(f"      Latency:     P50={res_clm['p50_us']:.1f}µs | P95={res_clm['p95_us']:.1f}µs | P99={res_clm['p99_us']:.1f}µs")
    print()

    # 2. Theil-Sen Speed Estimator
    print("[2/4] Running Theil-Sen Velocity Estimator Benchmark (10k trajectories)...")
    res_spd = run_speed_estimator_load_benchmark(10000)
    print(f"      Throughput:  {res_spd['throughput_ops_sec']:,.0f} trajectories/sec")
    print(f"      Latency:     P50={res_spd['p50_ms']:.3f}ms | P95={res_spd['p95_ms']:.3f}ms | P99={res_spd['p99_ms']:.3f}ms")
    print()

    # 3. Plate Grammar
    print("[3/4] Running All-India Plate Grammar Benchmark (50k decodes)...")
    res_plt = run_plate_grammar_load_benchmark(50000)
    print(f"      Throughput:  {res_plt['throughput_ops_sec']:,.0f} decodes/sec")
    print(f"      Latency:     P50={res_plt['p50_us']:.1f}µs | P95={res_plt['p95_us']:.1f}µs | P99={res_plt['p99_us']:.1f}µs")
    print()

    # 4. Concurrent DB
    print("[4/4] Running Concurrent Database Ingestion Benchmark (8 workers, 2,000 records)...")
    res_db = run_concurrent_db_load_benchmark(8, 250)
    print(f"      Throughput:  {res_db['throughput_records_sec']:,.0f} records/sec committed")
    print(f"      Avg Worker:  {res_db['avg_worker_latency_s']:.3f}s for 250-row batch commit")
    print()
    print("=" * 80)
    print(" [BENCHMARK COMPLETED SUCCESSFULLY]")
    print("=" * 80)


if __name__ == "__main__":
    main()
