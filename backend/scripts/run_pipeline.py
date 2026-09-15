"""Run the Sarvanetra pipeline as its own process.

This starts THE pipeline — all nine stages, one pass per frame: ingestion,
vehicle detection, tracking, ANPR, identity fusion, trajectory and speed,
traffic analytics, alerts, persistence. ANPR is stage 4 and runs inline; it is
not a second pipeline and nothing starts it separately. See
docs/UNIFIED_ARCHITECTURE.md for the stage-to-line map.


Ingestion and serving are different workloads and belong apart. The pipeline
decodes and infers over every camera continuously; the API answers short
requests and must stay responsive while it does. Sharing a process makes the
second impossible — measured on this machine, with the pipeline running inside
the API, /docs and /analytics/calibration/cameras both timed out after 30 s
while the process held 6.7 GB and 1,067 s of CPU. An operator could not open
the calibration page at all.

Both processes talk to the same database, so the API sees everything this one
writes. Run them side by side:

    python -m backend.scripts.run_pipeline          # this process
    uvicorn backend.main:app --port 8000            # the API

Ctrl-C stops it cleanly, draining queued vault crops before exit.

Environment:
    SENTINEL_READER_FPS   frames sampled per camera per second (default 10;
                          measured ceiling here is ~38 frames/s aggregate,
                          so 3-camera workers with 10 fps each have headroom)

    SENTINEL_ANPR_CAMERAS comma-separated list of cameras that get ANPR.
                          This worker intersects that list with its own
                          camera set, so ANPR can never accidentally run on
                          a camera this worker does not own.

    SENTINEL_PUBLISH_ASYNC unconditionally set to 1 by fleet_supervisor, but
                           respected here if run standalone.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

# SENTINEL_PUBLISH_ASYNC must be set BEFORE live_24x7_pipeline is imported,
# because the module reads it at load time to decide whether to use the
# publisher thread.  The fleet supervisor always sets it; when run standalone
# default to async-on so the standalone single-camera demo also benefits.
if not os.environ.get("SENTINEL_PUBLISH_ASYNC"):
    os.environ["SENTINEL_PUBLISH_ASYNC"] = "1"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── ANPR camera enforcement ───────────────────────────────────────────────────
# The supervisor sets SENTINEL_ANPR_CAMERAS to the cameras that get OCR
# treatment. If this worker owns none of them, disable ANPR entirely rather
# than letting it fall back to "all cameras in this process" — which would
# waste 31 ms/frame (measured) on cameras whose plate characters are < 10 px.
def _enforce_anpr_intersection() -> None:
    anpr_raw = os.environ.get("SENTINEL_ANPR_CAMERAS", "").strip()
    cams_raw = os.environ.get("SENTINEL_PIPELINE_CAMERAS", "").strip()
    if not anpr_raw or anpr_raw == "__none__":
        os.environ["SENTINEL_ANPR_CAMERAS"] = "__none__"
        return
    if not cams_raw:
        return  # single camera mode — let the pipeline decide

    anpr_set = {c.strip().upper() for c in anpr_raw.split(",") if c.strip() and c.strip() != "__none__"}
    cam_set  = {c.strip().upper() for c in cams_raw.split(",") if c.strip()}
    if "ALL" in anpr_set:
        effective = cam_set
    else:
        effective = anpr_set & cam_set

    if not effective:
        os.environ["SENTINEL_ANPR_CAMERAS"] = "__none__"
        return

    os.environ["SENTINEL_ANPR_CAMERAS"] = ",".join(sorted(effective))

    # Warn if ANPR was requested for a camera that cannot resolve plates.
    _check_plate_capability(effective)


def _check_plate_capability(anpr_cameras: set) -> None:
    """Log a warning for each ANPR-enabled camera whose plates are too small."""
    cap_path = ROOT / "backend" / "calibration_data" / "plate_capability.json"
    if not cap_path.exists():
        return
    try:
        cap = json.loads(cap_path.read_text(encoding="utf-8"))
        floor = cap.get("floor_px", 50)
        cams = cap.get("cameras", {})
        for cam_id in sorted(anpr_cameras):
            info = cams.get(cam_id)
            if info is None:
                continue
            max_px = info.get("max_plate_px", 0)
            if max_px < floor:
                logging.getLogger("run_pipeline").warning(
                    "ANPR requested for %s but measured max plate size is %d px "
                    "(floor %d px) — OCR will return no reads and costs %.0f ms/frame. "
                    "Reason: %s",
                    cam_id, max_px, floor, 31, info.get("reason", "unknown"))
    except Exception:  # noqa: BLE001
        pass


_enforce_anpr_intersection()
# ─────────────────────────────────────────────────────────────────────────────

from backend.services.live_24x7_pipeline import (                # noqa: E402
    READER_FPS, YOLO_IMGSZ, get_live_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
)
log = logging.getLogger("run_pipeline")


def main() -> int:
    frac_str = os.environ.get("SENTINEL_CUDA_MEM_FRACTION", "").strip()
    if frac_str:
        try:
            frac = float(frac_str)
            if 0.05 <= frac <= 0.95:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(frac)
                    log.info("Set per-process CUDA memory fraction to %.2f", frac)
        except Exception as _e:
            pass

    pipeline = get_live_pipeline()
    pipeline.start()

    n = len(pipeline._reader_threads)
    if not n:
        log.error("No cameras started. Is data/clips populated?")
        pipeline.stop()
        return 1

    anpr_effective = os.environ.get("SENTINEL_ANPR_CAMERAS", "")
    log.info(
        "Ingesting %d cameras on %s, yolov8s@%d, %d fps per camera, ANPR: [%s]",
        n, pipeline.device, YOLO_IMGSZ, READER_FPS,
        anpr_effective if anpr_effective != "__none__" else "disabled",
    )

    stopping = False

    def handle_stop(signum, _frame):
        nonlocal stopping
        if stopping:
            log.warning("Second signal — exiting immediately.")
            sys.exit(1)
        stopping = True
        log.info("Signal %s received; draining and stopping.", signum)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    # Report the counters that say whether the fleet is keeping up, rather
    # than only that the process is alive. A pipeline that keeps up by
    # dropping frames looks identical to one that genuinely does, unless the
    # drops are counted and printed.
    last_report = 0.0

    # One heartbeat file PER WORKER, not one for the whole machine.
    #
    # The fleet runs as several of these processes (see fleet_supervisor), and
    # they all wrote output/pipeline_heartbeat.json — each overwriting the
    # last, so "is the pipeline running" answered for whichever process
    # happened to write most recently and no per-camera detail survived at
    # all. The legacy file is still written for older readers of it.
    worker_id = os.environ.get("SENTINEL_WORKER_ID", "").strip() or f"w{os.getpid()}"
    fleet_dir = ROOT / "output" / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    hb_path = fleet_dir / f"worker_{worker_id}.json"
    legacy_hb = ROOT / "output" / "pipeline_heartbeat.json"
    prev_frames: dict = {}
    prev_processed: int = 0
    prev_hb = time.time()

    def write_heartbeat(running: bool = True) -> None:
        nonlocal prev_hb, prev_processed
        now = time.time()
        dt = max(1e-6, now - prev_hb)
        cams = []
        for r in pipeline._reader_threads:
            s = r.stats()
            read = int(s.get("frames_queued") or 0)
            # Instantaneous fps over the last heartbeat interval, not since
            # start. A camera that stalled an hour ago still has a healthy
            # cumulative average — this is meant to surface stalls.
            s["fps_now"] = round(max(0, read - prev_frames.get(r.cam_id, read)) / dt, 2)
            prev_frames[r.cam_id] = read
            # Report whether the camera is reading live or recorded clips.
            s["stream_state"] = (
                "LIVE" if not s.get("in_failover") and s.get("source_type") in ("rtsp", "http", "onvif")
                else "RECORDED" if s.get("source_type") in ("clips", "file")
                else "OFFLINE" if not s.get("is_connected")
                else "UNKNOWN"
            )
            cams.append(s)
        prev_hb = now
        h = pipeline.health

        # Instantaneous processed-frame rate over the last interval.
        cur_processed = int(h.get("total_frames_processed", 0))
        frames_processed_delta = max(0, cur_processed - prev_processed)
        fps_processed_now = round(frames_processed_delta / dt, 2)
        prev_processed = cur_processed

        gpu_mb = None
        gpu_util = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mb = round(torch.cuda.memory_reserved() / 2 ** 20)
        except Exception:
            pass
        try:
            import subprocess as sp
            result = sp.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                gpu_util = int(result.stdout.strip().split("\n")[0])
        except Exception:
            pass

        anpr_att = getattr(pipeline, "_anpr_attempts", 0)
        anpr_succ = getattr(pipeline, "_anpr_successes", 0)
        anpr_drop = getattr(pipeline, "_anpr_dropped", 0)
        anpr_ms = getattr(pipeline, "_anpr_total_ms", 0.0)
        anpr_async = getattr(pipeline, "_anpr_async", False)
        with getattr(pipeline, "_track_plates_lock", threading.Lock()):
            track_plates_copy = dict(getattr(pipeline, "_track_plates", {}))

        payload = {
            "worker_id": worker_id,
            "pid": os.getpid(),
            "running": running,
            "timestamp": now,
            "cameras": len(cams),
            "camera_stats": cams,
            "gpu_mb_held": gpu_mb,
            "gpu_util_pct": gpu_util,
            "fps_processed_now": fps_processed_now,
            "health": {k: v for k, v in h.items() if k != "per_camera_lag"},
            "anpr_cameras": [c for c in os.environ.get("SENTINEL_ANPR_CAMERAS", "").split(",")
                             if c and c != "__none__"],
            "anpr_attempts": anpr_att,
            "anpr_successes": anpr_succ,
            "anpr_dropped": anpr_drop,
            "anpr_avg_ms": round(anpr_ms / max(1, anpr_att), 1),
            "anpr_async": anpr_async,
            "confirmed_plates": track_plates_copy,
        }
        try:
            # Atomic write: temp file on same filesystem → os.replace()
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=fleet_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, hb_path)
            legacy_hb.write_text(json.dumps({
                "running": running, "pid": os.getpid(),
                "timestamp": now, "cameras": len(cams),
            }), encoding="utf-8")
        except Exception:
            pass

    try:
        while not stopping:
            time.sleep(1.0)
            write_heartbeat(True)

            if time.time() - last_report >= 60.0:
                last_report = time.time()
                h = pipeline.health
                # GPU memory: what is in use, the peak ever needed, and what
                # the allocator is holding. Two pipelines on one 12 GB card
                # froze each other once "held" filled it, so it is reported.
                gpu = ""
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu = (f" | GPU mem {torch.cuda.memory_allocated() / 2**20:.0f} MB in use,"
                               f" peak {torch.cuda.max_memory_allocated() / 2**20:.0f},"
                               f" held {torch.cuda.memory_reserved() / 2**20:.0f}")
                except Exception:
                    pass
                # Where each processed frame's time goes, so a backlog can be
                # traced to the stage causing it rather than guessed at.
                ms = h.get("ms_per_frame") or {}
                if ms:
                    gpu += " | ms/frame " + ", ".join(
                        f"{k} {v:.0f}" for k, v in sorted(ms.items(), key=lambda kv: -kv[1])
                        if v >= 1)
                log.info(
                    "%d/%d cameras at real time | rtf mean %s | "
                    "%d frames processed, %d dropped (%s%%) | "
                    "%d tracks, %d vault crops%s",
                    h.get("cameras_keeping_up", 0), n,
                    h.get("realtime_factor_mean"),
                    h.get("total_frames_processed", 0),
                    h.get("frames_dropped_total", 0),
                    h.get("frames_dropped_pct", 0),
                    h.get("total_tracks_persisted", 0),
                    h.get("total_vault_harvested", 0),
                    gpu,
                )
    finally:
        write_heartbeat(False)
        try:
            legacy_hb.write_text(json.dumps({"running": False, "timestamp": time.time()}), encoding="utf-8")
        except Exception:
            pass
        pipeline.stop()
        h = pipeline.health
        log.info("Stopped. %d frames processed, %d tracks, %d vault crops.",
                 h.get("total_frames_processed", 0),
                 h.get("total_tracks_persisted", 0),
                 h.get("total_vault_harvested", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
