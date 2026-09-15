"""Run the whole camera fleet: split the registry across worker processes.

WHY WORKERS AND NOT ONE PROCESS — measured 2026-09-12 on this machine
  All 34 registered cameras in a single pipeline process: every reader
  connected and 33 published frames, but the aggregate was 11.1 fps with 10-12%
  of frames dropped, and the GPU sat at 17% with 3.1 GB held. The time went
  track 45 ms, ANPR 31 ms, detect 12 ms, embed 6 ms per frame — i.e. the
  ceiling was one Python thread doing tracking and plate work, not the card.
  One process tops out near 10.6 frames/s however fast the GPU is.

  So the fleet runs as several pipeline processes, each owning a subset of the
  cameras. Four workers give four inference threads on a 32-core machine and
  roughly 1.5-2 GB of GPU each, which fits the 12 GB card with room to spare.

WHAT THIS PROCESS DOES
  * kills any orphaned workers from a previous run (closes the "3342 fps" bug)
  * reads the camera registry from the database (nothing is hardcoded here)
  * splits it across SENTINEL_FLEET_WORKERS processes, keeping the assignment
    stable so a restart puts the same camera back on the same worker
  * gives the focus cameras their own worker, with ANPR and a higher frame
    rate, since those are the ones a viewer is watching in detail
  * watches each worker's heartbeat and restarts one that dies or goes quiet,
    so a camera or a crash takes down its worker's share and nothing else
  * writes output/fleet/supervisor.json atomically for the API to serve

  Stop with Ctrl-C; every worker is terminated before this exits.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FLEET_DIR = ROOT / "output" / "fleet"
STATE_PATH = FLEET_DIR / "supervisor.json"
# Grace period: CUDA model loads across multiple workers can take up to
# 120 s on this machine (measured).
HEARTBEAT_STALE_S = 90.0
RESTART_BACKOFF_S = 15.0

# ANPR enabled across fleet; bounding box pixel thresholds gate compute automatically
DEFAULT_ANPR = os.environ.get("SENTINEL_ANPR_CAMERAS", "ALL")
DEFAULT_FOCUS = "CAM_09"


def _env_list(name: str, default: str = "") -> List[str]:
    raw = os.environ.get(name, default) or ""
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _kill_orphans() -> None:
    """Terminate any pipeline workers left over from a previous supervisor run.

    On Windows, killing the supervisor does NOT kill its children. They
    continue writing heartbeat files with steadily growing counters — which is
    how Fleet Operations ends up reporting "3342 fps" the moment a second run
    starts. Clearing them here, before launching the new generation, keeps the
    supervisor.json a faithful picture of the current run.
    """
    try:
        from backend.scripts.kill_workers import kill_workers
        n = kill_workers(dry=False, timeout=12.0)
        if n:
            print(f"Cleared {n} orphaned worker(s) from previous run.", flush=True)
            time.sleep(2)  # let OS release GPU memory before new workers load models
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: orphan kill failed ({exc}); proceeding anyway.", flush=True)


def discover_cameras() -> List[str]:
    """Return the cameras this supervisor run should process.

    If SENTINEL_ONLY_CAMERAS is set (e.g. to the top-10 list), only those
    cameras are returned — even if the database has 30+.  This is the
    primary knob for the top-10 fleet: set the env var, get 10 cameras.

    If SENTINEL_ONLY_CAMERAS is empty or unset, all non-deleted, non-test
    cameras in the registry are returned (legacy behaviour).
    """
    from backend.db.session import SessionLocal
    from backend.db.models import Camera

    # ── Top-10 (or any explicit subset) filter ───────────────────────────
    only_raw = os.environ.get("SENTINEL_ONLY_CAMERAS", "").strip()
    only_set: set = set()
    if only_raw:
        only_set = {c.strip().upper() for c in only_raw.split(",") if c.strip()}
        print(
            f"[supervisor] SENTINEL_ONLY_CAMERAS set — restricting to "
            f"{len(only_set)} camera(s): {', '.join(sorted(only_set))}",
            flush=True,
        )

    include_test = os.environ.get("SENTINEL_FLEET_INCLUDE_TEST", "0") == "1"
    db = SessionLocal()
    try:
        rows = db.query(Camera).filter(Camera.is_deleted == False).all()  # noqa: E712
        ids = []
        for c in rows:
            cam_id = c.camera_id or str(c.id)
            # Apply explicit camera filter first (top-10 mode)
            if only_set and cam_id.upper() not in only_set:
                continue
            if not include_test:
                try:
                    from backend.services.fleet_census import is_test_camera
                    if is_test_camera(c.id, c.name):
                        continue
                except Exception:
                    pass
            ids.append(cam_id)
    finally:
        db.close()

    result = sorted(set(ids))
    if only_set:
        # Warn about any requested cameras not found in the DB
        missing = only_set - {c.upper() for c in result}
        if missing:
            print(
                f"[supervisor] WARNING: {len(missing)} requested camera(s) not "
                f"found in DB: {', '.join(sorted(missing))}. They will be skipped.",
                flush=True,
            )
    return result


def plan_workers(cameras: List[str]) -> List[Dict]:
    """Assign cameras to workers.

    When SENTINEL_EQUAL_PRIORITY=1 (top-10 mode), all cameras are split
    equally across SENTINEL_FLEET_WORKERS workers with no focus asymmetry.
    Every camera gets the same fps, ANPR configuration, and GPU budget.

    Default behaviour (SENTINEL_EQUAL_PRIORITY=0): focus cameras get their
    own dedicated worker with higher fps and guaranteed ANPR.
    """
    equal_priority = os.environ.get("SENTINEL_EQUAL_PRIORITY", "0").strip() == "1"
    n_workers = max(1, int(os.environ.get("SENTINEL_FLEET_WORKERS", "8")))
    anpr_list = _env_list("SENTINEL_ANPR_CAMERAS", DEFAULT_ANPR)
    all_anpr = ("ALL" in anpr_list) or (not anpr_list)
    fleet_fps = os.environ.get("SENTINEL_FLEET_READER_FPS", "10")

    plan: List[Dict] = []

    if equal_priority:
        # Top-10 mode: round-robin all cameras equally, no focus worker.
        # Keeps assignment stable across restarts (same camera → same worker).
        groups: List[List[str]] = [[] for _ in range(n_workers)]
        for i, cam in enumerate(cameras):
            groups[i % n_workers].append(cam)
        for i, g in enumerate(groups):
            if not g:
                continue
            anpr_g = g if all_anpr else [c for c in anpr_list if c in g]
            plan.append({
                "worker_id": f"w{i + 1}",
                "cameras": g,
                "reader_fps": fleet_fps,
                "infer_every": "1",
                "infer_batch": "8",
                "anpr": ",".join(anpr_g),
            })
        print(
            f"[supervisor] Equal-priority mode: {n_workers} workers × "
            f"{len(cameras) // max(1, n_workers)} cameras each @ {fleet_fps}fps",
            flush=True,
        )
    else:
        # Legacy mode: focus cameras get a dedicated worker.
        focus = [c for c in _env_list("SENTINEL_FOCUS_CAMERAS", DEFAULT_FOCUS) if c in cameras]
        rest = [c for c in cameras if c not in focus]
        if focus:
            anpr_focus = focus if all_anpr else [c for c in anpr_list if c in focus]
            plan.append({
                "worker_id": "focus",
                "cameras": focus,
                "reader_fps": os.environ.get("SENTINEL_FOCUS_READER_FPS", "10"),
                "infer_every": "1",
                "infer_batch": "8",
                "anpr": ",".join(anpr_focus),
            })
        groups2: List[List[str]] = [[] for _ in range(n_workers)]
        for i, cam in enumerate(rest):
            groups2[i % n_workers].append(cam)
        for i, g in enumerate(groups2):
            if not g:
                continue
            anpr_g = g if all_anpr else [c for c in anpr_list if c in g]
            plan.append({
                "worker_id": f"w{i + 1}",
                "cameras": g,
                "reader_fps": fleet_fps,
                "infer_every": "1",
                "infer_batch": "8",
                "anpr": ",".join(anpr_g),
            })

    frac = os.environ.get("SENTINEL_WORKER_GPU_FRACTION", "").strip()
    force_rollups = os.environ.get("SENTINEL_ROLLUPS")
    for i, spec in enumerate(plan):
        spec["gpu_fraction"] = frac
        if force_rollups is not None:
            spec["rollups"] = force_rollups
        else:
            spec["rollups"] = "1" if i == 0 else "0"
    return plan


def worker_env(spec: Dict) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SENTINEL_")}
    cams = ",".join(spec["cameras"])
    anpr = spec.get("anpr") or "__none__"
    env.update({
        "SENTINEL_WORKER_ID": spec["worker_id"],
        "SENTINEL_PIPELINE_CAMERAS": cams,
        "SENTINEL_ONLY_CAMERAS": cams,
        "SENTINEL_ANPR_CAMERAS": anpr,
        "SENTINEL_READER_FPS": str(spec["reader_fps"]),
        "SENTINEL_INFER_EVERY": str(spec["infer_every"]),
        "SENTINEL_INFER_BATCH": str(spec["infer_batch"]),
        "SENTINEL_STRICT_LIVE": os.environ.get("SENTINEL_STRICT_LIVE", "0"),
        # Default 0: prefer live RTSP streams; fall back to clips only on disconnect.
        # Previously defaulted to 1 (always use clips) which blocked RTSP from ever connecting.
        "SENTINEL_FORCE_CLIPS": os.environ.get("SENTINEL_FORCE_CLIPS", "0"),
        "SENTINEL_GLOBALID_CAMERAS": os.environ.get("SENTINEL_GLOBALID_CAMERAS", "ALL"),
        "SENTINEL_SINGLE_FRAME_FLOOR_PX": os.environ.get("SENTINEL_SINGLE_FRAME_FLOOR_PX", "35"),
        "SENTINEL_FUSION_FLOOR_PX": os.environ.get("SENTINEL_FUSION_FLOOR_PX", "25"),
        "SENTINEL_YOLO_IMGSZ": os.environ.get("SENTINEL_YOLO_IMGSZ", "640"),
        "SENTINEL_CORP8_HEALTH": "0",
        "SENTINEL_CUDA_MEM_FRACTION": str(spec.get("gpu_fraction", "")),
        "SENTINEL_ROLLUPS": str(spec.get("rollups", "1")),
        "SENTINEL_PUBLISH_ASYNC": "1",
        "SENTINEL_PUBLISH_MAX_WIDTH": os.environ.get("SENTINEL_PUBLISH_MAX_WIDTH", "960"),
        "SENTINEL_PUBLISH_QUALITY": os.environ.get("SENTINEL_PUBLISH_QUALITY", "75"),
        # Async ANPR — critical for 10fps on 8 ANPR cameras simultaneously.
        # Must be forwarded explicitly because worker_env strips all SENTINEL_* vars.
        "SENTINEL_ANPR_ASYNC": os.environ.get("SENTINEL_ANPR_ASYNC", "0"),
    })
    # Knobs the supervisor does not decide but must not swallow. Every
    # SENTINEL_* name is stripped above so a worker cannot inherit a stale
    # single-camera setup; these are the ones a fleet run legitimately sets,
    # and leaving them out silently ignored them on the command line.
    for name in ("SENTINEL_EMBED_EVERY",      # appearance gate, the fleet's GPU budget
                 "SENTINEL_YOLO_WEIGHTS",     # detector choice
                 "SENTINEL_DETECT_INPUT_SIZE",
                 "SENTINEL_CLIP_MATCH",       # which recorded clips to play
                 "SENTINEL_PLATE_DET_CONF",   # plate detector confidence threshold
                 "SENTINEL_MIN_PLATE_ASPECT_RATIO", # minimum plate aspect ratio
                 "SENTINEL_SMALL_PLATE_DETECTOR",   # secondary detector for motorcycles/small plates
                 "SENTINEL_DOUBLE_LINE",            # two-line/stacked plate unstacking
                 "SENTINEL_HLS_SEEK_SEC",
                 "SENTINEL_PUBLISH_MIN_INTERVAL_S",
                 # Corp8 credentials — needed for HLS fallback and cameras.json API
                 "CORP8_EMAIL",
                 "CORP8_PASSWORD",
                 "CORP8_BASE",
                 "CORP8_SESSION_POOL",
                 "CORP8_LOGIN_INTERVAL",
                 # RTSP settings — required for live 30-camera RTSP mode
                 "CORP8_RTSP_HOST",           # default 103.250.160.189
                 "CORP8_RTSP_PORT",           # default 8554
                 # OpenCV FFMPEG RTSP transport — must be TCP for corp8 (cam21/25-30)
                 "OPENCV_FFMPEG_CAPTURE_OPTIONS"):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


def heartbeat_age(worker_id: str) -> float:
    path = FLEET_DIR / f"worker_{worker_id}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return time.time() - float(data.get("timestamp", 0))
    except Exception:
        return float("inf")


def _write_state_atomic(path: Path, data: dict) -> None:
    """Write JSON to path atomically so the API never reads a half-written file.

    A plain write_text() truncates and then fills, so a reader that opens
    during the fill sees an empty or truncated file. tempfile + os.replace()
    is atomic on both Windows (same volume) and POSIX.
    """
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        # Fall back to a plain write rather than leaving the state file absent.
        try:
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass


def main() -> int:
    FLEET_DIR.mkdir(parents=True, exist_ok=True)

    # Kill orphans BEFORE clearing heartbeats, so we don't delete a file that
    # is still being written by a live process.
    _kill_orphans()

    # Clear heartbeats from a previous run: a worker that no longer exists
    # would otherwise sit in the fleet view for ever as a stale row.
    for old in FLEET_DIR.glob("worker_*.json"):
        try:
            old.unlink()
        except OSError:
            pass

    cameras = discover_cameras()
    if not cameras:
        print("No cameras in the registry -- nothing to run.", flush=True)
        return 1
    plan = plan_workers(cameras)
    print(f"{len(cameras)} cameras across {len(plan)} workers:", flush=True)
    for spec in plan:
        print(f"   {spec['worker_id']:6s} {len(spec['cameras']):2d} cameras "
              f"@ {spec['reader_fps']} fps, ANPR on [{spec['anpr'] or 'none'}]", flush=True)

    procs: Dict[str, subprocess.Popen] = {}
    started_at: Dict[str, float] = {}
    restarts: Dict[str, int] = {c["worker_id"]: 0 for c in plan}
    stopping = False

    def launch(spec: Dict) -> None:
        wid = spec["worker_id"]
        log_path = ROOT / "output" / f"fleet_{wid}.log"
        log = open(log_path, "a", encoding="utf-8", errors="replace")
        procs[wid] = subprocess.Popen(
            [sys.executable, "-m", "backend.scripts.run_pipeline"],
            cwd=str(ROOT), env=worker_env(spec), stdout=log, stderr=subprocess.STDOUT)
        started_at[wid] = time.time()
        print(f"{time.strftime('%H:%M:%S')} started {wid} (pid {procs[wid].pid}, "
              f"{len(spec['cameras'])} cameras) -> {log_path.name}", flush=True)

    def handle_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    for spec in plan:
        launch(spec)
        time.sleep(3)   # stagger: model loads and portal logins do not pile up

    try:
        while not stopping:
            time.sleep(5)
            workers = []
            for spec in plan:
                wid = spec["worker_id"]
                proc = procs.get(wid)
                alive = proc is not None and proc.poll() is None
                age = heartbeat_age(wid)
                # HEARTBEAT_STALE_S is 90 s; add a 3-minute startup grace so a
                # worker loading CUDA models for the first time is not killed
                # before it emits its first heartbeat.
                grace = time.time() - started_at.get(wid, 0) < 180
                unhealthy = (not alive) or (age > HEARTBEAT_STALE_S and not grace)
                if unhealthy and not stopping:
                    reason = "process exited" if not alive else f"heartbeat {age:.0f}s old"
                    print(f"{time.strftime('%H:%M:%S')} {wid} unhealthy ({reason}) -- restarting",
                          flush=True)
                    if alive:
                        proc.terminate()
                        try:
                            proc.wait(timeout=15)
                        except Exception:
                            proc.kill()
                    restarts[wid] += 1
                    time.sleep(RESTART_BACKOFF_S)
                    launch(spec)
                    age = heartbeat_age(wid)
                workers.append({
                    "worker_id": wid,
                    "pid": procs[wid].pid if procs.get(wid) else None,
                    "cameras": spec["cameras"],
                    "camera_count": len(spec["cameras"]),
                    "reader_fps": spec["reader_fps"],
                    "anpr_cameras": [c for c in spec["anpr"].split(",") if c and c != "__none__"],
                    "alive": procs.get(wid) is not None and procs[wid].poll() is None,
                    "heartbeat_age_s": None if age == float("inf") else round(age, 1),
                    "restarts": restarts[wid],
                    "uptime_s": round(time.time() - started_at.get(wid, time.time()), 1),
                })
            _write_state_atomic(STATE_PATH, {
                "updated": time.time(),
                "supervisor_pid": os.getpid(),
                "cameras_total": len(cameras),
                "workers": workers,
            })
    finally:
        print(f"{time.strftime('%H:%M:%S')} stopping {len(procs)} workers", flush=True)
        for wid, proc in procs.items():
            if proc.poll() is None:
                proc.terminate()
        for wid, proc in procs.items():
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
        _write_state_atomic(STATE_PATH, {
            "updated": time.time(), "supervisor_pid": None,
            "cameras_total": len(cameras), "workers": [],
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
