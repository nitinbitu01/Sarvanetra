"""Launch the 30-camera Sarvanetra Fleet Supervisor.

Configures environment, clears old workers, and launches the supervisor
which manages 10 worker subprocesses (1 focus on CAM_09 + 9 fleet workers)
processing all 30 cameras simultaneously.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.kill_workers import kill_workers


def launch(mode: str = "clips", daemon: bool = False):
    print("=" * 60)
    print(f"  Starting Sarvanetra 30-Camera Fleet (Mode: {mode.upper()})")
    print("=" * 60)

    # 1. Kill any existing workers
    print("Terminating any existing workers...")
    kill_workers(dry=False, timeout=8.0)
    time.sleep(2)

    # 2. Environment Configuration
    env = dict(os.environ)
    if mode == "live":
        creds_file = ROOT / "config" / "corp8_credentials.json"
        if creds_file.exists():
            import json
            try:
                creds = json.loads(creds_file.read_text(encoding="utf-8"))
                env["CORP8_EMAIL"] = creds.get("email", "")
                env["CORP8_PASSWORD"] = creds.get("password", "")
                env["CORP8_BASE"] = creds.get("base", "https://cctv.corp8.cloud")
            except Exception:
                pass
        env["SENTINEL_FORCE_CLIPS"] = "0"
        env["SENTINEL_STRICT_LIVE"] = "0"
    else:
        env["SENTINEL_FORCE_CLIPS"] = "1"
        env["SENTINEL_STRICT_LIVE"] = "0"

    env["SENTINEL_FLEET_WORKERS"] = "9"
    env["SENTINEL_FLEET_READER_FPS"] = "10"
    env["SENTINEL_FOCUS_CAMERAS"] = "CAM_09"
    env["SENTINEL_ANPR_CAMERAS"] = "CAM_09"
    env["SENTINEL_YOLO_IMGSZ"] = "640"
    env["SENTINEL_PUBLISH_ASYNC"] = "1"
    env["SENTINEL_PUBLISH_MAX_WIDTH"] = "960"
    env["SENTINEL_PUBLISH_QUALITY"] = "75"
    env["SENTINEL_WORKER_GPU_FRACTION"] = "0.08"

    # 3. Launch supervisor
    log_dir = ROOT / "output"
    log_dir.mkdir(exist_ok=True)
    log_out = open(log_dir / f"fleet_{mode}.log", "w", encoding="utf-8")
    log_err = open(log_dir / f"fleet_{mode}.err", "w", encoding="utf-8")

    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)

    cmd = [sys.executable, "-m", "backend.scripts.fleet_supervisor"]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log_out,
        stderr=log_err,
        creationflags=flags,
    )

    print(f"\n[OK] Fleet Supervisor started successfully with PID {proc.pid}.")
    print(f"     Workers planned: 10 (1 focus for CAM_09 + 9 fleet workers)")
    print(f"     Log output: {log_dir / f'fleet_{mode}.log'}")
    print("\nModels are currently loading into GPU memory (~60-90s).")
    print("Monitor progress using: python -m backend.scripts.verify_30cam")
    print("Dashboard available at: http://localhost:3000 (SYSTEM -> Fleet Operations)")
    print("Proof API:              http://localhost:8000/api/v1/fleet/proof")
    print("=" * 60)
    return proc.pid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["clips", "live"], default="clips")
    args = parser.parse_args()
    launch(mode=args.mode)
