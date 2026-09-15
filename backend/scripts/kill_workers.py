"""Kill all running Sarvanetra pipeline worker processes.

Called automatically by fleet_supervisor at startup to clear orphaned workers
from a previous run, and can be invoked manually.

Works on both Windows and Linux via psutil process enumeration so no shell
quoting issues with subprocess-based kills.

Usage:
    python -m backend.scripts.kill_workers          # kills and exits
    python -m backend.scripts.kill_workers --dry    # prints what would be killed
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("kill_workers")

# Patterns that identify a Sarvanetra worker process.  Any process whose
# command line contains one of these strings is a pipeline worker.
WORKER_PATTERNS = [
    "run_pipeline",
    "fleet_supervisor",
    "live_24x7_pipeline",
    "cam09_live_relay",
]


def _is_worker(proc) -> bool:
    """Return True if the process is a Sarvanetra pipeline worker."""
    try:
        cmdline = " ".join(proc.cmdline())
        own_pid = os.getpid()
        # Never kill ourselves or the supervisor's own process.
        if proc.pid == own_pid:
            return False
        return any(pat in cmdline for pat in WORKER_PATTERNS)
    except Exception:  # noqa: BLE001
        return False


def find_workers():
    """Return a list of (pid, cmdline_summary) for every running worker."""
    try:
        import psutil
    except ImportError:
        log.warning("psutil not installed — cannot enumerate workers. "
                    "Run: pip install psutil")
        return []

    found = []
    own_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if proc.info["pid"] == own_pid:
                continue
            name = (proc.info.get("name") or "").lower()
            if "python" not in name and "py" not in name:
                continue
            if _is_worker(proc):
                cmd = " ".join((proc.cmdline() or [])[:6])
                found.append((proc.pid, cmd))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        except Exception:
            pass
    return found


def kill_workers(dry: bool = False, timeout: float = 10.0) -> int:
    """Terminate all pipeline workers.

    Args:
        dry:     If True, print what would be killed but don't kill.
        timeout: Seconds to wait for graceful termination before SIGKILL.

    Returns:
        Number of workers killed (0 in dry-run mode).
    """
    try:
        import psutil
    except ImportError:
        log.warning("psutil not installed — skipping worker kill. "
                    "Run: pip install psutil")
        return 0

    workers = find_workers()
    if not workers:
        log.info("No running pipeline workers found.")
        return 0

    log.info("Found %d running worker(s):", len(workers))
    for pid, cmd in workers:
        log.info("  PID %d  %s", pid, cmd[:100])

    if dry:
        log.info("Dry-run: not killing.")
        return 0

    procs = []
    for pid, _cmd in workers:
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            procs.append(proc)
            log.info("SIGTERM → PID %d", pid)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not SIGTERM PID %d: %s", pid, exc)

    # Wait for graceful shutdown.
    _, alive = psutil.wait_procs(procs, timeout=timeout)

    # Force-kill anything still alive.
    for proc in alive:
        try:
            proc.kill()
            log.warning("SIGKILL → PID %d (did not terminate in %.0fs)",
                        proc.pid, timeout)
        except Exception:  # noqa: BLE001
            pass

    # Also clear heartbeat files from a previous run so Fleet Operations
    # doesn't show ghost workers.
    fleet_dir = ROOT / "output" / "fleet"
    if fleet_dir.exists():
        for hb in fleet_dir.glob("worker_*.json"):
            try:
                hb.unlink()
                log.info("Removed stale heartbeat: %s", hb.name)
            except OSError:
                pass

    log.info("Killed %d worker(s).", len(workers))
    return len(workers)


def main() -> int:
    ap = argparse.ArgumentParser(description="Kill Sarvanetra pipeline workers")
    ap.add_argument("--dry", action="store_true", help="Print but don't kill")
    ap.add_argument("--wait", type=float, default=10.0,
                    help="Seconds to wait before SIGKILL (default 10)")
    args = ap.parse_args()
    n = kill_workers(dry=args.dry, timeout=args.wait)
    return 0


if __name__ == "__main__":
    sys.exit(main())
