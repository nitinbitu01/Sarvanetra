#!/usr/bin/env python3
"""
scripts/verify_demo_ready.py — Cross-platform Sentinel IQ Demo Ready Verification.
Exit 0 = ready. Exit 1 = not ready.
"""
import glob
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout/stderr handles UTF-8 on Windows consoles
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ERRORS = 0


def check(desc: str, val: str, expected: str) -> None:
    global ERRORS
    if str(val) != str(expected):
        print(f"[FAIL] {desc}: got '{val}', expected '{expected}'")
        ERRORS += 1
    else:
        print(f"[OK] {desc}")


db_path = PROJECT_ROOT / "output" / "sentinel.db"
if not db_path.exists():
    db_path = PROJECT_ROOT / "sentinel.db"

if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) FROM officers WHERE status='BUSY'")
        busy = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        busy = 0
    check("Officers BUSY (should be 0)", busy, 0)

    try:
        cursor.execute("SELECT COUNT(*) FROM routed_alerts WHERE status='ROUTED'")
        routed = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        routed = 0
    check("Active routed alerts (should be 0)", routed, 0)

    try:
        cursor.execute("SELECT COUNT(*) FROM alerts")
        alerts = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        alerts = 0
    check("Alerts in DB (should be 0)", alerts, 0)

    conn.close()
else:
    check("Database exists", "missing", "present")

# Check orphaned ffmpeg
ffmpeg_count = 0
try:
    if os.name == "nt":
        res = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq ffmpeg.exe"],
            capture_output=True,
            text=True,
        )
        if "ffmpeg.exe" in res.stdout:
            ffmpeg_count = res.stdout.count("ffmpeg.exe")
    else:
        res = subprocess.run(
            ["pgrep", "-c", "-f", "ffmpeg.*hls"],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            ffmpeg_count = int(res.stdout.strip() or "0")
except Exception:
    ffmpeg_count = 0
check("Orphaned ffmpeg processes (should be 0)", ffmpeg_count, 0)

# Check stale HLS segments
hls_ts_files = list((PROJECT_ROOT / "hls").glob("**/*.ts"))
check("Stale HLS segments (should be 0)", len(hls_ts_files), 0)

# Check config settings via sentinel.config
try:
    from sentinel.config import settings

    if settings.ACK_TIMEOUT_SECONDS > 20:
        print(f"[FAIL] ACK_TIMEOUT_SECONDS={settings.ACK_TIMEOUT_SECONDS} (demo max: 20)")
        ERRORS += 1
    else:
        print(f"[OK] ACK_TIMEOUT_SECONDS={settings.ACK_TIMEOUT_SECONDS}")

    if not settings.DEMO_MODE:
        print("[FAIL] DEMO_MODE=False (demo routes unavailable)")
        ERRORS += 1
    else:
        print("[OK] DEMO_MODE=True")
except Exception as exc:
    print(f"[FAIL] Failed to load settings from sentinel.config: {exc}")
    ERRORS += 1

print("")
if ERRORS == 0:
    print("[OK] DEMO-READY. Start server: uvicorn sentinel.main:app --reload")
    print("   Then verify in browser: DevTools -> Network -> WS = 1 connection")
    sys.exit(0)
else:
    print(f"[FAIL] {ERRORS} issue(s). Fix before starting Scene 1.")
    sys.exit(1)
