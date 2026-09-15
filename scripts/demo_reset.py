#!/usr/bin/env python3
"""
scripts/demo_reset.py — Cross-platform Sentinel IQ Demo Reset.
Performs the 4-step reset:
  1. Kill orphaned ffmpeg processes.
  2. Clear HLS directories.
  3. Clear evidence files (keeping directory structure).
  4. Reset database (officers AVAILABLE except Thompson; clear generated tables).
"""
import os
import shutil
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

print("[INFO] Sentinel IQ Demo Reset...")

# 1. Kill orphaned ffmpeg
try:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", "ffmpeg.exe", "/T"],
            capture_output=True,
            text=True,
        )
    else:
        subprocess.run(["pkill", "-f", "ffmpeg.*hls"], capture_output=True)
except Exception:
    pass

# 2. Clear HLS directories
hls_dir = PROJECT_ROOT / "hls"
if hls_dir.exists():
    shutil.rmtree(hls_dir, ignore_errors=True)
hls_dir.mkdir(parents=True, exist_ok=True)

# 3. Clear evidence files (keep directory structure)
evidence_dir = PROJECT_ROOT / "evidence"
if evidence_dir.exists():
    for ext in ("*.mp4", "*.jpg", "*.pdf", "*.png"):
        for f in evidence_dir.rglob(ext):
            try:
                f.unlink()
            except Exception:
                pass

# 4. Reset database
db_paths = [
    PROJECT_ROOT / "sentinel.db",
    PROJECT_ROOT / "output" / "sentinel.db",
]

for db_path in db_paths:
    if not db_path.exists():
        continue
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Ensure officers exist
    cursor.execute("SELECT COUNT(*) FROM officers;")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT OR IGNORE INTO officers
                (id, name, lat, lng, status, current_alert_id, last_updated)
            VALUES (?, ?, ?, ?, ?, NULL, datetime('now'))
        """, [
            (1, "Officer Chen", 37.7749, -122.4194, "AVAILABLE"),
            (2, "Officer Park", 37.7751, -122.4180, "AVAILABLE"),
            (3, "Officer Ramirez", 37.7740, -122.4210, "AVAILABLE"),
            (4, "Officer Thompson", 37.7760, -122.4220, "OFFLINE"),
        ])

    # Reset officers (keep Thompson OFFLINE)
    cursor.execute("""
        UPDATE officers
        SET status='AVAILABLE', current_alert_id=NULL, last_updated=datetime('now')
        WHERE name != 'Officer Thompson';
    """)

    # Reset cameras to ONLINE
    try:
        cursor.execute("UPDATE cameras SET status='ONLINE', consecutive_failures=0;")
    except Exception:
        pass

    # Clear generated tables
    tables_to_clear = [
        "routed_alerts",
        "alerts",
        "alert_feedback",
        "feedback_flag_log",
        "push_delivery_log",
        "officer_connectivity_log",
        "client_error_log",
    ]

    for table in tables_to_clear:
        try:
            cursor.execute(f"DELETE FROM {table};")
        except sqlite3.OperationalError:
            # Table may not exist yet in this schema version
            pass

    conn.commit()

    print(f"  Reset database: {db_path}")
    cursor.execute("SELECT name, status FROM officers;")
    officers = cursor.fetchall()
    print("  Officers after reset:")
    for name, status in officers:
        print(f"    {name}: {status}")
    conn.close()

print("\n⚠️  MANUAL BROWSER STEPS (open DevTools → Application → Local Storage):")
print("  Delete key: sentineliq.pending_acks")
print("  Delete key: sentineliq.dismissed_unrouted")
print("  Delete key: sentineliq.sync_lock")
print("  Then: Ctrl+R to reload dashboard\n")

# Run verification
verify_script = PROJECT_ROOT / "scripts" / "verify_demo_ready.py"
if verify_script.exists():
    print("Running verification...")
    rc = subprocess.run([sys.executable, str(verify_script)]).returncode
    sys.exit(rc)
