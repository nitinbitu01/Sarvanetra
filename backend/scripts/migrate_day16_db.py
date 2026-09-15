# scripts/migrate_day16_db.py
import sqlite3
from pathlib import Path

db_paths = [
    Path("output/sentinel.db"),
    Path("sentinel.db"),
]

for p in db_paths:
    if not p.exists():
        continue
    print(f"Migrating {p}...")
    conn = sqlite3.connect(p)
    cur = conn.cursor()

    # 1. Add columns to cameras if not present
    cur.execute("PRAGMA table_info(cameras)")
    cols = [row[1] for row in cur.fetchall()]

    if "stream_url" not in cols:
        print(f"  Adding stream_url to {p}")
        cur.execute("ALTER TABLE cameras ADD COLUMN stream_url TEXT;")

    if "location_label" not in cols:
        print(f"  Adding location_label to {p}")
        cur.execute("ALTER TABLE cameras ADD COLUMN location_label TEXT;")

    # 2. Create alert_feedback
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alert_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id TEXT NOT NULL UNIQUE,
        verdict TEXT NOT NULL,
        officer_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS ix_alert_feedback_alert_id ON alert_feedback (alert_id);")

    # 3. Create feedback_flag_log
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback_flag_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        zone_name TEXT NOT NULL,
        flag_type TEXT NOT NULL,
        flag_message TEXT NOT NULL,
        false_alarm_count INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved INTEGER DEFAULT 0
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS ix_feedback_flag_log_zone_name ON feedback_flag_log (zone_name);")

    # 4. Update demo cameras
    cur.execute("UPDATE cameras SET location_label='Main Entrance Zone', stream_url='demo/clips/entrance_loop.mp4' WHERE name LIKE '%Main Entrance%' OR id='1' OR id='CAM-01';")
    cur.execute("UPDATE cameras SET location_label='Parking Zone', stream_url='demo/clips/parking_loop.mp4' WHERE name LIKE '%Parking%' OR id='2' OR id='CAM-02';")

    conn.commit()
    conn.close()
    print(f"Successfully migrated {p}.")

print("All databases migrated!")
