"""Indexes for the columns the analytics dashboard filters on.

Every panel on the dashboard issues its queries on page load, so a query that
takes half a second is half a second of blank panel — and these grow with the
tables, which is exactly backwards for a system meant to run 24/7.

Measured on the current database (78,095 vault entries, 177,198 tracks) before
adding these:

    vault, crop_expires_at range      443 ms   full scan
    vault, training_eligible = 1      456 ms   full scan
    alerts, created_at window          --      full scan
    vault, group by compartment        11 ms   already indexed
    tracks, camera + first_seen        52 ms   already indexed

The two vault scans alone accounted for ~900 ms of the 1,125 ms /vault/stats
response. The alert indexes matter less today, at 558 rows, and matter a great
deal at the volume a 27-camera fleet reaches in a month.

Safe to re-run: every statement is IF NOT EXISTS.

Run:  python -m backend.scripts.add_analytics_indexes
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "output" / "sentinel.db"

INDEXES = [
    # /vault/stats: retention sweep and retraining-eligibility counts.
    ("ix_vault_crop_expires", "vault_entries", "(crop_expires_at)"),
    ("ix_vault_training_eligible", "vault_entries", "(training_eligible)"),
    ("ix_vault_used_in_training", "vault_entries", "(used_in_training)"),
    # /vault/stats groups by entity_type. The other grouped columns were
    # already indexed; this one was not, and stayed at ~820 ms while the
    # rest dropped to single digits.
    ("ix_vault_entity_type", "vault_entries", "(entity_type)"),
    # /analytics/overview, /trend, /zones, /officers all window on created_at.
    ("ix_alerts_created_at", "alerts", "(created_at)"),
    # /zones groups by severity within the window; /officers by assignee.
    ("ix_alerts_created_severity", "alerts", "(created_at, severity)"),
    ("ix_alerts_assigned_officer", "alerts", "(assigned_officer_id)"),
    # /fleet-health scans tracks by time across all cameras. The existing
    # index leads with camera_id, which does not serve a time-only range.
    ("ix_vehicle_track_first_seen", "vehicle_track", "(first_seen)"),
]

CHECKS = [
    ("vault, crop_expires_at range",
     "SELECT COUNT(*) FROM vault_entries "
     "WHERE crop_expires_at <= datetime('now','+7 day') "
     "AND crop_expires_at >= datetime('now') AND crop_minio_path IS NOT NULL"),
    ("vault, training_eligible",
     "SELECT COUNT(*) FROM vault_entries WHERE training_eligible = 1"),
    ("alerts, 24h window",
     "SELECT COUNT(*) FROM alerts WHERE created_at >= datetime('now','-1 day')"),
    ("tracks, 24h across fleet",
     "SELECT camera_id, COUNT(*) FROM vehicle_track "
     "WHERE first_seen >= datetime('now','-1 day') GROUP BY camera_id"),
]


def timed(con, sql):
    t = time.perf_counter()
    con.execute(sql).fetchall()
    return (time.perf_counter() - t) * 1000


def main() -> int:
    if not DB.is_file():
        print(f"no database at {DB}")
        return 1
    con = sqlite3.connect(str(DB))

    print("before:")
    before = {}
    for label, sql in CHECKS:
        before[label] = timed(con, sql)
        print(f"   {label:<28} {before[label]:8.1f} ms")

    print("\ncreating:")
    for name, table, cols in INDEXES:
        con.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} {cols}")
        print(f"   {name}")
    con.commit()
    # Let the planner know the new selectivity, or it may keep the old plan.
    con.execute("ANALYZE")
    con.commit()

    print("\nafter:")
    for label, sql in CHECKS:
        after = timed(con, sql)
        delta = before[label] / after if after > 0 else 0
        print(f"   {label:<28} {after:8.1f} ms   "
              f"{delta:5.1f}x faster" if delta >= 1.2 else
              f"   {label:<28} {after:8.1f} ms")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
