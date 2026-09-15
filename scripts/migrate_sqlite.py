"""
scripts/migrate_sqlite.py — Adds any missing master architecture columns to SQLite database.
"""

import sqlite3
import os
import glob

def migrate_db(db_path: str):
    if not os.path.exists(db_path):
        return
    print(f"Migrating {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Columns to add to journey_events
    journey_columns = [
        ("object_class", "TEXT DEFAULT 'person'"),
        ("color", "TEXT"),
        ("subtype", "TEXT"),
        ("plate_text", "TEXT"),
        ("visual_score", "FLOAT DEFAULT 0.0"),
        ("final_score", "FLOAT DEFAULT 0.0"),
        ("match_type", "TEXT DEFAULT 'visual_reid'"),
    ]

    for col_name, col_type in journey_columns:
        try:
            cur.execute(f"ALTER TABLE journey_events ADD COLUMN {col_name} {col_type}")
            print(f"  Added column journey_events.{col_name}")
        except sqlite3.OperationalError as e:
            # Column likely already exists
            pass

    # Columns for cameras
    camera_columns = [
        ("vendor", "TEXT DEFAULT 'unknown'"),
        ("channel", "INTEGER DEFAULT 1"),
        ("connects_to", "TEXT"),
    ]
    for col_name, col_type in camera_columns:
        try:
            cur.execute(f"ALTER TABLE cameras ADD COLUMN {col_name} {col_type}")
            print(f"  Added column cameras.{col_name}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()
    print(f"Migration completed for {db_path}.")

if __name__ == "__main__":
    db_candidates = [
        "sentinel.db",
        "output/sentinel.db",
        "backend/sentinel.db",
    ] + glob.glob("**/*.db", recursive=True)
    for db in set(db_candidates):
        migrate_db(db)
