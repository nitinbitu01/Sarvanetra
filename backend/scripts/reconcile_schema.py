"""backend/scripts/reconcile_schema.py — add model columns missing from the DB.

THE PROBLEM THIS SOLVES
  SQLAlchemy's create_all() only CREATES absent tables; it never ALTERs one
  that already exists. So every time a column is added to backend/db/models.py
  against an existing database, the model and the table silently diverge.

  The failure is delayed and misleading. Nothing complains at startup - the
  app boots fine - and the error only appears when a query happens to touch
  the missing column:

      sqlite3.OperationalError: no such column: cameras.last_heartbeat_at
      sqlite3.OperationalError: table alerts has no column named track_id

  Found in this codebase on cameras.last_heartbeat_at (which would break the
  camera-heartbeat/fleet-health feature) and users.department. Both were
  invisible until the exact code path ran.

WHAT IT DOES
  Compares every mapped model against the live SQLite schema and adds the
  columns that are missing. Idempotent; safe to re-run; safe to run before
  each deploy.

LIMITS - deliberately conservative
  * Only ADDS columns. Never drops, renames or retypes anything, so it can
    never destroy data.
  * SQLite cannot ALTER-ADD a NOT NULL column without a default. Those are
    reported for manual handling rather than guessed at.
  * Does not create indexes or constraints. Use Alembic for real migrations;
    this is a safety net for drift that has already happened.

USAGE
  python -m backend.scripts.reconcile_schema            # report only
  python -m backend.scripts.reconcile_schema --apply    # actually ALTER
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml

from backend.db.models import Base

# SQLAlchemy type -> SQLite column type. SQLite is dynamically typed, so the
# declared type is mostly documentation; keeping it close to the model avoids
# surprising a human reading the schema later.
_TYPE_MAP = {
    "INTEGER": "INTEGER", "BIGINT": "INTEGER", "SMALLINT": "INTEGER",
    "BOOLEAN": "BOOLEAN", "FLOAT": "FLOAT", "REAL": "FLOAT",
    "NUMERIC": "NUMERIC", "DATETIME": "DATETIME", "DATE": "DATE",
    "TEXT": "TEXT", "JSON": "JSON", "BLOB": "BLOB",
}


def _sqlite_type(col) -> str:
    try:
        compiled = str(col.type.compile(dialect=sqlite3 and None)) if False else str(col.type)
    except Exception:
        compiled = "TEXT"
    upper = compiled.upper()
    if upper.startswith("VARCHAR") or upper.startswith("STRING"):
        return compiled if upper.startswith("VARCHAR") else "TEXT"
    for key, val in _TYPE_MAP.items():
        if upper.startswith(key):
            return val
    return "TEXT"


def _db_path(explicit: str) -> str:
    if explicit:
        return explicit
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    return str(cfg.get("database", {}).get("path", "output/sentinel.db"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="")
    ap.add_argument("--apply", action="store_true",
                    help="Without this the script only reports the drift.")
    args = ap.parse_args()

    path = _db_path(args.db)
    if not Path(path).is_file():
        print(f"Database not found: {path}", file=sys.stderr)
        sys.exit(1)
    print(f"database: {path}\n")

    conn = sqlite3.connect(path)
    existing_tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}

    added = skipped = 0
    missing_tables: list[str] = []

    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            missing_tables.append(table_name)
            continue
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table_name})")}
        for col in table.columns:
            if col.name in have:
                continue
            # SQLite refuses ALTER-ADD of NOT NULL without a default, and
            # inventing one could corrupt meaning. Report instead.
            if not col.nullable and col.default is None and col.server_default is None:
                print(f"  MANUAL  {table_name}.{col.name} is NOT NULL with no "
                      f"default - add it via a proper migration")
                skipped += 1
                continue
            ddl = (f"ALTER TABLE {table_name} "
                   f"ADD COLUMN {col.name} {_sqlite_type(col)}")
            if args.apply:
                try:
                    conn.execute(ddl)
                    conn.commit()
                    print(f"  ADDED   {table_name}.{col.name}")
                    added += 1
                except sqlite3.OperationalError as exc:
                    print(f"  FAILED  {table_name}.{col.name}: {exc}")
                    skipped += 1
            else:
                print(f"  MISSING {table_name}.{col.name}   ({ddl})")
                added += 1

    conn.close()

    if missing_tables:
        print(f"\ntables in the models but not in the DB "
              f"({len(missing_tables)}): {', '.join(sorted(missing_tables)[:8])}"
              f"{' ...' if len(missing_tables) > 8 else ''}")
        print("  create_all() will make these on next startup.")

    print()
    if args.apply:
        print(f"added {added} column(s); {skipped} needing manual handling")
    else:
        print(f"{added} column(s) would be added; {skipped} need manual handling")
        print("re-run with --apply to perform the changes")


if __name__ == "__main__":
    main()
