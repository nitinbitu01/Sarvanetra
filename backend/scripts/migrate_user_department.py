"""backend/scripts/migrate_user_department.py — add users.department.

WHY A SCRIPT
  SQLAlchemy's create_all() only CREATES missing tables; it never ALTERs an
  existing one. The users table already exists, so adding the column to the
  model alone leaves the live database without it, and every department
  scoping query fails at runtime rather than at startup.

WHAT IT DOES
  Adds users.department if absent, then assigns departments to existing
  users so scoping is demonstrable immediately. Idempotent - safe to re-run.

  Admins are deliberately left NULL: NULL means state-level, which is the
  scope that crosses all 26 departments.

USAGE
  python -m backend.scripts.migrate_user_department
  python -m backend.scripts.migrate_user_department --assign Transport --user op1
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml


def _db_path() -> str:
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    return str(cfg.get("database", {}).get("path", "output/sentinel.db"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="")
    ap.add_argument("--assign", default="",
                    help="Department to assign (with --user).")
    ap.add_argument("--user", default="",
                    help="username or id to assign --assign to.")
    args = ap.parse_args()

    path = args.db or _db_path()
    if not Path(path).is_file():
        print(f"Database not found: {path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(users)")}

    if "department" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN department VARCHAR(100)")
        conn.commit()
        print("added column users.department")
    else:
        print("users.department already present")

    if args.assign and args.user:
        cur = conn.execute(
            "UPDATE users SET department = ? WHERE username = ? OR id = ?",
            (args.assign, args.user, args.user),
        )
        conn.commit()
        print(f"assigned department {args.assign!r} to {cur.rowcount} user(s) "
              f"matching {args.user!r}")

    print("\ncurrent users:")
    print(f"  {'id':<24} {'username':<16} {'role':<10} department")
    for r in conn.execute(
        "SELECT id, username, role, department FROM users ORDER BY role, username"
    ):
        dept = r[3] or "(state-level)"
        print(f"  {str(r[0])[:22]:<24} {str(r[1] or ''):<16} {str(r[2] or ''):<10} {dept}")
    conn.close()


if __name__ == "__main__":
    main()
