"""Give the seeded users real usernames, so login works without the backdoor.

All three users were created with username = NULL. The login route looks a
user up by username, badge_number, id or name, so "admin" matched nothing —
and every real login fell through to a "demo fallback" branch that issued an
ADMIN token with no password check at all.

Removing that branch without this script would lock everyone out: there would
be no username anyone could type. So the two changes belong together.

Passwords are NOT touched. Each user already has a genuine $2b$12$ bcrypt
hash, and those keep working.

Run:  python -m backend.scripts.fix_user_logins --dry-run
      python -m backend.scripts.fix_user_logins
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# id -> the username an operator will actually type.
USERNAMES = {
    "USER_ADMIN_01": "admin",
    "USER_OPS_01": "operator",
    "USER_VIEW_01": "viewer",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from backend.db.session import SessionLocal
    from backend.db.models import User

    db = SessionLocal()
    users = db.query(User).all()
    print(f"{len(users)} users\n")
    print(f"{'id':<16}{'name':<26}{'username now':<16}{'->':<4}{'username after'}")
    print("-" * 80)
    changed = 0
    for u in users:
        want = USERNAMES.get(str(u.id))
        current = u.username
        if want and current != want:
            print(f"{str(u.id):<16}{str(u.name)[:25]:<26}{str(current):<16}{'->':<4}{want}")
            if not args.dry_run:
                u.username = want
            changed += 1
        else:
            print(f"{str(u.id):<16}{str(u.name)[:25]:<26}{str(current):<16}"
                  f"{'':<4}(unchanged)")

    # A user with no username cannot log in now that the fallback is gone, so
    # say so plainly rather than leaving it to be discovered at the demo.
    orphaned = [u for u in users if not USERNAMES.get(str(u.id)) and not u.username]
    if orphaned:
        print(f"\nWARNING: {len(orphaned)} user(s) still have no username and "
              f"cannot log in:")
        for u in orphaned:
            print(f"  id={u.id} name={u.name} role={u.role}")

    if args.dry_run:
        db.rollback()
        print(f"\nDRY RUN — nothing written. {changed} would change.")
    else:
        db.commit()
        print(f"\ncommitted: {changed} usernames set.")
        print("Log in with admin / operator / viewer and the existing "
              "passwords.")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
