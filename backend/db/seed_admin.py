"""
backend/db/seed_admin.py — Create the first admin user.

Run once before starting the server:
    python -m backend.db.seed_admin

Prints a generated password to stdout exactly once — never committed to git.
If the admin user already exists, no-ops silently.

WARNING: This script reads .env for DATABASE_URL. Ensure .env exists first.
"""
from __future__ import annotations

import secrets
import sys

import bcrypt
from sqlalchemy.exc import IntegrityError

# Make project root importable when run as module
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.db.models import Base, User
from backend.db.session import SessionLocal, engine

ADMIN_USERNAME = "admin"
DEFAULT_BADGE  = "ADMIN-001"


def seed_admin() -> None:
    # Ensure all tables exist (safe — no-op if they're already there)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == ADMIN_USERNAME).first()
        if existing:
            print(f"[seed_admin] Admin user '{ADMIN_USERNAME}' already exists. No changes made.")
            return

        # Generate a secure random password — print once, never store in code
        raw_password = secrets.token_urlsafe(16)
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(raw_password.encode("utf-8"), salt).decode("utf-8")

        admin = User(
            username=ADMIN_USERNAME,
            hashed_password=hashed,
            role="admin",
            badge_number=DEFAULT_BADGE,
            is_active=True,
        )
        db.add(admin)
        db.commit()

        print("=" * 60)
        print(f"[seed_admin] Admin user created.")
        print(f"  Username : {ADMIN_USERNAME}")
        print(f"  Password : {raw_password}")
        print("  SAVE THIS PASSWORD — it will not be shown again.")
        print("=" * 60)

    except IntegrityError:
        db.rollback()
        print(f"[seed_admin] Admin already exists (concurrent creation). No changes.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_admin()
