"""Point the camera rows at the URLs that actually serve video.

The database held https://live.corp8.cloud/stream/1 .. /stream/30. None of
those ever returned a frame: wrong host, wrong protocol, wrong id format. They
failed quietly, answering HTTP 200 with the login page as text/html, so a
status-code health check called every camera online while the pipeline received
HTML where it expected H.264.

This replaces them with the endpoints measured working — cctv.corp8.cloud/
<id>/index.m3u8 for cam01..cam30 — taken from the portal's own /cameras.json
rather than constructed by hand, so the mapping cannot drift from what the
portal serves.

Matching is by the number embedded in each side's id: the DB's CAM_07 to the
portal's cam07. Every match is printed before anything is written, and
--dry-run stops there.

Run:  python -m backend.scripts.repoint_corp8_cameras --dry-run
      python -m backend.scripts.repoint_corp8_cameras
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def num(s: str) -> int | None:
    m = re.search(r"(\d+)", s or "")
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from backend.db.session import SessionLocal
    from backend.db.models import Camera
    from backend.services.corp8_session import get_corp8_session, Corp8AuthError

    try:
        sess = get_corp8_session()
        portal = sess.cameras()
    except Corp8AuthError as exc:
        print(f"Cannot authenticate: {exc}", file=sys.stderr)
        return 1
    print(f"portal reports {len(portal)} cameras")

    by_num = {}
    for c in portal:
        n = num(c["id"])
        if n is not None:
            by_num[n] = c

    db = SessionLocal()
    cams = db.query(Camera).filter(Camera.is_deleted == False).all()  # noqa: E712
    print(f"database holds {len(cams)} cameras\n")

    print(f"{'db camera':<11}{'db name':<32}{'portal id':<11}{'new url'}")
    print("-" * 104)
    changed = unmatched = 0
    for c in sorted(cams, key=lambda x: x.id):
        key = c.camera_id or str(c.id)
        n = num(key)
        entry = by_num.get(n) if n is not None else None
        if not entry:
            print(f"{key:<11}{(c.name or '')[:31]:<32}{'-':<11}"
                  f"NO PORTAL MATCH — left unchanged")
            unmatched += 1
            continue
        url = sess.stream_url(entry["id"])
        print(f"{key:<11}{(c.name or '')[:31]:<32}{entry['id']:<11}{url}")
        if not args.dry_run:
            c.url = url
            c.stream_url = url
        changed += 1

    if args.dry_run:
        db.rollback()
        print(f"\nDRY RUN — nothing written. {changed} would change, "
              f"{unmatched} unmatched.")
    else:
        db.commit()
        print(f"\ncommitted: {changed} cameras repointed, {unmatched} unmatched.")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
