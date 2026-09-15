"""
backend/scripts/warm_snapshot_cache.py

Fills the snapshot cache for every camera so the Control Room grid shows real
imagery immediately instead of placeholder tiles.

WHY THIS IS NEEDED
  A cold snapshot costs 33-50s. That is not a bug and not fixable from here:
  the portal serves twelve-hour VOD playlists, and the capture reads them with
  ffmpeg's `-re` (media-rate) on purpose, because reading flat out is what
  previously earned this account HTTP 429 and then 403 across every camera.

  The snapshot endpoint serves stale-while-revalidate, so ONCE a camera has any
  cached frame the UI returns it in ~0.2s and refreshes behind the response.
  The only slow moment is the very first frame per camera — which is exactly
  what an operator (or an evaluator) sees when they open the grid cold.

  So warm it beforehand, sequentially and paced, rather than letting thirty
  browser tiles each trigger their own cold grab at once. Firing thirty
  simultaneous captures at the portal is the documented way to get the account
  rate-limited; this deliberately does the opposite.

USAGE
    python -m backend.scripts.warm_snapshot_cache
    python -m backend.scripts.warm_snapshot_cache --spacing 3 --only CAM_08,CAM_11

Run it a few minutes before a demonstration and leave the dashboard open: the
background refresh in the snapshot endpoint keeps the frames current from then
on, so every tile stays populated.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://127.0.0.1:8000/api/v1")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")
    ap.add_argument("--spacing", type=float, default=2.0,
                    help="seconds to wait between cameras (default 2)")
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="per-camera timeout (default 90; a cold grab is 33-50s)")
    ap.add_argument("--only", default="",
                    help="comma-separated camera ids; default is every camera")
    args = ap.parse_args()

    try:
        r = requests.post(f"{args.api}/auth/login",
                          json={"username": args.username, "password": args.password},
                          timeout=30)
        r.raise_for_status()
        token = r.json()["access_token"]
    except Exception as exc:
        print(f"Could not log in to {args.api}: {exc}", file=sys.stderr)
        print("Is the API running? uvicorn backend.main:app --port 8000", file=sys.stderr)
        return 2

    headers = {"Authorization": f"Bearer {token}"}

    try:
        cams = requests.get(f"{args.api}/cameras", headers=headers, timeout=30).json()
    except Exception as exc:
        print(f"Could not list cameras: {exc}", file=sys.stderr)
        return 2

    wanted = {c.strip() for c in args.only.split(",") if c.strip()}
    ids = [c["id"] for c in cams if not wanted or c["id"] in wanted]
    if not ids:
        print("No cameras matched.", file=sys.stderr)
        return 1

    print(f"Warming {len(ids)} camera(s), {args.spacing}s apart. "
          f"A cold frame takes 33-50s, so this is not quick — that is the point "
          f"of doing it before anyone is watching.\n")

    warmed = failed = 0
    for i, cam_id in enumerate(ids, 1):
        t0 = time.monotonic()
        try:
            resp = requests.get(f"{args.api}/cameras/{cam_id}/snapshot",
                                headers=headers, timeout=args.timeout)
            dt = time.monotonic() - t0
            if resp.status_code == 200 and resp.content:
                state = resp.headers.get("X-Snapshot-Cache", "?")
                print(f"  [{i}/{len(ids)}] {cam_id:12s} ok   {dt:5.1f}s "
                      f"{len(resp.content):>7,d} bytes  ({state})")
                warmed += 1
            else:
                print(f"  [{i}/{len(ids)}] {cam_id:12s} HTTP {resp.status_code} "
                      f"after {dt:5.1f}s")
                failed += 1
        except Exception as exc:
            dt = time.monotonic() - t0
            print(f"  [{i}/{len(ids)}] {cam_id:12s} failed after {dt:5.1f}s: "
                  f"{type(exc).__name__}")
            failed += 1

        if i < len(ids):
            time.sleep(args.spacing)

    print(f"\nWarmed {warmed}, failed {failed}.")
    if warmed:
        print("Those cameras now answer in ~0.2s and refresh in the background.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
