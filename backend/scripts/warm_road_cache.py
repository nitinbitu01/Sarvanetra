"""Precompute the road route between every pair of cameras that traffic
actually uses, so trajectory rendering needs no network at runtime.

WHY PRECOMPUTE
  A police control room may sit behind a network that cannot reach the
  internet, and a journey must render instantly when an officer types a plate.
  Camera positions do not move, so the road between any two of them is a
  constant: fetch it once, keep it, and the runtime is a dictionary lookup.

WHICH PAIRS
  By default only pairs that have actually been observed in journey_events —
  the corridors traffic really uses — which is a few dozen rather than the
  n*(n-1) = 870 that a full fleet mesh would need. --all forces the mesh.

  python -m backend.scripts.warm_road_cache
  python -m backend.scripts.warm_road_cache --all --delay 1.0
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text                                         # noqa: E402

from backend.db.session import SessionLocal                         # noqa: E402
from backend.services.fleet_census import is_test_camera            # noqa: E402
from backend.services.road_router import get_road_router            # noqa: E402


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="every camera pair, not only observed corridors")
    ap.add_argument("--delay", type=float, default=0.6,
                    help="seconds between calls; the public demo server is "
                         "rate limited and this is basic courtesy")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    db = SessionLocal()
    cams = {}
    for r in db.execute(text(
            "SELECT id, name, lat, lon FROM cameras "
            "WHERE is_deleted=0 AND lat IS NOT NULL AND lon IS NOT NULL")).fetchall():
        if is_test_camera(r[0], r[1]):
            continue          # placeholder coordinates: routing them is meaningless
        cams[r[0]] = (float(r[2]), float(r[3]))
    log(f"{len(cams)} deployed cameras with coordinates")

    if args.all:
        pairs = list(itertools.permutations(sorted(cams), 2))
        log(f"warming the full mesh: {len(pairs)} ordered pair(s)")
    else:
        rows = db.execute(text("""
            SELECT a.camera_id, b.camera_id, COUNT(*) n
            FROM journey_events a
            JOIN journey_events b
              ON a.reid_id = b.reid_id AND a.camera_id <> b.camera_id
             AND b.timestamp > a.timestamp
            WHERE a.object_class='vehicle'
            GROUP BY a.camera_id, b.camera_id
            ORDER BY n DESC""")).fetchall()
        pairs = [(r[0], r[1]) for r in rows
                 if r[0] in cams and r[1] in cams]
        log(f"observed corridors: {len(pairs)} ordered pair(s)")
    db.close()

    pairs = pairs[:args.limit]
    router = get_road_router()
    done = ok = failed = cached = 0
    for a, b in pairs:
        (alat, alon), (blat, blon) = cams[a], cams[b]
        pre = router.route(alat, alon, blat, blon, allow_live=False)
        if pre.source == "cache":
            cached += 1
            continue
        leg = router.route(alat, alon, blat, blon, allow_live=True)
        done += 1
        if leg.matched:
            ok += 1
            log(f"  {a} -> {b}: road {leg.distance_km:.2f} km vs straight "
                f"{leg.straight_km:.2f} km  ({leg.detour_factor:.2f}x), "
                f"{len(leg.geometry)} pts")
        else:
            failed += 1
            log(f"  {a} -> {b}: NO ROUTE (kept as straight line)")
        time.sleep(args.delay)

    log(f"\n  {cached} already cached, {done} fetched: {ok} routed, "
        f"{failed} unroutable")
    st = router.stats()
    log(f"  cache now holds {st['cached_legs']} leg(s) "
        f"({st['cached_routed']} routed) at {st['cache_path']}")
    log("\n  Runtime needs no network now. To refresh after moving a camera, "
        "delete that pair from the cache and re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
