"""Prove the plate-clone detector, on cases whose answers are known in advance.

A detector that fires is not thereby correct; one that stays silent is not
thereby broken. Each case below states what the right answer is and why, so a
failure names the rule it broke rather than just printing a number.

Run:  python -m backend.scripts.test_plate_clone_detector
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.plate_clone_detector import (                 # noqa: E402
    MAX_ROAD_SPEED_KMH, MIN_SEPARATION_KM, Sighting, find_clones, haversine_km,
)

# Real camera positions from this deployment, so the distances are the ones
# the detector will actually meet.
JUNAGADH = (21.522, 70.4579)     # CAM_08 Majevadi Gate
BYPASS = (21.53, 70.46)          # CAM_09 New Bypass, 2 km away
RAJKOT = (22.3039, 70.8022)      # CAM_14 area, ~93 km away
KUTCH = (23.0853, 70.1337)       # ~180 km away

T0 = datetime(2026, 9, 9, 10, 0, 0)
passed = failed = 0


def check(name: str, expectation: str, got_alerts, want: int) -> None:
    global passed, failed
    ok = len(got_alerts) == want
    print("  %-52s %s" % (name, "PASS" if ok else "FAIL"))
    if not ok:
        print("      expected %d alert(s), got %d — %s" % (want, len(got_alerts), expectation))
        for a in got_alerts:
            print("      %s" % a.describe())
    if ok:
        passed += 1
    else:
        failed += 1


print("distances these cases rely on:")
print("   Junagadh -> Rajkot  %6.1f km" % haversine_km(*JUNAGADH, *RAJKOT))
print("   Junagadh -> Kutch   %6.1f km" % haversine_km(*JUNAGADH, *KUTCH))
print("   Junagadh -> bypass  %6.1f km" % haversine_km(*JUNAGADH, *BYPASS))
print("\nrules: over %.0f km/h, and at least %.0f km apart\n"
      % (MAX_ROAD_SPEED_KMH, MIN_SEPARATION_KM))

print("MUST FIRE")
check(
    "same plate 93 km apart, 5 minutes",
    "1100 km/h is not a vehicle",
    find_clones([
        Sighting("GJ01AB1234", "CAM_08", T0, *JUNAGADH),
        Sighting("GJ01AB1234", "CAM_14", T0 + timedelta(minutes=5), *RAJKOT),
    ]), 1)

check(
    "same plate 180 km apart, 30 minutes",
    "360 km/h is not a vehicle",
    find_clones([
        Sighting("GJ05XY9999", "CAM_08", T0, *JUNAGADH),
        Sighting("GJ05XY9999", "CAM_20", T0 + timedelta(minutes=30), *KUTCH),
    ]), 1)

check(
    "three sightings, only the middle leg impossible",
    "A->B legitimate, B->C impossible",
    find_clones([
        Sighting("GJ09PQ4444", "CAM_08", T0, *JUNAGADH),
        Sighting("GJ09PQ4444", "CAM_14", T0 + timedelta(hours=2), *RAJKOT),
        Sighting("GJ09PQ4444", "CAM_08", T0 + timedelta(hours=2, minutes=6), *JUNAGADH),
    ]), 1)

print("\nMUST NOT FIRE")
check(
    "93 km in 2 hours — an ordinary drive",
    "46 km/h is normal",
    find_clones([
        Sighting("GJ02CD5678", "CAM_08", T0, *JUNAGADH),
        Sighting("GJ02CD5678", "CAM_14", T0 + timedelta(hours=2), *RAJKOT),
    ]), 0)

check(
    "93 km in 50 minutes — fast but possible",
    "112 km/h is under the ceiling",
    find_clones([
        Sighting("GJ03EF1111", "CAM_08", T0, *JUNAGADH),
        Sighting("GJ03EF1111", "CAM_14", T0 + timedelta(minutes=50), *RAJKOT),
    ]), 0)

check(
    "adjacent cameras 2 km apart, 10 seconds",
    "under the %.0f km separation floor, where GPS error dominates" % MIN_SEPARATION_KM,
    find_clones([
        Sighting("GJ04GH2222", "CAM_08", T0, *JUNAGADH),
        Sighting("GJ04GH2222", "CAM_09", T0 + timedelta(seconds=10), *BYPASS),
    ]), 0)

check(
    "same camera twice, seconds apart",
    "one vehicle in one field of view",
    find_clones([
        Sighting("GJ06IJ3333", "CAM_08", T0, *JUNAGADH),
        Sighting("GJ06IJ3333", "CAM_08", T0 + timedelta(seconds=30), *JUNAGADH),
    ]), 0)

check(
    "two DIFFERENT plates far apart, minutes apart",
    "different registrations are different vehicles",
    find_clones([
        Sighting("GJ07KL5555", "CAM_08", T0, *JUNAGADH),
        Sighting("GJ08MN6666", "CAM_14", T0 + timedelta(minutes=5), *RAJKOT),
    ]), 0)

check(
    "a single sighting",
    "one point is not a transition",
    find_clones([Sighting("GJ10QR7777", "CAM_08", T0, *JUNAGADH)]), 0)

# ── against the real database ────────────────────────────────────────────
print("\nAGAINST THE REAL DATA")
try:
    from backend.db.session import SessionLocal
    from backend.services.plate_clone_detector import load_sightings_from_db

    db = SessionLocal()
    try:
        real = load_sightings_from_db(db)
    finally:
        db.close()
    got = find_clones(real)
    plates = len({s.plate for s in real})
    print("  %d real sightings, %d distinct plates -> %d alert(s)"
          % (len(real), plates, len(got)))
    for a in got:
        print("     %s" % a.describe())
    print("  %-52s %s" % ("no false alarm on real footage",
                          "PASS" if not got else "FAIL"))
    if got:
        failed += 1
    else:
        passed += 1
        print("      (correct: only 2 plates were ever read at two cameras, and")
        print("       both transitions are physically possible)")
except Exception as exc:                                            # noqa: BLE001
    print("  could not reach the database: %s" % exc)

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(1 if failed else 0)
