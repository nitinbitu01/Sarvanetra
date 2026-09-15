"""Vehicle speed in km/h from camera-to-camera travel time.

Per-camera instantaneous speed needs a metric ground plane, and this footage
does not supply one. Auto-calibration from the video itself was attempted and
reported honestly in fit_camera_calibration: vanishing points come out fine,
but camera height — recovered from vehicles of regulated width — has an
interquartile spread as large as the estimate on the only two cameras with
enough end-on observations to attempt it. A homography that uncertain produces
a number, not a measurement, which is precisely what the hand-typed
calibrations did.

This route needs no calibration at all. When the plate reader places the same
vehicle at camera A and later at camera B, and both cameras have surveyed GPS
positions, the average speed over that leg is distance divided by elapsed
time. Both inputs are measured: the positions come from the camera register,
the times from the clips.

What it is and is not:

  it is      the mean speed over a leg, which is what a traffic engineer or an
             investigator actually asks for — how long did this vehicle take
             between these two junctions
  it is not  instantaneous speed at a point, so it cannot support a speeding
             prosecution at a specific location

Straight-line distance underestimates road distance, so the speed derived from
it is a lower bound. That direction is stated rather than corrected with a
guessed road-network factor.

Run:  python -m backend.scripts.intercamera_speed
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "output" / "journey_index" / "records.jsonl"
DB = ROOT / "output" / "sentinel.db"
OUT = ROOT / "output" / "intercamera_speed.json"

# A leg shorter than this is within GPS and timing noise; longer than this and
# the vehicle has probably stopped or taken a different route than the
# straight line suggests.
MIN_LEG_M = 150.0
MAX_LEG_SECONDS = 1800.0
# Physically possible ground speeds for road vehicles. Anything outside is a
# mis-association or a clock error, not a fast car.
MAX_PLAUSIBLE_KMH = 140.0
# A partial read is not an identity. An Indian registration is 9-10 characters
# (GJ01AB1234); a two-character fragment like "GJ" matches a third of the
# state's vehicles, so pairing two of them across cameras produces a "leg" for
# vehicles that were never the same car. Those fragments were what generated
# every one of the 55,000 km/h outliers in the first run.
MIN_PLATE_CHARS = 8


def haversine_m(a: tuple, b: tuple) -> float:
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = p2 - p1
    dl = math.radians(b[1] - a[1])
    x = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(x))


def main() -> int:
    if not INDEX.is_file():
        print(f"no journey index at {INDEX}")
        return 1

    con = sqlite3.connect(str(DB))
    gps = {c: (float(la), float(lo)) for c, la, lo in con.execute(
        "select camera_id, coalesce(lat, gps_lat), coalesce(lon, gps_lon) "
        "from cameras where coalesce(lat, gps_lat) is not null")}
    con.close()
    print(f"{len(gps)} cameras with a surveyed position")

    by_plate: dict[str, list] = defaultdict(list)
    n = short = 0
    for line in INDEX.open(encoding="utf-8"):
        r = json.loads(line)
        plate = r.get("plate") or r.get("text")
        cam = r.get("camera")
        ts = r.get("timestamp") or r.get("ts")
        if not plate or cam not in gps or ts is None:
            continue
        if len(str(plate).strip()) < MIN_PLATE_CHARS:
            short += 1
            continue
        when = (datetime.fromisoformat(ts) if isinstance(ts, str)
                else datetime.utcfromtimestamp(float(ts)))
        by_plate[plate].append((when, cam))
        n += 1
    print(f"{n} usable sightings of {len(by_plate)} plates "
          f"({short} discarded as partial reads under {MIN_PLATE_CHARS} chars)\n")

    legs = []
    for plate, sightings in by_plate.items():
        sightings.sort()
        for (t1, c1), (t2, c2) in zip(sightings, sightings[1:]):
            if c1 == c2:
                continue
            dt = (t2 - t1).total_seconds()
            if not (0 < dt <= MAX_LEG_SECONDS):
                continue
            d = haversine_m(gps[c1], gps[c2])
            if d < MIN_LEG_M:
                continue
            kmh = d / dt * 3.6
            legs.append({"plate": plate, "from": c1, "to": c2,
                         "metres": round(d, 1), "seconds": round(dt, 1),
                         "kmh": round(kmh, 1),
                         "plausible": kmh <= MAX_PLAUSIBLE_KMH})

    if not legs:
        print("no camera-to-camera legs found in the index")
        return 1

    ok = [l for l in legs if l["plausible"]]
    speeds = np.array([l["kmh"] for l in ok])
    print(f"{len(legs)} legs, {len(ok)} within a physically possible speed")
    print(f"\nspeed over ground, km/h (straight-line, so a lower bound):")
    for q in (5, 25, 50, 75, 95):
        print(f"   p{q:<3} {np.percentile(speeds, q):6.1f}")
    print(f"   mean {speeds.mean():6.1f}   n = {len(speeds)}")

    dists = np.array([l["metres"] for l in ok])
    print(f"\nleg length, metres:  median {np.median(dists):,.0f}  "
          f"range {dists.min():,.0f}-{dists.max():,.0f}")

    print("\nbusiest camera pairs:")
    pairs: dict[tuple, list] = defaultdict(list)
    for l in ok:
        pairs[(l["from"], l["to"])].append(l["kmh"])
    for (a, b), v in sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:8]:
        print(f"   {a} -> {b:<9} {len(v):>4} legs   median "
              f"{np.median(v):5.1f} km/h")

    rejected = [l for l in legs if not l["plausible"]]
    if rejected:
        print(f"\n{len(rejected)} legs rejected above {MAX_PLAUSIBLE_KMH} km/h "
              f"— these are plate mis-matches or clock offsets, not vehicles:")
        for l in sorted(rejected, key=lambda x: -x["kmh"])[:5]:
            print(f"   {l['plate']:<12} {l['from']} -> {l['to']:<9} "
                  f"{l['metres']:>7.0f} m in {l['seconds']:>6.1f} s = "
                  f"{l['kmh']:>7.1f} km/h")

    OUT.write_text(json.dumps({
        "legs": legs,
        "n_plausible": len(ok),
        "median_kmh": float(np.median(speeds)),
        "method": "haversine between surveyed camera positions divided by "
                  "elapsed time between plate sightings; straight-line "
                  "distance makes this a lower bound on road speed",
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
