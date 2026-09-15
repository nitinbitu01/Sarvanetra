"""backend/services/corridor_legs.py — cross-camera journey legs, keyed by
CORRIDOR first and camera second.

WHY CORRIDOR IS THE PRIMARY KEY
  A "leg" is one plate read at two cameras: distance from GPS, time from the
  clocks, so a speed that touches no camera model. That makes legs the only
  independent yardstick this deployment has, and they are used both to solve
  for focal length and to validate the result.

  Aggregating them per CAMERA was the obvious thing to do and it is wrong.
  Measured on CAM_08, whose legs split cleanly in two:

      ~26 km/h   the urban link from CAM_11
      ~65 km/h   the highway link towards CAM_07

  "The median leg speed at CAM_08" is not a property of CAM_08 at all — it is
  an artefact of which corridor happened to contribute more legs that week. A
  focal length solved against it inherits that, and CAM_08's confidence
  interval duly blew out to 153% of the estimate while CAM_11 — whose legs are
  15/17 from a single corridor — came out at 13%.

  So corridor is the unit of aggregation. Per-camera views are derived from it
  and always carry how many corridors they span, because a camera fed by two
  corridors of different character needs a different treatment from one fed by
  a single corridor, and that distinction has to survive into whatever consumes
  this.

DIRECTION IS KEPT
  A -> B and B -> A are stored separately. They are usually similar and
  occasionally not — a gradient, a signal on one approach, a merge — and
  collapsing them would hide exactly the asymmetry worth seeing.
"""
from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "output" / "sentinel.db"

# Below this the two sightings are effectively the same place and the speed is
# dominated by GPS and timestamp error.
MIN_LEG_METRES = 50.0

# ABOVE this a leg stops being a proxy for speed at either camera.
#
# A leg is only useful because a vehicle's average over a short link is close
# to its speed where the camera stands. Over a long one it is not: the average
# is dominated by whatever happened in between — a different road class, a
# stop, a detour the straight-line distance knows nothing about.
#
# Keying legs by corridor exposed two that had been invisible when they were
# pooled per camera:
#
#     CAM_08 -> CAM_07     69,896 m  at 65 km/h
#     CAM_21 -> CAM_08    311,456 m  at 20 km/h   (15.5 hours)
#
# The second is not a journey at all — it is two different vehicles read as one
# plate, three hundred kilometres apart. The first is a real trip and still
# useless here, and together they formed the "highway cluster" that pulled
# CAM_08's focal length off and widened its interval.
#
# 5 km keeps the links that behave like one road (the useful CAM_11 -> CAM_08
# corridor is 2.4 km) and discards the rest.
MAX_LEG_METRES = 5000.0
# An Indian expressway limit is 100-120 km/h. A "leg" above this is two
# different vehicles read as one plate, not a fast car — an OCR collision. Two
# such rows (141 and 147 km/h) were on their own responsible for CAM_08's
# focal-length interval widening from 13% of the estimate to 153%.
MIN_LEG_KMH, MAX_LEG_KMH = 5.0, 110.0


@dataclass(frozen=True)
class Leg:
    plate: str
    from_cam: str
    to_cam: str
    metres: float
    seconds: float
    kmh: float
    t_from: str
    t_to: str

    @property
    def corridor(self) -> tuple:
        """Directed. See the module note on why direction is not collapsed."""
        return (self.from_cam, self.to_cam)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def extract_legs(db_path: Optional[Path] = None) -> list:
    """Every plausible leg in `journey_events`, newest schema-agnostic."""
    con = sqlite3.connect(str(db_path or DEFAULT_DB))
    try:
        rows = con.execute("""
            SELECT reid_id, camera_id, lat, lon, timestamp
            FROM journey_events
            WHERE lat IS NOT NULL AND lon IS NOT NULL AND timestamp IS NOT NULL
            ORDER BY reid_id, timestamp""").fetchall()
    finally:
        con.close()

    by_plate = defaultdict(list)
    for reid, cam, la, lo, ts in rows:
        by_plate[reid].append((ts, cam, la, lo))

    legs = []
    for plate, seq in by_plate.items():
        seq.sort()
        for (t1, c1, a1, o1), (t2, c2, a2, o2) in zip(seq, seq[1:]):
            if c1 == c2:
                continue
            m = haversine_m(a1, o1, a2, o2)
            if not (MIN_LEG_METRES <= m <= MAX_LEG_METRES):
                continue
            try:
                s = (datetime.fromisoformat(t2)
                     - datetime.fromisoformat(t1)).total_seconds()
            except Exception:
                continue
            if s <= 0:
                continue
            kmh = m / s * 3.6
            if not (MIN_LEG_KMH <= kmh <= MAX_LEG_KMH):
                continue
            legs.append(Leg(plate, c1, c2, m, s, kmh, t1, t2))
    return legs


def by_corridor(legs: Iterable[Leg]) -> dict:
    out = defaultdict(list)
    for l in legs:
        out[l.corridor].append(l)
    return dict(out)


def by_camera(legs: Iterable[Leg]) -> dict:
    """Legs touching each camera, grouped BY CORRIDOR within it.

    Deliberately not a flat list. A caller that wants one can flatten it, and
    in doing so has to notice how many corridors it just merged.
    """
    out = defaultdict(lambda: defaultdict(list))
    for l in legs:
        out[l.from_cam][l.corridor].append(l)
        out[l.to_cam][l.corridor].append(l)
    return {cam: dict(cor) for cam, cor in out.items()}


def _median(xs: list) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def camera_profile(cam: str, cam_legs: dict) -> dict:
    """What a camera's legs look like, and whether merging them is safe."""
    corridors = []
    for cor, ls in sorted(cam_legs.items(), key=lambda kv: -len(kv[1])):
        speeds = [l.kmh for l in ls]
        corridors.append({"corridor": "%s->%s" % cor, "n": len(ls),
                          "median_kmh": round(_median(speeds), 1),
                          "metres": round(_median([l.metres for l in ls]))})
    total = sum(c["n"] for c in corridors)
    meds = [c["median_kmh"] for c in corridors if c["n"] >= 3]
    # If the well-sampled corridors disagree by more than this, a single
    # per-camera median is not describing anything real.
    spread = (max(meds) / max(min(meds), 1e-6)) if len(meds) >= 2 else 1.0
    return {
        "camera": cam,
        "legs_total": total,
        "corridors": corridors,
        "n_corridors": len(corridors),
        "dominant_share": (round(corridors[0]["n"] / total, 2)
                           if corridors and total else 0.0),
        "corridor_speed_spread": round(spread, 2),
        "single_median_is_safe": bool(spread < 1.5),
    }


def profiles(db_path: Optional[Path] = None) -> dict:
    legs = extract_legs(db_path)
    per_cam = by_camera(legs)
    return {cam: camera_profile(cam, cl) for cam, cl in per_cam.items()}
