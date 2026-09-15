"""backend/scripts/validate_tier1_speeds.py — does a Tier 1 calibration agree
with a speed that never touched it?

THE ONLY INDEPENDENT YARDSTICK AVAILABLE
  A vehicle read at two cameras gives a speed from GPS positions and
  timestamps alone. It uses no homography, no vanishing point and no vehicle
  width, so it cannot inherit an error from the thing it is checking. That is
  what makes it worth comparing against.

  It is also a LOWER bound: straight-line distance between two cameras
  understates the road distance, and a vehicle that stopped at a light is
  averaged over the stop. A calibration reading somewhat FASTER than the leg
  figure is expected; one reading slower, or several times faster, is not.

WHAT IS COMPUTED
  For a camera with a `full` Tier 1 mapping, each tracked vehicle's ground
  positions come from the planar-VP model

      X = h (u - vp_x) / (v - vp_y)        lateral, metres
      Y = f h / (v - vp_y)                 along the road, metres

  and its speed from the displacement between the first and last points of the
  track divided by their time separation. The median over tracks is compared
  with the median of that camera's camera-to-camera legs.

Run:  python -m backend.scripts.validate_tier1_speeds [camera ...]
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.calibration_validation import (      # noqa: E402
    summarise, validate_calibration)

VP2 = ROOT / "output" / "camera_calibration" / "vp2_from_edges.json"
CALIB = ROOT / "output" / "camera_calibration" / "calibration.json"
GEOM = ROOT / "output" / "road_geometry"
DB = ROOT / "output" / "sentinel.db"

# The clips were written at this rate; road_geometry stores the raw frame
# index, so a frame difference only becomes a time difference through it.
NATIVE_FPS = 25.0
# A track must span enough time for a displacement to mean anything.
MIN_TRACK_SECONDS = 0.6
# Speeds outside this are tracker failures, not vehicles.
SPEED_SANE_MIN, SPEED_SANE_MAX = 2.0, 200.0


def haversine_m(a, b) -> float:
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def leg_speeds_by_camera() -> dict:
    """Camera-to-camera speeds, keyed by BOTH endpoint cameras."""
    con = sqlite3.connect(str(DB))
    rows = con.execute("""
        SELECT reid_id, camera_id, lat, lon, timestamp FROM journey_events
        WHERE lat IS NOT NULL AND lon IS NOT NULL AND timestamp IS NOT NULL
        ORDER BY reid_id, timestamp""").fetchall()
    con.close()

    by_v = defaultdict(list)
    for reid, cam, la, lo, ts in rows:
        by_v[reid].append((ts, cam, la, lo))

    out = defaultdict(list)
    for seq in by_v.values():
        seq.sort()
        for (t1, c1, a1, o1), (t2, c2, a2, o2) in zip(seq, seq[1:]):
            if c1 == c2:
                continue
            m = haversine_m((a1, o1), (a2, o2))
            try:
                s = (datetime.fromisoformat(t2) - datetime.fromisoformat(t1)).total_seconds()
            except Exception:
                continue
            if s <= 0 or m < 50:
                continue
            kmh = m / s * 3.6
            if 5.0 <= kmh <= 120.0:
                out[c1].append(kmh)
                out[c2].append(kmh)
    return out


def camera_speeds(cam: str, mapping: dict) -> list:
    p = GEOM / f"{cam}.json"
    if not p.is_file():
        return []
    tracks = json.loads(p.read_text(encoding="utf-8"))["tracks"]
    vpx, vpy = mapping["vp_x"], mapping["vp_y"]
    h, f = mapping["height_m"], mapping["focal_px"]

    speeds = []
    for pts in tracks.values():
        if len(pts) < 3:
            continue
        a, b = pts[0], pts[-1]
        dt = (b["frame"] - a["frame"]) / NATIVE_FPS
        if dt < MIN_TRACK_SECONDS:
            continue
        try:
            xy = []
            for q in (a, b):
                dv = q["v"] - vpy
                if dv <= 1.0:            # at or above the horizon
                    raise ValueError
                xy.append((h * (q["u"] - vpx) / dv, f * h / dv))
        except ValueError:
            continue
        d = math.hypot(xy[1][0] - xy[0][0], xy[1][1] - xy[0][1])
        kmh = d / dt * 3.6
        if SPEED_SANE_MIN <= kmh <= SPEED_SANE_MAX:
            speeds.append(kmh)
    return speeds


def main() -> int:
    fits = {r["camera"]: r for r in json.loads(VP2.read_text(encoding="utf-8"))}
    priors = {r["camera"]: r for r in json.loads(CALIB.read_text(encoding="utf-8"))}
    legs = leg_speeds_by_camera()
    cams = sys.argv[1:] or sorted(fits)

    results = []
    for cam in cams:
        fit = fits.get(cam)
        if not fit or fit.get("status") != "full":
            continue
        mapping = fit["mapping"]
        own = camera_speeds(cam, mapping)
        prior = priors.get(cam, {})
        params = {"height_m": mapping["height_m"],
                  "height_iqr_m": prior.get("height_iqr_m"),
                  "focal_px": mapping["focal_px"],
                  "frame_size": fit["frame_size"],
                  "vp_y": mapping["vp_y"],
                  "inliers": fit.get("inliers")}
        v = validate_calibration(cam, "vp_single_stream_edges", params,
                                 leg_speeds_kmh=legs.get(cam),
                                 camera_speeds_kmh=own)
        v.detail["own_speed_samples"] = len(own)
        if own:
            o = sorted(own)
            v.detail["own_speed_p25_median_p75"] = [round(o[len(o) // 4], 1),
                                                    round(o[len(o) // 2], 1),
                                                    round(o[3 * len(o) // 4], 1)]
        results.append(v)

    if not results:
        print("no camera has a `full` Tier 1 mapping to validate")
        return 0

    print(summarise(results))
    print()
    for v in results:
        print(f"{v.camera_id}:")
        for k in ("own_speed_samples", "own_speed_p25_median_p75",
                  "leg_median_kmh", "calibrated_median_kmh", "speed_ratio",
                  "legs_n", "fov_deg", "height_iqr_ratio"):
            if k in v.detail:
                print(f"   {k:<28} {v.detail[k]}")

    out = ROOT / "reports" / "tier1_validation_20260906.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([v.as_dict() for v in results], indent=1),
                   encoding="utf-8")
    print(f"\nwritten -> {out}")
    print("\nNothing is written to camera_calibrations by this script. Promotion"
          "\nis a separate, explicit step once a result has been read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
