"""Propose a road class per camera, for a human to confirm.

Speed limits depend on road class, and the cameras table has no road column.
Rather than invent one, this writes config/camera_road_class.json with a
SUGGESTION per camera and `confirmed: false` against each. Nothing is treated
as known until someone sets confirmed to true — speed_limits.load_road_classes
ignores unconfirmed entries entirely, so an unreviewed file changes no flag.

The suggestions come from two weak signals that are already in the database:

  camera name        "Circle", "Chowk", "Teen Rasta", "Cross Road", "Gate"
                     are junction words and indicate urban roads; "Highway",
                     "Bypass", "Toll", "NH" indicate a highway
  owning department  highway_patrol owns highway cameras, municipality and
                     traffic_police own city ones

Both are inference about the camera, not measurement of the road, which is
exactly why the output is a suggestion. A wrong road class produces a
confident overspeed flag against a driver who was obeying the limit, and that
is worse than not flagging at all.

Run:  python -m backend.scripts.suggest_camera_road_class
      python -m backend.scripts.suggest_camera_road_class --confirm-all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.speed_limits import (                        # noqa: E402
    EXPRESSWAY, NH_DIVIDED, NH_UNDIVIDED, STATE_HIGHWAY, URBAN,
    ROAD_CLASS_CONFIG,
)

URBAN_WORDS = ("circle", "chowk", "teen rasta", "cross road", "crossroad",
               "gate", "market", "square", "junction", "bridge", "flyover",
               "station", "temple", "road")
HIGHWAY_WORDS = ("highway", "bypass", "toll", " nh", "nh-", "expressway",
                 "corridor", "ring road")


def suggest(name: str, department: str) -> tuple[str, str]:
    n = (name or "").lower()
    d = (department or "").lower()
    for word in HIGHWAY_WORDS:
        if word in n:
            if "expressway" in n:
                return EXPRESSWAY, f"name contains '{word.strip()}'"
            return NH_DIVIDED, f"name contains '{word.strip()}'"
    if d == "highway_patrol":
        return NH_DIVIDED, "owned by highway_patrol"
    for word in URBAN_WORDS:
        if word in n:
            return URBAN, f"name contains '{word.strip()}'"
    if d in ("municipality", "traffic_police"):
        return URBAN, f"owned by {d}"
    return STATE_HIGHWAY, "no strong signal — defaulted, please review"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-all", action="store_true",
                    help="Mark every suggestion confirmed. Only for a "
                         "deployment where someone has actually checked them.")
    args = ap.parse_args()

    from backend.db.session import SessionLocal
    from backend.db.models import Camera

    db = SessionLocal()
    cams = db.query(Camera).filter(Camera.is_deleted == False).all()  # noqa: E712

    existing = {}
    if ROAD_CLASS_CONFIG.is_file():
        try:
            existing = (json.loads(ROAD_CLASS_CONFIG.read_text(encoding="utf-8"))
                        .get("cameras") or {})
        except Exception:                                          # noqa: BLE001
            existing = {}

    out = {}
    print(f"{'camera':<8}{'name':<34}{'department':<16}{'suggested':<18}why")
    print("-" * 96)
    for c in cams:
        key = str(c.id)
        prev = existing.get(key) or {}
        # A confirmed entry is never overwritten — the whole point is that a
        # human decision outranks this script's guess.
        if prev.get("confirmed") is True:
            out[key] = prev
            print(f"{key:<8}{(c.name or '')[:33]:<34}"
                  f"{(c.department or '')[:15]:<16}"
                  f"{prev.get('road_class', ''):<18}(confirmed, kept)")
            continue
        rc, why = suggest(c.name, c.department)
        out[key] = {
            "camera_name": c.name,
            "department": c.department,
            "road_class": rc,
            "confirmed": bool(args.confirm_all),
            "suggested_because": why,
        }
        print(f"{key:<8}{(c.name or '')[:33]:<34}"
              f"{(c.department or '')[:15]:<16}{rc:<18}{why}")
    db.close()

    ROAD_CLASS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    ROAD_CLASS_CONFIG.write_text(json.dumps({
        "_comment": (
            "road_class per camera id. Speed-limit flags use an entry ONLY "
            "when confirmed is true; until then the camera counts as unknown "
            "and only egregious speeds are flagged. Valid values: "
            "EXPRESSWAY, NH_4LANE_DIVIDED, NH_2LANE, STATE_HIGHWAY, URBAN."
        ),
        "cameras": out,
    }, indent=1), encoding="utf-8")

    n_conf = sum(1 for v in out.values() if v.get("confirmed"))
    print(f"\nwrote {ROAD_CLASS_CONFIG}")
    print(f"{len(out)} cameras, {n_conf} confirmed, "
          f"{len(out) - n_conf} awaiting review.")
    if not n_conf:
        print("\nNothing is confirmed, so no camera-specific limit applies "
              "yet and\nonly speeds above the national maximum are flagged. "
              "Set confirmed:true\nfor the cameras you have checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
