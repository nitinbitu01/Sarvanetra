"""backend/scripts/journey_discover.py — find every vehicle the index saw on more
than one camera, and check each link against physics.

WHAT THIS ANSWERS
  The search is measured: 93.8% recall at rank one, 91.1% precision with a
  threshold. Neither figure says how many multi-camera journeys actually EXIST
  in this footage, and that decides what can be demonstrated. Clips are
  synchronised ten-minute windows, so a vehicle has to cross two camera views
  inside one window to produce a journey at all.

WHY LINKS ARE CHECKED AND NOT ASSUMED
  Two different vehicles can be read as the same string - the recogniser is
  93% accurate per character, so collisions happen. Every candidate link is
  therefore put through the speed check the camera GPS makes possible: a pair
  of sightings implies a distance and a time, and above roughly 150 km/h no
  vehicle produced both.

  Links that fail are reported rather than dropped. A rejected link is
  evidence about the system's discrimination, and hiding it would make the
  output look cleaner than it is.

READ THE OUTPUT AS A CENSUS, NOT AS ACCURACY
  These journeys are discovered by the system, so a wrong one appears here as
  confidently as a right one. The verified column is what can be trusted
  without further checking: those are links where a human had already read the
  same plate on both cameras.

USAGE
  python -m backend.scripts.journey_discover --min-score -10
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from backend.scripts.plate_beam_decode import complete
from backend.services.journey_search import JourneySearch, haversine_km

REAL = Path("data/plate_real")
INDEX = Path("output/journey_index")
OUT = Path("output/journeys.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--max-speed", type=float, default=150.0)
    ap.add_argument("--min-frames", type=int, default=2,
                    help="Frames a sighting must be built from. A one-frame "
                         "read is what the multi-frame vote exists to avoid "
                         "trusting.")
    ap.add_argument("--link-score", type=float, default=-8.0,
                    help="CTC log-likelihood floor for calling two sightings "
                         "the same vehicle. Measured on held-out queries: "
                         "present plates score around -1.2 at the median, "
                         "absent ones around -42, and -8 was the point where "
                         "precision reached 96.8%%.")
    args = ap.parse_args()

    js = JourneySearch(Path(args.index))
    recs = js.records
    print(f"indexed vehicles : {len(recs)}")

    truth = {}
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text"):
            truth[(v["camera"], v["track"])] = v["text"]

    # Link by CTC likelihood, NOT by string equality.
    #
    # Grouping on the decoded string is the obvious approach and it is close to
    # useless here. The recogniser is 59% exact, so one vehicle seen twice
    # produces the same string on both sides only about 0.59 x 0.59 = 35% of
    # the time; with detection succeeding on roughly half of crossings, exact
    # grouping recovers around a sixth of the journeys that exist. Running it
    # that way found one.
    #
    # The index stores log-probabilities precisely so this comparison can be
    # made properly: take each vehicle's reading as a query and ask how much
    # probability every OTHER vehicle's pixels assign to it. A vehicle whose
    # own reading was wrong still scores highly for its true plate, so the two
    # sides link even when neither was read exactly right.
    # Only well-formed plates may anchor a link. Without this the run produced
    # journeys for GJ0006, GJ00 and GJ030005 - degenerate readings, mostly
    # zeros, that are short and generic enough to score highly against a great
    # many tracks. Garbage matched garbage and the result looked like nine
    # extra journeys. A reading that does not fit any Indian plate pattern is
    # not a vehicle identity and cannot link two sightings.
    def linkable(r) -> bool:
        """A reading solid enough to assert a vehicle's identity from.

        Two conditions, and the second is not redundant. The plate grammar
        admits an older short series of the form AA-DD-DDDD, which is a real
        format - and also exactly what a degenerate reading like GJ030005
        looks like. Requiring nine characters keeps the modern format, which
        is what these cameras actually see, and drops the readings that only
        pass because the permissive pattern exists.
        """
        p = r.get("plate") or ""
        if len(p) < 9 or not complete(p):
            return False
        # RTO district codes are issued from 01 upwards; 00 does not exist.
        # Checked against all 417 human-read plates: none uses 00, so this
        # discards readings and never a real plate. It is what remained of the
        # degenerate links after the format check - GJ00H0008 parses cleanly
        # as GJ-00-H-0008 and is not a plate.
        if p[:2].isalpha() and p[2:4] == "00":
            return False
        return r.get("n_frames", 0) >= args.min_frames

    cand = [r for r in recs if linkable(r)]
    ok_idx = {i for i, r in enumerate(recs) if linkable(r)}
    print(f"candidate sightings : {len(cand)} "
          f"(of {len(recs)}, after requiring a well-formed plate)")

    idx_of = {id(r): i for i, r in enumerate(recs)}
    groups: dict[int, list] = {}
    assigned: dict[int, int] = {}
    for n, r in enumerate(cand, 1):
        i = idx_of[id(r)]
        if i in assigned:
            continue
        scores = js.ctc_scores(r["plate"])
        gid = len(groups)
        members = [i]
        for j in np.argsort(-scores):
            j = int(j)
            if j == i or j in assigned:
                continue
            if scores[j] < args.link_score:
                break
            if recs[j]["camera"] == r["camera"]:
                continue                     # same camera is not a journey leg
            # Members are NOT required to be well-formed. Requiring it on both
            # sides defeats the point of scoring by likelihood: the case this
            # exists to catch is a vehicle read cleanly on one camera and
            # poorly on the other, and demanding a clean read at both ends
            # loses exactly those links. Applying that filter here dropped
            # GJ03CR1031, a human-verified crossing at a plausible 29.6 km/h.
            members.append(j)
        if len(members) > 1:
            # The anchor names the group. Naming it after whichever member had
            # the most frames is what produced journeys called GJ00 and
            # GJ0006: a degenerate reading with many frames outvoted the
            # well-formed plate that opened the group.
            groups[gid] = (i, members)
            for m in members:
                assigned[m] = gid
        if n % 200 == 0:
            print(f"  {n}/{len(cand)} scanned, {len(groups)} groups",
                  flush=True)

    multi = {}
    for gid, (anchor, members) in groups.items():
        multi[recs[anchor]["plate"]] = [recs[m] for m in members]
    print(f"\nplates linked across cameras by CTC : {len(multi)}")

    journeys, rejected = [], []
    for plate, sights in multi.items():
        sights.sort(key=lambda r: r.get("timestamp") or "")
        legs, path = [], [sights[0]]
        ok = True
        for prev, cur in zip(sights, sights[1:]):
            if prev["camera"] == cur["camera"]:
                path.append(cur)
                continue
            if not (prev.get("timestamp") and cur.get("timestamp")
                    and prev.get("lat") and cur.get("lat")):
                path.append(cur)
                continue
            t0 = datetime.fromisoformat(prev["timestamp"])
            t1 = datetime.fromisoformat(cur["timestamp"])
            gap = abs((t1 - t0).total_seconds())
            km = haversine_km(prev["lat"], prev["lon"], cur["lat"], cur["lon"])
            kmh = km / (gap / 3600.0) if gap > 1 else float("inf")
            leg = {"from": prev["camera"], "to": cur["camera"],
                   "km": round(km, 2), "gap_s": round(gap, 1),
                   "kmh": round(kmh, 1) if kmh != float("inf") else None}
            if kmh > args.max_speed:
                rejected.append({"plate": plate, **leg,
                                 "why": "speed implausible"})
                ok = False
                continue
            legs.append(leg)
            path.append(cur)

        cams = [x["camera"] for x in path]
        if len(set(cams)) < 2:
            continue
        verified = sum(1 for x in path
                       if truth.get((x["camera"], x["track"])) == plate)
        journeys.append({
            "plate": plate,
            "cameras": list(dict.fromkeys(cams)),
            "sightings": [{
                "camera": x["camera"], "camera_name": x.get("camera_name"),
                "department": x.get("department"),
                "timestamp": x.get("timestamp"),
                "lat": x.get("lat"), "lon": x.get("lon"),
                "frames": x.get("n_frames"), "plate_w": x.get("plate_w"),
                "crops": x.get("crops", [])[:2],
                "human_verified": truth.get((x["camera"], x["track"])) == plate,
            } for x in path],
            "legs": legs,
            "verified_sightings": verified,
            "all_legs_plausible": ok,
        })

    journeys.sort(key=lambda j: (-j["verified_sightings"],
                                 -len(j["cameras"])))
    OUT.write_text(json.dumps({"journeys": journeys,
                               "rejected_legs": rejected}, indent=1),
                   encoding="utf-8")

    print("\n" + "=" * 66)
    print("MULTI-CAMERA JOURNEYS DISCOVERED")
    print("=" * 66)
    print(f"journeys            : {len(journeys)}")
    print(f"legs rejected on speed : {len(rejected)}")
    print(f"with a human-verified sighting : "
          f"{sum(1 for j in journeys if j['verified_sightings'])}")
    print(f"with TWO human-verified sightings : "
          f"{sum(1 for j in journeys if j['verified_sightings'] >= 2)}")

    print(f"\n{'plate':<12} {'cameras':<24} {'legs':>5} {'verified':>9}")
    print("-" * 66)
    for j in journeys[:15]:
        print(f"{j['plate']:<12} {','.join(j['cameras'])[:23]:<24} "
              f"{len(j['legs']):>5} {j['verified_sightings']:>9}")

    if journeys:
        j = journeys[0]
        print(f"\nexample - {j['plate']}")
        for s in j["sightings"]:
            v = "  [human-verified]" if s["human_verified"] else ""
            print(f"  {s['timestamp']}  {s['camera']:<8} "
                  f"{(s.get('camera_name') or '')[:32]:<34}{v}")
        for leg in j["legs"]:
            print(f"    {leg['from']} -> {leg['to']}: {leg['km']} km in "
                  f"{leg['gap_s']}s = {leg['kmh']} km/h")

    if rejected:
        print(f"\nrejected legs (speed implausible), first 8:")
        for r in rejected[:8]:
            print(f"  {r['plate']:<12} {r['from']}->{r['to']} "
                  f"{r['km']} km in {r['gap_s']}s")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
