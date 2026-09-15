"""backend/services/appearance_bridge.py — continue a plate trajectory across
cameras where the plate could not be read.

THE PROBLEM THIS SOLVES
  A journey is built from plate reads, so it stops at the first camera that
  could not read one. On this fleet that is most cameras most of the time: the
  committed operating point covers 61.4% of vehicles and only above 80 px of
  plate width. A vehicle that passed six cameras and was read at two produces a
  two-stop journey, and the four cameras in between look like they never saw
  it.

  The appearance channel can fill those gaps. The vehicle's own appearance —
  taken from the cameras that DID read its plate — is matched against every
  track the intermediate cameras recorded but could not identify.

THE THRESHOLD IS MEASURED, NOT CHOSEN
  This is where an appearance system usually goes wrong, and this project has
  the scar: a threshold picked by eye grouped vehicles by colour. Here the
  fleet supplies its own labels. Two tracks with DIFFERENT confirmed plates are
  certainly different vehicles; two with the SAME confirmed plate are certainly
  the same one. So:

      negatives = similarity between tracks with different plates
      positives = similarity between tracks with the same plate

  The threshold is a high quantile of the negatives, chosen for a stated
  false-accept rate, and `measure()` reports precision and recall on those
  labels. Nothing here is tuned by hand and nothing is asserted without a
  number behind it.

PHYSICS STILL APPLIES
  A candidate is rejected if reaching it would need an impossible speed. A
  vehicle cannot be two places at once, and appearance is not permitted to
  override that — the same rule the clone detector enforces.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
from sqlalchemy.orm import Session

from backend.db.models import Camera, TrackAppearance
from backend.services.road_router import get_road_router

logger = logging.getLogger(__name__)

# No road vehicle sustains this between two cameras. Above it the pair is not
# one vehicle, whatever it looks like.
ABSOLUTE_MAX_KMH = 200.0
# How far outside the journey's own span a bridging sighting may sit.
WINDOW_MARGIN_MIN = 45.0
# Target false-accept rate the threshold is set for.
DEFAULT_FAR = 0.001
# Below this many labelled negatives the distribution is not worth a quantile.
MIN_NEGATIVES = 200


@dataclass
class BridgedStop:
    camera_id: str
    timestamp: datetime
    similarity: float
    track_id: int
    vehicle_class: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    implied_kmh: Optional[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp_utc": self.timestamp.isoformat(),
            "similarity": round(float(self.similarity), 4),
            "track_id": int(self.track_id),
            "subtype": self.vehicle_class,
            "lat": self.lat, "lon": self.lon,
            "implied_kmh": (round(self.implied_kmh, 1)
                            if self.implied_kmh is not None else None),
            "match_type": "appearance_bridged",
        }


def _vectors(rows) -> np.ndarray:
    out = []
    for r in rows:
        e = r.embedding
        if not e:
            continue
        v = np.asarray(e, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            out.append(v / n)
    return np.vstack(out) if out else np.zeros((0, 512), dtype=np.float32)


def calibrate(db: Session, far: float = DEFAULT_FAR,
              sample_cap: int = 900) -> dict[str, Any]:
    """Derive the match threshold from the fleet's own plate labels.

    Only CROSS-CAMERA pairs count as negatives: two tracks on one camera may be
    the same vehicle fragmented by the tracker, which would poison the negative
    set with genuine matches.
    """
    rows = (db.query(TrackAppearance)
            .filter(TrackAppearance.plate_text.isnot(None))
            .filter(TrackAppearance.plate_text != "")
            .order_by(TrackAppearance.last_seen.desc())
            .limit(sample_cap).all())
    rows = [r for r in rows if r.embedding]
    if len(rows) < 8:
        return {"ok": False,
                "reason": f"only {len(rows)} plated appearance rows; need at "
                          f"least 8 before a threshold means anything",
                "plated_rows": len(rows)}

    V = _vectors(rows)
    plates = np.array([r.plate_text for r in rows])
    cams = np.array([r.camera_id for r in rows])
    S = V @ V.T
    iu = np.triu_indices(len(rows), k=1)
    same_plate = plates[iu[0]] == plates[iu[1]]
    diff_cam = cams[iu[0]] != cams[iu[1]]
    sims = S[iu]

    neg = sims[(~same_plate) & diff_cam]
    pos = sims[same_plate & diff_cam]
    if neg.size < MIN_NEGATIVES:
        return {"ok": False,
                "reason": f"only {int(neg.size)} cross-camera negative pairs; "
                          f"need {MIN_NEGATIVES}",
                "plated_rows": len(rows), "negatives": int(neg.size),
                "positives": int(pos.size)}

    thr = float(np.quantile(neg, 1.0 - far))
    out = {
        "ok": True,
        "threshold": thr,
        "far_target": far,
        "plated_rows": len(rows),
        "negatives": int(neg.size),
        "positives": int(pos.size),
        "negative_mean": float(neg.mean()),
        "negative_p99": float(np.quantile(neg, 0.99)),
        "negative_max": float(neg.max()),
    }
    if pos.size:
        tp = int((pos >= thr).sum())
        fp = int((neg >= thr).sum())
        out.update({
            "positive_mean": float(pos.mean()),
            "positive_min": float(pos.min()),
            "recall": tp / float(pos.size),
            "precision": (tp / float(tp + fp)) if (tp + fp) else None,
            "true_positives": tp, "false_positives": fp,
        })
    return out


def measure(db: Session, far: float = DEFAULT_FAR) -> dict[str, Any]:
    """What calibrate() found, phrased for display. Never invents a number."""
    c = calibrate(db, far=far)
    if not c.get("ok"):
        c["headline"] = ("Not enough labelled appearance data yet to state an "
                         "accuracy. " + c.get("reason", ""))
        return c
    if c.get("positives", 0) == 0:
        c["headline"] = (
            f"Threshold {c['threshold']:.4f}, set so that {c['negatives']:,} "
            f"pairs of DIFFERENT vehicles clear it {far*100:.2f}% of the time. "
            f"No two cameras have yet read the same plate, so recall is not "
            f"measurable — precision against known-different vehicles is.")
        return c
    c["headline"] = (
        f"Threshold {c['threshold']:.4f} from {c['negatives']:,} known-different "
        f"pairs at a {far*100:.2f}% false-accept target. On {c['positives']} "
        f"known-same pairs it recovers {c['recall']*100:.1f}%.")
    return c


def bridge(db: Session, plate: str, stops: list[dict], *,
           far: float = DEFAULT_FAR, max_candidates: int = 400
           ) -> dict[str, Any]:
    """Find sightings of this vehicle on cameras that never read its plate.

    `stops` are the plate-confirmed stops already known, each needing at least
    `camera_id` and `timestamp_utc`.
    """
    cal = calibrate(db, far=far)
    result: dict[str, Any] = {"calibration": cal, "bridged": [],
                              "considered": 0, "rejected_physics": 0}
    if not cal.get("ok"):
        result["note"] = cal.get("reason")
        return result

    known = (db.query(TrackAppearance)
             .filter(TrackAppearance.plate_text == plate).all())
    known = [r for r in known if r.embedding]
    if not known:
        result["note"] = (f"No stored appearance for {plate} — its sightings "
                          f"predate appearance capture, so there is nothing to "
                          f"match with.")
        return result

    q = _vectors(known).mean(axis=0)
    n = float(np.linalg.norm(q))
    if n <= 1e-9:
        result["note"] = "Stored appearance for this vehicle is degenerate."
        return result
    q = q / n

    times = []
    for s in stops:
        try:
            times.append(datetime.fromisoformat(str(s["timestamp_utc"])))
        except Exception:                                           # noqa: BLE001
            continue
    if not times:
        result["note"] = "Journey has no usable timestamps."
        return result
    lo = min(times) - timedelta(minutes=WINDOW_MARGIN_MIN)
    hi = max(times) + timedelta(minutes=WINDOW_MARGIN_MIN)
    seen_cams = {s.get("camera_id") for s in stops}

    cands = (db.query(TrackAppearance)
             .filter(TrackAppearance.last_seen >= lo)
             .filter(TrackAppearance.last_seen <= hi)
             .filter(~TrackAppearance.camera_id.in_(seen_cams or ["_"]))
             .limit(max_candidates * 4).all())
    # Only tracks the recogniser could NOT identify. A track that read a
    # different plate is a different vehicle by evidence stronger than
    # appearance, and must not be overridden.
    cands = [r for r in cands if r.embedding and not (r.plate_text or "").strip()]
    result["considered"] = len(cands)
    if not cands:
        result["note"] = ("No unidentified tracks on other cameras inside the "
                          "journey's time window.")
        return result

    V = _vectors(cands)
    sims = V @ q
    thr = cal["threshold"]

    cams = {c.id: c for c in db.query(Camera).all()}
    router = get_road_router()

    # Anchor for the physics test: the plate-confirmed stop nearest in time.
    anchors = []
    for s in stops:
        try:
            t = datetime.fromisoformat(str(s["timestamp_utc"]))
        except Exception:                                           # noqa: BLE001
            continue
        c = cams.get(s.get("camera_id"))
        if c is not None and c.lat is not None and c.lon is not None:
            anchors.append((t, float(c.lat), float(c.lon)))

    picked: list[BridgedStop] = []
    for i, r in enumerate(cands):
        s = float(sims[i])
        if s < thr:
            continue
        cam = cams.get(r.camera_id)
        lat = float(cam.lat) if cam is not None and cam.lat is not None else None
        lon = float(cam.lon) if cam is not None and cam.lon is not None else None

        implied = None
        if lat is not None and lon is not None and anchors:
            t_anchor, alat, alon = min(
                anchors, key=lambda a: abs((a[0] - r.last_seen).total_seconds()))
            dt_h = abs((r.last_seen - t_anchor).total_seconds()) / 3600.0
            leg = router.route(alat, alon, lat, lon)
            if dt_h > 1e-6:
                implied = leg.distance_km / dt_h
                if implied > ABSOLUTE_MAX_KMH:
                    # Looks like it, cannot be it.
                    result["rejected_physics"] += 1
                    continue
        picked.append(BridgedStop(
            camera_id=r.camera_id, timestamp=r.last_seen, similarity=s,
            track_id=r.track_id, vehicle_class=r.vehicle_class,
            lat=lat, lon=lon, implied_kmh=implied))

    # One bridged stop per camera: the strongest. A camera saw the vehicle once
    # on this journey, and listing three candidate tracks from it as three
    # separate visits would misrepresent the route.
    best_per_cam: dict[str, BridgedStop] = {}
    for b in picked:
        cur = best_per_cam.get(b.camera_id)
        if cur is None or b.similarity > cur.similarity:
            best_per_cam[b.camera_id] = b

    result["bridged"] = [b.as_dict() for b in
                         sorted(best_per_cam.values(), key=lambda x: x.timestamp)]
    result["threshold"] = thr
    return result
