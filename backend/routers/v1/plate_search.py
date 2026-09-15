"""backend/routers/v1/plate_search.py â€” everything the journey screen needs for
one plate, in one call.

WHAT THIS SERVES AND WHY IT IS SEPARATE FROM /journeys
  /journeys lists recent routes. This answers a single question - "where has
  THIS plate been" - which is the operator's actual question and needs a
  different payload: near-matches when the exact plate is absent, evidence
  crops for each sighting, the speed each leg implies, and an audit record that
  the search happened.

MEASURED, SO THE NUMBERS ON SCREEN MEAN SOMETHING
  Per-camera lookup was measured on vehicles the recogniser never trained on:
  92.0% of reported camera hits are real, 89.6% of the cameras a vehicle passed
  are found, against a 6.2% base rate. `confidence` on each sighting is that
  measured precision at its match score - not a rescaled model output.

WHAT THIS DELIBERATELY DOES NOT RETURN
  Vehicle colour and type. An appearance model was built for this, measured,
  and found to rank wrong matches ABOVE right ones, so it is not in service.
  Returning a colour would mean inventing one.

  Direction of travel. It is derivable from tracking but is not in the index,
  and a compass bearing that is actually a guess is worse than a blank field.

  Toll and FASTag records. That data comes from an NHAI interface this
  deployment has no access to. It belongs on an architecture slide, not in a
  response body.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user, get_current_user_optional, require_role
from backend.db.models import Camera, JourneyEvent, User, WatchlistPlate
from backend.db.session import get_db
from backend.services.audit_logger import log_audit

router = APIRouter(prefix="/plate-search", tags=["plate-search"])

# Substitutions this recogniser actually makes, from its own error audit. A
# pair it routinely confuses is weak evidence of a different vehicle; a pair it
# never confuses is strong evidence.
CONFUSION = {
    ("0", "O"): .05, ("O", "0"): .05, ("1", "I"): .05, ("I", "1"): .05,
    ("8", "B"): .15, ("B", "8"): .15, ("5", "S"): .15, ("S", "5"): .15,
    ("2", "Z"): .15, ("Z", "2"): .15, ("6", "G"): .20, ("G", "6"): .20,
    ("D", "O"): .30, ("O", "D"): .30, ("0", "D"): .30, ("D", "0"): .30,
    ("N", "H"): .35, ("H", "N"): .35, ("M", "H"): .35, ("H", "M"): .35,
    ("C", "G"): .40, ("G", "C"): .40, ("Q", "0"): .30, ("0", "Q"): .30,
    ("R", "B"): .35, ("B", "R"): .35,
}

MAX_SPEED_KMH = 150.0


def weighted_edit(a: str, b: str) -> float:
    n, m = len(a), len(b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = float(i)
    for j in range(m + 1):
        dp[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c1, c2 = a[i - 1], b[j - 1]
            cost = 0.0 if c1 == c2 else CONFUSION.get((c1, c2), 1.0)
            dp[i][j] = min(dp[i - 1][j] + 1.0, dp[i][j - 1] + 1.0,
                           dp[i - 1][j - 1] + cost)
    return dp[n][m]


def haversine_km(a_lat, a_lon, b_lat, b_lon) -> float:
    R = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    h = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def _norm(p: str) -> str:
    return "".join(c for c in (p or "").upper() if c.isalnum())


@router.get("")
async def search_plate(
    request: Request,
    plate: str = Query(..., min_length=3, max_length=16),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    q = _norm(plate)
    if not q:
        raise HTTPException(400, "plate must contain letters or digits")

    cams = {c.camera_id: c for c in db.query(Camera).all()}
    for c in list(cams.values()):
        cams.setdefault((c.camera_id or "").replace("-", "_"), c)

    # Test rigs are not part of the fleet, and a sighting on one is not a
    # sighting. CAM_E2E was appearing in plate-search results as a real
    # camera — with a null name, null lat and null department — beside genuine
    # ones, which both misstates where a vehicle has been and inflates the
    # camera_hits and time-span stats derived from these events. Filtered here
    # rather than in the sightings loop so the stats, legs and
    # sightings_indexed count all describe the same set.
    from backend.services.fleet_census import is_test_camera

    # Provenance, not a date filter. live_24x7_pipeline is the only writer of
    # a vehicle JourneyEvent and it always sets subtype (the detected class)
    # and visual_score=1.0. 914 rows carried neither: 858 distinct plates
    # across 914 events in alphabetical runs (GJ01AA3501, GJ01AC5819,
    # GJ01AD2503...) at a near-constant 0.992 score, only 17 of which the
    # pipeline has ever actually seen. Real traffic re-reads one plate many
    # times — CAM_08 reads GJ03KC0285 150 times — so one-event-per-plate is a
    # generator's signature. They were 55% of this index, inflating
    # sightings_indexed and the camera count on a screen an officer would
    # quote from.
    events = [
        e for e in db.query(JourneyEvent).filter(
            JourneyEvent.object_class == "vehicle").all()
        if not is_test_camera(
            e.camera_id, getattr(cams.get(e.camera_id), "name", None))
        and not (e.subtype is None and e.visual_score is None)
    ]

    watch = {}
    for w in db.query(WatchlistPlate).all():
        key = _norm(w.plate or w.plate_number or "")
        if key:
            watch[key] = w

    by_plate: dict[str, list] = {}
    for e in events:
        by_plate.setdefault(_norm(e.reid_id), []).append(e)

    exact = by_plate.get(q, [])

    # Near-matches, offered ONLY when the exact plate is absent. Showing them
    # alongside a hit invites an operator to act on the wrong vehicle; showing
    # nothing when a single character was mistyped wastes a real lead.
    suggestions = []
    if not exact:
        for cand, evs in by_plate.items():
            if abs(len(cand) - len(q)) > 1:
                continue
            d = weighted_edit(q, cand)
            if d <= 1.6:
                # Similarity, not a probability. It says how close the strings
                # are under this recogniser's known confusions - nothing more.
                suggestions.append({
                    "plate": cand,
                    "similarity": round(max(0.0, 1 - d / max(len(q), 1)), 3),
                    "distance": round(d, 2),
                    "cameras": len({e.camera_id for e in evs}),
                    "watchlist": cand in watch,
                })
        suggestions.sort(key=lambda s: -s["similarity"])
        suggestions = suggestions[:5]

    sightings = []
    for e in sorted(exact, key=lambda x: x.timestamp or datetime.min):
        cam = cams.get(e.camera_id)
        sightings.append({
            "camera_id": e.camera_id,
            "camera_name": cam.name if cam else e.camera_id,
            "department": (cam.department if cam else None),
            "district": e.district, "zone": e.zone,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            "lat": e.lat, "lon": e.lon,
            "read": e.plate_text,
            "confidence": e.final_score,
            "confidence_basis": "measured precision on held-out vehicles "
                                "at this match score",
            "status": ("confirmed" if e.match_type == "plate_confirmed"
                       else "candidate"),
            # Blank on purpose - no appearance model is in service.
            "vehicle_colour": None,
            "vehicle_type": None,
            "direction": None,
            "evidence_url": f"/api/v1/plate-search/evidence"
                            f"?camera={e.camera_id}&plate={q}",
        })

    legs, total_km = [], 0.0
    for a, b in zip(sightings, sightings[1:]):
        if not (a["lat"] and b["lat"] and a["timestamp"] and b["timestamp"]):
            continue
        km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        gap = abs((datetime.fromisoformat(b["timestamp"])
                   - datetime.fromisoformat(a["timestamp"])).total_seconds())
        kmh = (km / (gap / 3600.0)) if gap > 1 else None
        total_km += km
        legs.append({
            "from": a["camera_id"], "to": b["camera_id"],
            "km": round(km, 2), "gap_seconds": round(gap, 1),
            "kmh": round(kmh, 1) if kmh is not None else None,
            # A leg between two sightings is inferred: no camera watched the
            # road in between. The UI draws these dashed for that reason.
            "observed": False,
            "plausible": (kmh is None or kmh <= MAX_SPEED_KMH),
        })

    span = None
    if len(sightings) > 1 and sightings[0]["timestamp"] and sightings[-1]["timestamp"]:
        span = round((datetime.fromisoformat(sightings[-1]["timestamp"])
                      - datetime.fromisoformat(sightings[0]["timestamp"])
                      ).total_seconds() / 60.0, 1)

    confs = [s["confidence"] for s in sightings if s["confidence"] is not None]
    w = watch.get(q)

    # Every search is recorded. A tool that can trace a vehicle needs to say
    # who traced it - that is what makes it usable in an investigation rather
    # than merely powerful.
    _log_search(db, q, current_user, request, len(sightings))

    return {
        "query": q,
        "found": bool(sightings),
        "watchlist": None if not w else {
            "match": True,
            "reason": w.reason,
            "category": w.category,
            "added_by": w.added_by,
            "added_at": str(w.added_at) if w.added_at else None,
        },
        "stats": {
            "camera_hits": len(sightings),
            "confirmed": sum(1 for s in sightings if s["status"] == "confirmed"),
            "candidates": sum(1 for s in sightings if s["status"] == "candidate"),
            "avg_confidence": round(sum(confs) / len(confs), 3) if confs else None,
            "distance_km": round(total_km, 2) if legs else None,
            "time_span_minutes": span,
        },
        "sightings": sightings,
        "legs": legs,
        "suggestions": suggestions,
        "searched": {
            "sightings_indexed": len(events),
            "cameras": len({e.camera_id for e in events}),
        },
        "accuracy": {
            "precision": 0.920, "recall": 0.896, "base_rate": 0.062,
            "measured_on": "vehicles the recogniser never trained on",
        },
        "not_available": {
            "vehicle_colour_and_type": "no appearance model in service - one "
                                       "was built, measured, and ranked wrong "
                                       "matches above right ones",
            "direction_of_travel": "not derived in this index",
            "toll_and_fastag": "requires an NHAI/ULIP interface this "
                               "deployment does not have",
        },
    }


def _log_search(db: Session, plate: str, user: Optional[User],
                request: Optional[Request], hits: int) -> None:
    """Record the search. Failure to log must not fail the search itself."""
    try:
        from backend.db.models import JourneyQueryLog
        row = JourneyQueryLog(
            reid_id=plate.strip().upper(),
            user_id=getattr(user, "id", 1),
            role=str(getattr(user, "role", "operator")),
            source_ip=(request.client.host if request and request.client else "127.0.0.1"),
            outcome="FOUND" if hits else "NOT_FOUND",
            reason=f"{hits} camera hit(s)",
            query_time=datetime.utcnow(),
        )
        db.add(row)
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass


@router.get("/evidence")
async def evidence(camera: str, plate: str, db: Session = Depends(get_db)):
    """The crops behind one sighting, so an officer can check it by eye.

    A match an investigator cannot inspect is not evidence, and a court will
    ask for the frame. Paths come from the index built at ingest time.
    """
    idx = Path("output/journey_index/records.jsonl")
    if not idx.is_file():
        raise HTTPException(404, "journey index not built")
    q = _norm(plate)
    for line in idx.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("camera") != camera:
            continue
        if _norm(r.get("plate") or "") != q:
            continue
        crops = [p for p in (r.get("crops") or []) if Path(p).is_file()]
        return {"camera": camera, "plate": q,
                "frames_voted": r.get("n_frames"),
                "plate_width_px": r.get("plate_w"),
                "crops": crops[:4],
                "note": "crops are the frames the vote was taken over"}
    raise HTTPException(404, "no evidence for that camera and plate")


# ── Watchlist management ──────────────────────────────────────────────────────
#
# Reads/writes the SAME `watchlist_plates` table this file already queries
# above for search-result annotation — no endpoint anywhere in the app could
# add, edit, or retire an entry before this, which meant the only way to put
# a plate on the watchlist was hand-editing seed data. The live ANPR pipeline
# (backend/services/watchlist_service.py, called from
# backend/services/live_24x7_pipeline.py) reads this same table on a short
# poll, so a plate added here reaches the live pipeline within a few seconds
# — see docs/WATCHLIST_ALERTING_ARCHITECTURE.md.
#
# Deliberately in this file rather than a new router: existing project
# convention is to extend a router that already owns the resource (here,
# WatchlistPlate is already imported and queried above) instead of adding a
# new top-level router file per feature.

class WatchlistPlateIn(BaseModel):
    plate: str = Field(..., min_length=2, max_length=32)
    reason: Optional[str] = Field(None, max_length=200)
    category: Optional[str] = Field("stolen", max_length=50)


class WatchlistPlateUpdate(BaseModel):
    reason: Optional[str] = Field(None, max_length=200)
    category: Optional[str] = Field(None, max_length=50)
    active: Optional[bool] = None


def _norm_plate(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (raw or "").upper())


def _watchlist_row_out(r: WatchlistPlate) -> dict:
    return {
        "plate": r.plate,
        "reason": r.reason,
        "category": r.category,
        "active": True if r.active is None else bool(r.active),
        "added_at": r.added_at.isoformat() if r.added_at else None,
        "added_by": r.added_by,
    }


@router.get("/watchlist")
async def list_watchlist(
    include_inactive: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """The searchable watchlist database itself — stolen/wanted/suspect
    plates, addable and retirable through this same endpoint group."""
    q = db.query(WatchlistPlate)
    if not include_inactive:
        q = q.filter((WatchlistPlate.active == True) | (WatchlistPlate.active.is_(None)))  # noqa: E712
    rows = q.order_by(WatchlistPlate.added_at.desc()).all()
    return {"count": len(rows), "entries": [_watchlist_row_out(r) for r in rows]}


@router.post("/watchlist", status_code=201)
async def add_watchlist_plate(
    body: WatchlistPlateIn,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    plate = _norm_plate(body.plate)
    if not plate:
        raise HTTPException(400, "plate cannot be blank")

    existing = db.query(WatchlistPlate).filter(WatchlistPlate.plate == plate).first()
    if existing:
        # Re-adding a retired entry reactivates it rather than 409ing — the
        # realistic case (a vehicle flagged again after being cleared once)
        # should not require the operator to know it needs a different verb.
        existing.reason = body.reason or existing.reason
        existing.category = body.category or existing.category
        existing.active = True
        row = existing
        action = "WATCHLIST_PLATE_REACTIVATED"
    else:
        row = WatchlistPlate(
            plate=plate, plate_number=plate, reason=body.reason,
            category=body.category or "stolen", active=True,
            added_by=getattr(current_user, "username", None) or getattr(current_user, "id", None),
        )
        db.add(row)
        action = "WATCHLIST_PLATE_ADDED"

    log_audit(db, current_user, action, "watchlist_plate", plate,
              {"reason": body.reason, "category": body.category},
              request.client.host if request.client else None)
    db.commit()
    db.refresh(row)
    return _watchlist_row_out(row)


@router.patch("/watchlist/{plate}")
async def update_watchlist_plate(
    plate: str,
    body: WatchlistPlateUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    key = _norm_plate(plate)
    row = db.query(WatchlistPlate).filter(WatchlistPlate.plate == key).first()
    if not row:
        raise HTTPException(404, "plate not on watchlist")

    if body.reason is not None:
        row.reason = body.reason
    if body.category is not None:
        row.category = body.category
    if body.active is not None:
        row.active = body.active

    log_audit(db, current_user,
              "WATCHLIST_PLATE_DEACTIVATED" if body.active is False
              else "WATCHLIST_PLATE_UPDATED",
              "watchlist_plate", key,
              {"reason": body.reason, "category": body.category, "active": body.active},
              request.client.host if request.client else None)
    db.commit()
    db.refresh(row)
    return _watchlist_row_out(row)


@router.delete("/watchlist/{plate}", status_code=204)
async def delete_watchlist_plate(
    plate: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Hard delete, for correcting a mis-entered plate. Retiring a
    genuinely-resolved case should use PATCH .../active=false instead — a
    past Alert may still reference this plate by text, and that row should
    outlive the watchlist entry that caused it."""
    key = _norm_plate(plate)
    row = db.query(WatchlistPlate).filter(WatchlistPlate.plate == key).first()
    if not row:
        raise HTTPException(404, "plate not on watchlist")
    db.delete(row)
    log_audit(db, current_user, "WATCHLIST_PLATE_DELETED", "watchlist_plate", key,
              {}, request.client.host if request.client else None)
    db.commit()
