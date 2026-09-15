"""
backend/services/zone_incident_manager.py — Zone incident clustering (Day 11).

Called as the second half of the shared hook in each of the four alert-firing
modules, immediately after apply_dedup():

    primary = apply_dedup(alert, db)
    await check_zone_incident(primary, db)
    db.commit()

WHAT IT DOES
  After any of the four modules fires an alert, checks whether ZONE_INCIDENT_MIN_ALERTS
  (default 5) or more Alert rows now exist within ZONE_INCIDENT_RADIUS_METERS
  (default 100m, haversine) of the new alert's camera in the last
  ZONE_INCIDENT_WINDOW_SEC (default 120s). If so:

    - If no open ZoneIncident already covers that area (center within 100m,
      closed_at IS NULL): create one, referencing all nearby alert IDs.
    - If an open ZoneIncident exists: append the new alert id to its alert_ids
      list and update the row.

  "Events" means fired Alert rows — not raw per-frame detections (see the
  Day 11 prompt's clarifying note: this counts DB rows, not pipeline frames).

WHAT IT DOES NOT DO
  - Delete or hide any Alert row. The alert_ids JSON list is a reference list,
    not a replacement for the underlying rows.
  - Cluster across different global_ids or alert types (zone incident is a
    pure spatial+temporal cluster of ANY mix of alert types — a crowd alert,
    a loitering alert, and a watchlist match all within 100m in 2 min is a
    genuine zone incident regardless of what triggered each).
  - Move or merge the constituent alerts themselves. They remain independent
    rows with their own lifecycle_status.

GEOGRAPHIC CLUSTERING
  Uses real haversine distance against Camera.gps_lat / Camera.gps_lon (confirmed
  populated with real Ahmedabad-area coordinates in Step 0). The `zone` column
  on Camera is carried as a human-readable label onto the incident but is NOT
  the clustering key.

  The duplicate "Gandhinagar Hwy Cam-4" rows (Step 0, item 3) — four camera
  rows with different camera_id values but the same GPS coordinates — are
  correctly handled: haversine collapses them to a single point. Activity at
  that location recorded by any of the four camera_id variants clusters as one
  location, not four separate near-misses.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

ACTION_ZONE_INCIDENT_OPENED = "ZONE_INCIDENT_OPENED"
ACTION_ZONE_INCIDENT_UPDATED = "ZONE_INCIDENT_UPDATED"


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two GPS coordinates.

    Uses the Haversine formula. Accurate to within ~0.5% over distances of a
    few kilometres — more than sufficient for the 100m clustering threshold.
    """
    R = 6_371_000.0  # Earth radius in metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Main entry point ──────────────────────────────────────────────────────────

async def check_zone_incident(primary_alert: Any, db: Session) -> Any | None:
    """Check whether a zone incident should be opened or updated.

    Args:
        primary_alert: The winning Alert from apply_dedup() (has a populated id).
                       If the new alert was merged away (loser), the caller
                       should pass the winner here — we cluster by the winner's
                       camera location, which is the higher-quality signal.
        db:            Active session (caller commits after this returns).

    Returns:
        The ZoneIncident row (new or existing), or None if threshold not met
        or if the camera has no GPS coordinates.
    """
    from backend.core.config import settings
    from backend.db.models import Alert, Camera, ZoneIncident

    # ── 1. Look up the new alert's camera GPS coordinates ────────────────────
    if not primary_alert.camera_id:
        return None

    camera = db.query(Camera).filter(Camera.id == primary_alert.camera_id).first()
    if not camera or camera.gps_lat is None or camera.gps_lon is None:
        logger.debug(
            "check_zone_incident: camera_id=%s has no GPS coordinates — skipping",
            primary_alert.camera_id,
        )
        return None

    ref_lat = camera.gps_lat
    ref_lon = camera.gps_lon
    zone_label = camera.zone or "Unknown zone"

    # ── 2. Query all non-deleted alerts from the last ZONE_INCIDENT_WINDOW_SEC ──
    window_start = datetime.utcnow() - timedelta(seconds=settings.ZONE_INCIDENT_WINDOW_SEC)
    recent_alerts = (
        db.query(Alert, Camera)
        .join(Camera, Alert.camera_id == Camera.id, isouter=True)
        .filter(
            Alert.is_deleted == False,
            Alert.created_at >= window_start,
        )
        .all()
    )

    # ── 3. Filter to those within ZONE_INCIDENT_RADIUS_METERS ────────────────
    # Alerts whose camera has no GPS coordinates are excluded (skip silently —
    # they can't participate in geographic clustering).
    nearby_alert_ids: list[int] = []
    for alert_row, cam_row in recent_alerts:
        if cam_row is None or cam_row.gps_lat is None or cam_row.gps_lon is None:
            continue
        dist = haversine_meters(ref_lat, ref_lon, cam_row.gps_lat, cam_row.gps_lon)
        if dist <= settings.ZONE_INCIDENT_RADIUS_METERS:
            nearby_alert_ids.append(alert_row.id)

    # ── 4. Threshold check ───────────────────────────────────────────────────
    if len(nearby_alert_ids) < settings.ZONE_INCIDENT_MIN_ALERTS:
        return None

    # ── 5. Check for an existing open ZoneIncident covering this area ─────────
    open_incidents = (
        db.query(ZoneIncident)
        .filter(ZoneIncident.closed_at.is_(None))
        .all()
    )

    existing: Any | None = None
    for inc in open_incidents:
        dist_to_center = haversine_meters(ref_lat, ref_lon, inc.center_lat, inc.center_lon)
        if dist_to_center <= settings.ZONE_INCIDENT_RADIUS_METERS:
            existing = inc
            break

    if existing is not None:
        # ── 5a. Append to existing incident ──────────────────────────────────
        current_ids: list[int] = json.loads(existing.alert_ids)
        if primary_alert.id not in current_ids:
            current_ids.append(primary_alert.id)
            existing.alert_ids = json.dumps(current_ids)
            db.flush()

            logger.info(
                "ZONE_INCIDENT updated: incident_id=%d now has %d alerts (zone=%s)",
                existing.id, len(current_ids), zone_label,
            )
            _log_zone_audit(db, ACTION_ZONE_INCIDENT_UPDATED, existing.id, {
                "zone": zone_label, "alert_count": len(current_ids),
                "new_alert_id": primary_alert.id,
            })
            await _broadcast_zone({
                "type": "zone_incident_updated",
                "incident_id": existing.id,
                "zone": zone_label,
                "alert_ids": current_ids,
                "center_lat": existing.center_lat,
                "center_lon": existing.center_lon,
                "opened_at": existing.opened_at.isoformat(),
            })
        return existing

    else:
        # ── 5b. Open a new ZoneIncident ───────────────────────────────────────
        # Center = mean GPS of all nearby cameras that contributed to the trigger.
        # Collect unique camera GPS coords from the nearby alerts.
        cam_coords: list[tuple[float, float]] = []
        for alert_row, cam_row in recent_alerts:
            if alert_row.id in nearby_alert_ids and cam_row and \
               cam_row.gps_lat is not None and cam_row.gps_lon is not None:
                cam_coords.append((cam_row.gps_lat, cam_row.gps_lon))

        center_lat = sum(c[0] for c in cam_coords) / len(cam_coords)
        center_lon = sum(c[1] for c in cam_coords) / len(cam_coords)

        incident = ZoneIncident(
            zone=zone_label,
            center_lat=center_lat,
            center_lon=center_lon,
            alert_ids=json.dumps(nearby_alert_ids),
        )
        db.add(incident)
        db.flush()  # populate incident.id

        logger.info(
            "ZONE_INCIDENT opened: incident_id=%d zone=%s center=(%.6f, %.6f) "
            "alert_count=%d",
            incident.id, zone_label, center_lat, center_lon, len(nearby_alert_ids),
        )
        _log_zone_audit(db, ACTION_ZONE_INCIDENT_OPENED, incident.id, {
            "zone": zone_label, "alert_count": len(nearby_alert_ids),
            "alert_ids": nearby_alert_ids,
            "center_lat": center_lat, "center_lon": center_lon,
        })
        await _broadcast_zone({
            "type": "zone_incident_opened",
            "incident_id": incident.id,
            "zone": zone_label,
            "alert_ids": nearby_alert_ids,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "opened_at": incident.opened_at.isoformat(),
        })
        return incident


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_zone_audit(db: Session, action: str, incident_id: int, details: dict) -> None:
    try:
        from backend.services.audit_logger import log_audit
        log_audit(db, user=None, action=action,
                  resource_type="zone_incident", resource_id=incident_id,
                  details=details)
    except Exception:
        logger.exception("Failed to write zone incident audit log")


async def _broadcast_zone(payload: dict) -> None:
    try:
        from backend.ws.dashboard_ws import broadcast
        await broadcast(payload)
    except Exception as exc:
        logger.debug("Failed to broadcast zone incident event: %s", exc)
