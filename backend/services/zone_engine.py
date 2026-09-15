"""backend/services/zone_engine.py — per-camera zones that decide whether
loitering is worth alerting on.

THE PROBLEM THIS SOLVES
  The loitering detector fires on "stationary for 60 seconds", everywhere,
  at all hours. On real footage that flags a woman waiting at a busy junction
  at 7am - technically correct, operationally useless. Send those to a control
  room and the operators stop reading alerts within days. That is alert
  fatigue, and it is the usual reason CCTV analytics get switched off after
  deployment rather than any failure to detect.

  Standing still is not suspicious. Standing still IN A PARTICULAR PLACE, at a
  particular hour, for longer than that place warrants, is.

ZONES ARE PER CAMERA - not global
  Pixel (150, 150) is a shop doorway on one camera and open road on another.
  A zone table keyed only by coordinates would apply one camera's meaning to
  every other camera's frame. Every lookup here therefore requires a camera id.

THREE ZONE TYPES
  exempt     Bus stop, auto stand, hospital gate. People wait here; that is
             the point of the place. Never alert, at any duration.
  normal     Footpath, general area. Long threshold - ten minutes by day.
  sensitive  Parked vehicles, an ATM, an isolated corner. Short threshold,
             shorter still at night.

  Overlaps resolve to the most sensitive zone, so a cautious definition never
  gets silently overridden by a broad exempt rectangle drawn on top of it.

UNCONFIGURED CAMERAS FAIL OPEN, LOUDLY
  A camera with no zone file gets the `normal` default rather than being
  silenced. Silently skipping detection because someone forgot to draw zones
  would be a far worse failure than an occasional low-value alert, so the
  warning is logged once per camera.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

from backend.core.config import settings

ZONE_DIR = Path("config/zones")

# Ordering for overlap resolution: lower value wins.
_PRIORITY = {"sensitive": 0, "normal": 1, "exempt": 2}

# Used when a camera has no zone file at all. Uses configured system duration.
DEFAULT_ZONE = {
    "type": "normal",
    "threshold_day_sec": float(getattr(settings, "LOITER_DURATION_SEC", 60.0)),
    "threshold_night_sec": float(getattr(settings, "LOITER_DURATION_SEC", 60.0) * 0.75),
    "label": "unconfigured — default thresholds",
}


@dataclass
class ZoneVerdict:
    """Result of asking "should this dwell, here, now, raise an alert?"."""
    zone_id: str
    zone_type: str
    label: str
    threshold_sec: float | None       # None == never alert (exempt)
    should_alert: bool
    reason: str


class ZoneEngine:
    def __init__(self, zone_dir: Path | str = ZONE_DIR) -> None:
        self.zone_dir = Path(zone_dir)
        self._cameras: dict[str, list[dict]] = {}
        self._warned: set[str] = set()
        self._load_all()

    # ── loading ────────────────────────────────────────────────────────────
    @staticmethod
    def _norm(cam: str) -> str:
        """Ids appear as both CAM-01 and CAM_01 across this project."""
        return cam.strip().upper().replace("-", "_")

    def _load_all(self) -> None:
        if not self.zone_dir.is_dir():
            logger.info("[zones] no zone directory at %s — all cameras will "
                        "use default thresholds", self.zone_dir)
            return
        for f in sorted(self.zone_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as exc:                      # noqa: BLE001
                logger.warning("[zones] %s is not valid JSON: %s", f.name, exc)
                continue
            cam = self._norm(data.get("camera_id", f.stem))
            zones = []
            for z in data.get("zones", []):
                poly = z.get("polygon") or []
                if len(poly) < 3:
                    logger.warning("[zones] %s/%s has %d points — a polygon "
                                   "needs at least 3, skipping",
                                   cam, z.get("id", "?"), len(poly))
                    continue
                ztype = z.get("type", "normal")
                if ztype not in _PRIORITY:
                    logger.warning("[zones] %s/%s unknown type %r, treating "
                                   "as normal", cam, z.get("id"), ztype)
                    ztype = "normal"
                zones.append({
                    "id": z.get("id", "zone"),
                    "type": ztype,
                    "label": z.get("label", z.get("id", "zone")),
                    "polygon": [(float(p[0]), float(p[1])) for p in poly],
                    "threshold_day_sec": z.get("threshold_day_sec"),
                    "threshold_night_sec": z.get("threshold_night_sec"),
                })
            if zones:
                self._cameras[cam] = zones
        if self._cameras:
            logger.info("[zones] loaded %d cameras: %s", len(self._cameras),
                        ", ".join(f"{c}({len(z)})"
                                  for c, z in self._cameras.items()))

    def reload(self) -> None:
        self._cameras.clear()
        self._warned.clear()
        self._load_all()

    # ── geometry ───────────────────────────────────────────────────────────
    @staticmethod
    def _contains(polygon: list[tuple[float, float]], x: float, y: float) -> bool:
        """Ray-casting point-in-polygon.

        Deliberately dependency-free. shapely is faster for complex geometry,
        but this runs on a handful of 4-8 point polygons per person per frame,
        and not importing it keeps the pipeline runnable on a machine where
        only the detector's dependencies are installed.
        """
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    def zone_at(self, camera_id: str, x: float, y: float) -> dict:
        """The most sensitive zone containing (x, y) on this camera."""
        cam = self._norm(camera_id)
        zones = self._cameras.get(cam)
        if not zones:
            if cam not in self._warned:
                self._warned.add(cam)
                logger.warning(
                    "[zones] no zones defined for %s — using default "
                    "thresholds (day %.0fs / night %.0fs). Draw zones with "
                    "backend/scripts/draw_zones.py to cut false alerts.",
                    cam, DEFAULT_ZONE["threshold_day_sec"],
                    DEFAULT_ZONE["threshold_night_sec"])
            return {**DEFAULT_ZONE, "id": "unconfigured"}

        hits = [z for z in zones if self._contains(z["polygon"], x, y)]
        if not hits:
            # Inside the frame but outside every drawn zone. Treated as
            # normal, not exempt: an area nobody bothered to mark is not a
            # statement that loitering there is acceptable.
            return {**DEFAULT_ZONE, "id": "outside_zones",
                    "label": "outside defined zones"}
        hits.sort(key=lambda z: _PRIORITY[z["type"]])
        return hits[0]

    # ── decision ───────────────────────────────────────────────────────────
    def evaluate(self, camera_id: str, x: float, y: float,
                 dwell_sec: float, is_night: bool) -> ZoneVerdict:
        """Should a dwell of `dwell_sec` at (x, y) raise an alert?"""
        z = self.zone_at(camera_id, x, y)

        if z["type"] == "exempt":
            return ZoneVerdict(
                zone_id=z.get("id", "?"), zone_type="exempt",
                label=z.get("label", ""), threshold_sec=None,
                should_alert=False,
                reason=f"exempt zone ({z.get('label','')}) — waiting expected")

        key = "threshold_night_sec" if is_night else "threshold_day_sec"
        thr = z.get(key)
        if thr is None:
            thr = DEFAULT_ZONE[key]
        thr = float(thr)

        ok = dwell_sec >= thr
        when = "night" if is_night else "day"
        return ZoneVerdict(
            zone_id=z.get("id", "?"), zone_type=z["type"],
            label=z.get("label", ""), threshold_sec=thr, should_alert=ok,
            reason=(f"{z['type']} zone, {when} threshold {thr:.0f}s — "
                    f"dwelled {dwell_sec:.0f}s"
                    + ("" if ok else ", below threshold")))


_engine: ZoneEngine | None = None


def get_zone_engine() -> ZoneEngine:
    global _engine
    if _engine is None:
        _engine = ZoneEngine()
    return _engine
