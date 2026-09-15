"""backend/services/road_router.py — snap a leg between two cameras onto the
actual road network.

WHY THIS EXISTS
  Between two cameras there is a blind spot, often kilometres of it. Until now
  the map drew a straight geodesic across it and the code was careful to call
  it exactly that (`straight_line_km`, "the vehicle drove a road, which is
  always longer"). Honest, but it understates every distance and therefore
  every speed, and it draws a line through buildings.

  Measured on this fleet's own cameras: CAM_01 to CAM_03 is 2.669 km as the
  crow flies and 3.845 km by road — a 1.44x difference. A leg speed computed
  from the geodesic is 44% low there.

WHAT IT DOES
  Asks a routing engine for the driving route between the two points and keeps
  the road geometry, its true length, and the engine's own free-flow duration.
  The map then follows the road, and leg speed is computed over the distance
  the vehicle could actually have driven.

THE THREE RULES THIS FOLLOWS
  1. Never silently invent a road. If no route can be obtained the leg is
     returned with `matched=False` and the straight line, and every consumer
     labels it as unmatched. A wrong road drawn confidently is worse than an
     honest straight line.
  2. Cache everything. Camera positions do not move, so a route between a pair
     is computed once and reused for ever. After warm_road_cache.py has run,
     the whole feature works with no network at all — which is what makes it
     deployable inside a police network that cannot reach the internet.
  3. Point at your own engine in production. OSRM is open source and a force
     is expected to self-host it; SENTINEL_OSRM_URL exists for exactly that.
     The public demo server is a development convenience and is rate-limited,
     which is the other reason the cache is not optional.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "output" / "road_routes" / "cache.json"

# Self-host in production: `docker run osrm/osrm-backend` against a Gujarat
# extract. The default is the project's public demo server, which is fine for
# development and warming the cache but must not be a runtime dependency.
OSRM_URL = os.environ.get("SENTINEL_OSRM_URL",
                          "https://router.project-osrm.org").rstrip("/")
OSRM_TIMEOUT = float(os.environ.get("SENTINEL_OSRM_TIMEOUT", "8"))
# Off by default at runtime so a slow network can never stall a request that
# renders a journey. warm_road_cache.py turns it on to fill the cache.
OSRM_LIVE = os.environ.get("SENTINEL_OSRM_LIVE", "0") == "1"

# Coordinates are rounded to ~1 m before they become a cache key, so tiny
# float differences in the same camera position do not each fetch a route.
_KEY_DP = 5


@dataclass
class RoadLeg:
    """One camera-to-camera hop, on the road where possible."""
    matched: bool
    distance_km: float          # road distance if matched, else geodesic
    straight_km: float
    geometry: list              # [[lat, lon], ...] — the drawn path
    duration_min: Optional[float] = None   # engine's free-flow estimate
    source: str = "straight_line"          # "osrm" | "cache" | "straight_line"
    detour_factor: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    r = 6371.0088
    dlat, dlon = radians(b_lat - a_lat), radians(b_lon - a_lon)
    h = (sin(dlat / 2) ** 2
         + cos(radians(a_lat)) * cos(radians(b_lat)) * sin(dlon / 2) ** 2)
    return 2 * r * asin(sqrt(h))


class RoadRouter:
    """Cached road routing. Safe to call from a request handler."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._loaded = False
        self._misses = 0
        self._live_calls = 0
        self._failures = 0

    # ── cache ───────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                if CACHE_PATH.is_file():
                    self._cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                    logger.info("Road route cache: %d leg(s) from %s",
                                len(self._cache), CACHE_PATH)
            except Exception as exc:                                # noqa: BLE001
                logger.warning("Road route cache unreadable (%s); starting empty",
                               exc)
                self._cache = {}
            self._loaded = True

    def _save(self) -> None:
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = CACHE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache), encoding="utf-8")
            os.replace(tmp, CACHE_PATH)
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("Could not persist road route cache: %s", exc)

    @staticmethod
    def _key(a_lat, a_lon, b_lat, b_lon) -> str:
        return (f"{round(a_lat, _KEY_DP)},{round(a_lon, _KEY_DP)}"
                f"|{round(b_lat, _KEY_DP)},{round(b_lon, _KEY_DP)}")

    # ── routing ─────────────────────────────────────────────────────────────
    def _fetch(self, a_lat, a_lon, b_lat, b_lon) -> Optional[dict]:
        """One OSRM call. Returns None on any failure — never raises upward."""
        url = (f"{OSRM_URL}/route/v1/driving/"
               f"{a_lon},{a_lat};{b_lon},{b_lat}"
               f"?overview=full&geometries=geojson&alternatives=false")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "SarvanetraAI/1.0 (traffic analytics)"})
            with urllib.request.urlopen(req, timeout=OSRM_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:                                    # noqa: BLE001
            self._failures += 1
            logger.debug("OSRM request failed: %s", exc)
            return None
        if data.get("code") != "Ok" or not data.get("routes"):
            self._failures += 1
            return None
        r0 = data["routes"][0]
        coords = r0.get("geometry", {}).get("coordinates") or []
        if len(coords) < 2:
            return None
        return {
            # GeoJSON is [lon, lat]; Leaflet wants [lat, lon]. Getting this
            # round the wrong way puts Junagadh in the Indian Ocean.
            "geometry": [[float(c[1]), float(c[0])] for c in coords],
            "distance_km": float(r0.get("distance", 0.0)) / 1000.0,
            "duration_min": float(r0.get("duration", 0.0)) / 60.0,
        }

    def route(self, a_lat, a_lon, b_lat, b_lon, *,
              allow_live: Optional[bool] = None) -> RoadLeg:
        """The road leg between two points, from cache or the engine.

        Always returns a RoadLeg. `matched` says whether it is a real road.
        """
        # Guard BEFORE the arithmetic. A JourneyEvent may carry a NULL lat/lon
        # when its camera has no surveyed position, and subtracting None throws
        # — which took down the whole journey response rather than degrading
        # that one leg.
        if (a_lat is None or a_lon is None or b_lat is None or b_lon is None):
            return RoadLeg(matched=False, distance_km=0.0, straight_km=0.0,
                           geometry=[], source="no_coordinates")

        straight = haversine_km(a_lat, a_lon, b_lat, b_lon)
        blank = RoadLeg(matched=False, distance_km=straight,
                        straight_km=straight,
                        geometry=[[a_lat, a_lon], [b_lat, b_lon]],
                        source="straight_line")

        self._load()
        key = self._key(a_lat, a_lon, b_lat, b_lon)
        hit = self._cache.get(key)
        if hit is not None:
            if not hit.get("ok"):
                return blank                    # a known, remembered failure
            return RoadLeg(
                matched=True, distance_km=hit["distance_km"],
                straight_km=straight, geometry=hit["geometry"],
                duration_min=hit.get("duration_min"), source="cache",
                detour_factor=(hit["distance_km"] / straight) if straight > 1e-6 else None)

        self._misses += 1
        live = OSRM_LIVE if allow_live is None else allow_live
        if not live:
            return blank

        got = self._fetch(a_lat, a_lon, b_lat, b_lon)
        self._live_calls += 1
        with self._lock:
            if got is None:
                # Remember the failure so a dead route is not retried on every
                # single page load.
                self._cache[key] = {"ok": False, "at": time.time()}
            else:
                self._cache[key] = {"ok": True, **got, "at": time.time()}
            self._save()
        if got is None:
            return blank
        return RoadLeg(
            matched=True, distance_km=got["distance_km"], straight_km=straight,
            geometry=got["geometry"], duration_min=got.get("duration_min"),
            source="osrm",
            detour_factor=(got["distance_km"] / straight) if straight > 1e-6 else None)

    def stats(self) -> dict[str, Any]:
        self._load()
        ok = sum(1 for v in self._cache.values() if v.get("ok"))
        return {
            "cached_legs": len(self._cache),
            "cached_routed": ok,
            "cached_unroutable": len(self._cache) - ok,
            "cache_path": str(CACHE_PATH),
            "engine": OSRM_URL,
            "live_lookups_enabled": OSRM_LIVE,
            "live_calls_this_process": self._live_calls,
            "failures_this_process": self._failures,
        }


_router: Optional[RoadRouter] = None


def get_road_router() -> RoadRouter:
    global _router
    if _router is None:
        _router = RoadRouter()
    return _router
