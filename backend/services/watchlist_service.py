"""
backend/services/watchlist_service.py

Replaces the DEFAULT_WATCHLIST dict + check_watchlist() method.

Upgrades:
  • Fuzzy 1-char match — catches single OCR errors on watchlist plates
  • 30-second per-plate alert cooldown — no duplicate alerts for same vehicle
  • SQLite persistence for every detection + every alert
  • Watchlist hot-reload without restarting the engine
  • check_and_alert() is thread-safe (GIL-protected dict ops)

WATCHLIST DATA SOURCE (changed — see docs/WATCHLIST_ALERTING_ARCHITECTURE.md):
  This used to load its plate list once from watchlist.json and never see
  another source again — a standalone file nothing in the app's UI or API
  could reach, while GET /plate-search and the journey lookup endpoints were
  already reading a completely different table, `watchlist_plates`, for the
  exact same kind of question ("is this plate on a watchlist"). Two
  watchlists, only one of them editable, and the live ANPR pipeline was
  checking the one nobody could edit — so a plate added the "normal" way
  (through the app) was invisible to the pipeline that is supposed to catch
  it driving past a camera.
  `_load_from_db()` below now reads `watchlist_plates` instead, refreshed on
  a short TTL (not per-call: this runs on every confirmed plate lock, and a
  DB round trip there is unnecessary when a few seconds of staleness is
  immaterial to whether a match fires). watchlist.json / DEFAULT_WATCHLIST
  remain ONLY as a fallback for when the DB genuinely cannot be reached
  (e.g. a standalone harvesting script run outside the app's DB config) —
  not a second intended data path.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DB_PATH        = "sentinel_detections.db"
WATCHLIST_PATH = "watchlist.json"
WATCHLIST_DB_RELOAD_INTERVAL_S = 5.0


# ── Default watchlist (empty by default; populated only by genuine user entries) ───
DEFAULT_WATCHLIST: dict[str, str] = {}


def _load_watchlist(path: str) -> dict[str, str]:
    """
    Load watchlist from JSON.
    Accepts two formats:
      • list   ["GJ01AB1234", ...]          → reason = "Watchlist"
      • dict   {"GJ01AB1234": "reason", ...}
    Falls back to DEFAULT_WATCHLIST if file missing.
    """
    p = Path(path)
    if not p.exists():
        log.warning("watchlist.json not found at %s — using default", path)
        _save_default(path)
        return dict(DEFAULT_WATCHLIST)

    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            wl = {v.upper().replace(" ", ""): "Watchlist" for v in data}
        elif isinstance(data, dict):
            wl = {k.upper().replace(" ", ""): v for k, v in data.items()}
        else:
            wl = {}
        log.info("Watchlist loaded: %d plates from %s", len(wl), path)
        return wl
    except Exception as exc:
        log.error("Failed to load watchlist: %s", exc)
        return dict(DEFAULT_WATCHLIST)


def _save_default(path: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_WATCHLIST, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


class WatchlistService:
    """Thread-safe watchlist checker with SQLite persistence."""

    COOLDOWN_SECS = 30

    def __init__(self, watchlist_path: str = WATCHLIST_PATH, db_path: str = DB_PATH):
        self.watchlist_path = watchlist_path
        self.db_path        = db_path
        self._last_db_load  = 0.0
        _from_db = self._load_from_db()
        self.watchlist = _from_db if _from_db is not None else _load_watchlist(watchlist_path)
        self._alert_times:  dict[str, datetime] = {}
        self._init_db()

    # ── watchlist_plates (the real, app-editable watchlist) ──────────────────

    def _load_from_db(self) -> Optional[dict[str, str]]:
        """Read active entries from `watchlist_plates`. Returns None (not {})
        on any failure, so the caller can tell "DB unreachable, keep what we
        had" apart from "DB reachable, watchlist is genuinely empty right
        now" — an empty dict would otherwise be indistinguishable from a
        connection error, and a connection hiccup should never blank the
        live watchlist out from under the pipeline mid-shift."""
        try:
            from backend.db.models import WatchlistPlate
            from backend.db.session import SessionLocal
        except Exception:
            return None

        db = None
        try:
            db = SessionLocal()
            rows = db.query(WatchlistPlate).filter(
                (WatchlistPlate.active == True) | (WatchlistPlate.active.is_(None))  # noqa: E712
            ).all()
            wl: dict[str, str] = {}
            for r in rows:
                key = (r.plate or r.plate_number or "").upper().replace(" ", "")
                if not key:
                    continue
                wl[key] = r.reason or f"Watchlist ({r.category or 'unspecified'})"
            return wl
        except Exception as exc:
            log.debug("watchlist_plates unreachable, keeping cached watchlist: %s", exc)
            return None
        finally:
            if db is not None:
                db.close()

    def _refresh_if_stale(self) -> None:
        now = time.monotonic()
        if now - self._last_db_load < WATCHLIST_DB_RELOAD_INTERVAL_S:
            return
        self._last_db_load = now
        fresh = self._load_from_db()
        if fresh is not None:
            self.watchlist = fresh

    # ── Database setup ────────────────────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c    = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                cam_id         INTEGER,
                plate          TEXT,
                confidence     REAL,
                lighting       TEXT,
                quality        REAL,
                timestamp      TEXT,
                watchlist_hit  INTEGER DEFAULT 0,
                match_type     TEXT    DEFAULT ''
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                cam_id     INTEGER,
                plate      TEXT,
                matched    TEXT,
                reason     TEXT,
                match_type TEXT,
                timestamp  TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ── Watchlist operations ──────────────────────────────────────────────────

    def reload_watchlist(self):
        """Force an immediate reload from watchlist_plates (falls back to
        disk only if the DB is unreachable). No engine restart needed."""
        self._last_db_load = 0.0
        fresh = self._load_from_db()
        self.watchlist = fresh if fresh is not None else _load_watchlist(self.watchlist_path)

    def _fuzzy_match(self, plate: str) -> Optional[tuple[str, str]]:
        """
        Returns (matched_watchlist_plate, match_type) or None.
        match_type ∈ {"exact", "fuzzy_1char"}
        """
        self._refresh_if_stale()
        if plate in self.watchlist:
            return plate, "exact"

        # One-character tolerance is what makes this work at all. Measured on
        # 74 held-out vehicles against a 10,000-entry watchlist: exact-string
        # matching identifies 64.9% of them, distance-1 identifies 86.5%, and
        # neither produced a single false hit on vehicles that were not on the
        # list. Distance-2 would reach 95.9% but starts flagging innocent
        # vehicles (1.4%), which is the wrong trade for a police alert.
        #
        # Ambiguity is rejected rather than resolved arbitrarily. This used to
        # return the FIRST entry within one character, so if two watchlist
        # plates were both one character from the read — which sequential
        # registrations make entirely possible — the alert named whichever
        # happened to be listed first. A confident alert pointing at the wrong
        # vehicle is worse than no alert, so a tie now yields nothing.
        best: Optional[str] = None
        ties = 0
        for wp in self.watchlist:
            if len(wp) != len(plate):
                continue
            if sum(a != b for a, b in zip(plate, wp)) == 1:
                if best is None:
                    best = wp
                    ties = 1
                else:
                    ties += 1
        if best is None or ties > 1:
            return None
        return best, "fuzzy_1char"

    # ── Main entry point ──────────────────────────────────────────────────────

    def check_and_alert(self, detection: dict) -> bool:
        """
        Saves detection to DB and fires an alert if watchlist match found.
        Returns True on watchlist hit.
        """
        plate = (detection.get("plate") or "").upper().replace(" ", "")
        if not plate:
            return False

        match = self._fuzzy_match(plate)
        hit   = match is not None

        self._save_detection(detection, watchlist_hit=int(hit),
                             match_type=match[1] if match else "")

        if not hit:
            return False

        matched_plate, match_type = match
        reason = self.watchlist.get(matched_plate, "Watchlist")

        # Cooldown dedup
        now  = datetime.now()
        last = self._alert_times.get(plate)
        if last and (now - last).total_seconds() < self.COOLDOWN_SECS:
            return True  # Hit, but suppressed by cooldown

        self._alert_times[plate] = now
        self._fire_alert(detection, matched_plate, reason, match_type)
        return True

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _save_detection(self, det: dict, watchlist_hit: int, match_type: str):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO detections
                (cam_id, plate, confidence, lighting, quality,
                 timestamp, watchlist_hit, match_type)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                det.get("cam_id", 0),
                det.get("plate", ""),
                det.get("confidence", 0.0),
                det.get("lighting", ""),
                det.get("quality", 0.0),
                det.get("timestamp", datetime.now().isoformat()),
                watchlist_hit,
                match_type,
            ))
            conn.commit()
            conn.close()
        except Exception as exc:
            log.error("DB write failed: %s", exc)

    def _fire_alert(
        self,
        det:        dict,
        matched:    str,
        reason:     str,
        match_type: str,
    ):
        cam  = det.get("cam_id", "?")
        plate = det.get("plate", "")
        conf  = det.get("confidence", 0.0)
        ts    = det.get("timestamp", datetime.now().isoformat())

        border = "=" * 58
        print(f"\n{border}")
        print(f"  🚨  WATCHLIST ALERT  🚨")
        print(f"  Camera     : CAM{cam:02d}" if isinstance(cam, int) else f"  Camera     : {cam}")
        print(f"  Detected   : {plate}")
        print(f"  Matched    : {matched}  [{match_type}]")
        print(f"  Reason     : {reason}")
        print(f"  Confidence : {conf:.1%}")
        print(f"  Time       : {ts}")
        print(f"{border}\n")

        log.warning(
            "WATCHLIST HIT | CAM%s | %s matched %s (%s) | conf=%.2f",
            cam, plate, matched, match_type, conf,
        )

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO alerts
                (cam_id, plate, matched, reason, match_type, timestamp)
                VALUES (?,?,?,?,?,?)
            """, (cam, plate, matched, reason, match_type, ts))
            conn.commit()
            conn.close()
        except Exception as exc:
            log.error("Alert DB write failed: %s", exc)