"""backend/services/alert_cooldown.py — cap how often one camera can alert.

THE FAILURE THIS PREVENTS
  The loitering detector already debounces per PERSON: one alert per continuous
  episode, refreshed while the track stays active. What it cannot see is the
  camera as a whole.

  On a busy junction that gap matters. Fifteen people can cross a dwell
  threshold in the same ten minutes and each is, individually, a valid first
  alert. Fifteen notifications arrive, an operator reads the first two, and
  from then on the feed is noise. The alerts that follow - including real ones
  - go unread. Detection quality is irrelevant once that has happened.

  So there is a budget: one alert per camera per window. Everything else is
  suppressed and counted, not silently dropped, so the suppression rate itself
  becomes visible evidence of how noisy a camera is.

DELIBERATELY NOT PERSISTED
  State lives in memory. A restart clears it, which risks one extra alert - an
  acceptable trade against a stale cooldown surviving a restart and silencing a
  camera nobody realises is muted.

PRIORITY BYPASS
  A high-severity alert can override an active cooldown. A cooldown exists to
  protect attention, not to suppress the one event that warranted it; without
  the bypass a critical alert could be lost behind a trivial one that happened
  to arrive first.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SEC = 1800.0        # 30 minutes


class AlertCooldown:
    def __init__(self, window_sec: float = DEFAULT_WINDOW_SEC) -> None:
        self.window_sec = float(window_sec)
        self._last: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}
        # Detectors run on the behaviour bridge's loop while the API server may
        # call in from its own thread; without this two callers can both read a
        # stale timestamp and both decide to alert.
        self._lock = threading.Lock()

    @staticmethod
    def _key(camera_id: str, alert_type: str) -> str:
        # Per (camera, type): a crowd surge should not be muted because a
        # loitering alert fired on the same camera a minute earlier - they are
        # different events an operator would want to see separately.
        return f"{camera_id.strip().upper().replace('-', '_')}:{alert_type}"

    def allow(self, camera_id: str, alert_type: str,
              force: bool = False) -> tuple[bool, str]:
        """May this alert be sent? Returns (allowed, reason)."""
        k = self._key(camera_id, alert_type)
        now = time.monotonic()
        with self._lock:
            last = self._last.get(k)
            if last is not None and (now - last) < self.window_sec:
                remaining = self.window_sec - (now - last)
                if not force:
                    self._suppressed[k] = self._suppressed.get(k, 0) + 1
                    return False, (f"cooldown active — {remaining/60:.1f} min "
                                   f"left, {self._suppressed[k]} suppressed "
                                   f"since last alert")
                # High-priority bypass still resets the window, so a burst of
                # critical alerts cannot itself become the flood.
                self._last[k] = now
                return True, (f"cooldown bypassed (high priority) — "
                              f"{remaining/60:.1f} min remained")
            n = self._suppressed.pop(k, 0)
            self._last[k] = now
            return True, (f"sent ({n} suppressed during previous window)"
                          if n else "sent")

    def stats(self) -> dict[str, dict]:
        with self._lock:
            now = time.monotonic()
            return {
                k: {"suppressed": self._suppressed.get(k, 0),
                    "minutes_since_last": round((now - t) / 60.0, 1)}
                for k, t in self._last.items()
            }

    def reset(self) -> None:
        with self._lock:
            self._last.clear()
            self._suppressed.clear()


_cooldown: AlertCooldown | None = None


def get_cooldown() -> AlertCooldown:
    global _cooldown
    if _cooldown is None:
        window = DEFAULT_WINDOW_SEC
        try:
            from backend.core.config import settings
            window = float(getattr(settings, "ALERT_COOLDOWN_SEC",
                                   DEFAULT_WINDOW_SEC))
        except Exception:                                  # noqa: BLE001
            pass
        _cooldown = AlertCooldown(window)
        logger.info("[cooldown] one alert per camera per %.0f min",
                    window / 60.0)
    return _cooldown
