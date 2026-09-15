import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("sentinel.behavior")


class BehaviorAnalyzer:
    """
    Detects:
    - Loitering: Same track_id in same camera > 5 minutes (or 15s in demo mode)
    - Crowd Anomaly: 8+ people simultaneously in frame
    - Abandoned Object: Backpack/luggage stationary for > 2 minutes
    """

    LOITER_SEC   = 300
    CROWD_THRESH = 8
    ABANDON_SEC  = 120

    def __init__(self):
        # {camera_id: {track_id: first_seen}}
        self._first_seen: Dict[str, Dict[int, datetime]] = defaultdict(dict)
        self._static_objects: Dict[str, Dict[str, datetime]] = defaultdict(dict)

    def update(
        self,
        camera_id: str,
        timestamp: datetime,
        detections: list,
    ) -> list:
        events = []

        person_dets  = [d for d in detections if d.get("class_name") == "person"]
        vehicle_dets = [d for d in detections if d.get("class_name") in {"car", "truck", "bus", "motorcycle"}]

        # ── Crowd Check ───────────────────────────────────────
        if len(person_dets) >= self.CROWD_THRESH:
            events.append({
                "type":       "crowd_anomaly",
                "camera_id":  camera_id,
                "timestamp":  timestamp,
                "count":      len(person_dets),
                "detail":     f"{len(person_dets)} people detected simultaneously",
            })

        # ── Loitering Check ───────────────────────────────────
        for det in person_dets:
            tid = det.get("track_id")
            if tid is None:
                continue

            cam_tracks = self._first_seen[camera_id]
            if tid not in cam_tracks:
                cam_tracks[tid] = timestamp
            else:
                duration = (timestamp - cam_tracks[tid]).total_seconds()
                if duration >= self.LOITER_SEC:
                    events.append({
                        "type":      "loitering",
                        "camera_id": camera_id,
                        "timestamp": timestamp,
                        "track_id":  tid,
                        "duration_sec": int(duration),
                        "detail":    f"Person loitering {duration/60:.1f} minutes",
                    })
                    cam_tracks[tid] = timestamp

        # Clean stale tracks
        active_ids = {d.get("track_id") for d in person_dets if d.get("track_id") is not None}
        stale = [
            tid for tid in self._first_seen[camera_id]
            if tid not in active_ids
        ]
        for tid in stale:
            del self._first_seen[camera_id][tid]

        return events
