"""preflight_check.py - GO / NO-GO check before a live demo.

WHY THIS EXISTS
  Measured across four 8-camera runs, roughly one in four produced ZERO
  detections while every health signal looked fine: all pipeline threads
  alive, frames flowing, no errors. A dead demo and a working demo look
  identical from the outside until you check whether detections are actually
  coming out the far end.

  This runs the real pipeline against the real cameras for a short window
  and answers the only question that matters before you present: is anything
  actually being detected right now?

USAGE
  python preflight_check.py            # 8 cameras, 60s
  python preflight_check.py 4 45       # 4 cameras, 45s

EXIT CODE
  0 = GO      detections are flowing
  1 = NO-GO   restart the pipeline, or switch to the recorded fallback
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

N_CAMERAS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 60

_tmp = tempfile.mkdtemp(prefix="preflight_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/preflight.db"
os.environ["SENTINEL_LIVE_CAMERAS"] = "1"
os.environ["SENTINEL_MAX_CAMERAS"] = str(N_CAMERAS)


def main() -> int:
    from sqlalchemy import text
    from starlette.testclient import TestClient

    from backend.config import load_config
    from backend.db.models import Base
    from backend.db.session import SessionLocal, engine
    import backend.db.seed_data as seed_data

    Base.metadata.create_all(bind=engine)
    seed_data.seed_cameras(load_config())

    counts = {"frames": 0, "detections": 0, "events": 0}
    per_camera_events: dict[str, int] = {}

    import detector as detector_mod
    _orig_track = detector_mod.PersonDetector.track_frame

    def counting_track(self, frame, frame_number):
        counts["frames"] += 1
        result = _orig_track(self, frame, frame_number)
        counts["detections"] += len(result)
        return result

    detector_mod.PersonDetector.track_frame = counting_track

    from backend.connection_manager import ConnectionManager
    _orig_handle = ConnectionManager.handle_detection_event

    async def counting_handle(self, event):
        counts["events"] += 1
        cam = event.get("camera_id", "?")
        per_camera_events[cam] = per_camera_events.get(cam, 0) + 1
        return await _orig_handle(self, event)

    ConnectionManager.handle_detection_event = counting_handle

    import backend.main as backend_main

    print(f"Pre-flight: {N_CAMERAS} camera(s), {DURATION}s ...")
    with TestClient(backend_main.app):
        started = len(backend_main.bridges)
        time.sleep(DURATION)
        alive = sum(1 for b in backend_main.bridges if b.is_running)

        db = SessionLocal()
        tracks = db.execute(text("SELECT COUNT(*) FROM tracks")).scalar() or 0
        db.close()

    print()
    print(f"  pipelines started/alive : {started}/{alive}")
    print(f"  frames processed        : {counts['frames']}")
    print(f"  raw detections          : {counts['detections']}")
    print(f"  confirmed track events  : {counts['events']}")
    print(f"  track rows written      : {tracks}")
    print(f"  cameras producing       : {per_camera_events or '(none)'}")
    print()

    # A pipeline can be fully "healthy" - every thread alive, frames flowing -
    # and still emit nothing. Each of these is checked separately so the
    # failure message says WHICH stage is dead, not just that something is.
    problems: list[str] = []
    if alive == 0:
        problems.append("no pipeline threads alive")
    elif alive < started:
        problems.append(f"{started - alive} pipeline thread(s) died")
    if counts["frames"] == 0:
        problems.append("no frames arrived (streams not delivering)")
    elif counts["detections"] == 0:
        problems.append("frames arrived but NOTHING was detected in them")
    elif counts["events"] == 0:
        problems.append("detections found but none confirmed into tracks "
                        "(per-camera frame rate likely too low)")

    if counts["events"] > 0 and not any(p.startswith("no pipeline") for p in problems):
        print("  RESULT: GO - detections are flowing.")
        if problems:
            print("  (warnings: " + "; ".join(problems) + ")")
        print(f"  Best camera(s) to demo: "
              f"{sorted(per_camera_events, key=per_camera_events.get, reverse=True)[:3]}")
        return 0

    print("  RESULT: NO-GO - " + "; ".join(problems or ["nothing detected"]))
    print()
    print("  Try: restart the pipeline (stream delivery is bursty and one run")
    print("  in four produced nothing in testing), or fall back to recorded")
    print("  footage:")
    print('     SENTINEL_SOURCES="CAM_01=demo/clips/your_clip.mp4" '
          "uvicorn backend.main:app")
    return 1


if __name__ == "__main__":
    sys.exit(main())
