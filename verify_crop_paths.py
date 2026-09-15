"""verify_crop_paths.py - the crop path contract between writer and reader.

WHY THIS EXISTS
  event_emitter.py WRITES person crops; connection_manager.crop_path_for()
  RECONSTRUCTS that path to store in Track.best_crop_path. They are two
  separate string constructions in two files that must agree exactly.

  When they disagree, nothing raises. Track.best_crop_path simply points at
  a file that does not exist, and the ReID review queue - where an officer
  compares "is this the same person" - renders a placeholder instead of the
  crop. A reviewer sees a broken image, not an error.

  The camera_id segment is the specific hazard. BoT-SORT track ids are
  per-camera locals that restart at 1 on every feed, so a path without the
  camera id puts CAM_01's person and CAM_02's person in the same track_N
  directory. That silently corrupts two things at once: the crop shown in
  review, and any ReID validation set built from those folders.

Run:  python verify_crop_paths.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} - {name}{(': ' + detail) if detail else ''}")


def main() -> None:
    from event_emitter import EventEmitter, _fs_safe
    from backend.connection_manager import crop_path_for
    from tracker import ConfirmedTrack

    tmp = tempfile.mkdtemp(prefix="verify_crop_paths_")
    cfg = {
        "output": {
            "events_file": f"{tmp}/events.jsonl",
            "object_events_file": f"{tmp}/objects.jsonl",
            "crops_dir": f"{tmp}/crops",
            "crop_save_interval_frames": 1,
        }
    }
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    crops_root = Path(tmp) / "crops"

    TRACK_ID = 1        # deliberately the SAME local id on both cameras
    FRAME_NO = 42

    for cam in ("CAM_01", "CAM_02"):
        emitter = EventEmitter(cfg, event_queue=None, camera_id=cam)
        emitter.emit(
            [ConfirmedTrack(track_id=TRACK_ID, bbox=[10, 10, 200, 400],
                            confidence=0.9, frame_number=FRAME_NO, timestamp=1.0)],
            frame,
        )

    written = sorted(crops_root.rglob("*.jpg"))
    check("Two cameras sharing one local track id write TWO distinct files "
          "(no cross-camera overwrite)",
          len(written) == 2, f"{len(written)} file(s)")

    for cam in ("CAM_01", "CAM_02"):
        rel = crop_path_for(TRACK_ID, FRAME_NO, cam)
        # crop_path_for returns a project-root-relative path; re-root it at
        # the temp crops dir to check the file the writer actually created.
        suffix = rel.split("output/crops/", 1)[-1]
        check(f"{cam}: reader path resolves to the file the writer created",
              (crops_root / suffix).is_file(), rel)

    check("Reader path contains the camera id segment",
          "CAM_01" in crop_path_for(TRACK_ID, FRAME_NO, "CAM_01"))

    check("Reader without a camera id degrades to an explicit marker rather "
          "than silently colliding with a real camera's directory",
          "unknown_camera" in crop_path_for(TRACK_ID, FRAME_NO, ""))

    # A camera id is used as a single path segment; a stray separator must
    # not let a crop escape the crops directory.
    check("Path traversal in a camera id is neutralised",
          "/" not in _fs_safe("../../etc") and ".." not in _fs_safe("../../etc"),
          _fs_safe("../../etc"))

    print(f"\n{'=' * 66}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 66}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
