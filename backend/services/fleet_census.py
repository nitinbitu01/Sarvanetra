"""backend/services/fleet_census.py — one authoritative answer to "how many
cameras are there".

WHY THIS EXISTS
  The fleet size was stated three different ways inside a single working
  session — 22, 42 and 29 — because three different things were counted and
  each was called "the fleet":

      42   every row in `cameras`, including 11 soft-deleted ones
      31   is_deleted = 0, but including a demo rig created during testing
      29   directories under data/clips, two of which are empty
      27   directories that actually contain an mp4

  None of those is wrong as a count; all of them are wrong as *the* count.
  Ratios built on them ("13 of 42 cameras have no footage") then inherit the
  error and invert the finding — that particular one collapsed from 13 to 3
  once the denominator was correct.

  Every report should call `census()` rather than counting for itself.

WHAT COUNTS AS THE FLEET
  A camera that is not soft-deleted and is not a test or demonstration rig.
  Both exclusions are deliberate: a deleted camera is not deployed, and a
  demonstration source created to rehearse an alert is not part of the estate
  being measured. Both are reported separately rather than silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "data" / "clips"

# Identifiers that exist for testing rather than deployment.
# CAM_M1..M4 are the four handheld phone clips registered for the re-ID
# cross-camera test. They are deliberately real `cameras` rows so the ordinary
# pipeline treats them like any other source, which is the whole point of that
# test — but they are not deployed estate, and without this they push the fleet
# to 34. They are not caught by the name marker below: their names say "Test",
# not "DEMO".
_TEST_PREFIXES = ("CAM-RT-",)
_TEST_IDS = {"CAM_E2E", "CAM_M1", "CAM_M2", "CAM_M3", "CAM_M4"}
_TEST_NAME_MARKERS = ("DEMO",)


# The four handheld re-ID clips, kept separate from the rest of _TEST_IDS.
#
# They are excluded from every fleet STATISTIC — the headline count, gap
# analysis, the network map — because their coordinates are placeholders and
# they are not deployed estate. But they are genuine footage being processed by
# the ordinary pipeline, and the control room is meant to show them, so the
# camera LIST endpoint admits them by name. Two different questions: "how big
# is the fleet" and "what can an operator watch".
_REID_TEST_IDS = {"CAM_M1", "CAM_M2", "CAM_M3", "CAM_M4"}


def is_reid_test_camera(camera_id: Optional[str]) -> bool:
    """True for the handheld re-ID test cameras specifically."""
    return (camera_id or "").strip().upper() in _REID_TEST_IDS


def is_test_camera(camera_id: Optional[str], name: Optional[str]) -> bool:
    cid = (camera_id or "").strip()
    nm = (name or "").upper()
    if cid in _TEST_IDS or cid.startswith(_TEST_PREFIXES):
        return True
    if not cid.startswith("CAM_"):
        return True
    return any(m in nm for m in _TEST_NAME_MARKERS)


@dataclass
class FleetCensus:
    fleet: int = 0                    # the number to quote
    with_gps: int = 0
    with_footage: int = 0
    dark: list = field(default_factory=list)      # registered, no footage
    unregistered_footage: list = field(default_factory=list)
    excluded_deleted: int = 0
    excluded_test: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"fleet": self.fleet, "with_gps": self.with_gps,
                "with_footage": self.with_footage, "dark": self.dark,
                "unregistered_footage": self.unregistered_footage,
                "excluded_deleted": self.excluded_deleted,
                "excluded_test": self.excluded_test}

    def headline(self) -> str:
        """The phrasing every report should use, with both halves present."""
        return (f"{self.with_footage} of {self.fleet} deployed cameras have "
                f"footage on disk"
                + (f"; {len(self.dark)} have none ({', '.join(self.dark)})"
                   if self.dark else "; none are dark")
                + f". Excluded: {self.excluded_deleted} soft-deleted, "
                  f"{len(self.excluded_test)} test/demo.")


def _dirs_with_clips() -> set:
    if not CLIPS.is_dir():
        return set()
    # A directory is not footage. Two of them were empty and still counted.
    return {p.name for p in CLIPS.iterdir()
            if p.is_dir() and p.name.startswith("CAM_") and any(p.glob("*.mp4"))}


def census(db: Session) -> FleetCensus:
    from backend.db.models import Camera

    c = FleetCensus()
    rows = db.query(Camera).all()
    deployed = set()
    for cam in rows:
        if getattr(cam, "is_deleted", False):
            c.excluded_deleted += 1
            continue
        if is_test_camera(cam.id, cam.name):
            c.excluded_test.append(cam.id)
            continue
        deployed.add(cam.id)
        c.fleet += 1
        if cam.lat is not None and cam.lon is not None:
            c.with_gps += 1

    on_disk = _dirs_with_clips()
    c.with_footage = len(deployed & on_disk)
    c.dark = sorted(deployed - on_disk)
    c.unregistered_footage = sorted(on_disk - deployed)
    return c
