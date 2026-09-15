"""Recover a clip's real capture time from the timestamp burnt into the video.

Why this exists: nothing else in the system knows when a clip was actually
recorded. The filename suffix looks like an hour but is an ingestion slot -
CAM_13_0730 carries an overlay reading 23:19:59, and CAM_14_0830 reads
00:20:50. Using the suffix would put a third of the fleet's night footage
under a daytime label.

Why it matters: day versus night decides how a crop should be reviewed and
which model it belongs to for retraining, and it is not reliably visible in
the pixels. An indoor corridor lit at 21:04 is identical to the same corridor
at 13:04, and a pixel classifier fitted across these 26 cameras scored 50%
held-out once a scene-clutter shortcut was removed. The overlay clock plus the
camera's GPS gives solar elevation exactly, so the question stops being a
guess.

Measured on twelve clips spanning both cities and both halves of the day, the
clock was recovered on ten. Cost is 200-750 ms, paid once per clip and cached,
against a ten-minute clip.

Cameras whose own clock is wrong are a real field condition, not an error
here: CAM_24 stamps 08-08-2026 on June footage. This module reports what the
camera says and leaves that judgement to the caller.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Overlays in this fleet use dd-mm-yyyy or yyyy-mm-dd, with '-' or '/'.
_DMY = re.compile(r"(\d{2})[-/](\d{2})[-/](\d{4})")
_YMD = re.compile(r"(\d{4})[-/](\d{2})[-/](\d{2})")
# Seconds are optional; OCR frequently reads ':' as '.' on thin strokes.
_CLOCK = re.compile(r"(\d{1,2})[:.](\d{2})(?:[:.](\d{2}))?")
_MERIDIEM = re.compile(r"\b([AP])\.?M\b", re.I)

_CACHE_LOCK = threading.Lock()
_MEM: dict[str, Optional[str]] = {}


def _reader():
    """EasyOCR is loaded lazily: most processes never read a clip clock."""
    global _READER
    try:
        return _READER
    except NameError:
        pass
    import easyocr
    _READER = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _READER


def _parse(text: str) -> Optional[datetime]:
    clock = _CLOCK.search(text)
    if not clock:
        return None
    hh, mm = int(clock.group(1)), int(clock.group(2))
    ss = int(clock.group(3) or 0)

    mer = _MERIDIEM.search(text)
    if mer:
        # 12-hour overlays: 12:04 AM is 00:04, 09:04 PM is 21:04.
        if mer.group(1).upper() == "P" and hh != 12:
            hh += 12
        elif mer.group(1).upper() == "A" and hh == 12:
            hh = 0
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        return None

    d = _DMY.search(text)
    if d:
        day, month, year = int(d.group(1)), int(d.group(2)), int(d.group(3))
    else:
        d = _YMD.search(text)
        if not d:
            return None
        year, month, day = int(d.group(1)), int(d.group(2)), int(d.group(3))
    try:
        return datetime(year, month, day, hh, mm, ss)
    except ValueError:
        return None


def read_clip_start(clip_path: Path, cache_dir: Optional[Path] = None
                    ) -> Optional[datetime]:
    """Capture time stamped on the clip's first frame, or None if unreadable.

    Cached in memory and, when `cache_dir` is given, on disk - the OCR result
    for a given file never changes.
    """
    key = str(clip_path)
    with _CACHE_LOCK:
        if key in _MEM:
            v = _MEM[key]
            return datetime.fromisoformat(v) if v else None

    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "clip_clocks.json"
        if cache_file.is_file():
            try:
                disk = json.loads(cache_file.read_text(encoding="utf-8"))
                if key in disk:
                    with _CACHE_LOCK:
                        _MEM[key] = disk[key]
                    return (datetime.fromisoformat(disk[key])
                            if disk[key] else None)
            except (OSError, ValueError):
                pass

    result = _ocr_first_frame(clip_path)
    with _CACHE_LOCK:
        _MEM[key] = result.isoformat() if result else None
    if cache_file is not None:
        try:
            disk = ({} if not cache_file.is_file()
                    else json.loads(cache_file.read_text(encoding="utf-8")))
            disk[key] = result.isoformat() if result else None
            cache_file.write_text(json.dumps(disk, indent=1), encoding="utf-8")
        except (OSError, ValueError) as exc:
            logger.debug("clip clock cache write failed: %s", exc)
    return result


def _ocr_first_frame(clip_path: Path) -> Optional[datetime]:
    cap = cv2.VideoCapture(str(clip_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    h = frame.shape[0]
    # Placement is not consistent across vendors in this fleet, so both bands
    # are read and whichever yields a parseable clock wins.
    bands = (frame[: int(h * 0.10), :], frame[int(h * 0.88):, :])
    try:
        reader = _reader()
    except Exception as exc:                                      # noqa: BLE001
        logger.warning("clip clock OCR unavailable: %s", exc)
        return None

    for band in bands:
        # The digits are small against a 1080p frame; recall on thin strokes
        # improves sharply with a 2x upscale.
        big = cv2.resize(band, None, fx=2.0, fy=2.0,
                         interpolation=cv2.INTER_CUBIC)
        try:
            found = reader.readtext(big)
        except Exception as exc:                                  # noqa: BLE001
            logger.debug("OCR failed on %s: %s", clip_path.name, exc)
            continue
        text = " ".join(t for _, t, c in found if c > 0.30)
        parsed = _parse(text)
        if parsed:
            return parsed
    return None


def solar_elevation_deg(when: datetime, lat: float, lon: float) -> float:
    """Sun elevation above the horizon, in degrees.

    NOAA's low-precision algorithm, good to about half a degree - far tighter
    than needed to separate day from night. `when` is local Indian time
    (UTC+5:30), which is what the camera overlays record.
    """
    utc = when - timedelta(hours=5, minutes=30)
    utc = utc.replace(tzinfo=timezone.utc)

    # Fractional year, radians.
    day_of_year = utc.timetuple().tm_yday
    hour = utc.hour + utc.minute / 60 + utc.second / 3600
    gamma = 2 * math.pi / 365 * (day_of_year - 1 + (hour - 12) / 24)

    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma)
                       - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))

    time_offset = eqtime + 4 * lon
    true_solar = hour * 60 + time_offset
    hour_angle = math.radians(true_solar / 4 - 180)

    lat_r = math.radians(lat)
    cos_zenith = (math.sin(lat_r) * math.sin(decl)
                  + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle))
    return math.degrees(math.asin(max(-1.0, min(1.0, cos_zenith))))


def is_daylight(when: datetime, lat: float, lon: float) -> bool:
    """True when the sun is above the horizon.

    The threshold is -0.833 degrees rather than zero: that is the standard
    sunrise/sunset definition, accounting for atmospheric refraction and the
    sun's apparent radius.
    """
    return solar_elevation_deg(when, lat, lon) > -0.833
