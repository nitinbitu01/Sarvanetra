"""
backend/services/environmental_classifier.py — Automated environmental
and lighting condition classification for Gujarat cameras.
"""
import math
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple
import cv2
import numpy as np

from backend.core.vault_config import LightingCondition

logger = logging.getLogger("sentinel.environment")


class EnvironmentalClassifier:
    """
    Classifies camera environmental state into LightingCondition:
    - DAY: Direct daylight / ambient sun
    - NIGHT: Low ambient light
    - NIGHT_GLARE: Night with high-intensity vehicle headlight glare / lens flares
    - MONSOON_RAIN: Low contrast, high atmospheric scattering, rain streaks
    - DUST_STORM: High yellow/brown hue shift, reduced visibility (Gujarat summer/Kutch)
    """

    @staticmethod
    def calculate_solar_elevation(lat: float, lon: float, dt: Optional[datetime] = None) -> float:
        """
        Computes solar elevation angle (degrees above horizon) for given coordinates.
        Positive = Sun is above horizon (day); Negative = Twilight / Night.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Day of year
        day_of_year = dt.timetuple().tm_yday
        # Solar declination
        declination = 23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 81)))

        # Time in hours UTC + longitude offset
        time_hours = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        lstm = 15 * round(lon / 15.0)
        eot = (
            9.87 * math.sin(math.radians(2 * (360 * (day_of_year - 81) / 365)))
            - 7.53 * math.cos(math.radians(360 * (day_of_year - 81) / 365))
            - 1.5 * math.sin(math.radians(360 * (day_of_year - 81) / 365))
        )
        tc = 4 * (lon - lstm) + eot
        lst = time_hours + tc / 60.0
        hra = 15 * (lst - 12)

        # Elevation angle
        sin_elev = (
            math.sin(math.radians(declination)) * math.sin(math.radians(lat))
            + math.cos(math.radians(declination))
            * math.cos(math.radians(lat))
            * math.cos(math.radians(hra))
        )
        elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))
        return elevation

    @classmethod
    def classify_frame(
        cls,
        frame: Optional[np.ndarray],
        lat: Optional[float] = 23.0225,
        lon: Optional[float] = 72.5714,
        captured_at: Optional[datetime] = None,
    ) -> LightingCondition:
        """
        Classifies frame lighting condition combining solar physics with image metrics.
        """
        lat = lat or 23.0225
        lon = lon or 72.5714
        solar_elev = cls.calculate_solar_elevation(lat, lon, captured_at)
        is_astronomical_night = solar_elev < -6.0  # Civil twilight boundary

        if frame is None or frame.size == 0:
            return LightingCondition.NIGHT if is_astronomical_night else LightingCondition.DAY

        try:
            # Convert to HSV and Grayscale
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            mean_v = float(np.mean(v))
            std_v = float(np.std(v))

            # 1. Night Glare Detection: Night + intense local white clusters (headlights)
            if is_astronomical_night or mean_v < 60.0:
                glare_pixels = np.sum(v > 240) / float(v.size)
                if glare_pixels > 0.04:  # >4% of frame is blinding bright
                    return LightingCondition.NIGHT_GLARE
                return LightingCondition.NIGHT

            # 2. Dust Storm Detection (Kutch / North Gujarat summer dust):
            # High yellow/brown hue band (H: 15-35) with elevated saturation & low contrast
            yellow_dust_mask = (h >= 15) & (h <= 35) & (s >= 40)
            dust_ratio = np.sum(yellow_dust_mask) / float(h.size)
            if dust_ratio > 0.35 and std_v < 40.0:
                return LightingCondition.DUST_STORM

            # 3. Monsoon Rain / Heavy Fog:
            # Washed out colors (low saturation) with low standard deviation in brightness
            mean_s = float(np.mean(s))
            if mean_s < 30.0 and std_v < 35.0 and mean_v < 160.0:
                return LightingCondition.MONSOON_RAIN

            return LightingCondition.DAY
        except Exception as e:
            logger.warning(f"Environmental classification heuristic error: {e}")
            return LightingCondition.NIGHT if is_astronomical_night else LightingCondition.DAY
