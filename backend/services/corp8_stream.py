"""
backend/services/corp8_stream.py — Corp8 CCTV Cloud Camera Stream Receiver

Connects to https://cctv.corp8.cloud/ using authenticated session and streams
live HLS video feeds from Gujarat CCTV cameras (cam01 to cam30).
"""
import urllib.request
import urllib.parse
import http.cookiejar
import json
import re
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Generator, Tuple, Dict, List

DEFAULT_PORTAL = "https://cctv.corp8.cloud"
DEFAULT_PASSWORD = "6KL6-3ZBX-UJMA"


class Corp8CCTVClient:
    """Session-managed client for Corp8 CCTV cloud camera grid."""

    def __init__(self, base_url: str = DEFAULT_PORTAL, password: str = DEFAULT_PASSWORD):
        self.base_url = base_url.rstrip("/")
        self.password = password
        self.cj = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.opener.addheaders = [
            ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"),
            ("Referer", f"{self.base_url}/"),
            ("Origin", self.base_url),
        ]
        self._authenticated = False
        self.cameras: List[Dict[str, str]] = []
        self.login()

    def login(self) -> bool:
        """Authenticate with the portal using the access password."""
        data = urllib.parse.urlencode({"password": self.password}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/auth/login", data=data, method="POST")
        try:
            resp = self.opener.open(req, timeout=10)
            self._authenticated = (resp.status == 200)
            if self._authenticated:
                self._load_camera_manifest()
            return self._authenticated
        except Exception as e:
            print(f"[Corp8Client] Login error: {e}")
            self._authenticated = False
            return False

    def _load_camera_manifest(self):
        """Load list of available cameras from cameras.json."""
        try:
            resp = self.opener.open(f"{self.base_url}/cameras.json", timeout=5)
            self.cameras = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[Corp8Client] Failed to load cameras.json: {e}")
            # Fallback list
            self.cameras = [{"id": f"cam{i:02d}", "name": f"Camera {i}"} for i in range(1, 31)]

    def get_camera_stream_url(self, cam_id: str) -> str:
        """Get the HLS playlist URL for a given camera ID (e.g. 'cam06' or '6')."""
        clean_id = cam_id.lower().strip()
        if not clean_id.startswith("cam"):
            clean_id = f"cam{int(clean_id):02d}"
        return f"{self.base_url}/{clean_id}/index.m3u8"

    def fetch_live_frame(self, cam_id: str) -> Tuple[bool, Optional[np.ndarray]]:
        """Fetch the most recent live frame from the camera's HLS stream."""
        if not self._authenticated:
            if not self.login():
                return False, None

        clean_id = cam_id.lower().strip()
        if not clean_id.startswith("cam"):
            clean_id = f"cam{int(clean_id):02d}"

        playlist_url = f"{self.base_url}/{clean_id}/index.m3u8"
        try:
            resp = self.opener.open(playlist_url, timeout=5)
            playlist = resp.read().decode("utf-8")
            ts_files = [l.strip() for l in playlist.splitlines() if l.strip() and not l.startswith("#")]
            if not ts_files:
                return False, None

            # Pick a recent segment
            target_ts = ts_files[-1] if len(ts_files) > 1 else ts_files[0]
            ts_url = f"{self.base_url}/{clean_id}/{target_ts}"

            ts_data = self.opener.open(ts_url, timeout=8).read()

            # Decode frame from byte stream via temporary memory buffer
            tmp_path = Path(__file__).resolve().parent.parent.parent / "temp_live.ts"
            tmp_path.write_bytes(ts_data)

            cap = cv2.VideoCapture(str(tmp_path))
            ret, frame = cap.read()
            cap.release()
            try:
                tmp_path.unlink()
            except Exception:
                pass

            return ret, frame
        except Exception as e:
            print(f"[Corp8Client] Error fetching frame for {cam_id}: {e}")
            return False, None


if __name__ == "__main__":
    client = Corp8CCTVClient()
    print("Logged in successfully:", client._authenticated)
    print("Cameras available:", len(client.cameras))
    
    # Test grabbing a frame from Camera 6 (Timbavadi gate Junagadh)
    print("\nFetching live frame from cam06 (Timbavadi gate Junagadh)...")
    ok, frame = client.fetch_live_frame("cam06")
    if ok and frame is not None:
        h, w = frame.shape[:2]
        print(f"SUCCESS: Captured live frame {w}x{h} px from cam06!")
    else:
        print("Failed to capture frame.")
