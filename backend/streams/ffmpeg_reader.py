import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("sentinel.stream")


@dataclass
class StreamState:
    camera_id: str
    url: str
    name: str
    lat: float
    lon: float
    district: str
    zone: str
    crime_level: str
    is_restricted: bool
    department: str = "unknown"

    is_online: bool = False
    consecutive_failures: int = 0
    last_frame_time: float = 0.0
    total_frames: int = 0
    reconnect_count: int = 0


class FFmpegStreamReader:
    """
    Production-grade stream reader using FFmpeg subprocess with OpenCV fallback.
    """

    def __init__(
        self,
        url: str,
        camera_id: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 5,
        stream_user: str = "",
        stream_pass: str = "",
    ):
        self.url       = url
        self.camera_id = camera_id
        self.width     = width
        self.height    = height
        self.fps       = fps
        self.frame_size = width * height * 3

        if stream_user and stream_pass and "://" in url:
            proto, rest = url.split("://", 1)
            self.url = f"{proto}://{stream_user}:{stream_pass}@{rest}"

        self._process: Optional[subprocess.Popen] = None
        self._running  = False
        self._use_opencv = False

    def start(self) -> bool:
        """Launch FFmpeg subprocess. Returns True if started."""
        cmd = [
            "ffmpeg",
            "-loglevel",            "error",
            "-reconnect",           "1",
            "-reconnect_streamed",  "1",
            "-reconnect_delay_max", "5",
            "-rtsp_transport",      "tcp",
            "-i",                   self.url,
            "-vf",                  f"fps={self.fps},scale={self.width}:{self.height}",
            "-f",                   "rawvideo",
            "-pix_fmt",             "bgr24",
            "-",
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.frame_size * 3,
            )
            self._running = True
            logger.info(f"[{self.camera_id}] FFmpeg started (PID {self._process.pid})")
            return True
        except FileNotFoundError:
            return self._start_opencv_fallback()
        except Exception as e:
            logger.warning(f"[{self.camera_id}] FFmpeg start failed: {e} — falling back to OpenCV/Mock")
            return self._start_opencv_fallback()

    def _start_opencv_fallback(self) -> bool:
        """OpenCV fallback when FFmpeg is unavailable"""
        try:
            self._opencv_cap = cv2.VideoCapture(self.url)
            self._opencv_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._use_opencv = True
            self._running = True
            return True
        except Exception:
            self._running = True
            return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame. Returns None on failure or EOF."""
        if getattr(self, "_use_opencv", False):
            if hasattr(self, "_opencv_cap") and self._opencv_cap and self._opencv_cap.isOpened():
                ret, frame = self._opencv_cap.read()
                if ret and frame is not None:
                    return cv2.resize(frame, (self.width, self.height))
            return None

        if not self._process or self._process.poll() is not None:
            return None

        try:
            raw = self._process.stdout.read(self.frame_size)
            if len(raw) != self.frame_size:
                return None
            return np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            ).copy()
        except Exception:
            return None

    def stop(self):
        self._running = False
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        if hasattr(self, "_opencv_cap") and self._opencv_cap:
            try:
                self._opencv_cap.release()
            except Exception:
                pass
            self._opencv_cap = None
        logger.info(f"[{self.camera_id}] Stream stopped")

    @property
    def is_alive(self) -> bool:
        if getattr(self, "_use_opencv", False):
            return True
        return self._process is not None and self._process.poll() is None
