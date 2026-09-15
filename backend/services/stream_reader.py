import asyncio
import logging
import os
import subprocess
import time
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("sentinel.stream_reader")


class FFmpegStreamReader:
    """
    Subprocess FFmpeg stream reader with OpenCV fallback.
    Decodes real-time RTSP/HLS/HTTP streams directly to raw BGR frames.
    """

    def __init__(
        self,
        stream_url: str,
        camera_id: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 5,
    ):
        self.stream_url = stream_url
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = width * height * 3

        self._process: Optional[subprocess.Popen] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._use_opencv_fallback = False
        self._last_frame: Optional[np.ndarray] = None
        self._consecutive_fails = 0

    def start(self):
        self._running = True
        self._start_ffmpeg()

    def _start_ffmpeg(self):
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel", "quiet",
            "-reconnect", "1",
            "-reconnect_at_eof", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-rtsp_transport", "tcp",
            "-i", self.stream_url,
            "-vf", f"fps={self.fps},scale={self.width}:{self.height}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.frame_size * 2,
            )
            self._use_opencv_fallback = False
        except Exception as e:
            logger.warning(
                f"FFmpeg binary unavailable for {self.camera_id}: {e}. Using OpenCV fallback."
            )
            self._use_opencv_fallback = True
            self._cap = cv2.VideoCapture(self.stream_url)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._running:
            return False, None

        if self._use_opencv_fallback:
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.stream_url)
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._consecutive_fails = 0
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                self._last_frame = frame
                return True, frame
            else:
                self._consecutive_fails += 1
                return self._read_failed()

        if self._process and self._process.stdout:
            try:
                raw = self._process.stdout.read(self.frame_size)
                if len(raw) == self.frame_size:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                    self._consecutive_fails = 0
                    self._last_frame = frame
                    return True, frame
                else:
                    self._consecutive_fails += 1
                    if self._consecutive_fails > 10:
                        self._restart()
                    return self._read_failed()
            except Exception:
                return self._read_failed()

        return self._read_failed()

    def _read_failed(self) -> Tuple[bool, None]:
        """Report a read failure. Returns (False, None) - always.

        This replaces a method named `_generate_synthetic_frame()`, which
        synthesized a black frame captioned "LIVE CCTV FEED [<camera>]" with
        a live timestamp and returned it as `(True, frame)`. That is the most
        dangerous shape of bug available in a surveillance system: a dead
        camera reported as healthy, with a frame that LOOKS live because the
        caption says so and the clock keeps advancing.

        Downstream the damage was silent. YOLO finds nothing in a black
        frame, so the feed yields zero detections, zero alerts and zero
        errors - indistinguishable from "a quiet camera where nothing is
        happening". An operator could watch an offline camera for a whole
        shift and never know.

        Callers must treat (False, None) as "no frame this tick" and let the
        reconnect path (_restart after repeated failures) run. Surfacing a
        dead feed to the operator is the camera-heartbeat / offline path's
        job, not something to paper over with a fabricated frame.
        """
        return False, None

    def _restart(self):
        self.stop()
        time.sleep(1)
        self.start()

    def stop(self):
        self._running = False
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                pass
            self._process = None
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
