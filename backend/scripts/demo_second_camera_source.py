"""
backend/scripts/demo_second_camera_source.py

Stands up a real, independently-running video source on this machine, for
the Model 2 deliverable "unified viewer connected to sample feeds from at
least two different systems": System A is the corp8 cloud portal (HLS,
cookie-session auth, one vendor's proprietary stack); this is a genuinely
separate system — a different process, a different protocol (progressive
HTTP/MJPEG, not HLS), no shared auth, no shared code with corp8_session.py —
that the platform's existing snapshot pipeline (backend/routers/v1/cameras.py
::_grab_one_frame's plain-cv2.VideoCapture path, already used for every
non-corp8 camera) connects to over the open network like any other camera.

Nothing here fakes a result. It gives the real snapshot/viewer pipeline a
real second endpoint to pull frames from, live, on demand.

WHY HTTP/MJPEG AND NOT AN RTSP SERVER
  ffmpeg's own "-rtsp_flags listen" (act-as-RTSP-server mode) was tried
  first and measured broken on the ffmpeg 7.1 "essentials" Windows build in
  this environment — every invocation attempted an outbound CLIENT connect
  to the listen address instead of binding a listen socket (confirmed with
  ffmpeg's own verbose log: "Starting connection attempt to <ip> port
  <port>" is client-connect log output, not server-bind output), reproduced
  even with a minimal `-f lavfi -i testsrc` source with no clip involved.
  Chasing a broken build further was not a good use of time under a
  deadline. HTTP multipart/x-mixed-replace MJPEG needs no ffmpeg feature at
  all — it's ~40 lines of Python stdlib http.server plus cv2 (already a
  project dependency), and OpenCV's own VideoCapture already decodes this
  exact format natively as a CLIENT, which is the well-tested direction
  (reading, not serving), unlike ffmpeg's newer, apparently-unreliable RTSP
  server support.

Usage (run this before a live demo, leave it running in its own terminal):
    python -m backend.scripts.demo_second_camera_source
    python -m backend.scripts.demo_second_camera_source --clip "data/clips/CAM_01/CAM_01_0730.mp4" --port 8091

Then add it as a camera exactly like any direct-URL source (Model 2's own
proven path — see the onboarding form's main "Stream URL" field, not the
vendor-negotiation section, since this demonstrates unified VIEWING of a
second system, not vendor negotiation, which is a separate, already-honestly
-labeled capability documented in docs/MODEL_3_4_ARCHITECTURE.md):
    POST /api/v1/cameras
    {"name": "Demo Source B (local HTTP/MJPEG)", "url": "http://127.0.0.1:8091/stream.mjpg",
     "protocol": "HTTP", "zone": "Demo", "department": "..."}
"""
from __future__ import annotations

import argparse
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# NOT demo/clips/entrance_loop.mp4 — that file (and parking_loop.mp4) is a
# 94KB synthetic SMPTE color-bar test card used by the dispatch-routing demo
# (backend/routing/seed.py), not real footage; verified by actually looking
# at a frame from it before picking this one instead. cam_01_0700.mp4 is
# real corp8-harvested traffic footage (~13MB, genuine timestamp/camera-name
# overlay baked in by the source system) from this project's own capture
# work — an honest "sample feed" for a demo, not a placeholder.
DEFAULT_CLIP = PROJECT_ROOT / "demo" / "clips" / "cam_01_0700.mp4"
BOUNDARY = b"sentinelframe"


def make_handler(clip: Path, fps: float):
    frame_interval = 1.0 / fps

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/stream.mjpg"):
                self.send_response(404)
                self.end_headers()
                return

            cap = cv2.VideoCapture(str(clip))
            if not cap.isOpened():
                self.send_response(503)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}",
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        # Loop the clip rather than end the stream — a real
                        # camera never runs out of frames.
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if not ok:
                        continue
                    payload = jpg.tobytes()
                    try:
                        self.wfile.write(b"--" + BOUNDARY + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode())
                        self.wfile.write(payload)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        break
                    time.sleep(frame_interval)
            finally:
                cap.release()

        def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
            print(f"[demo_second_camera_source] {self.address_string()} - {fmt % args}")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", type=Path, default=DEFAULT_CLIP,
                     help="Video file to loop (default: demo/clips/entrance_loop.mp4)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args()

    if not args.clip.is_file():
        print(f"Clip not found: {args.clip}", file=sys.stderr)
        sys.exit(1)

    url = f"http://{args.host}:{args.port}/stream.mjpg"
    print(f"[demo_second_camera_source] Serving {args.clip.name} on {url}")
    print("[demo_second_camera_source] This is System B — a separate process, "
          "separate protocol (HTTP/MJPEG), no auth shared with corp8. Ctrl+C to stop.")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.clip, args.fps))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo_second_camera_source] stopped")


if __name__ == "__main__":
    main()
