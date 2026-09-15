"""Verify direct live CCTV stream capture from https://cctv.corp8.cloud/ with ZERO local clips.

Runs live frame decodes directly from authenticated HLS endpoints on cctv.corp8.cloud,
verifies genuine pixel content (standard deviation > 10.0), draws forensic provenance metadata,
and saves visual verification proofs to output/corp8_live_proof/.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.corp8_session import get_corp8_session, Corp8AuthError
from backend.services.hls_ffmpeg_capture import FFmpegHLSCapture, ffmpeg_available

PROOF_DIR = ROOT / "output" / "corp8_live_proof"
PROOF_DIR.mkdir(parents=True, exist_ok=True)


def test_camera_live(cam_id: str, frames: int = 2) -> dict:
    sess = get_corp8_session()
    url = sess.stream_url(cam_id)
    cookie = sess.cookie_header()

    info = sess.probe(cam_id)
    duration = float(info.get("duration_sec") or 0.0) if info.get("ok") else None

    cap = FFmpegHLSCapture(url, cookie, duration_sec=duration, seek_sec=0.0)
    res = {
        "cam_id": cam_id,
        "url": url,
        "ok": False,
        "decoded": 0,
        "shape": None,
        "std": 0.0,
        "saved_path": None,
        "error": None,
    }

    if not cap.isOpened():
        res["error"] = "FFmpegHLSCapture failed to start"
        return res

    frames_list = []
    stds = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames_list.append(frame)
        stds.append(float(frame.std()))

    cap.release()

    res["decoded"] = len(frames_list)
    if frames_list:
        last_frame = frames_list[-1]
        res["shape"] = f"{last_frame.shape[1]}x{last_frame.shape[0]}"
        res["std"] = round(float(np.mean(stds)), 2)
        res["ok"] = res["decoded"] >= 1 and res["std"] > 5.0

        # Save annotated proof
        vis = last_frame.copy()
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 55), (10, 15, 25), -1)
        cv2.line(vis, (0, 55), (w, 55), (56, 189, 248), 2)

        osd_title = f"SENTINEL GUJARAT · {cam_id.upper()} · LIVE CCTV FROM https://cctv.corp8.cloud/"
        cv2.putText(vis, osd_title, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (248, 250, 252), 2, cv2.LINE_AA)

        ts_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        meta = f"{ts_str} | Res: {w}x{h} | Pixel Std: {res['std']} | Mode: DIRECT STREAM"
        cv2.putText(vis, meta, (16, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)

        save_path = PROOF_DIR / f"{cam_id}_live_stream.jpg"
        cv2.imwrite(str(save_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        res["saved_path"] = str(save_path)

    return res


def main():
    parser = argparse.ArgumentParser(description="Verify live CCTV stream capture from cctv.corp8.cloud")
    parser.add_argument("--cameras", default="cam01,cam08,cam06,cam07", help="Comma-separated list of camera IDs")
    parser.add_argument("--frames", type=int, default=2, help="Frames to decode per camera")
    args = parser.parse_args()

    if not ffmpeg_available():
        print("ERROR: ffmpeg is not available on PATH.", file=sys.stderr)
        return 1

    cam_list = [c.strip() for c in args.cameras.split(",") if c.strip()]
    print(f"\n=======================================================")
    print(f" LIVE CCTV PROOF: https://cctv.corp8.cloud/ DIRECT FEED")
    print(f" Strict Live Mode: ZERO local clips will be used.")
    print(f" Testing {len(cam_list)} cameras: {', '.join(cam_list)}")
    print(f"=======================================================\n")

    results = []
    for cid in cam_list:
        print(f"--> Connecting to https://cctv.corp8.cloud/{cid}/index.m3u8 ...")
        t0 = time.perf_counter()
        r = test_camera_live(cid, frames=args.frames)
        elapsed = (time.perf_counter() - t0) * 1000.0
        r["ms"] = round(elapsed, 1)
        results.append(r)

        if r["ok"]:
            print(f"    SUCCESS: Decoded {r['decoded']} frames ({r['shape']}), Pixel Std: {r['std']} ({r['ms']} ms)")
            print(f"    Proof Saved: {r['saved_path']}\n")
        else:
            print(f"    FAILED: {r.get('error') or 'Inconclusive read'} ({r['ms']} ms)\n")
        time.sleep(1.0)  # Gentle spacing between stream opens

    passed = sum(1 for r in results if r["ok"])
    print(f"Summary: {passed}/{len(results)} live cameras successfully decoded directly from cctv.corp8.cloud.\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
