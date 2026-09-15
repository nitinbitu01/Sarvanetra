"""Prove every corp8 camera decodes real frames through the production path.

RUN THIS GENTLY. The portal rate-limits hard and the penalty escalates:
  ~45 rapid requests on one session   -> HTTP 429 "slow down"
  answering that with fresh logins    -> HTTP 403 across every camera
  repeating that                      -> 403 within seconds, for much longer
Spacing works where volume does not: thirty playlists on ONE session at 1.5s
intervals all succeeded. Hence workers=1 and a 3s stagger by default. Raising
them will lock the account out for minutes, and it is the same account the
live pipeline needs.

Not a reachability check. The old URLs passed a reachability check — they
answered HTTP 200 all day with a login page as text/html. This opens each
camera exactly as CameraReaderThread does, decodes frames, and checks the
pixels are an image rather than a blank or a colour bar.

Reported per camera:
  playlist   is the HLS manifest served, how long the recording is
  decode     frames actually pulled through ffmpeg
  content    pixel standard deviation — a black or frozen frame has almost
             none, so this is what separates "a frame arrived" from "a frame
             arrived and it shows something"

Run:  python -m backend.scripts.verify_corp8_live
      python -m backend.scripts.verify_corp8_live --frames 3 --workers 6
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(cam: dict, frames: int, delay: float = 0.0) -> dict:
    # `delay` is the gap BEFORE this camera starts, not its position in a
    # global schedule. An earlier version passed index * stagger, which is
    # right when cameras run concurrently and badly wrong when they run one at
    # a time: each camera then waited its own cumulative offset AND for every
    # camera before it, so a 20s stagger over 30 cameras came to 20*(0+1+..+29)
    # = 2.4 hours instead of 10 minutes.
    if delay:
        time.sleep(delay)
    from backend.services.corp8_session import get_corp8_session
    from backend.services.hls_ffmpeg_capture import FFmpegHLSCapture

    sess = get_corp8_session()
    cid = cam["id"]
    out = {"id": cid, "name": cam.get("name", ""), "ok": False,
           "segments": 0, "duration": 0.0, "decoded": 0, "std": 0.0,
           "size": "-", "ms": 0.0, "note": ""}

    info = sess.probe(cid)
    out["segments"] = info.get("segments", 0)
    out["duration"] = info.get("duration_sec", 0.0)
    if not info.get("ok"):
        out["note"] = f"playlist not served ({info.get('status')})"
        return out

    t0 = time.perf_counter()
    # ONE shared cookie for every camera. Not a pooled or per-camera session:
    # those mint fresh logins, and logins are what the portal punishes with
    # HTTP 403. Measured directly, eight streams opened on a single shared
    # cookie all stayed up simultaneously, while the pooled version failed
    # 25 of 30 — the failures were caused by my own re-authentication, not by
    # any limit on streaming.
    cap = FFmpegHLSCapture(sess.stream_url(cid), sess.cookie_header(),
                           duration_sec=out["duration"])
    if not cap.isOpened():
        out["note"] = "ffmpeg would not start"
        return out
    stds = []
    for _ in range(frames):
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        out["decoded"] += 1
        out["size"] = f"{fr.shape[1]}x{fr.shape[0]}"
        stds.append(float(fr.std()))
    cap.release()
    out["ms"] = (time.perf_counter() - t0) * 1000.0
    out["std"] = max(stds) if stds else 0.0
    # A decoded frame that is uniformly one colour is not a working camera.
    out["ok"] = out["decoded"] >= 1 and out["std"] > 5.0
    if out["decoded"] and not out["ok"]:
        out["note"] = "frames decoded but image is blank/uniform"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--stagger", type=float, default=3.0,
                    help="Seconds between camera starts. The portal limits request RATE, so starting thirty streams at once is what trips it.")
    ap.add_argument("--per-session", type=int, default=10,
                    help="Cameras to open per login before rotating the "
                         "session. A session's request budget runs out after "
                         "roughly 11 stream opens.")
    ap.add_argument("--session-pause", type=float, default=45.0,
                    help="Seconds to wait before each fresh login. Logins are "
                         "rate-limited too; too many too quickly is what "
                         "causes the 403 lockouts.")
    args = ap.parse_args()

    from backend.services.corp8_session import get_corp8_session, Corp8AuthError
    from backend.services.hls_ffmpeg_capture import ffmpeg_available

    if not ffmpeg_available():
        print("ffmpeg is not on PATH — these streams cannot be decoded without "
              "it, because OpenCV cannot carry the session cookie into the AES "
              "key request.", file=sys.stderr)
        return 1

    try:
        cams = get_corp8_session().cameras()
    except Corp8AuthError as exc:
        print(f"Cannot authenticate: {exc}", file=sys.stderr)
        return 1

    print(f"{len(cams)} cameras on the portal, decoding {args.frames} frame(s) "
          f"each, {args.workers} at a time\n", flush=True)
    # Sequential runs already space themselves out by taking turns, so the
    # stagger is a fixed pause between cameras. Only when several run at once
    # does each need its own offset to avoid a simultaneous burst of stream
    # starts, which is what the portal actually rate-limits.
    if args.workers <= 1:
        results = []
        for i, cam in enumerate(cams):
            # Rotate the session every `--per-session` cameras.
            #
            # A session has a request budget — measured at roughly 45, after
            # which everything returns HTTP 429 "slow down" and only a new
            # login clears it. Opening one stream costs about four requests
            # (playlist, AES key, first segments), so a session is good for
            # about eleven cameras: exactly where sweeps kept stopping,
            # cam01-cam11 succeeding and cam12 onward refused.
            #
            # A new login fixes that, but logins are themselves limited and
            # answering every 429 with one is what produced the 403 lockouts.
            # So the rotation is deliberate and slow: pause, then log in once,
            # rather than re-authenticating per camera.
            if i and args.per_session and i % args.per_session == 0:
                print(f"  -- session budget reached after {i} cameras; "
                      f"pausing {args.session_pause:.0f}s for a fresh login --",
                      flush=True)
                time.sleep(args.session_pause)
                get_corp8_session().session(force=True)
            r = check(cam, args.frames, args.stagger if i else 0.0)
            results.append(r)
            # Print as we go: a sweep of thirty cameras takes minutes, and
            # collecting silently then printing at the end gives no way to
            # tell progress from a hang.
            status = "LIVE" if r["ok"] else f"FAIL {r['note']}"
            print(f"  {r['id']:<8}{r['name'][:28]:<30}{r['decoded']:>2} frames "
                  f"{r['size']:>10}  {status}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(
                lambda ic: check(ic[1], args.frames, ic[0] * args.stagger),
                list(enumerate(cams))))

    print(f"{'id':<8}{'name':<30}{'segs':>6}{'rec(h)':>8}{'frames':>8}"
          f"{'size':>11}{'detail':>8}{'ms':>8}  status")
    print("-" * 100)
    good = 0
    for r in sorted(results, key=lambda x: x["id"]):
        status = "LIVE" if r["ok"] else f"FAIL {r['note']}"
        good += bool(r["ok"])
        print(f"{r['id']:<8}{r['name'][:29]:<30}{r['segments']:>6}"
              f"{r['duration']/3600:>8.1f}{r['decoded']:>8}{r['size']:>11}"
              f"{r['std']:>8.1f}{r['ms']:>8.0f}  {status}")
    print("-" * 100)
    print(f"decoding real video: {good}/{len(results)} cameras")
    return 0 if good == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
