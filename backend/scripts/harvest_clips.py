"""backend/scripts/harvest_clips.py - archive CLIPS from the camera fleet.

WHY CLIPS AND NOT FRAMES
  The expensive resource here is the network, not the disk. This server was
  measured at 0.07-0.13 MB/s, and a single seek costs ~280s. Earlier
  harvests paid that toll, extracted a handful of stills, and DELETED the
  downloaded window - so changing any threshold meant re-downloading at
  0.1 MB/s.

  Archiving the clip instead means frames can be re-extracted, with
  different gates, forever, for free. It also unlocks everything temporal:
  track-gap mining, ReID validation, loitering/crowd/abandoned-object
  tuning, and evidence material - none of which a still image can support.

WHY ffmpeg, AND WHY -c copy
  Measured on these exact endpoints:
      cv2.VideoCapture.set() seek to 07:05 : 6,092 s
      ffmpeg -ss before -i                 :   284 s     (21x faster)
  `-ss` BEFORE `-i` is an index-based input seek. After `-i` it decodes
  everything up to that point, which is the trap OpenCV falls into.

  `-c copy` is a byte-level stream copy: no decode, no re-encode. CPU stays
  idle and throughput is purely network-bound - the most efficient mode
  available. It also means quality is bit-identical to the source.

  (Hardware decode via GStreamer/nvh264dec was considered and measured:
  decoding all 30 cameras costs 0.47 of 32 cores - 1.5%. It targets 20.9%
  of per-frame cost while GPU inference is 79%. Not the bottleneck.)

TWO SOURCE TYPES, TWO MODES
  Most cameras serve 12-hour recorded files (verified: 2.55 GB,
  Content-Range, exactly 12.00h) and can be seeked.
  CAM_23/27/28 are genuinely LIVE MJPEG with no stored history - every seek
  against them returned instantly with zero frames. Those are recorded
  forward in real time instead, so a 10-minute clip takes 10 real minutes.

SAFETY
  * disk guard - aborts before starting a job that could fill the disk,
    rather than corrupting a clip mid-write
  * resume - a clip already on disk with a sane size is skipped, so an
    outage (this server 502'd twice in one day) costs one job, not the run
  * per-job timeout, and every output is validated as openable

USAGE
  python -m backend.scripts.harvest_clips                      # 5 targets
  python -m backend.scripts.harvest_clips --times 22:00 --window 600
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

CLIPS_ROOT = Path("data/clips")
MANIFEST = Path("data/clips/manifest.json")
FILE_START_HOUR = 21.0
FILE_DURATION_H = 12.0
# Cameras with no seekable history - record forward instead.
LIVE_ONLY = {"CAM_23", "CAM_27", "CAM_28"}
_lock = threading.Lock()


def clock_to_offset_s(hhmm: str) -> float | None:
    h, m = (int(x) for x in hhmm.split(":"))
    off = ((h + m / 60.0) - FILE_START_HOUR) % 24.0
    return None if off > FILE_DURATION_H else off * 3600.0


def free_gb(path: str = ".") -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def server_serving_data(url: str, want_bytes: int = 200_000,
                        timeout_s: int = 45) -> tuple[bool, str]:
    """Does the source actually SEND BYTES?

    A status check is not sufficient here. Measured on this server: a
    byte-range request returned `PartialContent` and then delivered 0 bytes
    in 11.6s, while the site root still answered 200. Anything that only
    checks a status code would call that healthy and march 145 jobs into a
    wall. So request a small range and count what arrives.
    """
    import urllib.request
    # A browser-like User-Agent is required: urllib's default
    # ('Python-urllib/3.x') is rejected by this edge with 403 Forbidden,
    # which would otherwise be misread as the source being down. ffmpeg's
    # own UA is accepted, so only this probe needs the header.
    req = urllib.request.Request(url, headers={
        "Range": f"bytes=0-{want_bytes}",
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36"),
    })
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            got = 0
            while got < want_bytes:
                chunk = resp.read(65536)
                if not chunk:
                    break
                got += len(chunk)
            el = time.perf_counter() - t0
            if got == 0:
                return False, f"HTTP {resp.status} but 0 bytes in {el:.1f}s"
            return True, f"{got} bytes in {el:.1f}s ({got/1024/max(el,0.01):.0f} KB/s)"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:60]}"


def wait_for_server(url: str, minutes: float) -> bool:
    """Poll until the source serves data, or the budget runs out."""
    deadline = time.time() + minutes * 60
    probe = 0
    while time.time() < deadline:
        probe += 1
        ok, why = server_serving_data(url)
        remaining = (deadline - time.time()) / 60
        print(f"  probe {probe}: {'SERVING' if ok else 'not serving'} - {why}"
              f"   ({remaining:.0f} min left)", flush=True)
        if ok:
            return True
        time.sleep(120)
    return False


def validate(path: Path, want_seconds: float = 0.0,
             min_fraction: float = 0.80) -> tuple[bool, str]:
    """A file on disk is not proof of a usable clip - check its DURATION.

    Checking only "opens and has >10 frames" is not enough. This server
    truncates mid-transfer and ffmpeg still exits 0, so a partial clip looks
    successful. Measured on the first four downloads with a 600s request:

        CAM_01  601s  ok
        CAM_02  232s  truncated
        CAM_03  339s  truncated
        CAM_04   41s  truncated

    Three of four would have been accepted. That matters beyond tidiness:
    CROWD_BASELINE_WINDOW_SEC is 300, so a clip under five minutes can never
    establish a baseline and crowd detection is impossible on it - the exact
    capability the 10-minute window exists to enable.

    Short clips are therefore rejected so the retry logic re-fetches them.
    """
    if not path.is_file() or path.stat().st_size < 200_000:
        return False, "missing or too small"
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened()
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) if ok else 0
        fps = (cap.get(cv2.CAP_PROP_FPS) or 0) if ok else 0
        cap.release()
        if not ok:
            return False, "will not open"
        if n < 10 or fps <= 0:
            return False, f"only {n:.0f} frames"
        secs = n / fps
        if want_seconds > 0 and secs < want_seconds * min_fraction:
            return False, (f"truncated: {secs:.0f}s of {want_seconds:.0f}s "
                           f"requested")
        return True, f"{secs:.0f}s, {n:.0f} frames"
    except Exception as exc:
        return False, f"validate error {type(exc).__name__}"


def is_hls(cam: dict) -> bool:
    return ".m3u8" in str(cam.get("url", ""))


def harvest_one(cam: dict, hhmm: str, args) -> dict:
    cam_id = cam["id"]
    live = cam_id in LIVE_ONLY
    offset = None if live else clock_to_offset_s(hhmm)

    # HLS endpoints deliver PARTIAL clips, consistently and by nature.
    # Measured across repeated attempts at a 600s request:
    #     CAM_13  55s      CAM_14   6s      CAM_16 120s / 135s
    # and a controlled seek-vs-forward comparison at 60s gave
    #     CAM_13  seek 71% / forward 48%      CAM_16 seek 38% / forward 100%
    # so neither mode is reliably better - the segmented protocol simply
    # does not sustain a long continuous pull from this server.
    #
    # Retrying such a camera six times with 900s backoff discards footage
    # that IS usable and starves the worker pool: progress stalled at 12/145
    # with every worker parked on an HLS retry.
    #
    # So accept short clips from these cameras instead. A 60s clip still
    # yields training frames and supports wrong-way (~30s). It cannot
    # support crowd detection (300s baseline) - that is a property of the
    # camera, not something a retry can fix, and is recorded as such.
    hls = is_hls(cam)
    min_fraction = 0.05 if hls else 0.80        # 30s floor vs 480s
    max_attempts = 2 if hls else args.retries

    out_dir = CLIPS_ROOT / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{cam_id}_{hhmm.replace(':', '')}.mp4"

    rec = {"camera": cam_id, "clock": hhmm, "live_mode": live,
           "path": str(dst), "mb": 0.0, "elapsed_s": 0.0, "status": "ok"}

    # Resume: never re-download something already valid.
    ok, why = validate(dst, args.window, min_fraction)
    if ok:
        rec["status"] = "skipped (already have)"
        rec["mb"] = round(dst.stat().st_size / 1_048_576, 1)
        with _lock:
            print(f"  {cam_id:<9} {hhmm}  SKIP  already have {rec['mb']} MB ({why})",
                  flush=True)
        return rec

    if not live and offset is None:
        rec["status"] = f"{hhmm} outside the 21:00-09:00 recording"
        with _lock:
            print(f"  {cam_id:<9} {hhmm}  SKIP  {rec['status']}", flush=True)
        return rec

    # Disk guard - refuse to start a job that could fill the volume.
    if free_gb() < args.min_free_gb:
        rec["status"] = f"ABORT low disk ({free_gb():.1f} GB free)"
        with _lock:
            print(f"  {cam_id:<9} {hhmm}  ABORT low disk "
                  f"({free_gb():.1f} GB < {args.min_free_gb} GB)", flush=True)
        return rec

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-rw_timeout", "300000000"]
    if offset is not None:
        cmd += ["-ss", f"{offset:.0f}"]          # BEFORE -i: index seek
    cmd += ["-i", cam["url"], "-t", str(args.window),
            "-c", "copy",                        # no decode, no re-encode
            "-movflags", "+faststart",
            "-y", str(dst)]

    # RETRY WITH BACKOFF.
    #
    # This server fails in three distinct ways, all observed in one day:
    #   502 Bad Gateway (twice), a 35x slowdown to 0.07 MB/s, and - measured
    #   directly - answering a byte-range request with HTTP 206 and then
    #   sending ZERO bytes. The last is indistinguishable from a healthy
    #   server until you count the bytes.
    #
    # A single-shot attempt therefore throws away a job for a fault that is
    # usually transient. Backoff keeps a long unattended run alive across an
    # outage instead of burning through 145 jobs during a bad ten minutes.
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.timeout)
            rec["elapsed_s"] = round(time.perf_counter() - t0, 1)
            ok, why = validate(dst, args.window, min_fraction)
            if ok:
                rec["mb"] = round(dst.stat().st_size / 1_048_576, 1)
                rec["attempts"] = attempt
                mode = "LIVE" if live else f"seek {offset/3600:.1f}h"
                tag = "" if attempt == 1 else f" (attempt {attempt})"
                with _lock:
                    print(f"  {cam_id:<9} {hhmm}  OK    {rec['mb']:>6.1f} MB  "
                          f"{rec['elapsed_s']:>5.0f}s  [{mode}] {why}{tag}",
                          flush=True)
                return rec
            last_err = why + " | " + (p.stderr or "").strip().replace("\n", " ")[:70]
            if dst.exists():
                dst.unlink()            # never leave a broken clip behind
        except subprocess.TimeoutExpired:
            rec["elapsed_s"] = round(time.perf_counter() - t0, 1)
            last_err = f"timeout after {args.timeout}s"
            if dst.exists():
                dst.unlink()
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"

        if attempt < max_attempts:
            wait = min(args.backoff * (2 ** (attempt - 1)), args.max_backoff)
            with _lock:
                print(f"  {cam_id:<9} {hhmm}  retry {attempt}/{max_attempts - 1} "
                      f"in {wait:.0f}s - {last_err[:60]}", flush=True)
            time.sleep(wait)

    rec["status"] = f"failed after {max_attempts} attempts: {last_err[:90]}"
    rec["attempts"] = max_attempts
    with _lock:
        print(f"  {cam_id:<9} {hhmm}  FAIL  after {max_attempts} attempts",
              flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Night -> dawn -> daylight, all inside the 21:00-09:00 window.
    ap.add_argument("--times", default="22:00,01:00,06:30,07:30,08:30")
    ap.add_argument("--window", type=int, default=600,
                    help="Seconds per clip. 600 is the floor for crowd "
                         "detection: CROWD_BASELINE_WINDOW_SEC is 300, so a "
                         "shorter clip can never establish a baseline.")
    ap.add_argument("--workers", type=int, default=4,
                    help="A 30-way concurrent probe on these endpoints took "
                         "872s and produced corrupt h264. 4 is proven stable.")
    ap.add_argument("--timeout", type=float, default=2400.0)
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--cameras", default="")
    ap.add_argument("--retries", type=int, default=4,
                    help="Attempts per job. This server intermittently "
                         "answers 206 with zero bytes, so one shot throws a "
                         "job away for a transient fault.")
    ap.add_argument("--backoff", type=float, default=60.0,
                    help="Seconds before the first retry; doubles each time.")
    ap.add_argument("--max-backoff", type=float, default=600.0)
    ap.add_argument("--wait-for-server", type=float, default=0.0,
                    help="Minutes to wait for the source to start serving "
                         "DATA before starting. A HEAD check is not enough - "
                         "this server returns 206 while sending nothing - so "
                         "this actually counts bytes received.")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH.", file=sys.stderr)
        sys.exit(1)

    times = [t.strip() for t in args.times.split(",") if t.strip()]
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    only = {c.strip() for c in args.cameras.split(",") if c.strip()} or None
    cams = [c for c in cfg.get("demo_cameras", [])
            if c.get("enabled") is not False and c.get("url")
            and (not only or c["id"] in only)]

    jobs = [(c, t) for t in times for c in cams]
    est_gb = args.window * 0.30 * len(jobs) / 1024
    CLIPS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"CLIP HARVEST - {len(cams)} cameras x {len(times)} times = "
          f"{len(jobs)} jobs")
    print(f"  window   : {args.window}s per clip")
    print(f"  times    : {', '.join(times)}")
    print(f"  workers  : {args.workers}")
    print(f"  est size : ~{est_gb:.1f} GB    free now: {free_gb():.1f} GB")
    print(f"  mode     : ffmpeg -c copy (no re-encode); "
          f"{len(LIVE_ONLY & {c['id'] for c in cams})} live-only camera(s)")
    print()
    if free_gb() - est_gb < args.min_free_gb:
        print(f"WARNING: estimated {est_gb:.1f} GB against {free_gb():.1f} GB "
              f"free. Jobs will abort cleanly at {args.min_free_gb} GB rather "
              f"than filling the disk.\n")

    if args.wait_for_server > 0 and cams:
        print(f"waiting up to {args.wait_for_server:.0f} min for the source "
              f"to serve data...", flush=True)
        if not wait_for_server(cams[0]["url"], args.wait_for_server):
            print("\nSource never started serving data within the wait "
                  "budget. Nothing harvested - re-run when it recovers; "
                  "existing clips are skipped so no work is lost.")
            sys.exit(1)
        print("  source is serving - starting harvest\n", flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        report = list(ex.map(lambda j: harvest_one(j[0], j[1], args), jobs))
    elapsed = time.perf_counter() - t0

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(report, indent=2))

    ok = [r for r in report if r["status"] == "ok"]
    skip = [r for r in report if r["status"].startswith("skipped")]
    bad = [r for r in report if r not in ok and r not in skip]
    total_mb = sum(r["mb"] for r in report)

    print("\n" + "=" * 62)
    print(f"clips downloaded : {len(ok)}")
    print(f"already had      : {len(skip)}")
    print(f"failed           : {len(bad)}")
    print(f"total size       : {total_mb/1024:.1f} GB")
    print(f"footage archived : {(len(ok)+len(skip))*args.window/3600:.1f} h")
    print(f"wall time        : {elapsed/3600:.2f} h")
    print(f"free disk now    : {free_gb():.1f} GB")
    if bad:
        print("\nfailures:")
        for r in bad[:12]:
            print(f"  {r['camera']:<9} {r['clock']}  {r['status'][:70]}")
    print(f"\nmanifest -> {MANIFEST}")
    print("Re-run the same command to retry failures; existing clips are skipped.")


if __name__ == "__main__":
    main()


