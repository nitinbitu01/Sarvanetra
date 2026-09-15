"""Pull a window of a camera's LIVE recording to disk, so it can be decoded smoothly.

WHY THIS EXISTS
  Streaming corp8 through ffmpeg delivers ~1.8 fps for this fleet, measured
  repeatedly with and without -re, with nothing else running and with the raw
  pipe swapped for MJPEG. It is not the GPU, the pipe, or the pipeline: it is
  how fast the portal serves a deeply-seeked VOD over one connection. At 1.8 fps
  the camera view is a slideshow no matter what the rest of the system does.

  But these playlists are RECORDINGS, not live edges, so nothing requires the
  frames to arrive in real time. Measured on this connection:

      sequential segment fetch   0.10 MB/s
      6 parallel workers         0.31 MB/s   -> 120 s of footage in 86 s

  Parallel fetching runs at ~1.4x real time. So a window can be pulled ahead to
  local disk and then decoded from there at whatever frame rate the footage
  actually contains — the network leaves the decode path entirely.

WHAT IT WRITES
  output/hls_buffer/<CAM>/seg*.ts   the segments, still AES-128 encrypted
  output/hls_buffer/<CAM>/enc.key   the key, fetched once
  output/hls_buffer/<CAM>/index.m3u8  a local playlist referencing both

  ffmpeg opens that local playlist and does the decryption itself, exactly as it
  does for the remote one — so nothing downstream needs to know the difference.
  Point the pipeline at it with:

      SENTINEL_HLS_LOCAL_BUFFER=output/hls_buffer/CAM_09/index.m3u8

USAGE
  python -m backend.scripts.prefetch_live_buffer --camera CAM_09 --minutes 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests                                                   # noqa: E402

from backend.services.corp8_session import get_corp8_session       # noqa: E402

# 07:30 in the recording — full daylight and the only offset measured to hold
# frame rate. See start_cam09_live_demo.ps1 for how this number was derived
# from the camera's own burnt-in clock.
DEFAULT_SEEK_SEC = 37680
SEGMENT_SECONDS = 10.0        # CAM_09's #EXTINF; other cameras may differ
WORKERS = 6                   # measured sweet spot: 3x sequential, no 403s


def _playlist_url(base: str, cam: str) -> str:
    return f"{base}/{cam.lower().replace('_', '')}/index.m3u8"


def fetch_playlist(sess, base: str, cam: str) -> tuple[list[str], float]:
    """Segment names and the per-segment duration, from the real playlist."""
    url = _playlist_url(base, cam)
    r = requests.get(url, headers={"Cookie": sess.cookie_header(),
                                   "User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    # The FIRST #EXTINF, not the last: the final segment of a VOD is usually a
    # short remainder (3 s here against a 10 s nominal), and taking that as the
    # segment duration threw the seek-to-index arithmetic out by a factor of
    # three — it asked for segment 14718 of 4306 and bought a negative window.
    segs, dur = [], None
    for line in r.text.splitlines():
        line = line.strip()
        if line.startswith("#EXTINF:") and dur is None:
            try:
                dur = float(line.split(":", 1)[1].rstrip(","))
            except ValueError:
                pass
        elif line and not line.startswith("#"):
            segs.append(line)
    return segs, dur or SEGMENT_SECONDS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="CAM_09")
    ap.add_argument("--minutes", type=float, default=5.0,
                    help="how much footage to buffer")
    ap.add_argument("--seek", type=int, default=DEFAULT_SEEK_SEC,
                    help="seconds into the recording to start at")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cam = args.camera.upper()
    cam_path = cam.lower().replace("_", "")
    sess = get_corp8_session()
    base = os.environ.get("CORP8_BASE", "https://cctv.corp8.cloud").rstrip("/")
    hdrs = {"Cookie": sess.cookie_header(), "User-Agent": "Mozilla/5.0"}

    out_dir = Path(args.out) if args.out else (
        ROOT / "output" / "hls_buffer" / cam)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{cam}] reading playlist ...")
    segs, seg_dur = fetch_playlist(sess, base, cam)
    if not segs:
        print("  playlist had no segments — aborting")
        return 1
    print(f"  {len(segs)} segments of {seg_dur:.0f}s "
          f"({len(segs) * seg_dur / 3600:.1f} h of footage)")

    first = int(args.seek / seg_dur)
    count = max(1, int(args.minutes * 60 / seg_dur))
    if first + count > len(segs):
        count = len(segs) - first
    wanted = segs[first:first + count]
    print(f"  buffering {count} segments from #{first} "
          f"= {count * seg_dur / 60:.1f} min of footage")

    # The key. ffmpeg will use it to decrypt from the local playlist.
    kr = requests.get(f"{base}/enc.key", headers=hdrs, timeout=30)
    kr.raise_for_status()
    (out_dir / "enc.key").write_bytes(kr.content)
    print(f"  enc.key {len(kr.content)} bytes")

    def grab(name: str):
        dest = out_dir / name
        if dest.is_file() and dest.stat().st_size > 4096:
            return name, dest.stat().st_size, True          # already have it
        r = requests.get(f"{base}/{cam_path}/{name}", headers=hdrs, timeout=90)
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(r.content)
        os.replace(tmp, dest)
        return name, len(r.content), False

    t0 = time.time()
    total = 0
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for name, size, cached in ex.map(grab, wanted):
            total += size
            done += 1
            if done % 6 == 0 or done == len(wanted):
                el = time.time() - t0
                print(f"    {done}/{len(wanted)}  {total/1e6:5.1f} MB  "
                      f"{el:5.1f}s  ({total/1e6/max(el,.01):.2f} MB/s)")
    dt = time.time() - t0
    footage = count * seg_dur
    print(f"  fetched {total/1e6:.1f} MB in {dt:.1f}s "
          f"= {footage/max(dt,.01):.1f}x real time")

    # A local playlist over the local files. Same AES declaration as the real
    # one, so ffmpeg decrypts it identically — it just never touches the network.
    lines = ["#EXTM3U", "#EXT-X-VERSION:6",
             f"#EXT-X-TARGETDURATION:{int(seg_dur) + 1}",
             "#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-PLAYLIST-TYPE:VOD",
             "#EXT-X-INDEPENDENT-SEGMENTS",
             '#EXT-X-KEY:METHOD=AES-128,URI="enc.key",'
             "IV=0x00000000000000000000000000000000"]
    for name in wanted:
        lines += [f"#EXTINF:{seg_dur:.6f},", name]
    lines.append("#EXT-X-ENDLIST")
    pl = out_dir / "index.m3u8"
    pl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n  playlist -> {pl}")
    print("  run the pipeline against it with:")
    print(f'    $env:SENTINEL_HLS_LOCAL_BUFFER = "{pl}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
