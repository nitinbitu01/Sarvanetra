"""Relay one corp8 camera as a local real-time MPEG-TS stream, so the pipeline
can decode it at full frame rate without ever waiting on the portal.

WHAT THE PORTAL ACTUALLY SERVES — measured 2026-09-11 on CAM_09
  * A fixed 12-hour VOD: PLAYLIST-TYPE:VOD, EXT-X-ENDLIST, 4,306 segments of
    10 s. Re-fetched 25 s later it had not grown. There is no live edge.
  * The camera's own burnt-in clock on the last segment reads
    "14/06/2026 08:59:47 Sun" at offset 43,031.16 s, so offset 0 is
    21:02:35.84 on 13/06/2026. The portal's "live" feed is that recording.
  * Source video is 1920x1080 H.264 at 25 fps — 250 frames per segment. The
    dashboard's frame rate was never limited by the source: the pipeline was
    decoding it at 5 fps.
  * One 10 s segment (2.94 MB) fetched in 0.7-1.8 s, several times real time.
    But the portal answers HTTP 429 after ~45 requests on one session and
    needs a re-login, and the documented deep-seek rate has been as low as
    0.10 MB/s. A player streaming straight from the portal stalls through
    both, and a stall is what "lag" is.

WHY A TS STREAM AND NOT A LOCAL HLS PLAYLIST — measured
  The first version of this relay served a sliding local HLS playlist. ffmpeg
  read it at 0.3x real time — 30 s of video took 100 s — because its live-HLS
  reader waits a target-duration between playlist reloads. Served as one
  continuous MPEG-TS byte stream there is no playlist to wait on: ffmpeg reads
  it like any live camera feed.

WHAT THIS DOES
    download head   segments fetched from the portal AHEAD of the live point,
                    in parallel, and decrypted as they land (AES-128), so a
                    slow segment or a re-login costs cushion, never a frozen
                    picture
    live point      advances one segment per 10 s of wall-clock time
    stream          http://127.0.0.1:8091/CAM_09.ts — each client starts at the
                    live point and is fed at no more than real time

  Every byte shown comes from the portal. Nothing is read from data/clips.

WHERE PLAYBACK STARTS
  "auto" aligns the recording to the wall clock: at 08:10 it plays what the
  camera recorded at 08:10, so the burnt-in clock matches the real time of day
  (the date stays 14/06/2026 — that is the portal's recording). Outside the
  recording's daylight it starts at 07:30, and at the end of the recording it
  loops back to 07:00; a client is disconnected at the loop and reconnects
  cleanly rather than being fed a backwards timestamp jump.

THE PORTAL METERS WATCH TIME — measured 2026-09-11
  After 13 min 21 s of continuous streaming the portal refused every request
  for the account with HTTP 403 "watch time limit reached — please wait for
  your cooldown, then watch again". A fresh login does not lift it. The relay
  waits it out with one playlist check every 150 s and resumes on its own;
  it does not try to get around the quota. For a demo, start the relay just
  before it is needed so the quota is spent while someone is watching.

ONE CONSUMER
  The portal session is shared across processes and has a ~45-request budget.
  Run the API with SENTINEL_CORP8_HEALTH=0 and SENTINEL_SNAPSHOT_PORTAL_GRAB=0
  so this relay is the only thing spending it.

  python -m backend.scripts.cam09_live_relay --camera CAM_09
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests                                                    # noqa: E402
from cryptography.hazmat.primitives.ciphers import (                # noqa: E402
    Cipher, algorithms, modes,
)

from backend.services.corp8_session import get_corp8_session        # noqa: E402

BASE = os.environ.get("CORP8_BASE", "https://cctv.corp8.cloud").rstrip("/")

# Burnt-in clock at offset 0 of CAM_09's recording, in seconds after midnight:
# 08:59:47 at offset 43,031.16 s  ->  21:02:35.84 at offset 0.
REC_START_CLOCK_S = 8 * 3600 + 59 * 60 + 47 - 43031.16 + 86400
DAYLIGHT_FROM_CLOCK_S = 6 * 3600 + 15 * 60      # "auto" only aligns after 06:15
FALLBACK_START_OFFSET = 37680                   # 07:30, the known-good start
LOOP_START_CLOCK_S = 7 * 3600                   # loop back to 07:00 at the end

LOGIN_MIN_GAP_S = 20.0
LOCKOUT_BACKOFF_S = 100.0                       # HTTP 403: ~90 s lockout measured
LOCKOUT_MAX_BACKOFF_S = 400.0                   # 100, 200, 400 if it persists
WATCH_PROBE_S = 150.0                           # cooldown check interval
MAX_AHEAD_OF_LIVE = 2                           # segments a client may lead by
# The live clock starts only once this many segments are on disk. Measured
# 08:47: started with an empty buffer while the portal took 31.7 s a segment,
# the first client waited out its 30 s deadline, was dropped and reconnected —
# a 14 s freeze at start-up. A few segments of lead up front cost ~a minute of
# lag behind the wall clock and remove that.
WARMUP_SEGMENTS = 3


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} relay  {msg}", flush=True)


def clock_to_offset(clock_s: float) -> float:
    return (clock_s - REC_START_CLOCK_S) % 86400


def offset_to_clock(off: float) -> str:
    t = (REC_START_CLOCK_S + off) % 86400
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


class Relay:
    def __init__(self, cam: str, out_dir: Path, start: str, ahead: int,
                 back: int, workers: int):
        self.cam = cam.upper()
        self.cam_path = self.cam.lower().replace("_", "")
        self.out = out_dir
        self.ahead = ahead
        self.back = back
        self.workers = workers
        self.sess = get_corp8_session()
        self._cookie = self.sess.cookie_header()
        self._login_gen = 0
        self._login_busy = False
        self._lockouts = 0
        self._last_login = 0.0
        self.blocked_until = 0.0
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.have: dict[int, int] = {}          # local n -> real segment index
        self.in_flight: set[int] = set()
        self.active: dict[int, int] = {}        # client id -> local n it is on
        self._bad_sync_logged = False
        self.stats = {"downloads": 0, "bytes": 0, "errors": 0, "relogins": 0,
                      "last_fetch_s": None, "clients": 0, "served_segments": 0,
                      "started": datetime.now().isoformat(timespec="seconds")}

        self._load_playlist()
        last_full = self.last_full
        if start == "auto":
            now = datetime.now()
            clock = now.hour * 3600 + now.minute * 60 + now.second
            off = clock_to_offset(clock)
            if not (clock_to_offset(DAYLIGHT_FROM_CLOCK_S) <= off <= last_full * self.seg_s):
                log(f"wall clock {now:%H:%M} is outside the recording's daylight; "
                    f"starting at {offset_to_clock(FALLBACK_START_OFFSET)} instead")
                off = FALLBACK_START_OFFSET
        else:
            off = float(start)
        self.start_seg = min(int(off // self.seg_s), last_full)
        self.loop_seg = min(int(clock_to_offset(LOOP_START_CLOCK_S) // self.seg_s),
                            last_full)
        self.t0: float | None = None       # set once WARMUP_SEGMENTS are on disk
        log(f"{self.cam}: {len(self.segs)} segments, starting at #{self.start_seg} "
            f"(burnt-in clock {offset_to_clock(self.start_seg * self.seg_s)}), "
            f"looping to #{self.loop_seg} at the end")

    # ── portal access ────────────────────────────────────────────────────────
    #
    # Measured 2026-09-11 08:31-08:34: one 403 hit all six download workers,
    # each slept 100 s on its own and then queued its own login, 20 s apart.
    # Every one of those logins restarted the portal's login lockout, so it
    # never cleared and the 190 s cushion drained to a frozen picture.
    #
    # So refusals are handled ONCE for the whole relay: the first worker to
    # see one pauses all portal traffic and does the single login; the others
    # wait for it and retry with the new cookie. A request refused on a cookie
    # that has since been replaced is just retried — a login invalidates the
    # previous session (one session per IP), so in-flight requests on the old
    # cookie are expected to fail and are not a new lockout. A lockout that
    # survives its login doubles the next pause instead of hammering.
    #
    # The relay owns its cookie: it is read once and replaced only here, so
    # the session's 30-minute age refresh can never log in silently mid-flight.

    def _wait_portal(self) -> None:
        while True:
            with self.lock:
                busy = self._login_busy
                left = self.blocked_until - time.time()
            if not busy and left <= 0:
                return
            time.sleep(min(max(left, 0.2), 1.0))

    def _login_once(self, why: str) -> None:
        log(f"re-login ({why})")
        try:
            self.sess.session(force=True)
            cookie = self.sess.cookie_header()
        except Exception as exc:                                   # noqa: BLE001
            log(f"re-login failed: {type(exc).__name__}: {exc}")
            cookie = None
        with self.lock:
            if cookie:
                self._cookie = cookie
            self._login_gen += 1
            self._last_login = time.time()
            self.stats["relogins"] += 1

    def _sit_out_cooldown(self) -> None:
        """Wait out the portal's watch-time cooldown without spending logins.

        Measured 2026-09-11: after 13 min 21 s of streaming (08:17:40 to
        08:31:01) every request — playlist included, on a session minted
        seconds earlier — answered HTTP 403 "watch time limit reached — please
        wait for your cooldown, then watch again". It is an account quota, not
        a login problem, so a login cannot lift it; one small playlist request
        every WATCH_PROBE_S finds the moment it ends. Every other worker stays
        parked in _wait_portal meanwhile.
        """
        since = time.time()
        self.stats["cooldowns"] = self.stats.get("cooldowns", 0) + 1
        self.stats["cooldown_since"] = datetime.now().isoformat(timespec="seconds")
        log("portal watch-time limit reached; waiting out the cooldown "
            f"(no login, one playlist check every {WATCH_PROBE_S:.0f}s)")
        url = f"{BASE}/{self.cam_path}/index.m3u8"
        while True:
            # The run loop writes status every 0.5 s, but a cooldown at start-up
            # blocks before that loop exists; the dashboard reads this file to
            # say WHY the picture stopped rather than show a stale frame as live.
            try:
                self._atomic_write(self.out / "relay_status.json", json.dumps({
                    "camera": self.cam, "recording_date": "14/06/2026",
                    "waited_min": round((time.time() - since) / 60, 1),
                    "updated": datetime.now().isoformat(timespec="seconds"),
                    **self.stats}, indent=1).encode())
            except Exception:                                      # noqa: BLE001
                pass
            time.sleep(WATCH_PROBE_S)
            with self.lock:
                cookie = self._cookie
            try:
                r = requests.get(url, headers={"Cookie": cookie,
                                               "User-Agent": "Mozilla/5.0"},
                                 timeout=30)
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.text.lstrip().startswith("#EXTM3U"):
                break
            if r.status_code in (401, 429) or "text/html" in \
                    (r.headers.get("content-type") or "").lower():
                self._login_once(f"session lapsed during cooldown (HTTP {r.status_code})")
                continue
            log(f"still in cooldown after {(time.time() - since) / 60:.1f} min")
        waited = (time.time() - since) / 60
        self.stats["cooldown_since"] = None
        self.stats["last_cooldown_wait_min"] = round(waited, 1)
        log(f"watch-time cooldown over after {waited:.1f} min here; resuming")

    def _on_refused(self, status: int, html: bool, used_gen: int,
                    reason: str = "") -> None:
        if status == 403 and "watch time" in reason.lower():
            with self.lock:
                if self._login_busy:
                    return
                self._login_busy = True
            try:
                self._sit_out_cooldown()
            finally:
                with self.lock:
                    self._login_busy = False
            return
        with self.lock:
            if used_gen != self._login_gen or self._login_busy:
                return                     # already handled, or being handled
            self._login_busy = True
            if status == 403:
                self._lockouts += 1
                pause = min(LOCKOUT_BACKOFF_S * 2 ** (self._lockouts - 1),
                            LOCKOUT_MAX_BACKOFF_S)
            else:                          # 401 / 429 / login page: session spent
                pause = max(0.0, LOGIN_MIN_GAP_S - (time.time() - self._last_login))
            self.blocked_until = time.time() + pause
        try:
            if status == 403:
                log(f"HTTP 403 — portal lockout #{self._lockouts}; pausing all "
                    f"portal traffic {pause:.0f}s, then one login")
            time.sleep(pause)
            self._login_once(f"HTTP {status}" + (" login page" if html else ""))
        finally:
            with self.lock:
                self._login_busy = False

    def _get(self, url: str, timeout: float = 45.0) -> requests.Response:
        for attempt in range(6):
            self._wait_portal()
            with self.lock:
                cookie, gen = self._cookie, self._login_gen
            try:
                r = requests.get(url, headers={"Cookie": cookie,
                                               "User-Agent": "Mozilla/5.0"},
                                 timeout=timeout)
            except requests.RequestException as exc:
                self.stats["errors"] += 1
                log(f"network error ({type(exc).__name__}), retry {attempt + 1}")
                time.sleep(1.5 * (attempt + 1))
                continue
            ctype = (r.headers.get("content-type") or "").lower()
            html = "text/html" in ctype
            if r.status_code in (401, 403, 429) or html:
                self.stats["errors"] += 1
                # The reason text tells a login lockout from "one session per
                # IP" (another logged-in browser); markup is stripped, and the
                # body never carries the cookie or the key.
                reason = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text or ""))
                with self.lock:
                    first = gen == self._login_gen and not self._login_busy
                if first:
                    log(f"refused: HTTP {r.status_code} — {reason.strip()[:160]!r}")
                self._on_refused(r.status_code, html, gen, reason)
                continue
            r.raise_for_status()
            with self.lock:
                self._lockouts = 0
            return r
        raise RuntimeError(f"gave up on {url}")

    def _load_playlist(self) -> None:
        r = self._get(f"{BASE}/{self.cam_path}/index.m3u8", timeout=30)
        segs, durs, key_line = [], [], None
        for raw in r.text.splitlines():
            line = raw.strip()
            if line.startswith("#EXT-X-KEY"):
                key_line = line
            elif line.startswith("#EXTINF:"):
                durs.append(float(line.split(":", 1)[1].rstrip(",").split(",")[0]))
            elif line and not line.startswith("#"):
                segs.append(line)
        if not segs:
            raise RuntimeError("portal playlist has no segments")
        self.segs, self.durs = segs, durs
        self.seg_s = durs[0]
        # The final segment of a VOD is a short remainder; never schedule it.
        self.last_full = len(segs) - 2 if durs[-1] < durs[0] else len(segs) - 1

        self.key_method, self.key_iv, key_uri = "NONE", None, None
        if key_line:
            m = re.search(r'METHOD=([A-Z0-9-]+)', key_line)
            self.key_method = m.group(1) if m else "AES-128"
            m = re.search(r'URI="([^"]+)"', key_line)
            key_uri = m.group(1) if m else "enc.key"
            m = re.search(r'IV=0x([0-9A-Fa-f]+)', key_line)
            if m:
                self.key_iv = bytes.fromhex(m.group(1).rjust(32, "0"))
        self.key = None
        if self.key_method == "AES-128":
            if key_uri.startswith("http"):
                key_url = key_uri
            elif key_uri.startswith("/"):
                key_url = BASE + key_uri
            elif key_uri == "enc.key":
                key_url = f"{BASE}/enc.key"      # where prefetch found it
            else:
                key_url = f"{BASE}/{self.cam_path}/{key_uri}"
            # Held in memory only. The key unlocks the feed; it is not written
            # to disk, since nothing here needs it after decryption.
            self.key = self._get(key_url, timeout=30).content
        log(f"playlist loaded; {self.key_method}, "
            f"IV {'explicit' if self.key_iv else 'implicit (segment number)'}")

    def _decrypt(self, data: bytes, r_idx: int) -> bytes:
        if self.key_method != "AES-128":
            return data
        iv = self.key_iv or r_idx.to_bytes(16, "big")
        dec = Cipher(algorithms.AES(self.key), modes.CBC(iv)).decryptor()
        out = dec.update(data) + dec.finalize()
        pad = out[-1] if out else 0
        if 1 <= pad <= 16 and out[-pad:] == bytes([pad]) * pad:
            out = out[:-pad]
        if out[:1] != b"\x47" and not self._bad_sync_logged:
            self._bad_sync_logged = True
            log("WARNING: decrypted segment does not start with the TS sync byte")
        return out

    # ── schedule ─────────────────────────────────────────────────────────────
    def real_index(self, n: int) -> int:
        r = self.start_seg + n
        if r <= self.last_full:
            return r
        span = self.last_full - self.loop_seg + 1
        return self.loop_seg + (r - self.last_full - 1) % span

    def live_index(self) -> int:
        if self.t0 is None:
            return 0
        return int((time.time() - self.t0) // self.seg_s)

    # ── download ─────────────────────────────────────────────────────────────
    @staticmethod
    def _atomic_write(dest: Path, data: bytes) -> None:
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        for attempt in range(8):
            try:
                os.replace(tmp, dest)
                return
            except PermissionError:
                time.sleep(0.02 * (attempt + 1))
        os.replace(tmp, dest)

    def _grab(self, n: int) -> None:
        r_idx = self.real_index(n)
        try:
            t = time.time()
            r = self._get(f"{BASE}/{self.cam_path}/{self.segs[r_idx]}", timeout=90)
            plain = self._decrypt(r.content, r_idx)
            self._atomic_write(self.out / f"p{n:06d}.ts", plain)
            dt = time.time() - t
            with self.cond:
                self.have[n] = r_idx
                self.stats["downloads"] += 1
                self.stats["bytes"] += len(r.content)
                self.stats["last_fetch_s"] = round(dt, 2)
                self.cond.notify_all()
        except Exception as exc:                                   # noqa: BLE001
            self.stats["errors"] += 1
            log(f"segment #{r_idx} failed: {type(exc).__name__}: {exc}")
        finally:
            with self.lock:
                self.in_flight.discard(n)

    # ── serving ──────────────────────────────────────────────────────────────
    def stream(self, wfile) -> None:
        """Feed one client from the live point onward, never ahead of it."""
        cid = id(wfile)
        n = self.live_index()
        prev_r = None
        with self.lock:
            self.active[cid] = n
            self.stats["clients"] = len(self.active)
        log(f"client connected at live #{n} "
            f"(clock {offset_to_clock(self.real_index(n) * self.seg_s)})")
        try:
            while True:
                while n > self.live_index() + MAX_AHEAD_OF_LIVE:
                    time.sleep(0.2)
                with self.cond:
                    deadline = time.time() + 30.0
                    while n not in self.have:
                        # Rejoin only if the live point has actually moved past
                        # n. A later segment landing before n (downloads run in
                        # parallel) is not "behind" — re-picking the same n
                        # there spun this loop while holding the lock.
                        if self.have and n < min(self.have) and self.live_index() > n:
                            n = self.live_index()          # fell behind: rejoin
                            continue
                        if self.t0 is None:                # still warming up
                            deadline = time.time() + 30.0
                        left = deadline - time.time()
                        if left <= 0:
                            log(f"segment {n} never arrived; closing client")
                            return
                        self.cond.wait(min(left, 1.0))
                    r_idx = self.have[n]
                    self.active[cid] = n
                if prev_r is not None and r_idx != prev_r + 1:
                    log("recording looped; closing client so it reconnects cleanly")
                    return
                data = (self.out / f"p{n:06d}.ts").read_bytes()
                wfile.write(data)
                wfile.flush()
                self.stats["served_segments"] += 1
                prev_r = r_idx
                n += 1
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
                OSError):
            pass
        finally:
            with self.lock:
                self.active.pop(cid, None)
                self.stats["clients"] = len(self.active)
            log("client disconnected")

    def serve(self, port: int) -> None:
        relay = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self):                                      # noqa: N802
                name = self.path.split("?")[0].strip("/").upper()
                if name not in (f"{relay.cam}.TS", "LIVE.TS"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "video/mp2t")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                relay.stream(self.wfile)

            def log_message(self, *args):                          # noqa: D401
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True,
                         name="relay-http").start()
        log(f"serving http://127.0.0.1:{port}/{self.cam}.ts")

    # ── housekeeping ─────────────────────────────────────────────────────────
    def _cleanup(self, c: int) -> None:
        with self.lock:
            floor = c - self.back
            if self.active:
                floor = min(floor, min(self.active.values()))
            old = [n for n in self.have if n < floor - 1]
            for n in old:
                self.have.pop(n, None)
        for n in old:
            try:
                (self.out / f"p{n:06d}.ts").unlink()
            except OSError:
                pass

    def run(self) -> None:
        pool = ThreadPoolExecutor(max_workers=self.workers)
        last_report = 0.0
        last_reload = time.time()
        try:
            while True:
                c = self.live_index()
                with self.lock:
                    todo = [n for n in range(max(0, c - 1), c + self.ahead + 1)
                            if n not in self.have and n not in self.in_flight]
                    todo = todo[: max(0, self.workers - len(self.in_flight))]
                    self.in_flight.update(todo)
                for n in todo:
                    pool.submit(self._grab, n)
                self._cleanup(c)
                if self.t0 is None:
                    with self.lock:
                        warm = all(i in self.have for i in range(WARMUP_SEGMENTS))
                    if warm:
                        self.t0 = time.time()
                        log(f"warm: {WARMUP_SEGMENTS} segments on disk, live clock started")

                with self.lock:
                    cushion, n = 0, c
                    while n in self.have:
                        cushion += 1
                        n += 1
                    positions = sorted(self.active.values())
                status = {
                    "camera": self.cam,
                    "live_local_index": c,
                    "cushion_segments": cushion,
                    "cushion_seconds": round(cushion * self.seg_s, 1),
                    "now_playing_clock": offset_to_clock(self.real_index(c) * self.seg_s),
                    "recording_date": "14/06/2026",
                    "client_positions": positions,
                    "updated": datetime.now().isoformat(timespec="seconds"),
                    **self.stats,
                }
                try:
                    self._atomic_write(self.out / "relay_status.json",
                                       json.dumps(status, indent=1).encode())
                except Exception:                                  # noqa: BLE001
                    pass

                if time.time() - last_report > 10:
                    last_report = time.time()
                    log(f"live #{c} cushion {cushion * self.seg_s:.0f}s clock "
                        f"{status['now_playing_clock']} clients {len(positions)} "
                        f"dl {self.stats['downloads']} err {self.stats['errors']} "
                        f"relogin {self.stats['relogins']} last {self.stats['last_fetch_s']}s")

                if time.time() - last_reload > 1800:
                    last_reload = time.time()
                    try:
                        self._load_playlist()
                    except Exception as exc:                       # noqa: BLE001
                        log(f"playlist reload failed: {exc}")
                time.sleep(0.5)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="CAM_09")
    ap.add_argument("--start", default="auto",
                    help="'auto' to follow the wall clock, or an offset in seconds")
    ap.add_argument("--ahead", type=int, default=9,
                    help="segments downloaded ahead of the live point (cushion)")
    ap.add_argument("--back", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Surface the session module's own log lines, so every portal login this
    # process makes is visible in the relay log. A login here was invisible
    # before, which left the 08:31 lockout without a traceable cause.
    import logging
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(name)s  %(message)s")

    cam = args.camera.upper()
    out = Path(args.out) if args.out else ROOT / "output" / "hls_buffer" / f"{cam}_live"
    out.mkdir(parents=True, exist_ok=True)
    # Start clean, and remove anything a previous version left behind —
    # including a key file, which this version never writes.
    for p in list(out.glob("*.ts")) + list(out.glob("*.part")) + \
            [out / "index.m3u8", out / "relay_status.json", out / "enc.key"]:
        try:
            p.unlink()
        except OSError:
            pass

    relay = Relay(cam, out, args.start, args.ahead, args.back, args.workers)
    relay.serve(args.port)
    try:
        relay.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
