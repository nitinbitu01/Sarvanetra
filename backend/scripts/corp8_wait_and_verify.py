"""Wait for the portal's rate-limit block to lift, then verify all 30 cameras.

The portal punishes bursts, and the penalty escalates with repetition — so the
worst thing to do while blocked is keep trying. This polls with ONE request
every couple of minutes, which is far below any rate that has ever triggered a
block, and only starts the real verification once the portal answers normally.

Then it runs the camera sweep with a wide stagger between stream starts. The
limit that matters is on STARTS, not on steady-state traffic: each new stream
costs a playlist fetch, an AES key fetch and a segment burst. Measured, one
start every ~3 seconds gave 11 of 30; the theory this tests is that a longer
gap between starts lets all 30 through.

Run:  python -m backend.scripts.corp8_wait_and_verify
      python -m backend.scripts.corp8_wait_and_verify --stagger 20 --max-wait 60
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def portal_ready() -> tuple[bool, str]:
    """One cheap request. True when the portal serves the camera list again."""
    import requests
    from backend.services.corp8_session import Corp8Session, CORP8_BASE

    try:
        email, password = Corp8Session._resolve_credentials(None, None)
    except Exception as exc:                                       # noqa: BLE001
        return False, f"credentials: {exc}"
    try:
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0"
        r = s.post(f"{CORP8_BASE}/auth/login",
                   data={"email": email, "password": password}, timeout=25)
        if r.status_code == 403:
            return False, "403 on login (still rate-limited)"
        rr = s.get(f"{CORP8_BASE}/cameras.json", timeout=20)
        if rr.status_code == 200 and "json" in (rr.headers.get("content-type") or ""):
            return True, f"{len(rr.json())} cameras listed"
        return False, f"cameras.json -> HTTP {rr.status_code}"
    except Exception as exc:                                       # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=float, default=120.0,
                    help="Seconds between readiness checks. Deliberately slow: "
                         "polling hard is what caused the block.")
    ap.add_argument("--max-wait", type=float, default=45.0,
                    help="Give up after this many minutes.")
    ap.add_argument("--stagger", type=float, default=20.0,
                    help="Seconds between camera stream starts.")
    ap.add_argument("--frames", type=int, default=2)
    args = ap.parse_args()

    deadline = time.time() + args.max_wait * 60
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        ready, detail = portal_ready()
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] check {attempt}: "
              f"{'READY — ' if ready else 'blocked — '}{detail}", flush=True)
        if ready:
            break
        # One request per poll, minutes apart. Nothing here can deepen the
        # block; the point is to notice recovery, not to force it.
        time.sleep(args.poll)
    else:
        print(f"\nStill blocked after {args.max_wait:.0f} minutes. That is "
              f"longer than any previous penalty, which suggests the limit is "
              f"not purely time-based — worth asking the organisers whether "
              f"this account has a concurrency or daily cap.")
        return 1

    print(f"\nPortal is answering again. Sweeping all cameras with a "
          f"{args.stagger:.0f}s gap between starts "
          f"(~{30*args.stagger/60:.0f} min).\n", flush=True)
    cmd = [sys.executable, "-m", "backend.scripts.verify_corp8_live",
           "--frames", str(args.frames), "--stagger", str(args.stagger),
           "--workers", "1"]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
