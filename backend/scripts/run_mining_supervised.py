"""backend/scripts/run_mining_supervised.py — keep mining alive overnight.

WHY A SUPERVISOR
  Mining runs for hours unattended. Some failures cannot be handled inside the
  process at all: a CUDA context that faults badly enough poisons every later
  GPU call, so the only real fix is a fresh process. Left alone, that turns
  into hours of "sweeping" clips that write nothing - which is exactly what
  happened once here, when a fault on clip 2 let clips 3-43 fail silently.

  Because completed clips are recorded in clips_done.json, a restart RESUMES
  rather than repeating work. So the right response to a wedged process is to
  restart it, not to abandon the remaining clips.

THE ONE THING THAT DOES STOP IT
  Restarting only helps if each attempt makes progress. If an attempt finishes
  no new clips, restarting again would loop forever on the same broken state -
  a wrong model path, a missing dataset, an empty disk. Two consecutive
  no-progress attempts and it stops and says why.

  That is the worst case: not one clip failing, but nothing succeeding.

USAGE
  python -m backend.scripts.run_mining_supervised
  python -m backend.scripts.run_mining_supervised --max-restarts 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DONE_FILE = Path("output/plate_corpus/clips_done.json")
EXIT_RESTART = 75


def done_count() -> int:
    if not DONE_FILE.is_file():
        return 0
    try:
        return len(json.loads(DONE_FILE.read_text()))
    except Exception:                                    # noqa: BLE001
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--cameras", default="ALL")
    ap.add_argument("--times", default="0630,0730,0830")
    ap.add_argument("--min-plate-px", type=float, default=45.0)
    ap.add_argument("--max-frames-per-track", type=int, default=12)
    ap.add_argument("--max-tracks-per-clip", type=int, default=200)
    args = ap.parse_args()

    base = [sys.executable, "-m", "backend.scripts.plate_mine_corpus",
            "--cameras", args.cameras, "--times", args.times,
            "--min-plate-px", str(args.min_plate_px),
            "--max-frames-per-track", str(args.max_frames_per_track),
            "--max-tracks-per-clip", str(args.max_tracks_per_clip)]

    start_done = done_count()
    print(f"supervisor starting — {start_done} clips already complete")
    no_progress = 0

    for attempt in range(1, args.max_restarts + 2):
        before = done_count()
        print(f"\n{'='*60}\nattempt {attempt} — {before} clips done so far\n"
              f"{'='*60}", flush=True)

        rc = subprocess.call(base)
        after = done_count()
        gained = after - before
        print(f"\nattempt {attempt} finished: exit {rc}, "
              f"+{gained} clips (total {after})", flush=True)

        if rc == 0:
            print("mining completed normally.")
            break

        if gained == 0:
            no_progress += 1
            # Restarting only helps if attempts make progress. Two in a row
            # that complete nothing means the state itself is broken and
            # another restart would loop on it.
            if no_progress >= 2:
                print(f"\n*** Two attempts in a row completed no clips. "
                      f"Restarting again would loop on the same fault rather "
                      f"than make progress.", file=sys.stderr)
                print(f"*** Stopping with {after} clips done. Check "
                      f"model.path, the clips directory, and free disk.",
                      file=sys.stderr)
                sys.exit(1)
        else:
            no_progress = 0

        if attempt > args.max_restarts:
            print(f"\n*** restart budget ({args.max_restarts}) exhausted — "
                  f"stopping with {after} clips done.", file=sys.stderr)
            sys.exit(1)

        # Let the GPU settle before rebuilding a context on it.
        print(f"restarting in 20s (exit {rc}"
              f"{' = EXIT_RESTART' if rc == EXIT_RESTART else ''})...",
              flush=True)
        time.sleep(20)

    total = done_count()
    print(f"\nsupervisor done — {total} clips complete "
          f"(+{total - start_done} this session)")


if __name__ == "__main__":
    main()
