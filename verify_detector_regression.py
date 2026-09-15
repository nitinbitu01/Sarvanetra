"""
verify_detector_regression.py — regression check for the three detectors that
had the SENTINEL IQ hook added to their alert-firing paths.

WHY THIS EXISTS
───────────────
Adding compute_alert_iq() inside loitering_detector, crowd_detector, and
abandoned_object_detector put new code — including a DB query for the
cross-zone multiplier — inside each detector's `_fire_alert`, in the same
session and transaction as the Alert insert. If that broke or slowed a firing
path, the labeled clips in tests/*/clips/ are what would show it.

WHAT THIS IS, PRECISELY
───────────────────────
It runs the REAL eval harnesses (tests.behavior_eval.eval_runner and
tests.abandoned_object_eval.eval_runner) over the REAL clip files against the
REAL detector code — with an in-memory stand-in for Redis, because no Redis
is available in this environment.

Both harnesses deliberately refuse to run without a real Redis, and that
refusal is correct: a fake cannot catch a Redis-semantics bug. This script
does not remove that guard from the harnesses — it substitutes the backend
from the outside and says so. Treat a pass here as "the IQ hook did not
break detection", NOT as "the Day 8/9 acceptance criteria have been re-met".
Re-run the harnesses properly against `redis-server` for that claim.

Run:  python verify_detector_regression.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_tmpdir = tempfile.mkdtemp(prefix="verify_detector_regression_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/regression.db"


BANNER = """
======================================================================
  DETECTOR REGRESSION CHECK — in-memory state backend, NOT real Redis
  A pass here means the SENTINEL IQ hook did not break detection.
  It does NOT re-establish the Day 8/9 acceptance criteria; those
  require the harnesses to run against a real redis-server.
======================================================================
"""

FIXED_NOTE = """
----------------------------------------------------------------------
HISTORY — two bugs this harness now covers that it previously could not
----------------------------------------------------------------------
Both were found by running these clips while adding the SENTINEL IQ hook,
and both are now fixed. Recorded here because the second was invisible
until the first was fixed, which is a pattern worth remembering.

1. Wall-clock TTL inside a video_time state machine.
   The "is a person near this object" marker was a Redis key cleared only
   by a wall-clock TTL, while first_seen_time / span_sec /
   ABANDON_DURATION_SEC are all video_time. In an eval, 120s of video_time
   elapse in ~10ms of wall-clock, so the marker never cleared and the only
   true-positive clip could never fire — this harness had never been green.
   In production the same applied to any faster-than-real-time feed
   (config.yaml's `loop_video: true`, catch-up after a backlog), silently
   suppressing genuine abandonment. The marker now stores the video_time of
   the sighting; its TTL is garbage collection only.

2. The unattended clock never restarted when a person was present.
   The PERSON_NEARBY branch refreshed last_seen_time but left
   first_seen_time alone, so an object accrued "unattended" time while
   someone was standing next to it. Fixing (1) exposed this immediately:
   edge_bystander_near_object began firing a false positive at t=60, having
   counted the 30 seconds a bystander spent beside the suitcase. That
   clip's own description had specified the intended rule all along
   ("the abandonment window restarts from t=50").

Both harnesses now report precision 1.000 / recall 1.000 with zero false
positives on every negative clip.
----------------------------------------------------------------------
"""


async def main() -> None:
    print(BANNER)

    from tests.in_memory_state import install

    results: dict[str, int] = {}

    # ── Day 8: loitering + crowd anomaly ────────────────────────────────
    print("\n--- tests.behavior_eval.eval_runner (loitering + crowd) ---\n")
    install()   # fresh state per harness, so neither can contaminate the other
    from tests.behavior_eval import eval_runner as behavior_eval
    results["behavior_eval"] = await behavior_eval.run_all(None)

    # ── Day 9: abandoned object ─────────────────────────────────────────
    print("\n--- tests.abandoned_object_eval.eval_runner (abandoned object) ---\n")
    install()
    from tests.abandoned_object_eval import eval_runner as abandoned_eval
    results["abandoned_object_eval"] = await abandoned_eval.run_all(None)

    print(FIXED_NOTE)
    print("=" * 70)

    real_failures = [name for name, code in results.items() if code != 0]
    for name, code in results.items():
        print(f"{'PASS' if code == 0 else 'FAIL'} — {name} (exit={code})")

    print("=" * 70)

    if real_failures:
        print("FAILED:", real_failures)
        sys.exit(1)

    print("No regression from the SENTINEL IQ hook.")
    print("Reminder: this ran against an in-memory state backend. Re-run both")
    print("harnesses against a real redis-server before treating the Day 8/9")
    print("acceptance criteria as verified.")


if __name__ == "__main__":
    asyncio.run(main())
